from typing import Tuple

import tensorflow as tf
from opennmt import END_OF_SENTENCE_ID
from opennmt.utils.decoding import BeamSearch, BestSampler, DecodingResult, GreedySearch, _penalize_token
from opennmt.utils.misc import shape_list


class DictionaryGuidedBeamSearch(BeamSearch):
    def __init__(
        self, dict_trg_words: tf.RaggedTensor, beam_size: int, length_penalty: float = 0, coverage_penalty: float = 0
    ):
        super().__init__(beam_size, length_penalty=length_penalty, coverage_penalty=coverage_penalty)
        self.dict_trg_words = dict_trg_words.to_tensor()
        self.dict_trg_words_length = tf.shape(self.dict_trg_words)[2]

    def initialize(self, start_ids, attention_size=None):
        batch_size = tf.shape(start_ids)[0]
        start_ids, finished, initial_log_probs, extra_vars = super().initialize(start_ids, attention_size)

        extra_vars["cur_dict_trg_words"] = tf.zeros(
            (batch_size * self.beam_size, self.dict_trg_words_length), dtype=tf.int64
        )
        extra_vars["cur_aligned_src_ids"] = tf.sparse.from_dense(
            tf.zeros((batch_size * self.beam_size, 0), dtype=tf.int64)
        )
        return start_ids, finished, initial_log_probs, extra_vars

    def step(self, step, sampler, log_probs, cum_log_probs, finished, state=None, attention=None, **kwargs):
        cur_dict_trg_words: tf.Tensor = kwargs["cur_dict_trg_words"]
        cur_aligned_src_ids = tf.sparse.SparseTensor = kwargs["cur_aligned_src_ids"]
        log_probs, cur_dict_trg_words, cur_aligned_src_ids = tf.cond(
            self.dict_trg_words_length > 0,
            true_fn=lambda: self.update_log_probs_from_dict(
                log_probs, attention, cur_dict_trg_words, cur_aligned_src_ids
            ),
            false_fn=lambda: (log_probs, cur_dict_trg_words, cur_aligned_src_ids),
        )

        word_ids, cum_log_probs, finished, state, extra_vars = super().step(
            step, sampler, log_probs, cum_log_probs, finished, state, attention, **kwargs
        )
        extra_vars["cur_dict_trg_words"] = cur_dict_trg_words
        extra_vars["cur_aligned_src_ids"] = cur_aligned_src_ids
        return word_ids, cum_log_probs, finished, state, extra_vars

    def update_log_probs_from_dict(
        self,
        log_probs: tf.Tensor,
        attention: tf.Tensor,
        cur_dict_trg_words: tf.Tensor,
        cur_aligned_src_ids: tf.sparse.SparseTensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        attn = tf.reshape(attention, [-1, self.beam_size, tf.shape(attention)[1]])
        aligned_src_ids = tf.math.argmax(attn, axis=2)
        dict_trg_words: tf.Tensor = tf.gather(self.dict_trg_words, aligned_src_ids, axis=1, batch_dims=1)
        dict_trg_words = tf.reshape(dict_trg_words, (-1, self.dict_trg_words_length))
        aligned_src_ids = tf.reshape(aligned_src_ids, (-1, 1)) + 1
        intersection = tf.sets.intersection(aligned_src_ids, cur_aligned_src_ids)
        intersection_sizes = tf.sets.size(intersection)
        nonzero_count = tf.math.count_nonzero(cur_dict_trg_words, axis=1, dtype=tf.int32)
        use_dict_trg_words = tf.reshape((intersection_sizes + nonzero_count) == 0, (-1, 1))
        cur_dict_trg_words = tf.where(use_dict_trg_words, dict_trg_words, cur_dict_trg_words)

        first_word: tf.Tensor = cur_dict_trg_words[:, :1]
        indices: tf.Tensor = tf.concat(
            [tf.reshape(tf.range(tf.shape(first_word)[0], dtype=tf.int64), (-1, 1)), first_word], axis=1
        )
        indices = tf.boolean_mask(indices, tf.reduce_min(indices[:, 1:], axis=1) > 0)
        log_probs = tf.cond(
            tf.shape(indices)[0] > 0,
            true_fn=lambda: tf.tensor_scatter_nd_update(log_probs, indices, tf.zeros(tf.shape(indices)[0])),
            false_fn=lambda: log_probs,
        )

        cur_dict_trg_words = cur_dict_trg_words[:, 1:]
        cur_dict_trg_words = tf.concat(
            [cur_dict_trg_words, tf.zeros((tf.shape(cur_dict_trg_words)[0], 1), dtype=tf.int64)], axis=1
        )
        cur_aligned_src_ids = tf.sets.union(aligned_src_ids, cur_aligned_src_ids)
        return log_probs, cur_dict_trg_words, cur_aligned_src_ids


def dynamic_decode(
    symbols_to_logits_fn,
    start_ids,
    end_id=END_OF_SENTENCE_ID,
    initial_state=None,
    decoding_strategy=None,
    sampler=None,
    maximum_iterations=None,
    minimum_iterations=0,
    attention_history=False,
    attention_size=None,
):
    if initial_state is None:
        initial_state = {}
    if decoding_strategy is None:
        decoding_strategy = GreedySearch()
    if sampler is None:
        sampler = BestSampler()

    def _cond(step, finished, state, inputs, outputs, attention, cum_log_probs, extra_vars):
        return tf.reduce_any(tf.logical_not(finished))

    def _body(step, finished, state, inputs, outputs, attention, cum_log_probs, extra_vars):
        # Get log probs from the model.
        result = symbols_to_logits_fn(inputs, step, state)
        logits, state = result[0], result[1]
        attn = result[2] if len(result) > 2 else None
        logits = tf.cast(logits, tf.float32)

        # Penalize or force EOS.
        batch_size, vocab_size = shape_list(logits)
        eos_max_prob = tf.one_hot(
            tf.fill([batch_size], end_id),
            vocab_size,
            on_value=logits.dtype.max,
            off_value=logits.dtype.min,
        )
        logits = tf.cond(
            step < minimum_iterations,
            true_fn=lambda: _penalize_token(logits, end_id),
            false_fn=lambda: tf.where(
                tf.broadcast_to(tf.expand_dims(finished, -1), tf.shape(logits)),
                x=eos_max_prob,
                y=logits,
            ),
        )
        log_probs = tf.nn.log_softmax(logits)

        # Run one decoding strategy step.
        (output, next_cum_log_probs, finished, state, extra_vars,) = decoding_strategy.step(
            step,
            sampler,
            log_probs,
            cum_log_probs,
            finished,
            state=state,
            attention=attn,
            **extra_vars,
        )

        # Update loop vars.
        if attention_history:
            if attn is None:
                raise ValueError("attention_history is set but the model did not return attention")
            attention = attention.write(step, tf.cast(attn, tf.float32))
        outputs = outputs.write(step, output)
        cum_log_probs = tf.where(finished, x=cum_log_probs, y=next_cum_log_probs)
        finished = tf.logical_or(finished, tf.equal(output, end_id))
        return (
            step + 1,
            finished,
            state,
            output,
            outputs,
            attention,
            cum_log_probs,
            extra_vars,
        )

    start_ids = tf.convert_to_tensor(start_ids)
    ids_dtype = start_ids.dtype
    start_ids = tf.cast(start_ids, tf.int32)
    start_ids, finished, initial_log_probs, extra_vars = decoding_strategy.initialize(
        start_ids, attention_size=attention_size
    )
    step = tf.constant(0, dtype=tf.int32)
    outputs = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    attention = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    _, _, state, _, outputs, attention, log_probs, extra_vars = tf.while_loop(
        _cond,
        _body,
        loop_vars=(
            step,
            finished,
            initial_state,
            start_ids,
            outputs,
            attention,
            initial_log_probs,
            extra_vars,
        ),
        shape_invariants=(
            step.shape,
            finished.shape,
            tf.nest.map_structure(_get_shape_invariants, initial_state),
            start_ids.shape,
            tf.TensorShape(None),
            tf.TensorShape(None),
            initial_log_probs.shape,
            tf.nest.map_structure(_get_shape_invariants, extra_vars),
        ),
        parallel_iterations=1,
        maximum_iterations=maximum_iterations,
    )

    ids, attention, lengths = decoding_strategy.finalize(
        outputs,
        end_id,
        attention=attention if attention_history else None,
        **extra_vars,
    )
    if attention is not None:
        attention = attention[:, :, :-1]  # Ignore attention for </s>.
    log_probs = tf.reshape(log_probs, [-1, decoding_strategy.num_hypotheses])
    ids = tf.cast(ids, ids_dtype)
    return DecodingResult(ids=ids, lengths=lengths, log_probs=log_probs, attention=attention, state=state)


def _get_shape_invariants(tensor):
    """Returns the shape of the tensor but sets middle dims to None."""
    if isinstance(tensor, tf.TensorArray):
        shape = None
    elif hasattr(tensor, "dense_shape"):
        # sparse tensor
        shape = [len(tensor.shape)]
    else:
        shape = tensor.shape.as_list()
        for i in range(1, len(shape) - 1):
            shape[i] = None
    return tf.TensorShape(shape)

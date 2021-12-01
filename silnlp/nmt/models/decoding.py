from typing import Tuple

import tensorflow as tf
from opennmt import END_OF_SENTENCE_ID
from opennmt.utils.decoding import (
    BeamSearch,
    BestSampler,
    DecodingResult,
    GreedySearch,
    _gather_from_word_indices,
    _penalize_token,
    _reorder_state,
)
from opennmt.utils.misc import shape_list


class DictionaryGuidedBeamSearch(BeamSearch):
    def __init__(
        self,
        src_entry_indices: tf.Tensor,
        trg_entries: tf.Tensor,
        beam_size: int,
        length_penalty: float = 0,
        coverage_penalty: float = 0,
        tflite_output_size=None,
    ):
        super().__init__(
            beam_size,
            length_penalty=length_penalty,
            coverage_penalty=coverage_penalty,
            tflite_output_size=tflite_output_size,
        )
        self.src_entry_indices = src_entry_indices
        self.trg_entries = trg_entries
        self.trg_variant_length = tf.shape(self.trg_entries)[-1]

    def initialize(self, start_ids, attention_size=None):
        batch_size = tf.shape(start_ids)[0]
        start_ids, finished, initial_log_probs, extra_vars = super().initialize(start_ids, attention_size)

        extra_vars["active_trg_entries"] = tf.zeros(
            (batch_size * self.beam_size, self.trg_variant_length), dtype=tf.int32
        )
        extra_vars["used_trg_entry_indices"] = tf.sparse.from_dense(
            tf.zeros((batch_size * self.beam_size, 0), dtype=tf.int32)
        )
        return start_ids, finished, initial_log_probs, extra_vars

    def step(self, step, sampler, log_probs, cum_log_probs, finished, state=None, attention=None, **kwargs):
        parent_ids = kwargs["parent_ids"]
        sequence_lengths = kwargs["sequence_lengths"]
        active_trg_entries: tf.Tensor = kwargs["active_trg_entries"]
        used_trg_entry_indices = tf.sparse.SparseTensor = kwargs["used_trg_entry_indices"]

        if self.trg_variant_length > 0:
            log_probs, step_trg_entries, step_trg_entry_indices = self._update_log_probs_from_dict(
                log_probs, attention, active_trg_entries, used_trg_entry_indices
            )
        else:
            step_trg_entries = tf.zeros((self.beam_size, self.trg_variant_length), dtype=tf.int32)
            step_trg_entry_indices = tf.zeros((self.beam_size, 1), dtype=tf.int32)

        if self.coverage_penalty != 0:
            if attention is None:
                raise ValueError("Coverage penalty is enabled but the model did not " "return an attention vector")
            not_finished = tf.math.logical_not(finished)
            attention *= tf.expand_dims(tf.cast(not_finished, attention.dtype), 1)
            accumulated_attention = kwargs["accumulated_attention"] + attention
        else:
            accumulated_attention = None

        # Compute scores from log probabilities.
        vocab_size = log_probs.shape[-1]
        total_probs = log_probs + tf.expand_dims(cum_log_probs, 1)  # Add current beam probability.
        scores = self._get_scores(
            total_probs,
            sequence_lengths,
            finished,
            accumulated_attention=accumulated_attention,
        )
        scores = tf.reshape(scores, [-1, self.beam_size * vocab_size])
        total_probs = tf.reshape(total_probs, [-1, self.beam_size * vocab_size])

        # Sample predictions.
        sample_ids, sample_scores = sampler(scores, num_samples=self.beam_size)
        cum_log_probs = tf.reshape(_gather_from_word_indices(total_probs, sample_ids), [-1])
        sample_ids = tf.reshape(sample_ids, [-1])
        sample_scores = tf.reshape(sample_scores, [-1])

        # Resolve beam origin and word ids.
        word_ids = sample_ids % vocab_size
        beam_ids = sample_ids // vocab_size
        beam_indices = (tf.range(tf.shape(word_ids)[0]) // self.beam_size) * self.beam_size + beam_ids

        # Update sequence_length of unfinished sequence.
        sequence_lengths = tf.where(finished, x=sequence_lengths, y=sequence_lengths + 1)

        # Update state and flags.
        finished = tf.gather(finished, beam_indices)
        sequence_lengths = tf.gather(sequence_lengths, beam_indices)

        if self.trg_variant_length > 0:
            active_trg_entries, used_trg_entry_indices = self._reorder_dict_state(
                word_ids, beam_indices, step_trg_entries, step_trg_entry_indices, used_trg_entry_indices
            )

        parent_ids = parent_ids.write(step, beam_ids)
        extra_vars = {
            "parent_ids": parent_ids,
            "sequence_lengths": sequence_lengths,
            "active_trg_entries": active_trg_entries,
            "used_trg_entry_indices": used_trg_entry_indices,
        }
        if accumulated_attention is not None:
            extra_vars["accumulated_attention"] = tf.gather(accumulated_attention, beam_indices)
        if state is not None:
            state = _reorder_state(state, beam_indices, reorder_flags=self._state_reorder_flags)
        return word_ids, cum_log_probs, finished, state, extra_vars

    def _update_log_probs_from_dict(
        self,
        log_probs: tf.Tensor,
        attention: tf.Tensor,
        active_trg_entries: tf.Tensor,
        used_trg_entry_indices: tf.sparse.SparseTensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        attn = tf.reshape(attention, [-1, self.beam_size, tf.shape(attention)[1]])
        aligned_src_indices = tf.math.argmax(attn, axis=2)
        step_trg_entry_indices = tf.gather(self.src_entry_indices, aligned_src_indices, axis=1, batch_dims=1)
        step_trg_entries: tf.Tensor = tf.gather(self.trg_entries, step_trg_entry_indices, axis=1, batch_dims=1)

        batch_size = tf.shape(step_trg_entries)[0]
        step_trg_entries = tf.reshape(step_trg_entries, (batch_size * self.beam_size, -1, self.trg_variant_length))

        step_trg_entry_indices = tf.reshape(step_trg_entry_indices, (-1, 1))
        intersection = tf.sets.intersection(step_trg_entry_indices, used_trg_entry_indices)
        intersection_sizes = tf.sets.size(intersection)
        nonzero_count = tf.math.count_nonzero(active_trg_entries, axis=1, dtype=tf.int32)
        use_step_trg_entries = tf.reshape((intersection_sizes + nonzero_count) == 0, (-1, 1, 1))

        step_trg_entries = tf.where(use_step_trg_entries, step_trg_entries, tf.expand_dims(active_trg_entries, 1))

        first_token: tf.Tensor = step_trg_entries[:, :, 0]
        _, best_first_token_indices = tf.math.top_k(tf.gather(log_probs, first_token, batch_dims=1))
        step_trg_entries = tf.squeeze(tf.gather(step_trg_entries, best_first_token_indices, batch_dims=1), axis=1)
        best_first_token = step_trg_entries[:, :1]
        indices: tf.Tensor = tf.concat(
            [tf.reshape(tf.range(tf.shape(best_first_token)[0], dtype=tf.int32), (-1, 1)), best_first_token], axis=1
        )
        indices = tf.boolean_mask(indices, tf.reduce_min(indices[:, 1:], axis=1) > 0)
        if tf.shape(indices)[0] > 0:
            num_updates = tf.shape(indices)[0]
            vocab_size = tf.shape(log_probs)[1]
            log_probs_preserve = tf.gather_nd(log_probs, indices)
            log_probs = tf.tensor_scatter_nd_update(
                log_probs, indices[:, :1], tf.fill((num_updates, vocab_size), -float("inf"))
            )
            log_probs = tf.tensor_scatter_nd_update(log_probs, indices, log_probs_preserve)
        return log_probs, step_trg_entries, step_trg_entry_indices

    @tf.autograph.experimental.do_not_convert
    def _reorder_dict_state(
        self,
        word_ids: tf.Tensor,
        beam_indices: tf.Tensor,
        step_trg_entries: tf.Tensor,
        step_trg_entry_indices: tf.Tensor,
        used_trg_entry_indices: tf.sparse.SparseTensor,
    ) -> Tuple[tf.Tensor, tf.sparse.SparseTensor]:
        step_trg_entries = tf.gather(step_trg_entries, beam_indices)
        step_trg_entry_indices = tf.gather(step_trg_entry_indices, beam_indices)
        used_trg_entry_indices = tf.gather(tf.sparse.to_dense(used_trg_entry_indices), beam_indices)
        keep_dict_trg_entries = tf.reshape(step_trg_entries[:, 0] == word_ids, (-1, 1))
        step_trg_entries = tf.where(keep_dict_trg_entries, step_trg_entries, tf.zeros_like(step_trg_entries))
        step_trg_entry_indices = tf.where(
            keep_dict_trg_entries, step_trg_entry_indices, tf.zeros_like(step_trg_entry_indices)
        )

        active_trg_entries = step_trg_entries[:, 1:]
        active_trg_entries = tf.concat(
            [active_trg_entries, tf.zeros((tf.shape(active_trg_entries)[0], 1), dtype=tf.int32)], axis=1
        )
        used_trg_entry_indices = tf.sets.union(step_trg_entry_indices, used_trg_entry_indices)
        return active_trg_entries, used_trg_entry_indices


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
    tflite_output_size=None,
):
    if initial_state is None:
        initial_state = {}
    if decoding_strategy is None:
        decoding_strategy = GreedySearch()
    if sampler is None:
        sampler = BestSampler()
    is_tflite_run = tflite_output_size is not None

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
        outputs = outputs.write(step, output)
        if attention_history:
            if attn is None:
                raise ValueError("attention_history is set but the model did not return attention")
            attention = attention.write(step, tf.cast(attn, tf.float32))
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

    if is_tflite_run:
        output_shape = tf.TensorShape(None)
        outputs = tf.TensorArray(
            tf.int32,
            size=tflite_output_size,
            dynamic_size=False,
            element_shape=output_shape,
        )
        attn_shape = tf.TensorShape(None)
        attention = tf.TensorArray(
            tf.float32,
            size=tflite_output_size,
            dynamic_size=False,
            element_shape=attn_shape,
        )
        maximum_iterations = tflite_output_size if maximum_iterations > tflite_output_size else maximum_iterations
    else:
        output_shape = tf.TensorShape(None)
        outputs = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        attn_shape = tf.TensorShape(None)
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
            output_shape,
            attn_shape,
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

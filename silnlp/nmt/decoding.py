import tensorflow as tf
from opennmt.utils.decoding import BeamSearch


@tf.function
def replace_word(
    cur_dict_trg_word: tf.Tensor, dict_trg_word: tf.Tensor, intersection_size: tf.Tensor
) -> tf.RaggedTensor:
    if tf.shape(cur_dict_trg_word)[0] > 0:
        return cur_dict_trg_word
    return dict_trg_word if intersection_size == 0 else cur_dict_trg_word


class DictionaryGuidedBeamSearch(BeamSearch):
    def __init__(
        self, dict_trg_words: tf.RaggedTensor, beam_size: int, length_penalty: float = 0, coverage_penalty: float = 0
    ):
        super().__init__(beam_size, length_penalty=length_penalty, coverage_penalty=coverage_penalty)
        self.dict_trg_words = dict_trg_words

    def initialize(self, start_ids, attention_size=None):
        batch_size = tf.shape(start_ids)[0].numpy()
        start_ids, finished, initial_log_probs, extra_vars = super().initialize(start_ids, attention_size)
        extra_vars["cur_dict_trg_words"] = tf.ragged.constant([[]] * (self.beam_size * batch_size), dtype=tf.int64)
        extra_vars["cur_aligned_src_ids"] = tf.sparse.from_dense(
            tf.zeros((batch_size, self.beam_size, 0), dtype=tf.int64)
        )
        return start_ids, finished, initial_log_probs, extra_vars

    @tf.function
    def step(self, step, sampler, log_probs, cum_log_probs, finished, state=None, attention=None, **kwargs):
        cur_dict_trg_words: tf.RaggedTensor = kwargs["cur_dict_trg_words"]
        cur_aligned_src_ids = tf.Tensor = kwargs["cur_aligned_src_ids"]
        attn = tf.reshape(attention, [-1, self.beam_size, tf.shape(attention)[1]])
        aligned_src_ids = tf.math.argmax(attn, axis=2)
        dict_trg_words: tf.RaggedTensor = tf.gather(self.dict_trg_words, aligned_src_ids, axis=1, batch_dims=1)
        aligned_src_ids = tf.reshape(aligned_src_ids, (-1, self.beam_size, 1))
        intersection_sizes = tf.sets.size(tf.sets.intersection(aligned_src_ids, cur_aligned_src_ids))

        cur_dict_trg_words: tf.RaggedTensor = tf.map_fn(
            lambda x: replace_word(x[0], x[1], x[2]),
            (cur_dict_trg_words, dict_trg_words.values, tf.reshape(intersection_sizes, (-1))),
            fn_output_signature=tf.RaggedTensorSpec(ragged_rank=0, dtype=tf.int64),
        )

        first_word: tf.Tensor = cur_dict_trg_words[:, :1].to_tensor(default_value=-1)
        if tf.shape(first_word)[1] == 1:
            indices: tf.Tensor = tf.concat(
                [tf.reshape(tf.range(tf.shape(first_word)[0], dtype=tf.int64), (-1, 1)), first_word], axis=1
            )
            indices = tf.boolean_mask(indices, tf.reduce_min(indices, axis=1) >= 0)
            log_probs = tf.tensor_scatter_nd_update(log_probs, indices, tf.zeros(tf.shape(indices)[0]))
        word_ids, cum_log_probs, finished, state, extra_vars = super().step(
            step, sampler, log_probs, cum_log_probs, finished, state, attention, **kwargs
        )

        cur_dict_trg_words = cur_dict_trg_words[:, 1:]
        extra_vars["cur_dict_trg_words"] = cur_dict_trg_words

        cur_aligned_src_ids = tf.sets.union(aligned_src_ids, cur_aligned_src_ids)
        extra_vars["cur_aligned_src_ids"] = cur_aligned_src_ids

        return word_ids, cum_log_probs, finished, state, extra_vars

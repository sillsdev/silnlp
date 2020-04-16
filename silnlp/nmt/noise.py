import opennmt
import tensorflow as tf


class WordDropout(opennmt.data.Noise):
    def __init__(self, dropout, skip_first_word=False):
        self.dropout = dropout
        self.skip_first_word = skip_first_word

    def _apply(self, words):
        if self.dropout == 0:
            return tf.identity(words)
        num_words = tf.shape(words, out_type=tf.int64)[0]
        keep_mask = opennmt.data.noise.random_mask([num_words], 1 - self.dropout)
        if self.skip_first_word:
            indices = tf.constant([[0]])
            updates = tf.constant([True])
            keep_mask = tf.tensor_scatter_nd_update(keep_mask, indices, updates)
        keep_ind = tf.where(keep_mask)
        # Keep at least one word.
        keep_ind = tf.cond(
            tf.equal(tf.shape(keep_ind)[0], 1 if self.skip_first_word else 0),
            true_fn=lambda: self._get_rand_index(num_words),
            false_fn=lambda: tf.squeeze(keep_ind, -1),
        )
        return tf.gather(words, keep_ind)

    def _get_rand_index(self, num_words):
        if self.skip_first_word:
            keep_ind = tf.random.uniform([1], minval=1, maxval=num_words, dtype=tf.int64)
            keep_ind = tf.concat([tf.constant([0], dtype=tf.int64), keep_ind], 0)
        else:
            keep_ind = tf.random.uniform([1], maxval=num_words, dtype=tf.int64)
        return keep_ind

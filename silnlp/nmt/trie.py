from typing import List, Optional

import numpy as np
import tensorflow as tf
from scipy.sparse import dok_matrix


class Trie:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.num_states = 0
        self.build_matrix: Optional[dok_matrix] = dok_matrix((1, self.vocab_size), dtype=np.int32)
        self.build_states: Optional[List[tf.Tensor]] = [tf.constant([], dtype=tf.int32)]

        self.data: Optional[tf.Tensor] = None
        self.indices: Optional[tf.Tensor] = None
        self.indptr: Optional[tf.Tensor] = None
        self.states: Optional[tf.RaggedTensor] = None

    def add(self, ids: tf.Tensor, value: tf.Tensor) -> None:
        if self.build_matrix is None or self.build_states is None:
            raise RuntimeError("The trie is immutable.")

        value = tf.cast(value, tf.int32)
        np_ids = ids.numpy()
        cur_state: int = 0
        for i, id in enumerate(np_ids):
            next_state: int = self.build_matrix[cur_state, id]
            if next_state == 0:
                num_states = len(self.build_states)
                self.build_matrix.resize((num_states + 1, self.vocab_size))
                cur_value = tf.constant([], dtype=tf.int32) if i < len(np_ids) - 1 else value
                self.build_states.append(cur_value)
                self.num_states += 1
                next_state = num_states
                self.build_matrix[cur_state, id] = next_state
            elif i == len(np_ids) - 1:
                self.build_states[next_state] = value
            cur_state = next_state

    def compile(self) -> None:
        if self.build_matrix is None:
            return
        csc_matrix = self.build_matrix.tocsc()
        self.build_matrix = None
        self.data = tf.constant(csc_matrix.data)
        self.indices = tf.constant(csc_matrix.indices)
        self.indptr = tf.constant(csc_matrix.indptr)

        self.states = tf.ragged.stack(self.build_states)
        self.build_states = None

    def longest_prefix(self, ids: tf.Tensor) -> tf.Tensor:
        if self.data is None or self.indices is None or self.indptr is None or self.states is None:
            raise RuntimeError("The trie must be compiled.")

        cur_state = 0
        value: tf.Tensor = tf.constant([], dtype=tf.int32)
        for id in ids:
            tf.autograph.experimental.set_loop_options(shape_invariants=[(value, tf.TensorShape([None]))])
            col_start = self.indptr[id]
            col_end = self.indptr[id + 1]
            if col_start == col_end:
                break
            col_indices = tf.reshape(self.indices[col_start:col_end], (-1, 1))
            col_data = self.data[col_start:col_end]
            col = tf.scatter_nd(col_indices, col_data, tf.constant([self.num_states]))
            cur_state = col[cur_state]
            if cur_state == 0:
                break
            else:
                cur_value = self.states[cur_state]
                if tf.shape(cur_value)[0] != 0:
                    value = cur_value
        return value

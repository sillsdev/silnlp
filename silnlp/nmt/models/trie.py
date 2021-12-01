from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from scipy.sparse import dok_matrix


class Trie:
    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size
        self._num_states = 0
        self._build_matrix: Optional[dok_matrix] = dok_matrix((1, self._vocab_size), dtype=np.int32)
        self._build_states: Optional[List[tf.RaggedTensor]] = [
            tf.RaggedTensor.from_tensor(tf.zeros((0, 0), dtype=tf.int32), row_splits_dtype=tf.int32)
        ]
        self._build_ref_ids: Optional[List[tf.Tensor]] = [tf.convert_to_tensor([], dtype=tf.int32)]
        self._build_ref_lookup: Optional[Dict[str, int]] = {"": 1}

        self._data: Optional[tf.Tensor] = None
        self._indices: Optional[tf.Tensor] = None
        self._indptr: Optional[tf.Tensor] = None
        self._states: Optional[tf.RaggedTensor] = None
        self._ref_ids: Optional[tf.RaggedTensor] = None
        self._ref_ids_lookup: Optional[tf.lookup.StaticHashTable] = None

    @property
    def empty(self) -> bool:
        return self._num_states == 0

    def add(self, ids: tf.Tensor, values: List[tf.Tensor], refs: List[str]) -> None:
        if (
            self._build_matrix is None
            or self._build_states is None
            or self._build_ref_ids is None
            or self._build_ref_lookup is None
        ):
            raise RuntimeError("The trie is immutable.")

        ref_ids_list: List[int] = [1]
        for ref in refs:
            ref_id = self._build_ref_lookup.get(ref)
            if ref_id is None:
                ref_id = len(self._build_ref_lookup) + 1
                self._build_ref_lookup[ref] = ref_id
            ref_ids_list.append(ref_id)
        ref_ids = tf.convert_to_tensor(ref_ids_list, dtype=tf.int32)

        value = tf.cast(tf.ragged.stack(values), tf.int32)
        np_ids = ids.numpy()
        cur_state: int = 0
        for i, id in enumerate(np_ids):
            next_state: int = self._build_matrix[cur_state, id]
            if next_state == 0:
                num_states = len(self._build_states)
                self._build_matrix.resize((num_states + 1, self._vocab_size))
                if i < len(np_ids) - 1:
                    cur_value = tf.RaggedTensor.from_tensor(tf.zeros((0, 0), dtype=tf.int32), row_splits_dtype=tf.int32)
                    cur_ref_ids = tf.convert_to_tensor([], dtype=tf.int32)
                else:
                    cur_value = value
                    cur_ref_ids = ref_ids
                self._build_states.append(cur_value)
                self._build_ref_ids.append(cur_ref_ids)
                self._num_states += 1
                next_state = num_states
                self._build_matrix[cur_state, id] = next_state
            elif i == len(np_ids) - 1:
                self._build_states[next_state] = value
                self._build_ref_ids[next_state] = ref_ids
            cur_state = next_state

    def compile(self) -> None:
        if (
            self._build_matrix is None
            or self._build_states is None
            or self._build_ref_ids is None
            or self._build_ref_lookup is None
        ):
            return

        csc_matrix = self._build_matrix.tocsc()
        self._build_matrix = None
        self._data = tf.constant(csc_matrix.data)
        self._indices = tf.constant(csc_matrix.indices)
        self._indptr = tf.constant(csc_matrix.indptr)

        self._states = tf.ragged.stack(self._build_states)
        self._build_states = None
        self._ref_ids = tf.ragged.stack(self._build_ref_ids)
        self._build_ref_ids = None

        refs = tf.convert_to_tensor(list(self._build_ref_lookup.keys()), dtype=tf.string)
        ref_ids = tf.convert_to_tensor(list(self._build_ref_lookup.values()), dtype=tf.int32)
        initializer = tf.lookup.KeyValueTensorInitializer(refs, ref_ids)
        self._ref_ids_lookup = tf.lookup.StaticHashTable(initializer, default_value=0)
        self._build_ref_lookup = None

    def get_ref_id(self, ref: tf.RaggedTensor) -> tf.RaggedTensor:
        if self._ref_ids_lookup is None:
            raise RuntimeError("The trie is immutable.")
        ref_id = self._ref_ids_lookup.lookup(ref)
        return tf.ragged.boolean_mask(ref_id, ref_id != 0)

    def longest_prefix(self, ids: tf.Tensor, ref_id: tf.Tensor) -> Tuple[tf.RaggedTensor, tf.Tensor]:
        if (
            self._data is None
            or self._indices is None
            or self._indptr is None
            or self._states is None
            or self._ref_ids is None
        ):
            raise RuntimeError("The trie must be compiled.")

        value: tf.RaggedTensor = tf.RaggedTensor.from_tensor(
            tf.zeros((0, 0), dtype=tf.int32), row_splits_dtype=tf.int32
        )
        prefix_len = 0
        cur_state = 0
        ref_id = tf.expand_dims(ref_id, axis=0)
        i = 0
        for id in ids:
            tf.autograph.experimental.set_loop_options(shape_invariants=[(value, tf.TensorShape((None, None)))])
            col_start = self._indptr[id]
            col_end = self._indptr[id + 1]
            if col_start == col_end:
                break
            col_indices = tf.reshape(self._indices[col_start:col_end], (-1, 1))
            col_data = self._data[col_start:col_end]
            col = tf.scatter_nd(col_indices, col_data, tf.constant([self._num_states]))
            cur_state = col[cur_state]
            if cur_state == 0:
                break
            else:
                cur_value: tf.RaggedTensor = self._states[cur_state]
                if cur_value.nrows() != 0:
                    entry_ref_ids = self._ref_ids[cur_state]
                    match = False
                    if tf.size(entry_ref_ids) == 1:
                        match = True
                    else:
                        entry_ref_ids = tf.expand_dims(entry_ref_ids, axis=0)
                        matched = tf.sets.intersection(entry_ref_ids, ref_id)
                        if tf.size(matched) > 0:
                            match = True
                    if match:
                        value = cur_value
                        prefix_len = i + 1
            i += 1
        return value, prefix_len

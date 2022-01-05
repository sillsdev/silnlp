import tensorflow as tf
from opennmt.data.dataset import make_datasets
from opennmt.inputters import WordEmbedder


class SILSourceWordEmbedder(WordEmbedder):
    def make_dataset(self, data_file, training=None):
        if not isinstance(data_file, list):
            return super().make_dataset(data_file, training=training)

        text_dataset = super().make_dataset(data_file[0], training=training)
        if training:
            return text_dataset
        ref_dataset = make_datasets(tf.data.TextLineDataset, data_file[1])
        return tf.data.Dataset.zip((text_dataset, ref_dataset))

    def get_dataset_size(self, data_file):
        if not isinstance(data_file, list):
            return super().get_dataset_size(data_file)
        return super().get_dataset_size(data_file[0])

    def make_features(self, element=None, features=None, training=None):
        if element is None or not isinstance(element, tuple):
            features = super().make_features(element=element, features=features, training=training)
            if not training and "ref" not in features:
                features["ref"] = tf.constant([""], dtype=tf.string)
                features["ref_length"] = tf.shape(features["ref"])[0]
        else:
            features = super().make_features(element=element[0], features=features, training=training)
            if not training and "ref" not in features:
                if tf.strings.length(element[1]) == 0:
                    ref = tf.constant([""], dtype=tf.string)
                else:
                    ref = get_all_verse_refs(element[1])
                features["ref"] = ref
                features["ref_length"] = tf.shape(ref)[0]
        return features

    def input_signature(self):
        signature = super().input_signature()
        signature["ref"] = tf.TensorSpec([None], tf.string)
        signature["ref_length"] = tf.TensorSpec([], tf.int32)
        return signature


def get_all_verse_refs(ref: tf.Tensor) -> tf.Tensor:
    ref_parts = tf.strings.split(ref, sep=":", maxsplit=2)
    # split on sequence separators
    seq_parts = tf.strings.split(ref_parts[1], sep=",")
    # split on range separators
    range_parts = tf.strings.split(seq_parts, sep="-")
    # strip off segment letters
    range_parts = tf.strings.regex_replace(range_parts, "[a-z]", "")
    range_parts = tf.strings.to_number(range_parts, out_type=tf.int32)
    range_parts = range_parts.to_tensor(default_value=-1, shape=(None, 2))

    # fill in range numbers
    indices = tf.where(range_parts[:, 1] == -1)
    gather_indices = tf.pad(indices, [[0, 0], [0, 1]], constant_values=0)
    updates = tf.gather_nd(range_parts, gather_indices)
    update_indices = tf.pad(indices, [[0, 0], [0, 1]], constant_values=1)
    range_parts = tf.tensor_scatter_nd_update(range_parts, update_indices, updates)
    range_parts = tf.ragged.range(range_parts[:, 0], range_parts[:, 1] + 1)

    # flatten to 1-D tensor
    range_parts = range_parts.merge_dims(0, -1)
    range_parts = tf.strings.as_string(range_parts)
    # build tensor of simple refs
    return ref_parts[0] + ":" + range_parts

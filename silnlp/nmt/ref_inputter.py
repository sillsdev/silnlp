import tensorflow as tf
from opennmt.data.dataset import make_datasets
from opennmt.inputters import Inputter


class RefInputter(Inputter):
    def make_dataset(self, data_file, training=None):
        return make_datasets(tf.data.TextLineDataset, data_file)

    def input_signature(self):
        return {"ref": tf.TensorSpec((0,), tf.string)}

    def make_features(self, element=None, features=None, training=None):
        if features is None:
            features = {}
        if "ref" in features:
            return features
        features["ref"] = tf.convert_to_tensor(element, dtype=tf.string)
        return features

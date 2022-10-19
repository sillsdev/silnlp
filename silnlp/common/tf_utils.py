import logging
import os
from typing import Dict

import tensorflow as tf

_PYTHON_TO_TENSORFLOW_LOGGING_LEVEL: Dict[int, int] = {
    logging.CRITICAL: 3,
    logging.ERROR: 2,
    logging.WARNING: 1,
    logging.INFO: 0,
    logging.DEBUG: 0,
    logging.NOTSET: 0,
}


def set_tf_log_level(log_level: int = logging.INFO) -> None:
    tf.get_logger().setLevel(log_level)
    # Do not display warnings from TensorFlow C++, because of spurious "PredictCost()" errors.
    # See https://github.com/tensorflow/tensorflow/issues/50575.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def enable_memory_growth() -> None:
    gpus = tf.config.list_physical_devices(device_type="GPU")
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, enable=True)


def enable_eager_execution() -> None:
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

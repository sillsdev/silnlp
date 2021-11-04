import logging

# Initialize logger
LOGGER = logging.getLogger("silnlp")
LOGGER.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

# This is to prevent double logging on tensorflow: https://stackoverflow.com/questions/33662648/tensorflow-causes-logging-messages-to-double
import tensorflow as tf

logger = tf.get_logger()
logger.propagate = False
# This is to prevent double logging on tensorflow:
# https://stackoverflow.com/questions/33662648/tensorflow-causes-logging-messages-to-double
import tensorflow as tf

logger = tf.get_logger()
logger.propagate = False

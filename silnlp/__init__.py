import logging
import os

# Initialize logger
LOGGER = logging.getLogger("silnlp")
LOGGER.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

os.system("nvidia-smi")

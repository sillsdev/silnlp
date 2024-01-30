import logging
import os
import pathlib
import sys

# Initialize logger
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)


source_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.insert(0, source_path)

os.environ["SIL_NLP_DATA_PATH"] = source_path

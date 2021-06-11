import sys
import pathlib
import os

source_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.insert(0, source_path)

os.environ["SIL_NLP_DATA_PATH"] = source_path

import helper
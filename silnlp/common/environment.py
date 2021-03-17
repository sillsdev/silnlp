import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path.home() / r".silnlp"


def get_data_dir() -> Path:
    sil_nlp_data_path = os.getenv("SIL_NLP_DATA_PATH")
    if sil_nlp_data_path is not None:
        return Path(sil_nlp_data_path)
    gutenberg_path = Path(r"G:/Shared drives/Gutenberg")
    if gutenberg_path.exists():
        return gutenberg_path
    return ROOT_DIR / r"data"


# Root data directory
DATA_DIR = get_data_dir()

# Paratext directories
PT_DIR = DATA_DIR / r"Paratext"
PT_UNZIPPED_DIR = PT_DIR / r"Paratext.unzipped"
PT_PREPROCESSED_DIR = PT_DIR / r"Paratext.preprocessed"
PT_BIBLICAL_TERMS_LISTS_DIR = PT_DIR / r"BiblicalTermsLists"

ALIGN_DIR = DATA_DIR / r"Alignment"
ALIGN_GOLD_STANDARDS_DIR = ALIGN_DIR / r"Gold Standards"
ALIGN_EXPERIMENTS_DIR = ALIGN_DIR / r"Experiments"

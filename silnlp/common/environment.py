import os
import logging
from pathlib import Path

from s3path import S3Path
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

load_dotenv()

ROOT_DIR = Path.home() / ".silnlp"


class MTPath(S3Path):
    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        pass


def get_ml_path(filename):
    if filename.startswith("s3"):
        mtpath = MTPath(filename[4:])  # strip off 's3:/'
        if mtpath is None:
            LOGGER.error(
                "s3 bucket did not initialize correctly from env. variable SIL_NLP_DATA_PATH: " + sil_nlp_data_path
            )
        return mtpath
    else:
        return Path(filename)


def get_data_dir() -> Path:
    sil_nlp_data_path = os.getenv("SIL_NLP_DATA_PATH")
    if sil_nlp_data_path is not None:
        return Path(sil_nlp_data_path)
    try:
        s3_aqua_path = MTPath("/aqua-ml-data")
        if s3_aqua_path.isdir():
            return s3_aqua_path
    except:
        pass
    gutenberg_path = Path("G:/Shared drives/Gutenberg")
    if gutenberg_path.is_dir():
        return gutenberg_path
    return ROOT_DIR / "data"


# Root data directory
DATA_DIR = get_data_dir()

# Paratext directories
PT_DIR = DATA_DIR / "Paratext"
PT_PROJECTS_DIR = PT_DIR / "projects"
PT_TERMS_DIR = PT_DIR / "terms"

MT_DIR = DATA_DIR / "MT"
MT_CORPORA_DIR = MT_DIR / "corpora"
MT_TERMS_DIR = MT_DIR / "terms"
MT_SCRIPTURE_DIR = MT_DIR / "scripture"
MT_EXPERIMENTS_DIR = MT_DIR / "experiments"

ALIGN_DIR = DATA_DIR / "Alignment"
ALIGN_GOLD_DIR = ALIGN_DIR / "gold"
ALIGN_EXPERIMENTS_DIR = ALIGN_DIR / "experiments"

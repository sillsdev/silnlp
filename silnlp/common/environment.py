import os
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path.home() / ".silnlp"

ASSETS_DIR = Path(__file__).parent.parent / "assets"


def get_data_dir() -> Path:
    sil_nlp_data_path = os.getenv("SIL_NLP_DATA_PATH")
    if sil_nlp_data_path is not None:
        temp_path = Path(sil_nlp_data_path)
        if temp_path.is_dir():
            LOGGER.info(f"Using workspace: {sil_nlp_data_path} as per environment variable SIL_NLP_DATA_PATH.")
            return Path(sil_nlp_data_path)
        else:
            raise Exception(
                f"The path defined by environment variable SIL_NLP_DATA_PATH ({sil_nlp_data_path}) is not a directory."
            )
    auqa_ml_path = Path("G:/Shared drives/AQUA")
    if auqa_ml_path.is_dir():
        LOGGER.info(
            f"Using workspace: {auqa_ml_path}.  To change the workspace, set the environment variable SIL_NLP_DATA_PATH."
        )
        return auqa_ml_path
    gutenberg_path = Path("G:/Shared drives/Gutenberg")
    if gutenberg_path.is_dir():
        LOGGER.info(
            f"Using workspace: {gutenberg_path}.  To change the workspace, set the environment variable SIL_NLP_DATA_PATH."
        )
        return gutenberg_path
    data_dir = ROOT_DIR / "data"
    LOGGER.info(
        f"Using workspace: {data_dir}.  To change the workspace, set the environment variable SIL_NLP_DATA_PATH."
    )
    return data_dir


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

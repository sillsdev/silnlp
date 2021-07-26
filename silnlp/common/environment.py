import os
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger(__name__)


class SilNlpEnv:
    def __init__(self):
        self.root_dir = Path.home() / ".silnlp"
        self.assets_dir = Path(__file__).parent.parent / "assets"

        # Root data directory
        self.set_data_dir()

    def set_data_dir(self, DATA_DIR: Path = None):
        if DATA_DIR is None:
            DATA_DIR = self.resolve_data_dir()

        self.data_dir = Path(DATA_DIR)

        # Paratext directories
        self.set_paratext_dir()
        self.set_machine_translation_dir()
        self.set_alignment_dir()

    def set_paratext_dir(self, PT_DIR: Path = None):
        if PT_DIR is not None:
            self.pt_dir = Path(PT_DIR)
        elif hasattr(self, "pt_dir"):
            # it is already initialized
            return
        else:
            self.pt_dir = self.data_dir / "Paratext"
        self.pt_projects_dir = self.pt_dir / "projects"
        self.pt_terms_dir = self.pt_dir / "terms"

    def set_machine_translation_dir(self, MT_DIR: Path = None):
        if MT_DIR is not None:
            self.mt_dir = Path(MT_DIR)
        elif hasattr(self, "mt_dir"):
            # it is already initialized
            return
        else:
            self.mt_dir = self.data_dir / "MT"
        self.mt_corpora_dir = self.mt_dir / "corpora"
        self.mt_terms_dir = self.mt_dir / "terms"
        self.mt_scripture_dir = self.mt_dir / "scripture"
        self.mt_experiments_dir = self.mt_dir / "experiments"

    def set_alignment_dir(self, ALIGN_DIR: Path = None):
        if ALIGN_DIR is not None:
            self.align_dir = Path(ALIGN_DIR)
        elif hasattr(self, "align_dir"):
            # it is already initialized
            return
        else:
            self.align_dir = self.data_dir / "Alignment"
        self.align_gold_dir = self.align_dir / "gold"
        self.align_experiments_dir = self.align_dir / "experiments"

    def resolve_data_dir(self) -> Path:
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

        aqua_path = Path("G:/Shared drives/AQUA")
        if aqua_path.is_dir():
            LOGGER.info(
                f"Using workspace: {aqua_path}.  To change the workspace, set the environment variable SIL_NLP_DATA_PATH."
            )
            return aqua_path

        gutenberg_path = Path("G:/Shared drives/Gutenberg")
        if gutenberg_path.is_dir():
            LOGGER.info(
                f"Using workspace: {gutenberg_path}.  To change the workspace, set the environment variable SIL_NLP_DATA_PATH."
            )
            return gutenberg_path

        data_root = self.root_dir / "data"
        LOGGER.info(
            f"Using workspace: {data_root}.  To change the workspace, set the environment variable SIL_NLP_DATA_PATH."
        )
        return data_root


SIL_NLP_ENV = SilNlpEnv()

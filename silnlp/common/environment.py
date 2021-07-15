import os
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger(__name__)


class SilNlpEnv:
    def __init__(self):
        self._ROOT_DIR = Path.home() / ".silnlp"
        self._ASSETS_DIR = Path(__file__).parent.parent / "assets"

        # Root data directory
        self.set_data_dir()

    def set_data_dir(self, DATA_DIR: Path = None):
        if DATA_DIR is None:
            DATA_DIR = self.resolve_data_dir()

        self._DATA_DIR = Path(DATA_DIR)

        # Paratext directories
        self.set_paratext_dir()
        self.set_machine_translation_dir()
        self.set_alignment_dir()

    def set_paratext_dir(self, PT_DIR: Path = None):
        if PT_DIR is not None:
            self._PT_DIR = Path(PT_DIR)
        elif hasattr(self, "_PT_DIR"):
            # it is already initialized
            return
        else:
            self._PT_DIR = self._DATA_DIR / "Paratext"
        self._PT_PROJECTS_DIR = self._PT_DIR / "projects"
        self._PT_TERMS_DIR = self._PT_DIR / "terms"

    def set_machine_translation_dir(self, MT_DIR: Path = None):
        if MT_DIR is not None:
            self._MT_DIR = Path(MT_DIR)
        elif hasattr(self, "_MT_DIR"):
            # it is already initialized
            return
        else:
            self._MT_DIR = self._DATA_DIR / "MT"
        self._MT_CORPORA_DIR = self._MT_DIR / "corpora"
        self._MT_TERMS_DIR = self._MT_DIR / "terms"
        self._MT_SCRIPTURE_DIR = self._MT_DIR / "scripture"
        self._MT_EXPERIMENTS_DIR = self._MT_DIR / "experiments"

    def set_alignment_dir(self, ALIGN_DIR: Path = None):
        if ALIGN_DIR is not None:
            self._ALIGN_DIR = Path(ALIGN_DIR)
        elif hasattr(self, "_ALIGN_DIR"):
            # it is already initialized
            return
        else:
            self._ALIGN_DIR = self._DATA_DIR / "Alignment"
        self._ALIGN_GOLD_DIR = self._ALIGN_DIR / "gold"
        self._ALIGN_EXPERIMENTS_DIR = self._ALIGN_DIR / "experiments"

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

        data_root = self._ROOT_DIR / "data"
        LOGGER.info(
            f"Using workspace: {data_root}.  To change the workspace, set the environment variable SIL_NLP_DATA_PATH."
        )
        return data_root


SNE = SilNlpEnv()

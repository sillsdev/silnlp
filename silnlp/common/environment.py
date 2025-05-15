import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path, PurePath
from platform import system, uname
from typing import Callable, Iterable, List, Optional, Sequence, Union

from dotenv import load_dotenv

load_dotenv()

import atexit

# Suppress urllib3 warnings about unverified HTTPS requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOGGER = logging.getLogger(__name__)


class SilNlpEnv:
    def __init__(self):
        atexit.register(check_transfers)
        self.root_dir = Path.home() / ".silnlp"
        self.assets_dir = Path(__file__).parent.parent / "assets"

        self.set_data_dir()

    def set_data_dir(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = self.resolve_data_dir()

        self.data_dir = pathify(data_dir)

        # Paratext directories
        self.set_paratext_dir()
        self.set_machine_translation_dir()
        self.set_alignment_dir()

    def set_paratext_dir(self, pt_dir: Optional[Path] = None):
        if pt_dir is not None:
            self.pt_dir = pathify(pt_dir)
        elif hasattr(self, "pt_dir"):
            # it is already initialized
            return
        elif os.getenv("SIL_NLP_PT_DIR"):
            self.pt_dir = self.data_dir / os.getenv("SIL_NLP_PT_DIR", "")
        else:
            self.pt_dir = self.data_dir / "Paratext"
        self.pt_terms_dir = self.pt_dir / "terms"
        self.pt_projects_dir = self.pt_dir / "projects"

    def set_machine_translation_dir(self, mt_dir: Optional[Path] = None):
        if mt_dir is not None:
            self.mt_dir = pathify(mt_dir)
        elif hasattr(self, "mt_dir"):
            # it is already initialized
            return
        elif os.getenv("SIL_NLP_MT_DIR"):
            self.mt_dir = self.data_dir / os.getenv("SIL_NLP_MT_DIR", "")
        else:
            self.mt_dir = self.data_dir / "MT"
        self.mt_corpora_dir = self.mt_dir / "corpora"
        if os.getenv("SIL_NLP_MT_TERMS_DIR"):
            self.mt_terms_dir = self.data_dir / os.getenv("SIL_NLP_MT_TERMS_DIR")
        else:
            self.mt_terms_dir = self.mt_dir / "terms"
        if os.getenv("SIL_NLP_MT_SCRIPTURE_DIR"):
            self.mt_scripture_dir = self.data_dir / os.getenv("SIL_NLP_MT_SCRIPTURE_DIR")
        else:
            self.mt_scripture_dir = self.mt_dir / "scripture"

        self.mt_experiments_dir = self.mt_dir / "experiments"

    def set_alignment_dir(self, align_dir: Optional[Path] = None):
        if align_dir is not None:
            self.align_dir = pathify(align_dir)
        elif hasattr(self, "align_dir"):
            # it is already initialized
            return
        else:
            self.align_dir = self.data_dir / "Alignment"
        self.align_gold_dir = self.align_dir / "gold"
        self.align_experiments_dir = self.align_dir / "experiments"

    def resolve_data_dir(self) -> Path:
        sil_nlp_data_path = os.getenv("SIL_NLP_DATA_PATH", default="")
        if sil_nlp_data_path != "":
            temp_path = Path(sil_nlp_data_path)
            if temp_path.is_dir():
                LOGGER.info(f"Using workspace: {sil_nlp_data_path} as per environment variable SIL_NLP_DATA_PATH.")
                return Path(sil_nlp_data_path)
            else:
                LOGGER.warning(
                    f"The path defined by environment variable SIL_NLP_DATA_PATH ({sil_nlp_data_path}) is not a "
                    + "local directory."
                )

        gutenberg_path = Path("G:/Shared drives/Gutenberg")
        if gutenberg_path.is_dir():
            LOGGER.info(f"Using workspace: {gutenberg_path}.")
            return gutenberg_path

        raise FileExistsError("No valid path exists")


def check_transfers() -> None:
    # check if rclone is running or if CHECK_TRANSFERS is set
    if (
        not os.path.exists("/root/rclone_log.txt")
        or os.getenv("SIL_NLP_DATA_PATH", default="") == ""
        or os.getenv("CHECK_TRANSFERS", default=0) == 0
    ):
        return
    LOGGER.info("Checking rclone transfer progress.")
    time.sleep(60)  # wait for the latest poll interval
    while True:
        with open("/root/rclone_log.txt", "r", encoding="utf-8") as log_file:
            log_lines = log_file.readlines()
        transfers_complete = False
        for line in reversed(log_lines):
            if "vfs cache: cleaned" in line:
                transfers_complete = bool(re.match(r".*in use 0, to upload 0, uploading 0,.*", line))
                break
        if transfers_complete:
            LOGGER.info(line)
            LOGGER.info("rclone transfers are complete.")
            break
        else:
            LOGGER.info(line)
            LOGGER.info("rclone transfers are still in progress. Waiting one minute.")
        time.sleep(60)


def try_n_times(func: Callable, n=10):
    for i in range(n):
        try:
            func()
            break
        except Exception as e:
            if i < n - 1:
                LOGGER.exception(f"Failed {i+1} of {n} times.  Retrying.")
                time.sleep(2**i)
            else:
                raise e


def pathify(path: Path) -> Path:
    # If it does not act like a path, make it a path
    if isinstance(path, Path):
        return path
    else:
        return Path(path)


def wsl_path(win_path: Union[str, Path]) -> str:
    win_path_str = os.path.normpath(win_path).replace("\\", "\\\\")
    args: List[str] = []
    if system() == "Windows":
        args.append("wsl")
    args.extend(["wslpath", "-a", win_path_str])
    result = subprocess.run(args, capture_output=True, encoding="utf-8")
    return result.stdout.strip()


def is_wsl() -> bool:
    return "microsoft-standard" in uname().release


def get_env_path(name: str, default: str = ".") -> str:
    path = os.getenv(name, default)
    if is_wsl() and (re.match(r"^[a-zA-Z]:", path) is not None or "\\" in path):
        return wsl_path(path)
    return path


SIL_NLP_ENV = SilNlpEnv()

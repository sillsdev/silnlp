import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from platform import system, uname
from typing import Callable, List, Optional

from dotenv import load_dotenv

load_dotenv()

import atexit

# Suppress urllib3 warnings about unverified HTTPS requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOGGER = logging.getLogger(__name__)


class SilNlpEnv:
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        pt_dir: Optional[Path] = None,
        mt_dir: Optional[Path] = None,
        align_dir: Optional[Path] = None,
    ):
        atexit.register(self.delete_path)
        atexit.register(check_transfers)
        self.path_to_delete: Optional[Path] = None
        self._data_dir = self._resolve_data_dir(data_dir)
        self._pt_dir = self._resolve_paratext_dir(pt_dir)
        self._mt_dir = self._resolve_mt_dir(mt_dir)
        self._align_dir = self._resolve_align_dir(align_dir)

    def _resolve_data_dir(self, data_dir: Optional[Path] = None) -> Path:
        if data_dir is not None:
            return data_dir

        sil_nlp_data_path = os.getenv("SIL_NLP_DATA_PATH", default="")
        if sil_nlp_data_path != "":
            temp_path = Path(sil_nlp_data_path)
            if temp_path.is_dir():
                LOGGER.info(f"Using workspace: {sil_nlp_data_path} as per environment variable SIL_NLP_DATA_PATH.")
                return Path(sil_nlp_data_path)

            LOGGER.warning(
                f"The path defined by environment variable SIL_NLP_DATA_PATH ({sil_nlp_data_path}) is not a "
                + "local directory."
            )

        raise FileExistsError("No valid path exists")

    def _resolve_paratext_dir(self, pt_dir: Optional[Path] = None) -> Path:
        return self._resolve_relative_or_absolute_path(pt_dir, env_var_dir="SIL_NLP_PT_DIR", default_subdir="Paratext")

    def _resolve_mt_dir(self, mt_dir: Optional[Path] = None) -> Path:
        return self._resolve_relative_or_absolute_path(mt_dir, env_var_dir="SIL_NLP_MT_DIR", default_subdir="MT")

    def _resolve_align_dir(self, align_dir: Optional[Path] = None) -> Path:
        return self._resolve_relative_or_absolute_path(align_dir, env_var_dir=None, default_subdir="Alignment")

    def _resolve_relative_or_absolute_path(
        self, custom_dir: Optional[Path], env_var_dir: Optional[str], default_subdir: str
    ) -> Path:
        # Explicit paths have precedence over environment variables.
        if custom_dir is not None:
            if custom_dir.is_absolute():
                return custom_dir
            return self._data_dir / custom_dir

        if env_var_dir is not None:
            env_value = os.getenv(env_var_dir)
            if env_value:
                return self._data_dir / env_value

        return self._data_dir / default_subdir

    @property
    def assets_dir(self) -> Path:
        return Path(__file__).parent.parent / "assets"

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def pt_dir(self) -> Path:
        return self._pt_dir

    @property
    def pt_terms_dir(self) -> Path:
        return self.pt_dir / "terms"

    @property
    def pt_projects_dir(self) -> Path:
        return self.pt_dir / "projects"

    @property
    def mt_dir(self) -> Path:
        return self._mt_dir

    @property
    def mt_corpora_dir(self) -> Path:
        return self.mt_dir / "corpora"

    @property
    def mt_terms_dir(self) -> Path:
        mt_terms_dir = os.getenv("SIL_NLP_MT_TERMS_DIR")
        if mt_terms_dir:
            return self._data_dir / mt_terms_dir
        return self._mt_dir / "terms"

    @property
    def mt_scripture_dir(self) -> Path:
        mt_scripture_dir = os.getenv("SIL_NLP_MT_SCRIPTURE_DIR")
        if mt_scripture_dir:
            return self._data_dir / mt_scripture_dir
        return self._mt_dir / "scripture"

    @property
    def mt_experiments_dir(self) -> Path:
        return self.mt_dir / "experiments"

    @property
    def align_dir(self) -> Path:
        return self._align_dir

    @property
    def align_gold_dir(self) -> Path:
        return self.align_dir / "gold"

    @property
    def align_experiments_dir(self) -> Path:
        return self.align_dir / "experiments"

    @staticmethod
    def create_standard_environment() -> "SilNlpEnv":
        return SilNlpEnv()

    @staticmethod
    def create_environment_with_data_dir(data_dir: Path) -> "SilNlpEnv":
        return SilNlpEnv(data_dir=data_dir)

    @staticmethod
    def create_environment_with_paratext_dir(pt_dir: Path) -> "SilNlpEnv":
        return SilNlpEnv(pt_dir=pt_dir)

    @staticmethod
    def create_environment_with_mt_dir(mt_dir: Path) -> "SilNlpEnv":
        return SilNlpEnv(mt_dir=mt_dir)

    @staticmethod
    def create_environment_with_align_dir(align_dir: Path) -> "SilNlpEnv":
        return SilNlpEnv(align_dir=align_dir)

    @staticmethod
    def create_environment_with_custom_directories(
        data_dir: Optional[Path] = None,
        pt_dir: Optional[Path] = None,
        mt_dir: Optional[Path] = None,
        align_dir: Optional[Path] = None,
    ) -> "SilNlpEnv":
        return SilNlpEnv(data_dir=data_dir, pt_dir=pt_dir, mt_dir=mt_dir, align_dir=align_dir)

    def get_mt_exp_dir(self, exp_name: str) -> Path:
        return self.mt_experiments_dir / exp_name

    def delete_path_on_exit(self, path: Path) -> None:
        self.path_to_delete = path

    def delete_path(self) -> None:
        if self.path_to_delete and self.path_to_delete.is_dir():
            shutil.rmtree(self.path_to_delete)


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
    transfers_complete = False
    for i in range(7):
        with open("/root/rclone_log.txt", "r", encoding="utf-8") as log_file:
            log_lines = log_file.readlines()
        last_logged_line = ""
        for line in reversed(log_lines):
            if "vfs cache: cleaned" in line:
                transfers_complete = bool(re.match(r".*in use 0, to upload 0, uploading 0,.*", line))
                last_logged_line = line.strip()
                break
        if transfers_complete:
            LOGGER.info(last_logged_line)
            LOGGER.info("rclone transfers are complete.")
            break
        else:
            LOGGER.info(last_logged_line)
            LOGGER.info(f"rclone transfers are still in progress. Waiting another minute. Attempt {i+1} of 7.")
        time.sleep(60)
    if not transfers_complete:
        LOGGER.warning("rclone transfers could not be completed. Some data may be lost or corrupted.")


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


def wsl_path(win_path: Path) -> str:
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
        return wsl_path(Path(path))
    return path


SIL_NLP_ENV = SilNlpEnv()

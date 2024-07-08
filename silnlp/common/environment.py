import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path, PurePath
from platform import system, uname
from typing import Callable, Iterable, List, Optional, Sequence, Union

import boto3
from dotenv import load_dotenv
from s3path import S3Path

load_dotenv()

LOGGER = logging.getLogger(__name__)

boto3.setup_default_session(
     region_name='us-east-1',
     aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
     aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))

class SilNlpEnv:
    def __init__(self):
        self.root_dir = Path.home() / ".silnlp"
        self.assets_dir = Path(__file__).parent.parent / "assets"
        self.is_bucket = False

        # Root data directory
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
        if self.is_bucket and os.environ.get("SIL_NLP_PT_DIR", None) is None:
            sil_nlp_cache_dir = os.getenv("SIL_NLP_CACHE_PROJECT_DIR")
            if sil_nlp_cache_dir is not None:
                temp_path = Path(sil_nlp_cache_dir)
                if not hasattr(self, "pt_projects_dir"):
                    if temp_path.is_dir():
                        LOGGER.info(
                            f"Using cache dir: {sil_nlp_cache_dir} as per environment variable "
                            + "SIL_NLP_CACHE_PROJECT_DIR."
                        )
                        self.pt_projects_dir = temp_path
                    else:
                        raise Exception(
                            "The path in SIL_NLP_CACHE_PROJECT_DIR does not exist.  Create it first: "
                            + sil_nlp_cache_dir
                        )
            else:
                self.pt_projects_dir = Path(tempfile.TemporaryDirectory().name)
                self.pt_projects_dir.mkdir()
        else:
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
        if self.is_bucket:
            sil_nlp_cache_dir = os.getenv("SIL_NLP_CACHE_EXPERIMENT_DIR")
            if sil_nlp_cache_dir is not None:
                temp_path = Path(sil_nlp_cache_dir)
                if not hasattr(self, "mt_experiments_dir"):
                    if temp_path.is_dir():
                        LOGGER.info(
                            f"Using cache dir: {sil_nlp_cache_dir} as per environment variable "
                            + "SIL_NLP_CACHE_EXPERIMENT_DIR."
                        )
                        self.mt_experiments_dir = temp_path
                    else:
                        raise Exception(
                            "The path in SIL_NLP_CACHE_EXPERIMENT_DIR does not exist.  Create it first: "
                            + sil_nlp_cache_dir
                        )
            else:
                self.mt_experiments_dir = Path(tempfile.TemporaryDirectory().name)
                self.mt_experiments_dir.mkdir()
        else:
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
        self.is_bucket = False
        sil_nlp_data_path = get_env_path("SIL_NLP_DATA_PATH", default="")
        if sil_nlp_data_path != "":
            temp_path = Path(sil_nlp_data_path)
            if temp_path.is_dir():
                LOGGER.info(f"Using workspace: {sil_nlp_data_path} as per environment variable SIL_NLP_DATA_PATH.")
                return Path(sil_nlp_data_path)
            else:
                temp_s3_path = S3Path(sil_nlp_data_path)
                if temp_s3_path.is_dir():
                    LOGGER.info(
                        f"Using s3 workspace: {sil_nlp_data_path} as per environment variable SIL_NLP_DATA_PATH."
                    )
                    self.is_bucket = True
                    return S3Path(sil_nlp_data_path)
                else:
                    raise Exception(
                        f"The path defined by environment variable SIL_NLP_DATA_PATH ({sil_nlp_data_path}) is not a "
                        + "real or s3 directory."
                    )

        gutenberg_path = Path("G:/Shared drives/Gutenberg")
        if gutenberg_path.is_dir():
            LOGGER.info(
                f"Using workspace: {gutenberg_path}.  To change the workspace, set the environment variable "
                + "SIL_NLP_DATA_PATH."
            )
            return gutenberg_path

        s3root = S3Path("/aqua-ml-data")
        if s3root.is_dir():
            LOGGER.info(
                f"Using s3 workspace workspace: {s3root}.  To change the workspace, set the environment variable "
                + "SIL_NLP_DATA_PATH."
            )
            self.is_bucket = True
            return s3root

        raise FileExistsError("No valid path exists")

    def copy_pt_project_from_bucket(self, name: Union[str, Path], patterns: Union[str, Sequence[str]] = []):
        if not self.is_bucket:
            return
        name = str(name)
        pt_projects_path = str(self.data_dir / "Paratext" / "projects") if os.environ.get('SIL_NLP_PT_DIR', None) is not None else str(self.pt_dir.relative_to(self.data_dir) / "projects") + "/"
        name = name.split(pt_projects_path)[-1]
        if len(name) == 0:
            raise Exception(
                f"No Paratext project name is given.  Data still in the cache directory of {self.pt_projects_dir}"
            )
        s3 = boto3.resource("s3")
        data_bucket = s3.Bucket(str(self.data_dir).strip("\\/"))
        len_aqua_path = len(pt_projects_path)
        pt_projects_path = pt_projects_path + name
        objs = list(data_bucket.object_versions.filter(Prefix=pt_projects_path + "/"))
        if len(objs) == 0:
            LOGGER.info("No files found in the bucket under: " + pt_projects_path)
            return
        if isinstance(patterns, str):
            patterns = [patterns]
        for obj in objs:
            rel_path = str(obj.object_key)[len_aqua_path:]
            pure_path = PurePath(rel_path)
            if len(patterns) == 0 or any(pure_path.match(pattern) for pattern in patterns):
                # copy over project files and experiment files
                temp_dest_path = self.pt_projects_dir / rel_path
                temp_dest_path.parent.mkdir(parents=True, exist_ok=True)
                if temp_dest_path.exists():
                    LOGGER.debug("File already exists in local cache: " + rel_path)
                else:
                    LOGGER.info("Downloading " + rel_path)
                    try_n_times(lambda: data_bucket.download_file(obj.object_key, str(temp_dest_path)))
    
    def copy_pt_project_to_bucket(self, name: str, patterns: Union[str, Sequence[str]] = [], overwrite: bool = False):
        if not self.is_bucket:
            return
        name = str(name)
        if len(name) == 0:
            raise Exception(
                f"No Paratext project name is given.  Data still in the temp directory of {self.pt_projects_dir}"
            )
        pt_projects_path = str(self.pt_projects_dir) + "/"
        s3 = boto3.resource("s3")
        data_bucket = s3.Bucket(str(self.data_dir).strip("\\/"))
        temp_folder = str(self.pt_projects_dir / name)
        # we don't need to delete all existing files - it will just overwrite them
        len_projects_dir = len(str(self.pt_projects_dir))
        files_already_in_s3 = set()
        for obj in data_bucket.object_versions.filter(Prefix=pt_projects_path + name):
            files_already_in_s3.add(str(obj.object_key))

        if isinstance(patterns, str):
            patterns = [patterns]
        for root, _, files in os.walk(temp_folder, topdown=False):
            s3_dest_path = str(pt_projects_path + root[len_projects_dir + 1 :].replace("\\", "/"))
            for file in files:
                pure_path = PurePath(file)
                if len(patterns) == 0 or any(pure_path.match(pattern) for pattern in patterns):
                    source_file = os.path.join(root, file)
                    dest_file = s3_dest_path + "/" + file
                    if not overwrite and dest_file in files_already_in_s3:
                        LOGGER.debug("File already exists in S3 bucket: " + dest_file)
                    else:
                        LOGGER.info("Uploading " + dest_file)
                        try_n_times(lambda: data_bucket.upload_file(source_file, dest_file))

    def copy_experiment_from_bucket(self, name: Union[str, Path], patterns: Union[str, Sequence[str]] = []):
        if not self.is_bucket:
            return
        name = str(name)
        experiments_path = str(self.mt_dir.relative_to(self.data_dir) / "experiments") + "/"
        name = name.split(experiments_path)[-1]
        if len(name) == 0:
            raise Exception(
                f"No experiment name is given.  Data still in the cache directory of {self.mt_experiments_dir}"
            )
        s3 = boto3.resource("s3")
        data_bucket = s3.Bucket(str(self.data_dir).strip("\\/"))
        len_aqua_path = len(experiments_path)
        experiment_path = experiments_path + name
        objs = list(data_bucket.object_versions.filter(Prefix=experiment_path + "/"))
        if len(objs) == 0:
            LOGGER.info("No files found in the bucket under: " + experiment_path)
            return
        if isinstance(patterns, str):
            patterns = [patterns]
        for obj in objs:
            rel_path = str(obj.object_key)[len_aqua_path:]
            pure_path = PurePath(rel_path)
            if len(patterns) == 0 or any(pure_path.match(pattern) for pattern in patterns):
                # copy over project files and experiment files
                temp_dest_path = self.mt_experiments_dir / rel_path
                temp_dest_path.parent.mkdir(parents=True, exist_ok=True)
                if temp_dest_path.exists():
                    LOGGER.debug("File already exists in local cache: " + rel_path)
                else:
                    LOGGER.info("Downloading " + rel_path)
                    try_n_times(lambda: data_bucket.download_file(obj.object_key, str(temp_dest_path)))

    def copy_experiment_to_bucket(self, name: str, patterns: Union[str, Sequence[str]] = [], overwrite: bool = False):
        if not self.is_bucket:
            return
        name = str(name)
        if len(name) == 0:
            raise Exception(
                f"No experiment name is given.  Data still in the temp directory of {self.mt_experiments_dir}"
            )
        experiment_path = str(self.mt_dir.relative_to(self.data_dir) / "experiments") + "/"
        s3 = boto3.resource("s3")
        data_bucket = s3.Bucket(str(self.data_dir).strip("\\/"))
        temp_folder = str(self.mt_experiments_dir / name)
        # we don't need to delete all existing files - it will just overwrite them
        len_exp_dir = len(str(self.mt_experiments_dir))
        files_already_in_s3 = set()
        for obj in data_bucket.object_versions.filter(Prefix=experiment_path + name):
            files_already_in_s3.add(str(obj.object_key))

        if isinstance(patterns, str):
            patterns = [patterns]
        for root, _, files in os.walk(temp_folder, topdown=False):
            s3_dest_path = str(experiment_path + root[len_exp_dir + 1 :].replace("\\", "/"))
            for file in files:
                pure_path = PurePath(file)
                if len(patterns) == 0 or any(pure_path.match(pattern) for pattern in patterns):
                    source_file = os.path.join(root, file)
                    dest_file = s3_dest_path + "/" + file
                    if not overwrite and dest_file in files_already_in_s3:
                        LOGGER.debug("File already exists in S3 bucket: " + dest_file)
                    else:
                        LOGGER.info("Uploading " + dest_file)
                        try_n_times(lambda: data_bucket.upload_file(source_file, dest_file))

    def get_source_experiment_path(self, tmp_path: Path) -> str:
        end_of_path = str(tmp_path)[len(str(self.mt_experiments_dir)) :]
        source_path = end_of_path.replace("\\", "/")
        if source_path.startswith("/"):
            source_path = source_path[1:]
        return source_path


def download_if_s3_paths(paths: Iterable[S3Path]) -> List[Path]:
    return_paths = []
    s3_setup = False

    for path in paths:
        if type(path) is not S3Path:
            return_paths.append(path)
        else:
            if not s3_setup:
                temp_root = Path(tempfile.TemporaryDirectory().name)
                temp_root.mkdir()
                s3 = boto3.resource("s3")
                data_bucket = s3.Bucket(str(SIL_NLP_ENV.data_dir).strip("\\/"))
                s3_setup = True
            temp_path = temp_root / path.name
            try_n_times(lambda: data_bucket.download_file(path.key, str(temp_path)))
            return_paths.append(temp_path)
    return return_paths


def try_n_times(func: Callable, n=10):
    for i in range(n):
        try:
            func()
            break
        except Exception as e:
            if i < n - 1:
                LOGGER.exception(f"Failed {i+1} of {n} times.  Retrying.")
                time.sleep(5)
            else:
                raise e


def pathify(path: Path) -> Path:
    # If it does not act like a path, make it a path
    if type(path) in [Path, S3Path]:
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

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
from botocore.config import Config
from dotenv import load_dotenv
from s3path import PureS3Path, S3Path, register_configuration_parameter

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
        self.is_bucket = False
        self.bucket_service = os.getenv("BUCKET_SERVICE", "").lower()

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
        if self.is_bucket:
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
        sil_nlp_data_path = os.getenv("SIL_NLP_DATA_PATH", default="")
        if sil_nlp_data_path != "" and self.bucket_service == "":
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

        self.set_s3_bucket()
        sil_nlp_data_path = f"/{self.bucket.name}"
        s3root = S3Path(sil_nlp_data_path)
        if s3root.is_dir():
            LOGGER.info(f"Using s3 workspace: {s3root}.")
            self.is_bucket = True
            return s3root

        raise FileExistsError("No valid path exists")

    def set_resource(self, bucket_name: str, endpoint_url: str, access_key: str, secret_key: str):
        resource = boto3.resource(
            service_name="s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=generate_s3_config(),
            # Verify is false if endpoint_url is an IP address. Aqua/Cheetah connecting to MinIO need this disabled for now.
            verify=False if re.match(r"https://\d+\.\d+\.\d+\.\d+", endpoint_url) else True,
        )

        bucket = resource.Bucket(bucket_name)
        # Tests the connection to the bucket. Delete is used because it fails fast and is free of api cost from Backblaze.
        bucket.delete_objects(Delete={"Objects": [{"Key": "conn_test_key"}]})
        register_configuration_parameter(PureS3Path("/"), resource=resource)
        self.bucket = bucket

    def set_s3_bucket(self):
        # TEMPORARY: This allows users to still connect to AWS S3 if they have not set up MinIO or B2 yet. This will be removed in the future.
        if self.bucket_service == "aws" or (os.getenv("MINIO_ACCESS_KEY") is None and os.getenv("B2_KEY_ID") is None):
            LOGGER.warning("Support for AWS S3 will soon be removed. Please set up MinIO and/or B2 credentials.")
            resource = boto3.resource(
                service_name="s3",
                config=generate_s3_config(),
            )
            bucket = resource.Bucket("silnlp")
            register_configuration_parameter(PureS3Path("/"), resource=resource)
            self.bucket = bucket
            self.bucket_service = "aws"
            return

        if self.bucket_service == "":
            self.bucket_service = "minio"

        if self.bucket_service not in ["minio", "b2"]:
            LOGGER.warning("BUCKET_SERVICE environment variable must be either 'minio' or 'b2'. Default is 'minio'.")
            self.bucket_service = "minio"
        if self.bucket_service in ["minio"]:
            try:
                LOGGER.info("Trying to connect to MinIO bucket.")
                self.set_resource(
                    "nlp-research",
                    os.getenv("MINIO_ENDPOINT_URL"),
                    os.getenv("MINIO_ACCESS_KEY"),
                    os.getenv("MINIO_SECRET_KEY"),
                )
                LOGGER.info("Connected to MinIO bucket.")
            except Exception as e:
                LOGGER.exception("MinIO connection failed.")
                raise e
        if self.bucket_service in ["b2"]:
            try:
                LOGGER.info("Trying to connect to B2 bucket.")
                self.set_resource(
                    "silnlp",
                    os.getenv("B2_ENDPOINT_URL"),
                    os.getenv("B2_KEY_ID"),
                    os.getenv("B2_APPLICATION_KEY"),
                )
                LOGGER.info("Connected to B2 bucket.")
            except Exception as e:
                LOGGER.exception("B2 connection failed.")
                raise e

    def copy_pt_project_from_bucket(self, name: Union[str, Path], patterns: Union[str, Sequence[str]] = []):
        if not self.is_bucket:
            return
        name = str(name)
        pt_projects_path = str(self.pt_dir.relative_to(self.data_dir) / "projects") + "/"
        name = name.split(pt_projects_path)[-1]
        if len(name) == 0:
            raise Exception(
                f"No paratext project name is given.  Data still in the cache directory of {self.pt_projects_dir}"
            )
        len_silnlp_path = len(pt_projects_path)
        pt_projects_path = pt_projects_path + name
        objs = list(self.bucket.object_versions.filter(Prefix=pt_projects_path + "/"))
        if len(objs) == 0:
            LOGGER.info("No files found in the bucket under: " + pt_projects_path)
            return
        if isinstance(patterns, str):
            patterns = [patterns]
        for obj in objs:
            rel_path = str(obj.object_key)[len_silnlp_path:]
            pure_path = PurePath(rel_path)
            if len(patterns) == 0 or any(pure_path.match(pattern) for pattern in patterns):
                # copy over project files and experiment files
                temp_dest_path = self.pt_projects_dir / rel_path
                temp_dest_path.parent.mkdir(parents=True, exist_ok=True)
                if temp_dest_path.exists():
                    LOGGER.debug("File already exists in local cache: " + rel_path)
                else:
                    LOGGER.info("Downloading " + rel_path)
                    try_n_times(lambda: self.bucket.download_file(obj.object_key, str(temp_dest_path)))

    def copy_experiment_from_bucket(
        self, name: Union[str, Path], patterns: Union[str, Sequence[str]] = [], no_checkpoints: bool = False
    ):
        if not self.is_bucket:
            return
        name = str(name)
        experiments_path = str(self.mt_dir.relative_to(self.data_dir) / "experiments") + "/"
        name = name.split(experiments_path)[-1]
        if len(name) == 0:
            raise Exception(
                f"No experiment name is given.  Data still in the cache directory of {self.mt_experiments_dir}"
            )
        len_silnlp_path = len(experiments_path)
        experiment_path = experiments_path + name
        objs = list(self.bucket.object_versions.filter(Prefix=experiment_path + "/"))
        if len(objs) == 0:
            LOGGER.info("No files found in the bucket under: " + experiment_path)
            return
        if isinstance(patterns, str):
            patterns = [patterns]
        for obj in objs:
            rel_path = str(obj.object_key)[len_silnlp_path:]
            pure_path = PurePath(rel_path)
            if len(patterns) == 0 or any(pure_path.match(pattern) for pattern in patterns):
                # copy over project files and experiment files
                if no_checkpoints:
                    checkpoint_regex = re.compile(
                        r"^MT/experiments/.+/run/(checkpoint.*(pytorch_model\.bin|\.safetensors)$|ckpt.+\.(data-00000-of-00001|index)$)"
                    )
                    if checkpoint_regex.match(str(obj.object_key)):
                        continue
                temp_dest_path = self.mt_experiments_dir / rel_path
                temp_dest_path.parent.mkdir(parents=True, exist_ok=True)
                if temp_dest_path.exists():
                    LOGGER.debug("File already exists in local cache: " + rel_path)
                else:
                    LOGGER.info("Downloading " + rel_path)
                    try_n_times(lambda: self.bucket.download_file(obj.object_key, str(temp_dest_path)))

    def copy_experiment_to_bucket(self, name: str, patterns: Union[str, Sequence[str]] = [], overwrite: bool = False):
        if not self.is_bucket:
            return
        name = str(name)
        if len(name) == 0:
            raise Exception(
                f"No experiment name is given.  Data still in the temp directory of {self.mt_experiments_dir}"
            )
        experiment_path = str(self.mt_dir.relative_to(self.data_dir) / "experiments") + "/"
        temp_folder = str(self.mt_experiments_dir / name)
        # we don't need to delete all existing files - it will just overwrite them
        len_exp_dir = len(str(self.mt_experiments_dir))
        files_already_in_s3 = set()
        for obj in self.bucket.object_versions.filter(Prefix=experiment_path + name):
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
                        try_n_times(lambda: self.bucket.upload_file(source_file, dest_file))

    def get_source_experiment_path(self, tmp_path: Path) -> str:
        end_of_path = str(tmp_path)[len(str(self.mt_experiments_dir)) :]
        source_path = end_of_path.replace("\\", "/")
        if source_path.startswith("/"):
            source_path = source_path[1:]
        return source_path

    def download_if_s3_paths(self, paths: Iterable[S3Path]) -> List[Path]:
        return_paths = []
        s3_setup = False

        for path in paths:
            if type(path) is not S3Path:
                return_paths.append(path)
            else:
                if not s3_setup:
                    temp_root = Path(tempfile.TemporaryDirectory().name)
                    temp_root.mkdir()
                    self.set_s3_bucket()
                    s3_setup = True
                temp_path = temp_root / path.name
                try_n_times(lambda: self.bucket.download_file(path.key, str(temp_path)))
                return_paths.append(temp_path)
        return return_paths


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


def generate_s3_config() -> Config:
    s3_config = Config(
        s3={"addressing_style": "path"},
        retries={"mode": "adaptive", "max_attempts": 10},
        connect_timeout=600,
        read_timeout=600,
    )
    return s3_config


SIL_NLP_ENV = SilNlpEnv()

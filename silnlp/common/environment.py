import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import boto3
from dotenv import load_dotenv
from s3path import S3Path

load_dotenv()

LOGGER = logging.getLogger(__name__)


class SilNlpEnv:
    def __init__(self):
        self.root_dir = Path.home() / ".silnlp"
        self.assets_dir = Path(__file__).parent.parent / "assets"
        self.is_bucket = False

        # Root data directory
        self.set_data_dir()

    def set_data_dir(self, DATA_DIR: Path = None):
        if DATA_DIR is None:
            DATA_DIR = self.resolve_data_dir()

        self.data_dir = pathify(DATA_DIR)

        # Paratext directories
        self.set_paratext_dir()
        self.set_machine_translation_dir()
        self.set_alignment_dir()

    def set_paratext_dir(self, PT_DIR: Path = None):
        if PT_DIR is not None:
            self.pt_dir = pathify(PT_DIR)
        elif hasattr(self, "pt_dir"):
            # it is already initialized
            return
        else:
            self.pt_dir = self.data_dir / "Paratext"
        self.pt_projects_dir = self.pt_dir / "projects"
        self.pt_terms_dir = self.pt_dir / "terms"

    def set_machine_translation_dir(self, MT_DIR: Path = None):
        if MT_DIR is not None:
            self.mt_dir = pathify(MT_DIR)
        elif hasattr(self, "mt_dir"):
            # it is already initialized
            return
        else:
            self.mt_dir = self.data_dir / "MT"
        self.mt_corpora_dir = self.mt_dir / "corpora"

        self.mt_terms_dir = self.mt_dir / "terms"
        self.mt_scripture_dir = self.mt_dir / "scripture"
        if self.is_bucket:
            self.mt_experiments_dir = Path(tempfile.TemporaryDirectory().name)
            self.mt_experiments_dir.mkdir()
        else:
            self.mt_experiments_dir = self.mt_dir / "experiments"

    def set_alignment_dir(self, ALIGN_DIR: Path = None):
        if ALIGN_DIR is not None:
            self.align_dir = pathify(ALIGN_DIR)
        elif hasattr(self, "align_dir"):
            # it is already initialized
            return
        else:
            self.align_dir = self.data_dir / "Alignment"
        self.align_gold_dir = self.align_dir / "gold"
        self.align_experiments_dir = self.align_dir / "experiments"

    def resolve_data_dir(self) -> Path:
        self.is_bucket = False
        sil_nlp_data_path = os.getenv("SIL_NLP_DATA_PATH")
        if sil_nlp_data_path is not None:
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
                        f"The path defined by environment variable SIL_NLP_DATA_PATH ({sil_nlp_data_path}) is not a real or s3 directory."
                    )

        gutenberg_path = Path("G:/Shared drives/Gutenberg")
        if gutenberg_path.is_dir():
            LOGGER.info(
                f"Using workspace: {gutenberg_path}.  To change the workspace, set the environment variable SIL_NLP_DATA_PATH."
            )
            return gutenberg_path

        s3root = S3Path("/aqua-ml-data")
        if s3root.is_dir():
            LOGGER.info(
                f"Using s3 workspace workspace: {s3root}.  To change the workspace, set the environment variable SIL_NLP_DATA_PATH."
            )
            self.is_bucket = True
            return s3root

        raise FileExistsError("No valid path exists")

    def copy_experiment_from_bucket(self, name: str, extensions: Tuple[str] = ("")):
        if not self.is_bucket:
            return
        name = str(name)
        if len(name) == 0:
            raise Exception(
                f"No experiment name is given.  Data still in the temp directory of {self.mt_experiments_dir}"
            )
        s3 = boto3.resource("s3")
        data_bucket = s3.Bucket(str(self.data_dir).strip("\\/"))
        len_aqua_path = len("MT/experiments/")
        proj_name = name.split("/")[0]
        objs = list(data_bucket.object_versions.filter(Prefix="MT/experiments/" + proj_name))
        if len(objs) == 0:
            LOGGER.info("No files found in the bucket under: MT/experiments/" + proj_name)
            return
        for obj in objs:
            rel_path = str(obj.object_key)[len_aqua_path:]
            if rel_path.endswith(extensions):
                rel_folder = "/".join(rel_path.split("/")[:-1])
                if (rel_folder == proj_name) or rel_folder.startswith(name):
                    # copy over project files and experiment files
                    LOGGER.info("Copying from bucket to temp drive: " + rel_path)
                    temp_dest_path = self.mt_experiments_dir / rel_path
                    temp_dest_path.parent.mkdir(parents=True, exist_ok=True)
                    data_bucket.download_file(obj.object_key, str(temp_dest_path))

    def copy_experiment_to_bucket(self, name: str):
        if not self.is_bucket:
            return
        name = str(name)
        if len(name) == 0:
            raise Exception(
                f"No experiment name is given.  Data still in the temp directory of {self.mt_experiments_dir}"
            )
        s3 = boto3.resource("s3")
        data_bucket = s3.Bucket(str(self.data_dir).strip("\\/"))
        temp_folder = str(self.mt_experiments_dir / name)
        # we don't need to delete all existing files - it will just overwrite them
        len_exp_dir = len(str(self.mt_experiments_dir))
        files_already_in_s3 = set()
        for obj in data_bucket.object_versions.filter(Prefix="MT/experiments/" + name):
            files_already_in_s3.add(str(obj.object_key))

        for root, dirs, files in os.walk(temp_folder, topdown=False):
            s3_dest_path = str("MT/experiments/" + root[len_exp_dir + 1 :].replace("\\", "/"))
            for f in files:
                source_file = os.path.join(root, f)
                dest_file = s3_dest_path + "/" + f
                if dest_file in files_already_in_s3:
                    LOGGER.debug(f"{dest_file} already in s3 bucket")
                else:
                    LOGGER.debug(f"adding{dest_file} to s3 bucket")
                    data_bucket.upload_file(source_file, dest_file)


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
            data_bucket.download_file(path.key, str(temp_path))
            return_paths.append(temp_path)
    return return_paths


def download_if_s3_path(path: S3Path) -> Path:
    if type(path) is not S3Path:
        return path
    else:
        temp_root = Path(tempfile.TemporaryDirectory().name)
        temp_root.mkdir()
        s3 = boto3.resource("s3")
        data_bucket = s3.Bucket(str(SIL_NLP_ENV.data_dir).strip("\\/"))
        temp_path = temp_root / path.name
        data_bucket.download_file(path.key, str(temp_path))
        return temp_path


def pathify(path: Path) -> Path:
    # If it does not act like a path, make it a path
    if type(path) in [Path, S3Path]:
        return path
    else:
        return Path(path)


SIL_NLP_ENV = SilNlpEnv()

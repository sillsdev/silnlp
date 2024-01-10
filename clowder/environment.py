import warnings

warnings.filterwarnings("ignore", r"Blowfish")

import datetime
import os
from pathlib import Path
from typing import Any, Optional, Union

import gspread
import gspread_dataframe as gd
import pandas as pd
import s3path
import yaml
from clearml import Task
from oauth2client.service_account import ServiceAccountCredentials
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive, GoogleDriveFile
from pydrive2.files import MediaIoReadable
from status import Status

from clowder.configuration_exception import MissingConfigurationFile
from clowder.consts import (
    ENTRYPOINT_ATTRIBUTE,
    GDRIVE_SCOPE,
    NAME_ATTRIBUTE,
    RESULTS_CLEARML_METRIC_ATTRIBUTE,
    RESULTS_CSVS_ATTRIBUTE,
)
from clowder.investigation import Investigation


class DuplicateInvestigationException(Exception):
    """There is already an investigation in the current context with that name"""


class InvestigationNotFoundError(Exception):
    """No such investigation in the current context"""


class ClowderMeta:
    def __init__(self, meta_filepath: str) -> None:
        self.filepath = meta_filepath
        if not Path.exists(Path("..", ".clowder")):
            os.mkdir("../.clowder")
        if not Path.is_file(Path(self.filepath)):
            data = {"temp": {"investigations": {}}, "current_root": "temp"}
            with open(self.filepath, "w") as f:
                yaml.safe_dump(data, f)
        with open(self.filepath, "r") as f:
            self.data: Any = yaml.safe_load(f)

    def flush(self):
        with open(self.filepath, "w") as f:
            yaml.safe_dump(self.data, f)
        with open(self.filepath, "r") as f:
            self.data: Any = yaml.safe_load(f)


class ClowderEnvironment:
    def __init__(self):
        self.meta = ClowderMeta("../.clowder/clowder.master.meta.yml")
        self.INVESTIGATIONS_GDRIVE_FOLDER = self.root
        try:
            self.GOOGLE_CREDENTIALS_FILE = (
                self._get_env_var("GOOGLE_CREDENTIALS_FILE")
                if os.environ.get("GOOGLE_CREDENTIALS_FILE") is not None
                else "../.clowder/"
                + list(filter(lambda p: "clowder" in p and ".json" in p, os.listdir("../.clowder/")))[
                    0
                ]  # TODO more robust
            )
        except IndexError:
            raise MissingConfigurationFile("No google credentials file found in .clowder directory")
        self.EXPERIMENTS_S3_FOLDER = (
            "/aqua-ml-data/MT/experiments/clowder/"  # self._get_env_var("EXPERIMENTS_S3_FOLDER")
        )
        self._setup_google_drive()
        self.gc = gspread.service_account(filename=Path(self.GOOGLE_CREDENTIALS_FILE))

    @property
    def root(self):
        return self.meta.data["current_root"]

    @root.setter
    def root(self, value):
        self.meta.data["current_root"] = value
        self.meta.flush()

    @property
    def current_meta(self):
        return self.meta.data[self.root]

    @property
    def investigations(self) -> "list[Investigation]":
        return [self.get_investigation(inv_name) for inv_name in self.current_meta["investigations"].keys()]

    def get_investigation(self, investigation_name: str) -> Investigation:
        inv_data = self.current_meta["investigations"].get(investigation_name, None)
        if inv_data is None:
            raise InvestigationNotFoundError(
                f"Investigation {investigation_name} does not exist in the current context"
            )
        return Investigation.from_meta({investigation_name: inv_data})

    def investigation_exists(self, investigation_name: str):
        return investigation_name in self.current_meta["investigations"]

    def add_investigation(self, investigation_name: str, investigation_data: dict):
        self.current_meta["investigations"][investigation_name] = investigation_data
        self.meta.flush()

    def create_investigation(self, investigation_name: str) -> Investigation:
        if self.investigation_exists(investigation_name):
            raise DuplicateInvestigationException(
                f"There is already an investigation with name {investigation_name} in this context"
            )
        folder_id = self._create_gdrive_folder(investigation_name, self.root)
        clowder_log_id = self._write_gdrive_file_in_folder(folder_id, "clowder.log", "")
        clowder_config_yml_id = self._write_gdrive_file_in_folder(folder_id, "config.yml", "", "application/x-yaml")
        sheet = self.gc.create("investigation", folder_id)
        df: pd.DataFrame = pd.DataFrame(
            columns=[NAME_ATTRIBUTE, ENTRYPOINT_ATTRIBUTE, RESULTS_CSVS_ATTRIBUTE, RESULTS_CLEARML_METRIC_ATTRIBUTE]
        )
        gd.set_with_dataframe(sheet.sheet1, df)
        sheet.sheet1.update_title("ExperimentsSetup")
        sheet_id = sheet.id
        experiments_folder_id = self._create_gdrive_folder("experiments", folder_id)
        remote_meta_content: dict = {
            "name": investigation_name,
            "id": folder_id,
            "status": Status.Created.value,
            "experiments_folder_id": experiments_folder_id,
            "clowder_log_id": clowder_log_id,
            "clowder_config_yml_id": clowder_config_yml_id,
            "sheet_id": sheet_id,
        }
        clowder_meta_yml_id = self._write_gdrive_file_in_folder(
            folder_id, "clowder.meta.yml", yaml.safe_dump(remote_meta_content), "application/x-yaml"
        )

        investigation_data: dict = {
            "id": folder_id,
            "status": Status.Created.value,
            "experiments_folder_id": experiments_folder_id,
            "clowder_meta_yml_id": clowder_meta_yml_id,
            "clowder_log_id": clowder_log_id,
            "clowder_config_yml_id": clowder_config_yml_id,
            "sheet_id": sheet_id,
        }
        self.add_investigation(investigation_name, investigation_data)
        return self.get_investigation(investigation_name)

    def _get_clearml_tasks(self, investigation_name: str) -> "dict[str, Union[None,Task]]":
        if "experiments" not in self.current_meta["investigations"][investigation_name]:
            self.current_meta["investigations"][investigation_name]["experiments"] = {}
        experiments = self.current_meta["investigations"][investigation_name]["experiments"]
        tasks = {}
        for experiment_name, obj in experiments.items():
            clearml_id = obj["clearml_id"]
            if clearml_id is None or clearml_id == "unknown":
                continue
            task: Optional[Task] = Task.get_task(task_id=clearml_id)
            tasks[experiment_name] = task
        return tasks

    def track_investigation_by_name(self, investigation_name: str):
        try:
            self.get_investigation(investigation_name)
            raise DuplicateInvestigationException(
                f"Unable to track investigation with name {investigation_name}. An investigation with that name already exists in the current context"
            )
        except InvestigationNotFoundError:
            pass
        files = self._find_investigations(folder_id=self.root, by_name=investigation_name)
        if len(files) == 0:
            raise InvestigationNotFoundError(
                f"No such investigation {investigation_name} in current root folder {self.root}"
            )
        if len(files) > 1:
            raise DuplicateInvestigationException(
                f"Unable to track investigation with name {investigation_name}. Multiple investigations with that name exist in current root folder {self.root}"
            )
        investigation_folder_id = files.pop()
        self._track_investigation_in_folder(investigation_folder_id)

    def track_all_investigations(self):
        self._track_all_investigations_in_folder(self.root)

    def _get_env_var(self, name: str) -> str:
        var = os.environ.get(name)
        assert var is not None, name + " needs to be set."
        return var

    def _setup_google_drive(self):
        gauth = GoogleAuth()
        gauth.auth_method = "service"
        gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
            self.GOOGLE_CREDENTIALS_FILE, scopes=GDRIVE_SCOPE
        )
        self._google_drive = GoogleDrive(gauth)

    def _dict_of_gdrive_files(self, folder_id: str) -> "dict[str, GoogleDriveFile]":
        files = self._google_drive.ListFile({"q": f"trashed=false and '{folder_id}' in parents"}).GetList()
        return {f["title"]: f for f in files}

    def _list_gdrive_files(self, folder_id: str) -> "list[GoogleDriveFile]":
        return list(self._dict_of_gdrive_files(folder_id).values())

    def _read_gdrive_file_as_string(self, file_id: str) -> str:
        return self._read_gdrive_file_as_bytes(file_id).decode("utf-8")

    def _read_gdrive_file_as_bytes(self, file_id: str) -> bytes:
        file = self._google_drive.CreateFile({"id": file_id})
        buffer: MediaIoReadable = file.GetContentIOBuffer()
        b = buffer.read()
        return b if b is not None else b""  # type: ignore

    def _write_gdrive_file_in_folder(
        self, parent_folder_id: str, file_name: str, content: Union[str, bytes], file_type: Optional[str] = None
    ) -> str:
        files = self._dict_of_gdrive_files(parent_folder_id)
        if file_name in files:
            # overwrite the existing folder
            fh = self._google_drive.CreateFile({"id": files[file_name]["id"]})
        else:
            # create a new folder
            fh = self._google_drive.CreateFile(
                {
                    "title": file_name,
                    "parents": [{"id": parent_folder_id}],
                    "mimeType": "text/plain" if file_type is None else file_type,
                }
            )
        fh.SetContentString(content)
        fh.Upload()
        return fh["id"]

    def _delete_gdrive_folder(self, folder_id: str) -> str:
        fh = self._google_drive.CreateFile({"id": folder_id})
        fh.Delete()
        return fh["id"]

    def _create_gdrive_folder(self, folder_name: str, parent_folder_id: str) -> str:
        files = self._dict_of_gdrive_files(parent_folder_id)
        if folder_name in files:
            return files[folder_name]["id"]
        fh = self._google_drive.CreateFile(
            {
                "title": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [{"id": parent_folder_id}],
            }
        )
        fh.Upload()
        return fh["id"]

    def _find_investigations(
        self, folder_id: str, by_name: Optional[str] = None, files_acc: "set[str]" = set()
    ) -> "set[str]":
        files = self._dict_of_gdrive_files(folder_id)
        for filename, file in files.items():
            if filename == "clowder.meta.yml":
                files_acc.add(folder_id)
            if file["mimeType"] == "application/vnd.google-apps.folder":
                if (not by_name) or (file["title"] == by_name):
                    files_acc = files_acc.union(
                        self._find_investigations(file["id"], by_name=by_name, files_acc=files_acc)
                    )
        return files_acc

    def _track_investigation_in_folder(self, folder_id: str):
        files = self._dict_of_gdrive_files(folder_id)
        meta_file = files.get("clowder.meta.yml", None)
        if meta_file is None:
            raise MissingConfigurationFile(f"No clowder.meta.yml file could be found in folder with id {folder_id}")
        remote_meta = yaml.safe_load(self._read_gdrive_file_as_string(meta_file["id"]))
        if "experiments_folder_id" not in remote_meta:
            experiments_folder_id = self._create_gdrive_folder("experiments", folder_id)
        else:
            experiments_folder_id = remote_meta["experiments_folder_id"]

        if "sheet_id" not in remote_meta:
            sheet = self.gc.create("investigation", folder_id)
            sheet.sheet1.update_title("ExperimentsSetup")
            sheet_id = sheet.id
        else:
            sheet_id = remote_meta["sheet_id"]

        if "clowder_log_id" not in remote_meta:
            clowder_log_id = self._write_gdrive_file_in_folder(folder_id, "clowder.log", "")
        else:
            clowder_log_id = remote_meta["clowder_log_id"]

        if "clowder_config_yml_id" not in remote_meta:
            clowder_config_yml_id = self._write_gdrive_file_in_folder(folder_id, "config.yml", "", "application/x-yaml")
        else:
            clowder_config_yml_id = remote_meta["clowder_config_yml_id"]

        folder = self._google_drive.CreateFile({"id": folder_id})
        investigation_name = folder["title"]
        if self.investigation_exists(investigation_name):
            raise DuplicateInvestigationException(
                f"There is already an investigation with name {investigation_name} in this context"
            )
        self.add_investigation(
            investigation_name,
            {
                "id": folder_id,
                "status": remote_meta.get("status", "Created"),
                "experiments_folder_id": experiments_folder_id,
                "clowder_meta_yml_id": meta_file["id"],
                "clowder_log_id": clowder_log_id,
                "clowder_config_yml_id": clowder_config_yml_id,
                "sheet_id": sheet_id,
                "experiments": remote_meta.get("experiments", {}),
            },
        )

    def _track_all_investigations_in_folder(self, folder_id: str):
        files = self._find_investigations(folder_id)
        for file in files:
            try:
                self._track_investigation_in_folder(file)
            except DuplicateInvestigationException:
                pass

    # TODO types!

    def log(self, investigation_name: str, data: str):
        current_log = self._read_gdrive_file_as_string(
            self.current_meta["investigation"][investigation_name]["clowder_log_id"]
        )
        new_id = self._write_gdrive_file_in_folder(
            self.current_meta["investigation"][investigation_name]["id"],
            "clowder.log",
            current_log + "\n" + datetime.datetime.now().isoformat() + " | " + data,
        )
        self.current_meta["investigation"][investigation_name]["clowder_log_id"] = new_id
        self.meta.flush()

    def _copy_gdrive_folder_to_s3(self, folder_id: str, s3_path: s3path.S3Path):
        # print(f"Copying folder {folder_id} to {s3_path}")
        for file in self._list_gdrive_files(folder_id):
            s3_file = s3_path / file["title"]
            if file["mimeType"] == "application/vnd.google-apps.folder":
                self._copy_gdrive_folder_to_s3(file["id"], s3_file)
            else:
                with s3_file.open("wb") as f:
                    data = self._read_gdrive_file_as_bytes(file["id"])
                    # print(data.decode("utf-8"))
                    f.write(data)

    def _copy_s3_folder_to_gdrive(self, s3_path: s3path.S3Path, folder_id: str):
        for file in s3_path.iterdir():
            if file.is_dir():
                id = self._create_gdrive_folder(file.name, folder_id)
                self._copy_s3_folder_to_gdrive(file, id)
            else:
                try:
                    with file.open("r") as f:
                        self._write_gdrive_file_in_folder(folder_id, file.name, f.read())
                except:
                    print(f"Failed to copy file {file.name} to GDrive folder {folder_id}")

    def _delete_s3_file(self, s3_path: s3path.S3Path):
        s3_path.unlink(missing_ok=True)

    def _delete_s3_folder(self, s3_path: s3path.S3Path):
        for child in s3_path.iterdir():
            if child.is_dir():
                self._delete_s3_folder(child)
            else:
                self._delete_s3_file(child)
        s3_path.rmdir()

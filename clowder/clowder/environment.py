import warnings

warnings.filterwarnings("ignore", r"Blowfish")

import os
import datetime
from typing import Any, Optional, Union
from pathlib import Path
import gspread
from clearml import Task
import s3path
from oauth2client.service_account import ServiceAccountCredentials
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive, GoogleDriveFile
from pydrive2.files import MediaIoReadable
from io import StringIO
from pathlib import Path
import subprocess
import pandas as pd
import re
import jinja2
from clearml import Task
import yaml
import numpy as np
from gspread import Worksheet
import gspread_dataframe as gd
from status import Status
from time import sleep
from tqdm import tqdm

GDRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
CLEARML_QUEUE = "jobs_backlog"
CLEARML_URL = "app.sil.hosted.allegro.ai"
RESULTS_CSVS_ATTRIBUTE = "results-csvs"
RESULTS_CLEARML_METRIC_ATTRIBUTE = "results-clearml-metrics"
ENTRYPOINT_ATTRIBUTE = "entrypoint"
NAME_ATTRIBUTE = "name"


class MissingConfigurationFile(IOError):
    "Missing clowder configuration file"


class DuplicateExperimentException(Exception):
    "Duplicate experiments within investigation"


class DuplicateInvestigationException(Exception):
    """There is already an investigation in the current context with that name"""


class Investigation:
    def __init__(
        self,
        id: str,
        name: str,
        experiments_folder_id: str,
        meta_id: str,
        sheet_id: str,
        log_id: str,
        status: str,
    ):
        self.id = id
        self.name = name
        self.experiments_folder_id = experiments_folder_id
        self.meta_id = meta_id
        self.sheet_id = sheet_id
        self.log_id = log_id
        self._status: Status = Status(status)
        self.investigation_s3_path = s3path.S3Path(ENV.EXPERIMENTS_S3_FOLDER) / (self.name + "_" + self.id)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, enum: Status):
        ENV.current_meta["investigations"][self.name]["status"] = enum.value
        ENV.meta.flush()
        self._status = enum

    @property
    def experiments(self):
        return ENV.current_meta["investigations"][self.name]["experiments"]

    def _get_experiments_df(self):
        worksheet: gspread.Spreadsheet = ENV.gc.open_by_key(self.sheet_id)
        experiments_df: pd.DataFrame = pd.DataFrame(worksheet.sheet1.get_all_records())
        if NAME_ATTRIBUTE not in experiments_df.columns:
            raise MissingConfigurationFile("Missing name column in ExperimentsSetup sheet")
        if ENTRYPOINT_ATTRIBUTE not in experiments_df.columns:
            raise MissingConfigurationFile("Missing entrypoint column in ExperimentsSetup sheet")
        experiments_df.set_index(experiments_df.name, inplace=True)
        if experiments_df.index.duplicated().sum() > 0:
            raise MissingConfigurationFile(
                "Duplicate names in experiments google sheet.  Each name needs to be unique."
            )
        return experiments_df

    def setup(self):
        experiments_df = self._get_experiments_df()
        self.experiments_folder_id = ENV._create_gdrive_folder("experiments", self.id)
        ENV.current_meta["investigations"][self.name]["experiments_folder_id"] = self.experiments_folder_id
        ENV.meta.flush()
        for name, params in experiments_df.iterrows():
            experiment_folder_id = ENV._create_gdrive_folder(str(name), self.experiments_folder_id)
            self._setup_experiment(params, experiment_folder_id)
        ENV._copy_gdrive_folder_to_s3(self.experiments_folder_id, self.investigation_s3_path)

    def _setup_experiment(self, params: pd.Series, folder_id: str):
        files = ENV._dict_of_gdrive_files(self.id)
        # print(params, folder_id)
        silnlp_config_yml = ENV._read_gdrive_file_as_string(files["config.yml"]["id"])  # TODO save config? Per type?
        rtemplate = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(silnlp_config_yml)
        rendered_config = rtemplate.render(params.to_dict())
        # print(rendered_config)
        ENV._write_gdrive_file_in_folder(folder_id, "config.yml", rendered_config)

    def start_investigation(self, force_rerun: bool = False) -> bool:
        experiments_df: pd.DataFrame = self._get_experiments_df()
        now_running = False
        temp_meta = {}
        for _, row in experiments_df.iterrows():
            if row[NAME_ATTRIBUTE] not in ENV.current_meta["investigations"][self.name]["experiments"]:
                ENV.current_meta["investigations"][self.name]["experiments"][row[NAME_ATTRIBUTE]] = {}
            temp_meta[row[NAME_ATTRIBUTE]] = ENV.current_meta["investigations"][self.name]["experiments"][
                row[NAME_ATTRIBUTE]
            ]
            if (
                not force_rerun
                and ENV.current_meta["investigations"][self.name]["experiments"][row[NAME_ATTRIBUTE]].get("status")
                == Task.TaskStatusEnum.completed
            ):
                continue
            elif ENV.current_meta["investigations"][self.name]["experiments"][row[NAME_ATTRIBUTE]].get("status") in [
                Task.TaskStatusEnum.in_progress,
                Task.TaskStatusEnum.queued,
            ]:
                continue
            experiment_path: s3path.S3Path = self.investigation_s3_path / row[NAME_ATTRIBUTE]
            command = f"python -m {row['entrypoint']} --memory-growth --clearml-queue {CLEARML_QUEUE} {'/'.join(str(experiment_path.absolute()).split('/')[4:])}"
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            now_running = True
            match = re.search(r"new task id=(.*)", result.stdout)
            clearml_id = match.group(1) if match is not None else "unknown"
            temp_meta[row[NAME_ATTRIBUTE]]["clearml_id"] = clearml_id
        ENV.current_meta["investigations"][self.name]["experiments"] = temp_meta
        ENV.meta.flush()
        return now_running

    def sync(self, gather_results=True):
        # Fetch info from clearml
        clearml_tasks_dict: dict[str, Union[Task, None]] = ENV._get_clearml_tasks(self.name)
        # Update gdrive, fetch
        meta_folder_id = ENV.current_meta["investigations"][self.name]["clowder_meta_yml_id"]
        remote_meta_content = yaml.safe_load(ENV._read_gdrive_file_as_string(meta_folder_id))
        if len(clearml_tasks_dict) > 0:
            if "experiments" not in remote_meta_content:
                remote_meta_content["experiments"] = {}
            for name, task in clearml_tasks_dict.items():
                if task is None:
                    continue
                if name not in remote_meta_content["experiments"]:
                    remote_meta_content["experiments"][name] = {}
                remote_meta_content["experiments"][name]["clearml_id"] = task.id
                remote_meta_content["experiments"][name][
                    "clearml_task_url"
                ] = f"https://{CLEARML_URL}/projects/*/experiments/{task.id}/output/execution"
                remote_meta_content["experiments"][name]["status"] = task.get_status()
        ENV._write_gdrive_file_in_folder(
            self.id, "clowder.meta.yml", yaml.safe_dump(remote_meta_content), "application/x-yaml"
        )
        statuses = []

        # Update locally
        for exp in ENV.current_meta["investigations"][self.name]["experiments"].keys():
            if "experiments" not in remote_meta_content or exp not in remote_meta_content["experiments"]:
                continue
            ENV.current_meta["investigations"][self.name]["experiments"][exp]["clearml_id"] = remote_meta_content[
                "experiments"
            ][exp]["clearml_id"]
            ENV.current_meta["investigations"][self.name]["experiments"][exp]["clearml_task_url"] = remote_meta_content[
                "experiments"
            ][exp]["clearml_task_url"]
            ENV.current_meta["investigations"][self.name]["experiments"][exp]["status"] = remote_meta_content[
                "experiments"
            ][exp]["status"]
            statuses.append(remote_meta_content["experiments"][exp]["status"])
        ENV.meta.flush()
        self.status = Status.from_clearml_task_statuses(statuses, self.status)  # type: ignore
        if self.status == Status.Completed:
            print(f"Investigation {self.name} is complete!")
            if gather_results:
                print("Results of investigation must be collected. This may take a while.")
                self._generate_results()  # TODO aggregate over completed experiments even if incomplete overall
            else:
                print("In order to see results, rerun with gather_results set to True.")
        return True

    def _generate_results(self):
        spreadsheet = ENV.gc.open_by_key(self.sheet_id)
        worksheets = spreadsheet.worksheets()
        setup_sheet: Worksheet = list(filter(lambda s: s.title == "ExperimentsSetup", worksheets))[0]
        setup_df = pd.DataFrame(setup_sheet.get_all_records())
        results: dict[str, pd.DataFrame] = {}
        experiment_folders = ENV._dict_of_gdrive_files(self.experiments_folder_id)
        print("Copying over results...")
        for _, row in tqdm(setup_df.iterrows()):
            ENV._copy_s3_folder_to_gdrive(
                self.investigation_s3_path / row[NAME_ATTRIBUTE], experiment_folders[row[NAME_ATTRIBUTE]]["id"]
            )
            for name in row[RESULTS_CSVS_ATTRIBUTE].split(";"):
                name = name.strip()
                s3_filepath: s3path.S3Path = (
                    self.investigation_s3_path / row[NAME_ATTRIBUTE] / name
                )  # TODO - use result that's already been copied over to gdrive
                with s3_filepath.open() as f:
                    df = pd.read_csv(StringIO(f.read()))
                    if "scores" in name:
                        name = "scores"
                        df = self._process_scores_csv(df)
                    df.insert(0, NAME_ATTRIBUTE, [row[NAME_ATTRIBUTE]])
                    if name not in results:
                        results[name] = pd.DataFrame()
                    results[name] = pd.concat([results[name], df], join="outer", ignore_index=True)

        tasks = ENV._get_clearml_tasks(self.name)
        metrics_data = {}
        metrics_names = set()
        for _, row in setup_df.iterrows():
            cur_metrics_names = set(map(lambda x: x.strip(), row[RESULTS_CLEARML_METRIC_ATTRIBUTE].split(";")))
            metrics_names = metrics_names.union(cur_metrics_names)

        metrics_names_list = list(metrics_names)
        if len(metrics_names_list) > 0 and metrics_names_list[0] != "":
            for index, row in setup_df.iterrows():
                task = tasks[row[NAME_ATTRIBUTE]]
                if task is None:
                    continue
                cols = [row[NAME_ATTRIBUTE]]
                for metric in metrics_names_list:
                    metrics = task.get_last_scalar_metrics()["Summary"]
                    cols.append(metrics.get(metric, {"last": np.nan})["last"])
                    metrics_data[index] = cols
            metrics_df = pd.DataFrame.from_dict(
                metrics_data, orient="index", columns=[NAME_ATTRIBUTE] + metrics_names_list
            )
            results["clearml_metrics"] = metrics_df

        print("Processing results data...")
        for name, df in tqdm(results.items()):
            for w in spreadsheet.worksheets():
                if w.title == name:
                    spreadsheet.del_worksheet(w)
            s = spreadsheet.add_worksheet(name, rows=0, cols=0)
            gd.set_with_dataframe(s, df)
            min_max_df = self._min_and_max_per_col(df)
            for row_index, row in df.iterrows():
                col_index = 0
                for col in df.columns:
                    if not np.issubdtype(df.dtypes[col], np.number):
                        col_index += 1
                        continue
                    ref = s.cell(
                        row_index + 2, col_index + 1  # type: ignore
                    ).address  # +2 = 1 + 1 - 1 for zero- vs. one-indexed and 1 to skip column names
                    col: str
                    max = min_max_df.at[col, "max"]
                    min = min_max_df.at[col, "min"]
                    range = max - min
                    r, g, b = self._color_func((row[col] - min) / (range) if range != 0 else 1.0)
                    s.format(f"{ref}", {"backgroundColor": {"red": r, "green": g, "blue": b}})

                    col_index += 1
                    sleep(1)  # TODO avoids exceeded per minute read quota - find better solution

    def _process_scores_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        ret = df[["score"]]
        column_names = df[["scorer"]].values.flatten()
        ret = ret.transpose()
        ret.columns = pd.Index(column_names)
        ret["BLEU-details"] = ret["BLEU"]
        ret["BLEU"] = ret["BLEU"].apply(lambda x: x.split("/")[0])
        ret[["BLEU", "CHRF3", "WER", "TER", "spBLEU"]] = ret[["BLEU", "CHRF3", "WER", "TER", "spBLEU"]].apply(
            pd.to_numeric, axis=0
        )  # TODO more robust (ignore for mvp)
        return ret

    def _min_and_max_per_col(self, df: pd.DataFrame):
        df = df.select_dtypes(include="number")
        ret = {}
        col: str
        for col in df.columns:
            ret[col] = [df[col].max(), df[col].min()]
        return pd.DataFrame.from_dict(ret, orient="index", columns=["max", "min"])

    def _color_func(self, x: float) -> tuple:
        if x > 0.5:
            return ((209 - (209 - 27) * (x - 0.5) / 0.5) / 255, 209 / 255, 27 / 255)
        return (209 / 255, (27 + (209 - 27) * x / 0.5) / 255, 27 / 255)

    def cancel(self):
        for _, obj in ENV.current_meta["investigations"][self.name]["experiments"].items():
            task: Optional[Task] = Task.get_task(task_id=obj["clearml_id"])
            if task is not None:
                task.mark_stopped(status_message="Task was stopped by user")

    def delete(self, delete_from_clearml: bool = True, delete_from_gdrive: bool = True, delete_from_s3: bool = True):
        if delete_from_clearml:
            try:
                for _, obj in ENV.current_meta["investigations"][self.name].get("experiments", {}).items():
                    task: Optional[Task] = Task.get_task(task_id=obj["clearml_id"])
                    if task is not None:
                        task.delete()
            except:
                print(f"Failed to delete investigation {self.name} from ClearML")
        if delete_from_gdrive:
            try:
                ENV._delete_gdrive_folder(self.id)
            except:
                print(f"Failed to delete investigation {self.name} from Google Drive")
        if delete_from_s3:
            try:
                ENV._delete_s3_folder(self.investigation_s3_path)
            except:
                print(f"Failed to delete investigation {self.name} from the S3 bucket")

        del ENV.current_meta["investigations"][self.name]
        ENV.meta.flush()
        self = None

    def import_setup_from(self, other):
        other_sheet_df = other._get_experiments_df()
        sheet = ENV.gc.open_by_key(self.sheet_id).sheet1
        gd.set_with_dataframe(sheet, other_sheet_df)
        config_data = ENV._read_gdrive_file_as_string(ENV._dict_of_gdrive_files(other.id)["config.yml"]["id"])
        ENV._write_gdrive_file_in_folder(self.id, "config.yml", config_data, "application/x-yaml")

    @staticmethod
    def from_meta(data: dict):
        name = list(data.keys())[0]
        data = data[name]
        return Investigation(
            id=data["id"],
            name=name,
            experiments_folder_id=data["experiments_folder_id"],
            log_id=data["clowder_log_id"],
            sheet_id=data["sheet_id"],
            meta_id=data["clowder_meta_yml_id"],
            status=data["status"],
        )


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


class Environment:
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
        ENV.add_investigation(investigation_name, investigation_data)
        return ENV.get_investigation(investigation_name)

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
            ENV.get_investigation(investigation_name)
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


ENV = Environment()

from typing import Optional, Union
import gspread
from clearml import Task
import s3path
from io import StringIO
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
from clowder.consts import (
    NAME_ATTRIBUTE,
    RESULTS_CLEARML_METRIC_ATTRIBUTE,
    RESULTS_CSVS_ATTRIBUTE,
    ENTRYPOINT_ATTRIBUTE,
    CLEARML_QUEUE,
    CLEARML_URL,
    get_env,
)

ENV = None


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
        global ENV
        if ENV is None:
            ENV = get_env()
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

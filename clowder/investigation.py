import datetime
import os
import re
import subprocess
from io import StringIO
from pathlib import Path
from pprint import pformat
from time import sleep
from typing import Optional, Union

import gspread
import gspread_dataframe as gd
import jinja2
import numpy as np
import pandas as pd
import s3path
import yaml
from clearml import Task
from gspread import Worksheet
from rich import print
from tqdm import tqdm

from clowder.configuration_exception import MissingConfigurationFileError
from clowder.consts import (
    CLEARML_QUEUE,
    CLEARML_URL,
    ENTRYPOINT_ATTRIBUTE,
    NAME_ATTRIBUTE,
    RESULTS_CLEARML_METRIC_ATTRIBUTE,
    RESULTS_CSVS_ATTRIBUTE,
    get_env,
)
from clowder.status import Status

ENV = None


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
            raise MissingConfigurationFileError("Missing name column in ExperimentsSetup sheet")
        if ENTRYPOINT_ATTRIBUTE not in experiments_df.columns:
            raise MissingConfigurationFileError("Missing entrypoint column in ExperimentsSetup sheet")
        experiments_df.set_index(experiments_df.name, inplace=True)
        if "" in experiments_df.index:
            experiments_df.drop("", inplace=True)
        if experiments_df.index.duplicated().sum() > 0:
            raise MissingConfigurationFileError(
                "Duplicate names in experiments google sheet.  Each name needs to be unique."
            )
        return experiments_df

    def setup(self):
        self.lof("Attempting to set-up experiments")
        experiments_df = self._get_experiments_df()
        self.experiments_folder_id = ENV._create_gdrive_folder("experiments", self.id)
        ENV.current_meta["investigations"][self.name]["experiments_folder_id"] = self.experiments_folder_id
        ENV.meta.flush()
        for name, params in experiments_df.iterrows():
            experiment_folder_id = ENV._create_gdrive_folder(str(name), self.experiments_folder_id)
            self._setup_experiment(params, experiment_folder_id)
        ENV._copy_gdrive_folder_to_s3(self.experiments_folder_id, self.investigation_s3_path)
        self.log("Investigation experiments were set-up")

    def _setup_experiment(self, params: pd.Series, folder_id: str):
        files = ENV._dict_of_gdrive_files(self.id)
        silnlp_config_yml = ENV._read_gdrive_file_as_string(files["config.yml"]["id"])  # TODO save config? Per type?
        rtemplate = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(silnlp_config_yml)
        rendered_config = rtemplate.render(params.to_dict())
        ENV._write_gdrive_file_in_folder(folder_id, "config.yml", rendered_config)

    def start_investigation(self, force_rerun: bool = False) -> bool:
        experiments_df: pd.DataFrame = self._get_experiments_df()
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", None)
        self.log(f"Attempting to run investigation with set-up:\n{pformat(experiments_df)}")
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
            if row["entrypoint"] == "":
                continue
            experiment_path: s3path.S3Path = self.investigation_s3_path / row[NAME_ATTRIBUTE]
            complete_entrypoint = (
                row["entrypoint"]
                .replace("$EXP", "/".join(str(experiment_path.absolute()).split("/")[4:]))
                .replace("$ON_CLEARML", f"--clearml-queue {CLEARML_QUEUE}")
                .replace("$LOCAL_EXP_DIR", str(Path(os.environ.get("SIL_NLP_DATA_PATH")) / "MT/experiments"))
            )
            if "silnlp" not in complete_entrypoint:
                raise ValueError("Entrypoints must be silnlp jobs")  # TODO make more robust against misuse
            command = f"python -m {complete_entrypoint}"
            print("[green]Running command: [/green]", command)
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"[red]{result.stderr}[/red]")
                self.log(f"Experiments {row[NAME_ATTRIBUTE]} failed to run with error:\n{result.stderr}")
                temp_meta[row[NAME_ATTRIBUTE]]["status"] = Task.TaskStatusEnum.failed.value
                continue
            now_running = True
            match = re.search(r"task id=(.*)", result.stdout)
            clearml_id = match.group(1) if match is not None else "unknown"
            temp_meta[row[NAME_ATTRIBUTE]]["clearml_id"] = clearml_id
            temp_meta[row[NAME_ATTRIBUTE]]["results_already_gathered"] = False
        ENV.current_meta["investigations"][self.name]["experiments"] = temp_meta
        ENV.meta.flush()
        if now_running:
            self.log(f"Investigation started. Experiments started: {temp_meta.keys()}")
        else:
            self.log("Starting investigation was attempted")
        return now_running

    def sync(self, gather_results=True, copy_all_results_to_gdrive: bool = True):
        self.log(
            f"Attempting to sync investigation (gather-results={gather_results}, copy-all-results-to-gdrive={copy_all_results_to_gdrive})"
        )
        # Fetch info from clearml
        clearml_tasks_dict: dict[str, Union[Task, None]] = ENV._get_clearml_tasks(self.name)
        # Update gdrive, fetch
        meta_folder_id = ENV.current_meta["investigations"][self.name]["clowder_meta_yml_id"]
        remote_meta_content = yaml.safe_load(ENV._read_gdrive_file_as_string(meta_folder_id))
        if len(clearml_tasks_dict) > 0:
            if "experiments" not in remote_meta_content:
                remote_meta_content["experiments"] = {}
            for name, task in clearml_tasks_dict.items():
                if task is None and self.experiments[name].get("clearml_id", None) != "unknown":
                    continue
                if name not in remote_meta_content["experiments"]:
                    remote_meta_content["experiments"][name] = {}
                    remote_meta_content["experiments"][name]["results_already_gathered"] = False
                if self.experiments[name]["clearml_id"] != "unknown":
                    remote_meta_content["experiments"][name]["clearml_id"] = task.id
                    remote_meta_content["experiments"][name][
                        "clearml_task_url"
                    ] = f"https://{CLEARML_URL}/projects/*/experiments/{task.id}/output/execution"
                    remote_meta_content["experiments"][name]["status"] = task.get_status()
                else:
                    remote_meta_content["experiments"][name]["status"] = Task.TaskStatusEnum.completed.value
        ENV._write_gdrive_file_in_folder(
            self.id, "clowder.meta.yml", yaml.safe_dump(remote_meta_content), "application/x-yaml"
        )
        statuses = []
        completed_exp = []
        # Update locally
        for exp in ENV.current_meta["investigations"][self.name]["experiments"].keys():
            if "experiments" not in remote_meta_content or exp not in remote_meta_content["experiments"]:
                continue
            if (
                "clearml_id" in ENV.current_meta["investigations"][self.name]["experiments"][exp]
                and "clearml_id" in remote_meta_content["experiments"][exp]
            ):
                ENV.current_meta["investigations"][self.name]["experiments"][exp]["clearml_id"] = remote_meta_content[
                    "experiments"
                ][exp]["clearml_id"]
                ENV.current_meta["investigations"][self.name]["experiments"][exp][
                    "clearml_task_url"
                ] = remote_meta_content["experiments"][exp]["clearml_task_url"]
            ENV.current_meta["investigations"][self.name]["experiments"][exp]["status"] = remote_meta_content[
                "experiments"
            ][exp]["status"]
            ENV.current_meta["investigations"][self.name]["experiments"][exp][
                "results_already_gathered"
            ] = remote_meta_content["experiments"][exp].get("results_already_gathered", False)
            statuses.append(remote_meta_content["experiments"][exp]["status"])
            if remote_meta_content["experiments"][exp]["status"] == Task.TaskStatusEnum.completed.value:
                completed_exp.append(exp)
        ENV.meta.flush()
        self.status = Status.from_clearml_task_statuses(statuses, self.status)  # type: ignore
        if self.status == Status.Completed:
            print(f"Investigation {self.name} is complete!")
            self.log(f"Investigation is complete")
            if gather_results:
                print("Results of investigation are being collected. This may take a while.")
                self._generate_results(copy_all_results_to_gdrive=copy_all_results_to_gdrive)
            else:
                print("In order to see results, rerun with gather_results set to True.")
        elif len(completed_exp) > 0 and gather_results:
            print(f"Results of experiments [{', '.join(completed_exp)}] must be collected. This may take a while.")
            self.log(f"Collecting results of completed experiments {','.join(completed_exp)}")
            self._generate_results(completed_exp, copy_all_results_to_gdrive=copy_all_results_to_gdrive)
            remote_meta_content = yaml.safe_load(ENV._read_gdrive_file_as_string(meta_folder_id))
            for exp in completed_exp:
                if copy_all_results_to_gdrive:
                    remote_meta_content["experiments"][exp]["results_already_gathered"] = True
                ENV.current_meta["investigations"][self.name]["experiments"][exp]["results_already_gathered"] = True
            ENV._write_gdrive_file_in_folder(
                self.id, "clowder.meta.yml", yaml.safe_dump(remote_meta_content), "application/x-yaml"
            )
            ENV.meta.flush()
        self.log(
            f"Synced investigation: status={self.status.value} (gather-results={gather_results}, copy-all-results-to-gdrive={copy_all_results_to_gdrive})"
        )
        return True

    def _generate_results(self, for_experiments: Optional[list] = None, copy_all_results_to_gdrive: bool = True):
        spreadsheet = ENV.gc.open_by_key(self.sheet_id)
        setup_df = self._get_experiments_df()
        results: dict[str, pd.DataFrame] = {}
        experiment_folders = ENV._dict_of_gdrive_files(self.experiments_folder_id)
        print("Copying over results...")
        for _, row in tqdm(setup_df.iterrows()):
            if (for_experiments is not None and row[NAME_ATTRIBUTE] not in for_experiments) or (
                row[ENTRYPOINT_ATTRIBUTE] == ""
            ):
                continue
            if (
                not ENV.current_meta["investigations"][self.name]["experiments"][row[NAME_ATTRIBUTE]].get(
                    "results_already_gathered", False
                )
                and copy_all_results_to_gdrive
            ):
                ENV._copy_s3_folder_to_gdrive(
                    self.investigation_s3_path / row[NAME_ATTRIBUTE], experiment_folders[row[NAME_ATTRIBUTE]]["id"]
                )
            csv_results_files = row[RESULTS_CSVS_ATTRIBUTE].split(";")
            if len(csv_results_files) > 0 and csv_results_files[0].strip() != "":
                for name in csv_results_files:
                    name = name.strip()
                    if name == "scores-best":
                        scores = list((self.investigation_s3_path / row[NAME_ATTRIBUTE]).glob("scores*"))
                        scores_vals = list(map(lambda s: int(s.name.split("-")[1].split(".")[0]), scores))
                        s3_filepath: s3path.S3Path = scores[
                            scores_vals.index(min(scores_vals))
                        ]  # TODO - use result that's already been copied over to gdrive
                    elif name == "scores-last":
                        scores = list((self.investigation_s3_path / row[NAME_ATTRIBUTE]).glob("scores*"))
                        scores_vals = list(map(lambda s: int(s.name.split("-")[1].split(".")[0]), scores))
                        s3_filepath: s3path.S3Path = scores[
                            scores_vals.index(max(scores_vals))
                        ]  # TODO - use result that's already been copied over to gdrive
                    else:
                        s3_filepath: s3path.S3Path = list(
                            (self.investigation_s3_path / row[NAME_ATTRIBUTE]).glob(
                                name if len(name.split("?")) <= 1 else name.split("?")[0]
                            )
                        )[
                            0
                        ]  # TODO - use result that's already been copied over to gdrive
                    with s3_filepath.open() as f:
                        df = None
                        if "tokenization" in name:
                            df = pd.read_csv(StringIO(f.read()), header=[0, 1])
                        else:
                            df = pd.read_csv(StringIO(f.read()))
                        if "scores" in name:
                            num_steps = "unknown"
                            if "-" in s3_filepath.parts[-1]:
                                name_elements = s3_filepath.parts[-1].split("-")
                                if len(name_elements) > 1 and ".csv" in name_elements[1]:
                                    steps_element = name_elements[1][:-4]
                                    if steps_element.isdigit():
                                        num_steps = int(steps_element)
                            df = self._process_scores_csv(df, num_steps)  # TODO ADD STEPS
                        if len(df.index) == 1:
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
                if for_experiments is not None and row[NAME_ATTRIBUTE] not in for_experiments:
                    continue
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
            name_elements = name.split("?")
            name = name_elements[0]
            color_mode = "column"  # overall column, row, nocolor #TODO enum?
            if len(name_elements) > 1 and name_elements[1] in ["overall", "column", "row", "nocolor"]:
                color_mode = name_elements[1]
            for w in spreadsheet.worksheets():
                if w.title == name:
                    spreadsheet.del_worksheet(w)
            s = spreadsheet.add_worksheet(name, rows=0, cols=0)
            gd.set_with_dataframe(s, df)
            if color_mode != "nocolor":
                self._color_code(df, s, color_mode)

    def _color_code(self, df: pd.DataFrame, s: Worksheet, mode: str):
        quota = 0
        min_max_df = None
        if mode == "row":
            min_max_df = self._min_and_max_per_row(df)
        elif mode == "overall":
            min_max_df = self._min_and_max_overall(df)
        else:
            min_max_df = self._min_and_max_per_col(df)
        row_index = 0
        if len(min_max_df.index) > 0 and isinstance(min_max_df.index[0], tuple):
            row_index += len(min_max_df.index[0]) - 1
        for label, row in df.iterrows():
            col_index = 0
            for col in df.columns:
                if not np.issubdtype(df.dtypes[col], np.number):
                    col_index += 1
                    continue
                ref = s.cell(
                    row_index + 2, col_index + 1  # type: ignore
                ).address  # +2 = 1 + 1 - 1 for zero- vs. one-indexed and 1 to skip column names
                min_max_row: str = col
                if mode == "row":
                    min_max_row = label
                elif mode == "overall":
                    min_max_row = 0
                max = min_max_df.at[min_max_row, "max"]
                min = min_max_df.at[min_max_row, "min"]
                range = max - min
                r, g, b = self._color_func((row[col] - min) / (range) if range != 0 else 1.0)
                s.format(f"{ref}", {"backgroundColor": {"red": r, "green": g, "blue": b}})
                col_index += 1
                quota += 1
                if quota > 1:
                    sleep(
                        2
                    )  # TODO avoids exceeded per minute read/write quota - find better solution: batching and guide to change quotas
                    quota = 0
            row_index += 1

    def _process_scores_csv(self, df: pd.DataFrame, num_steps: str) -> pd.DataFrame:
        ret = df[["score"]]
        column_names = df[["scorer"]].values.flatten()
        ret = ret.transpose()
        ret.columns = pd.Index(column_names)
        ret["BLEU-details"] = ret["BLEU"]
        ret["BLEU"] = ret["BLEU"].apply(lambda x: x.split("/")[0])
        ret[["BLEU", "spBLEU", "CHRF3", "WER", "TER"]] = ret[["BLEU", "spBLEU", "CHRF3", "WER", "TER"]].apply(
            pd.to_numeric, axis=0
        )  # TODO more robust (ignore for mvp)
        ret["NumberOfSteps"] = num_steps
        ret = ret[["BLEU", "spBLEU", "CHRF3", "WER", "TER", "NumberOfSteps", "BLEU-details"]]
        return ret

    def _min_and_max_per_col(self, df: pd.DataFrame):
        df = df.select_dtypes(include="number")
        ret = {}
        col: str
        for col in df.columns:
            ret[col] = [df[col].max(), df[col].min()]
        return pd.DataFrame.from_dict(ret, orient="index", columns=["max", "min"])

    def _min_and_max_per_row(self, df: pd.DataFrame):  # TODO
        df = df.select_dtypes(include="number")
        ret = {}
        for index, row in df.iterrows():
            ret[index] = [row.max(), row.min()]
        return pd.DataFrame.from_dict(ret, orient="index", columns=["max", "min"])

    def _min_and_max_overall(self, df: pd.DataFrame):  # TODO
        max = df.max(numeric_only=True).max()
        min = df.min(numeric_only=True).min()
        return pd.DataFrame.from_dict({0: [max, min]}, orient="index", columns=["max", "min"])

    def _color_func(self, x: float) -> tuple:
        if x > 0.5:
            return ((209 - (209 - 27) * (x - 0.5) / 0.5) / 255, 209 / 255, 27 / 255)
        return (209 / 255, (27 + (209 - 27) * x / 0.5) / 255, 27 / 255)

    def cancel(self) -> bool:
        self.log("Attempting to cancel investigation")
        canceled_anything = False
        for _, obj in ENV.current_meta["investigations"][self.name]["experiments"].items():
            if "clearml_id" in obj:
                task: Optional[Task] = Task.get_task(task_id=obj["clearml_id"])
                if task is not None:
                    task.mark_stopped(status_message="Task was stopped by user")
                    canceled_anything = True
        if canceled_anything:
            self.log("Investigation canceled")
        else:
            self.log("No active experiments to cancel")
        return canceled_anything

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
        self.log(f"Imported set up from investigation {other.name}")

    def log(self, data):
        current_log = ENV._read_gdrive_file_as_string(self.log_id)
        id = ENV._write_gdrive_file_in_folder(
            self.id,
            "clowder.log",
            current_log + "\n" + datetime.datetime.now().isoformat() + " | " + data,
        )
        self.log_id = id
        ENV.current_meta["investigations"][self.name]["clowder_log_id"] = id
        ENV.meta.flush()

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

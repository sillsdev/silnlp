import datetime
import os
import re
import subprocess
from io import StringIO
from pathlib import Path
from pprint import pformat
from pprint import pprint as print
from time import sleep
from typing import Optional, Union

import gspread
import gspread_dataframe as gd
import jinja2
import numpy as np
import pandas as pd
from clearml import Task
from gspread import Worksheet
from openpyxl.utils import get_column_letter
from rich import print
from tqdm import tqdm

from clowder.configuration_exception import MissingConfigurationFileError
from clowder.consts import (
    CLEARML_QUEUE,
    CLEARML_QUEUE_CPU,
    CLEARML_URL,
    ENTRYPOINT_ATTRIBUTE,
    NAME_ATTRIBUTE,
    RESULTS_CLEARML_METRIC_ATTRIBUTE,
    RESULTS_CSVS_ATTRIBUTE,
    get_env,
)
from clowder.status import Status
from silnlp.common.environment import SIL_NLP_ENV


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
        self.investigation_storage_path = ENV.EXPERIMENTS_FOLDER / (self.name + "_" + self.id)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, enum: Status):
        with ENV.meta.lock:
            ENV.meta.load()
            ENV.current_meta["investigations"][self.name]["status"] = enum.value
            ENV.meta.flush()
            self._status = enum

    @property
    def experiments(self):
        return ENV.current_meta["investigations"][self.name].get("experiments", {})

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
        self.log("Attempting to set-up experiments")
        experiments_df = self._get_experiments_df()
        self.experiments_folder_id = ENV._create_gdrive_folder("experiments", self.id)
        with ENV.meta.lock:
            ENV.meta.load()
            ENV.current_meta["investigations"][self.name]["experiments_folder_id"] = self.experiments_folder_id
            ENV.meta.flush()
        for name, params in experiments_df.iterrows():
            experiment_folder_id = ENV._create_gdrive_folder(str(name), self.experiments_folder_id)
            self._setup_experiment(params, experiment_folder_id)
        ENV._copy_gdrive_folder_to_storage(self.experiments_folder_id, self.investigation_storage_path)
        self.log("Investigation experiments were set-up")

    def _setup_experiment(self, params: pd.Series, folder_id: str):
        files = ENV._dict_of_gdrive_files(self.id)
        silnlp_config_yml = ENV._read_gdrive_file_as_string(files["config.yml"]["id"])  # TODO save config? Per type?
        rtemplate = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(silnlp_config_yml)
        rendered_config = rtemplate.render(params.to_dict())
        ENV._write_gdrive_file_in_folder(folder_id, "config.yml", rendered_config)

    def start_investigation(self, force_rerun: bool = False, experiments: "list[str]" = []) -> bool:
        experiments_df: pd.DataFrame = self._get_experiments_df()
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", None)
        self.log(f"Attempting to run investigation with set-up:\n{pformat(experiments_df)}")
        now_running = False
        temp_meta = {}
        with ENV.meta.lock:
            ENV.meta.load()
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
                if len(experiments) > 0 and row[NAME_ATTRIBUTE] not in experiments:
                    continue
                experiment_running, experiment_meta = self._run_experiment(row)
                if experiment_running:
                    now_running = True
                temp_meta[row[NAME_ATTRIBUTE]] = experiment_meta
            ENV.current_meta["investigations"][self.name]["experiments"] = temp_meta
            ENV.meta.flush()
        if now_running:
            self.log(f"Investigation started. Experiments started: {temp_meta.keys()}")
        else:
            self.log("Starting investigation was attempted")
        return now_running

    def _run_experiment(self, experiment_row: pd.Series):
        temp_meta = {}
        experiment_name = experiment_row[NAME_ATTRIBUTE]
        if "_draft" in experiment_name:
            experiment_name = experiment_name[: experiment_name.index("_draft")]
        experiment_path: Path = self.investigation_storage_path / experiment_name
        complete_entrypoint = (
            experiment_row["entrypoint"]
            .replace("$EXP", "clowder" + str(experiment_path.absolute()).split("clowder")[1])
            .replace("$ON_CLEARML_CPU", f"--clearml-queue {CLEARML_QUEUE_CPU}")            
            .replace("$ON_CLEARML", f"--clearml-queue {CLEARML_QUEUE}")
            .replace("$LOCAL_EXP_DIR", str(Path(os.environ.get("SIL_NLP_DATA_PATH")) / "MT/experiments"))
        )
        data_dir_override = ""
        if "data_folder" in ENV.current_meta:
            folder_id = ENV.current_meta["data_folder"]
            data_dir_override = f"SIL_NLP_MT_SCRIPTURE_DIR=MT/experiments/clowder/data/{folder_id}/scripture/ SIL_NLP_MT_TERMS_DIR=MT/experiments/clowder/data/{folder_id}/terms/ "
        if "silnlp" not in complete_entrypoint:
            raise ValueError("Entrypoints must be silnlp jobs")  # TODO make more robust against misuse
        python_cmd = os.environ.get('PYTHON', 'python')
        command = f"{data_dir_override} {python_cmd} -m {complete_entrypoint}"
        print("[green]Running command: [/green]", command)
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            self.log(f"Experiment {experiment_name} failed to run with error:\n{result.stderr}")
            print(f"[red]Experiment {experiment_name} failed to run with error:\n{result.stderr}[/red]")
            temp_meta["status"] = Task.TaskStatusEnum.failed.value
            temp_meta["clearml_id"] = "unknown"
            return False, temp_meta
        elif result.stderr != "":
            print(f"{result.stderr}")
        match = re.search(r"task id=(.*)", result.stdout)
        clearml_id = match.group(1) if match is not None else "unknown"
        temp_meta["clearml_id"] = clearml_id
        temp_meta["results_already_gathered"] = False
        return True, temp_meta

    def sync(self, gather_results=True, copy_all_results_to_gdrive: bool = True):
        self.log(
            f"Attempting to sync investigation (gather-results={gather_results}, copy-all-results-to-gdrive={copy_all_results_to_gdrive})"
        )
        # Fetch info from clearml
        clearml_tasks_dict: dict[str, Union[Task, None]] = self._get_clearml_tasks()
        # Update gdrive, fetch
        remote_meta_content = ENV.get_remote_meta(self.name)
        if len(clearml_tasks_dict) > 0:
            if "experiments" not in remote_meta_content:
                remote_meta_content["experiments"] = {}
            for name, task in clearml_tasks_dict.items():
                if task is None and self.experiments[name].get("clearml_id", None) != "unknown":
                    continue
                if SIL_NLP_ENV.is_bucket:
                    print("Copying from bucket...")
                    SIL_NLP_ENV.copy_experiment_from_bucket(f"clowder/{self.name}_{self.id}/{name}")
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
                    remote_meta_content["experiments"][name]["status"] = self.experiments[name].get(
                        "status", Task.TaskStatusEnum.completed.value
                    )
        ENV.set_remote_meta(self.name, remote_meta_content)
        statuses = []
        completed_exp = []
        # Update locally
        with ENV.meta.lock:
            ENV.meta.load()
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
                print("In order to see results, run sync with gather_results set to True.")
        elif len(completed_exp) > 0 and gather_results:
            print(f"Results of experiments [{', '.join(completed_exp)}] must be collected. This may take a while.")
            self.log(f"Collecting results of completed experiments {','.join(completed_exp)}")
            self._generate_results(completed_exp, copy_all_results_to_gdrive=copy_all_results_to_gdrive)
            with ENV.meta.lock:
                ENV.meta.load()
                remote_meta_content = ENV.get_remote_meta(self.name)
                for exp in completed_exp:
                    if copy_all_results_to_gdrive:
                        remote_meta_content["experiments"][exp]["results_already_gathered"] = True
                    ENV.current_meta["investigations"][self.name]["experiments"][exp]["results_already_gathered"] = True
                ENV.set_remote_meta(self.name, remote_meta_content)
                ENV.meta.flush()
        self.log(
            f"Synced investigation: status={self.status.value} (gather-results={gather_results}, copy-all-results-to-gdrive={copy_all_results_to_gdrive})"
        )
        return True

    def _generate_results(self, for_experiments: Optional[list] = None, copy_all_results_to_gdrive: bool = True):
        spreadsheet = ENV.gc.open_by_key(self.sheet_id)
        setup_df = self._get_experiments_df()
        results: dict[str, pd.DataFrame] = {}
        if copy_all_results_to_gdrive:
            print("Copying over results...")
        for _, row in tqdm(setup_df.iterrows(), disable=not copy_all_results_to_gdrive):
            if (for_experiments is not None and row[NAME_ATTRIBUTE] not in for_experiments) or (
                row[ENTRYPOINT_ATTRIBUTE] == ""
            ):
                continue
            if copy_all_results_to_gdrive:
                ENV.copy_experiment_data(self.name, row[NAME_ATTRIBUTE])
            csv_results_files = row[RESULTS_CSVS_ATTRIBUTE].split(";")
            if len(csv_results_files) > 0 and csv_results_files[0].strip() != "":
                for name in csv_results_files:
                    name = name.strip()
                    if name == "draft":
                        draft_folder_id = ENV._create_gdrive_folder("drafts", self.experiments_folder_id)
                        exp_name = row[NAME_ATTRIBUTE][: row[NAME_ATTRIBUTE].index("_draft")]
                        model_drafts_folder_id = ENV._create_gdrive_folder(exp_name, draft_folder_id)
                        ENV._copy_storage_folder_to_gdrive(
                            list(list((self.investigation_storage_path / exp_name / "infer").glob("*"))[0].glob("*"))[
                                0
                            ],
                            model_drafts_folder_id,
                        )
                        continue
                    try:
                        if name == "scores-best":
                            scores = list((self.investigation_storage_path / row[NAME_ATTRIBUTE]).glob("scores*"))
                            scores_vals = list(map(lambda s: int(s.name.split("-")[1].split(".")[0]), scores))
                            storage_filepath: Path = scores[scores_vals.index(min(scores_vals))]
                        elif name == "scores-last":
                            scores = list((self.investigation_storage_path / row[NAME_ATTRIBUTE]).glob("scores*"))
                            scores_vals = list(map(lambda s: int(s.name.split("-")[1].split(".")[0]), scores))
                            storage_filepath: Path = scores[scores_vals.index(max(scores_vals))]
                        else:
                            storage_filepath: Path = list(
                                (self.investigation_storage_path / row[NAME_ATTRIBUTE]).glob(
                                    name if len(name.split("?")) <= 1 else name.split("?")[0]
                                )
                            )[
                                0
                            ]  # TODO - use result that's already been copied over to gdrive?
                    except IndexError:
                        raise FileNotFoundError(
                            f"No such results file {name} found in {self.investigation_storage_path / row[NAME_ATTRIBUTE]}"
                        )
                    with storage_filepath.open() as f:
                        df = None
                        if "tokenization" in name:
                            df = pd.read_csv(StringIO(f.read()), header=[0, 1])
                        else:
                            df = pd.read_csv(StringIO(f.read()))
                        if "scores" in name:
                            num_steps = "unknown"
                            if "-" in storage_filepath.parts[-1]:
                                name_elements = storage_filepath.parts[-1].split("-")
                                if len(name_elements) > 1 and ".csv" in name_elements[1]:
                                    steps_element = name_elements[1][:-4]
                                    if steps_element.isdigit():
                                        num_steps = int(steps_element)
                            df = self._process_scores_csv(df, num_steps)
                        df.insert(0, NAME_ATTRIBUTE, np.repeat(row[NAME_ATTRIBUTE], len(df.index)))
                        if name not in results:
                            results[name] = pd.DataFrame()
                        results[name] = pd.concat([results[name], df], join="outer", ignore_index=True)

        tasks = self._get_clearml_tasks()
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
        formats = []
        for label, row in df.iterrows():
            col_index = 0
            for col in df.columns:
                if not np.issubdtype(df.dtypes[col], np.number):
                    col_index += 1
                    continue
                ref = f"{get_column_letter(col_index + 1)}{row_index + 2}"  # +2 = 1 + 1 - 1 for zero- vs. one-indexed and 1 to skip column names
                min_max_row: Union[str, int] = col
                if mode == "row":
                    min_max_row = label
                elif mode == "overall":
                    min_max_row = 0
                max = min_max_df.at[min_max_row, "max"]
                min = min_max_df.at[min_max_row, "min"]
                range = max - min
                r, g, b = self._color_func((row[col] - min) / (range) if range != 0 else 1.0)
                formats.append(
                    {
                        "range": ref,
                        "format": {
                            "backgroundColor": {"red": r, "green": g, "blue": b},
                        },
                    },
                )
                col_index += 1
            row_index += 1
        if len(formats) > 0:
            s.batch_format(formats)

    def _get_clearml_tasks(self) -> "dict[str, Union[None,Task]]":
        if "experiments" not in ENV.current_meta["investigations"][self.name]:
            ENV.current_meta["investigations"][self.name]["experiments"] = {}
        tasks = {}
        for experiment_name, obj in self.experiments.items():
            clearml_id = obj.get("clearml_id")
            if clearml_id is None or clearml_id == "unknown":
                tasks[experiment_name] = None
            else:
                task: Optional[Task] = Task.get_task(task_id=clearml_id)
                tasks[experiment_name] = task
        return tasks

    def _process_scores_csv(self, df: pd.DataFrame, num_steps: str) -> pd.DataFrame:
        groups = df.groupby(["src_iso", "trg_iso"]).groups
        dfs = []
        for (src_iso, trg_iso), _ in groups.items():
            out_df = df.loc[(df["src_iso"] == src_iso) & (df["trg_iso"] == trg_iso), :]
            if src_iso == "ALL" and trg_iso == "ALL":
                out_df = out_df[["score"]]
                out_df = out_df.transpose()
                out_df.columns = pd.Index(["BLEU"])
                out_df["BLEU"] = out_df["BLEU"].apply(lambda x: str(x).split("/")[0])
                out_df["BLEU"] = out_df["BLEU"].apply(pd.to_numeric)
                out_df[["spBLEU", "CHRF3", "WER", "TER", "BLEU-details"]] = None
                out_df["NumberOfSteps"] = num_steps
                out_df = out_df[["BLEU", "spBLEU", "CHRF3", "WER", "TER", "NumberOfSteps", "BLEU-details"]]
            else:
                out_df = out_df[["score", "scorer"]]
                column_names = out_df[["scorer"]].values.flatten()
                out_df = out_df[["score"]]
                out_df = out_df.transpose()
                out_df.columns = pd.Index(column_names)
                out_df["BLEU-details"] = out_df["BLEU"]
                out_df["BLEU"] = out_df["BLEU"].apply(lambda x: str(x).split("/")[0])
                out_df[["BLEU", "spBLEU", "CHRF3", "WER", "TER"]] = out_df[
                    ["BLEU", "spBLEU", "CHRF3", "WER", "TER"]
                ].apply(
                    pd.to_numeric, axis=0
                )  # TODO more robust (ignore for mvp)
                out_df["NumberOfSteps"] = num_steps
                out_df = out_df[["BLEU", "spBLEU", "CHRF3", "WER", "TER", "NumberOfSteps", "BLEU-details"]]
            out_df.insert(0, "src_iso", src_iso)
            out_df.insert(0, "trg_iso", trg_iso)
            dfs.append(out_df)
        return pd.concat(dfs, join="outer", ignore_index=True)

    def _min_and_max_per_col(self, df: pd.DataFrame):
        df = df.select_dtypes(include="number")
        ret = {}
        col: str
        for col in df.columns:
            ret[col] = [df[col].max(), df[col].min()]
        return pd.DataFrame.from_dict(ret, orient="index", columns=["max", "min"])

    def _min_and_max_per_row(self, df: pd.DataFrame):
        df = df.select_dtypes(include="number")
        ret = {}
        for index, row in df.iterrows():
            ret[index] = [row.max(), row.min()]
        return pd.DataFrame.from_dict(ret, orient="index", columns=["max", "min"])

    def _min_and_max_overall(self, df: pd.DataFrame):
        max = df.max(numeric_only=True).max()
        min = df.min(numeric_only=True).min()
        return pd.DataFrame.from_dict({0: [max, min]}, orient="index", columns=["max", "min"])

    def _color_func(self, x: float) -> tuple:
        if np.isnan(x):
            return (1.0, 1.0, 1.0)
        if x > 0.5:
            return ((209 - (209 - 27) * (x - 0.5) / 0.5) / 255, 209 / 255, 27 / 255)
        return (209 / 255, (27 + (209 - 27) * x / 0.5) / 255, 27 / 255)

    def cancel(self) -> bool:
        self.log("Attempting to cancel investigation")
        canceled_anything = False
        for _, obj in ENV.current_meta["investigations"][self.name]["experiments"].items():
            if "clearml_id" in obj and obj["clearml_id"] != "unknown":
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
        with ENV.meta.lock:
            ENV.meta.load()
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
                    ENV._delete_storage_folder(self.investigation_storage_path)
                except:
                    print(f"Failed to delete investigation {self.name} from storage (s3/local)")

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
        with ENV.meta.lock:
            ENV.meta.load()
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

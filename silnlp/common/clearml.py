from dataclasses import dataclass
import shutil

import yaml
from clearml import Task

from .environment import SIL_NLP_ENV
from ..nmt.config import Config, get_mt_exp_dir


@dataclass
class SILClearML:
    name: str
    queue_name: str = None
    project_prefix: str = "LangTech_"
    project_suffix: str = ""
    experiment_suffix: str = ""

    def __post_init__(self):
        name_parts = self.name.split("/")
        project = name_parts[0]
        if len(name_parts) == 1:
            exp_name = name_parts[0]
        else:
            exp_name = name_parts[1]

        self.task = Task.init(
            project_name=self.project_prefix + project + self.project_suffix,
            task_name=exp_name + self.experiment_suffix,
        )

        self.task.set_base_docker(
            docker_cmd="silintlai/machine-silnlp:master-latest",
        )
        if self.queue_name is not None:
            self.task.execute_remotely(queue_name=self.queue_name)

    def get_remote_name(self):
        # after init, "project name" and "task name" could be different. Read them again and update.
        self.clearml_project_folder: str = self.task.get_project_name()
        if (self.clearml_project_folder.startswith(self.project_prefix)) and (
            self.clearml_project_folder.endswith(self.project_suffix)
        ):
            self.clearml_project_folder = self.clearml_project_folder[
                len(self.project_prefix) : -len(self.project_suffix)
            ]
        self.name = self.clearml_project_folder + "/" + self.task.name
        if self.name.endswith(self.experiment_suffix):
            self.name = self.name[: -len(self.experiment_suffix)]
        return self.name

    def load_config(self):

        # copy from S3 bucket to temp first
        SIL_NLP_ENV.copy_experiment_from_bucket(self.name, extensions=("config.yml"))
        # if the project/experiment yaml file already exists, use it to re-read the config.  If not, write it.
        exp_dir = get_mt_exp_dir(self.name)
        proj_dir = get_mt_exp_dir(self.clearml_project_folder)
        if (proj_dir / "config.yml").exists():
            # if there is no experiment yaml, copy the project one to it.
            if not (exp_dir / "config.yml").exists():
                exp_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(proj_dir / "config.yml"), str(exp_dir / "config.yml"))
        if (exp_dir / "config.yml").exists():
            # read in the project/experiment yaml file
            with (exp_dir / "config.yml").open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            # connect it with ClearML - if it is run remotely, it will update the params with the remote values
            self.task.connect(mutable=config, name="config")
        else:
            # else, read in the project only yaml file
            with (get_mt_exp_dir(self.clearml_project_folder) / "config.yml").open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            self.task.connect(mutable=config, name="config")

            # then, after connection (and a possible remote update) write it to the experiment folder
            exp_dir.mkdir(parents=True, exist_ok=True)
            with (exp_dir / "config.yml").open("w+", encoding="utf-8") as file:
                yaml.safe_dump(data=config, stream=file)

        return Config(exp_dir, config)
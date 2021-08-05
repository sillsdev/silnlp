import argparse
from dataclasses import dataclass

from .experiment import SILExperiment
from .config import Config, get_git_revision_hash, get_mt_exp_dir
from clearml import Task
import yaml


@dataclass
class SILExperimentCML(SILExperiment):
    def __post_init__(self):

        name_parts = self.name.split("/")
        project = name_parts[0]
        if len(name_parts) == 1:
            exp_name = name_parts[0]
        else:
            exp_name = name_parts[1]

        self.task = Task.init(project_name="LangTech_" + project, task_name=exp_name)

        # after init, "project name" and "task name" could be different. Read them again and update.
        self.clearml_project_folder: str = self.task.get_project_name()
        if self.clearml_project_folder.startswith("LangTech_"):
            self.clearml_project_folder = self.clearml_project_folder[len("LangTech_") :]
        self.name = self.clearml_project_folder + "/" + self.task.name

        self.config: Config = self.load_clearml_config()
        self.rev_hash = get_git_revision_hash()

    def load_clearml_config(self):

        # if the project/experiment yaml file already exists, use it to re-read the config.  If not, write it.
        exp_dir = get_mt_exp_dir(self.name)
        if (exp_dir / "config.yml").exists():
            # read in the project/experiment yaml file
            with open(exp_dir / "config.yml", "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            # connect it with ClearML - if it is run remotely, it will update the params with the remote values
            self.task.connect(mutable=config, name="config")

        else:
            # else, read in the project only yaml file
            with open(get_mt_exp_dir(self.clearml_project_folder) / "config.yml", "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            self.task.connect(mutable=config, name="config")

            # then, after connection (and a possible remote update) write it to the experiment folder
            exp_dir.mkdir(parents=True, exist_ok=True)
            with open(exp_dir / "config.yml", "w+", encoding="utf-8") as file:
                yaml.safe_dump(data=config, stream=file)

        return Config(exp_dir, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment - preprocesses, train and test")
    parser.add_argument("experiment", help="Experiment name")
    args = parser.parse_args()

    exp = SILExperimentCML(
        name=args.experiment,
        make_stats=True,  # limited by stats_max_size to process only Bibles
        mixed_precision=True,  # clearML GPU's can handle mixed precision
        memory_growth=False,  # we can allocate all memory all the time
        num_devices=-1,  # get all devices
    )
    exp.run()


if __name__ == "__main__":
    main()

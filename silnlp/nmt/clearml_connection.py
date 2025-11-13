import logging
import shutil
from dataclasses import dataclass
from typing import Optional

import yaml

from ..common.environment import SIL_NLP_ENV
from .config import get_mt_exp_dir
from .config_utils import create_config

LOGGER = logging.getLogger(__name__)

TAGS_LIST = ["research", "dev", "eitl", "onboarding"]


@dataclass
class SILClearML:
    name: str
    queue_name: Optional[str] = None
    project_prefix: str = "LangTech_"
    project_suffix: str = ""
    experiment_suffix: str = ""
    clearml_project_folder: str = ""
    commit: Optional[str] = None
    use_default_model_dir: bool = True
    tag: Optional[str] = None
    skip_config: bool = False

    def __post_init__(self) -> None:
        self.name = self.name.replace("\\", "/")
        name_parts = self.name.split("/")
        project = name_parts[0]
        exp_name = name_parts[-1]
        if len(name_parts) > 2:
            exp_name = "/".join(name_parts[1:])
        if self.queue_name is None:
            self.task = None
            self._load_config()
            LOGGER.info("No ClearML task initiated.")
            return

        from clearml import Task
        from clearml.backend_api.session.session import LoginError

        try:
            self.task: Task = Task.init(
                project_name=self.project_prefix + project + self.project_suffix,
                task_name=exp_name + self.experiment_suffix,
                tags=[f"silnlp-{self.tag}"] if self.tag else None,
            )

            self._determine_clearml_project_name()
            if not self.skip_config:
                self._load_config()

            self.task.set_base_docker(
                docker_image="ghcr.io/sillsdev/silnlp:latest",
                docker_arguments=[
                    "--env TOKENIZERS_PARALLELISM='false'",
                    "--cap-add SYS_ADMIN",
                    "--device /dev/fuse",
                    "--security-opt apparmor=docker-apparmor",
                    "--env CHECK_TRANSFERS=1",
                    "--env SIL_NLP_DATA_PATH=/root/M",
                ],
                docker_setup_bash_script=[
                    "apt install -y python3-venv",
                    "python3 -m pip install --user pipx",
                    "PATH=$PATH:/root/.local/bin",
                    "pipx install poetry==1.7.1",
                    # update config.toml and pyvenv.cfg to give poetry environment access to system site packages
                    "poetry config virtualenvs.options.system-site-packages true",
                    (
                        "sed -i 's/include-system-site-packages = .*/include-system-site-packages = true/' "
                        "/root/.local/share/pipx/venvs/poetry/pyvenv.cfg"
                    ),
                    # automatically connect to the MinIO bucket
                    "apt-get install --no-install-recommends -y fuse3 rclone",
                    "mkdir -p /root/M",
                    "mkdir -p /root/.config/rclone",
                    "cp scripts/rclone/rclone.conf /root/.config/rclone/",
                    'sed -i -e "s#access_key_id = x*#access_key_id = $MINIO_ACCESS_KEY#" ~/.config/rclone/rclone.conf',
                    'sed -i -e "s#secret_access_key = x*#secret_access_key = $MINIO_SECRET_KEY#" ~/.config/rclone/rclone.conf',
                    'sed -i -e "s#endpoint = .*#endpoint = $MINIO_ENDPOINT_URL#" ~/.config/rclone/rclone.conf',
                    "rclone mount --daemon --no-check-certificate --log-file=/root/rclone_log.txt --log-level=DEBUG --vfs-cache-mode full --use-server-modtime miniosilnlp:nlp-research /root/M",
                ],
            )
            if self.commit:
                self.task.set_script(commit=self.commit)
            if self.queue_name.lower() not in ("local", "locally"):
                SIL_NLP_ENV.delete_temp_model_dir()
                self.task.execute_remotely(queue_name=self.queue_name)
        except LoginError as e:
            if self.queue_name is None:
                LOGGER.info(
                    f"Was not able to connect to a ClearML task.  Proceeding only locally.  Error code: {e.args[0]}"
                )
            else:
                LOGGER.error(
                    f"Was not able to connect to ClearML to execute on queue {self.queue_name}).  Stopping execution."
                )
                exit()
            self.task = None

    def _determine_clearml_project_name(self) -> None:
        if self.task is None:
            self.clearml_project_folder = ""
            return
        # after init, "project name" and "task name" could be different. Read them again and update.
        self.clearml_project_folder = self.task.get_project_name()
        assert self.clearml_project_folder is not None
        if (self.clearml_project_folder.startswith(self.project_prefix)) and (
            self.clearml_project_folder.endswith(self.project_suffix)
        ):
            if len(self.project_suffix) > 0:
                self.clearml_project_folder = self.clearml_project_folder[
                    len(self.project_prefix) : -len(self.project_suffix)
                ]
            else:
                self.clearml_project_folder = self.clearml_project_folder[len(self.project_prefix) :]
        self.name = self.clearml_project_folder + "/" + self.task.name
        if len(self.experiment_suffix) > 0 and self.name.endswith(self.experiment_suffix):
            self.name = self.name[: -len(self.experiment_suffix)]

    def _load_config(self) -> None:
        # if the project/experiment yaml file already exists, use it to re-read the config.  If not, write it.
        exp_dir = get_mt_exp_dir(self.name)
        if self.task is None:
            with (exp_dir / "config.yml").open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            if config is None or len(config.keys()) == 0:
                raise RuntimeError("Config file has no contents.")
            config["use_default_model_dir"] = self.use_default_model_dir
            self.config = create_config(exp_dir, config)
            return
        # There is a ClearML task - lets' do more complex importing.
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
        else:
            config = {}
        if config is None or len(config.keys()) == 0:
            raise RuntimeError("Config file has no contents.")

        # connect it with ClearML
        # - if it is run locally, it will set the config parameters in the clearml server
        # - if it is run remotely, it will update the params with the remote values
        self.task.connect(mutable=config, name="config")
        # then, after connection (and a possible remote update) write it to the experiment folder
        exp_dir.mkdir(parents=True, exist_ok=True)
        with (exp_dir / "config.yml").open("w+", encoding="utf-8") as file:
            yaml.safe_dump(data=config, stream=file)
        config["use_default_model_dir"] = self.use_default_model_dir
        self.config = create_config(exp_dir, config)

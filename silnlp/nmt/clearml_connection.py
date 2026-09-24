import logging
from dataclasses import dataclass
from typing import Optional

from clearml import Task
from clearml.backend_api.session.session import LoginError

from silnlp.common.environment import SilNlpEnv

from .config_utils import experiment_config_file

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
    tag: Optional[str] = None
    records_config: bool = True
    environment: SilNlpEnv = SilNlpEnv.create_standard_environment()

    def __post_init__(self) -> None:
        self.name = self.name.replace("\\", "/")
        name_parts = self.name.split("/")
        project = name_parts[0]
        exp_name = name_parts[-1]
        if len(name_parts) > 2:
            exp_name = "/".join(name_parts[1:])
        if self.queue_name is None:
            self.task = None
            LOGGER.info("No ClearML task initiated.")
            return

        try:
            self.task: Task = Task.init(
                project_name=self.project_prefix + project + self.project_suffix,
                task_name=exp_name + self.experiment_suffix,
                tags=[f"silnlp-{self.tag}"] if self.tag else None,
            )

            self._determine_clearml_project_name()
            if self.records_config:
                self._record_config()

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
                    "rclone mount --daemon --no-check-certificate --log-file=/root/rclone_log.txt --log-level=DEBUG --vfs-cache-mode full --vfs-cache-max-size 15G --use-server-modtime miniosilnlp:nlp-research /root/M",
                ],
            )
            if self.commit:
                self.task.set_script(commit=self.commit)
            if self.queue_name.lower() not in ("local", "locally"):
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

    def _record_config(self) -> None:
        # Recorded one way: an experiment runs from its config.yml, never from edits made in the UI.
        config = experiment_config_file(self.name, self.environment).read()
        self.task.connect(mutable=config, name="config", ignore_remote_overrides=True)

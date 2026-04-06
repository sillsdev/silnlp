import argparse
import os

from clearml import Task
from clearml.backend_api.session.session import LoginError

parser = argparse.ArgumentParser()
parser.add_argument("main-project", type=str, help="The name of the main Paratext project to onboard.")
parser.add_argument(
    "--dir",
    type=str,
    help="The name of the directory in ONBOARDING_PATH to use as the source for onboarding.",
)
args = parser.parse_args()


def setup_base_docker(task: Task, volume_path: str = None) -> Task:
    task.set_base_docker(
        docker_image="ghcr.io/sillsdev/silnlp:latest",
        docker_arguments=[
            "--env TOKENIZERS_PARALLELISM='false'",
            "--cap-add SYS_ADMIN",
            "--device /dev/fuse",
            "--security-opt apparmor=docker-apparmor",
            "--env CHECK_TRANSFERS=1",
            "--env SIL_NLP_DATA_PATH=/root/M",
            f"-v {volume_path}" if volume_path else "",
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

    return task


try:
    task: Task = Task.init(
        project_name="Onboarding",
        task_name=args.task_name,
        tags=["silnlp-auto-onboarding"],
    )
    task = setup_base_docker(
        task,
        f"-v {os.getenv('ONBOARDING_PATH')}/{args.dir}:/root/OnboardingProjects/{args.dir}",
    )

    task.execute_remotely(queue_name="jobs_backlog.cpu_only")

    import sys

    from silnlp.common import onboard_project

    old_argv = sys.argv
    onboard_projects_dir = f"/root/OnboardingProjects/{args.dir}"
    ref_projects = os.listdir(onboard_projects_dir)
    ref_projects.remove(args.main_project)
    try:
        sys.argv = [
            args.main_project,
            "--ref-projects",
            *ref_projects,
            "--copy-from",
            onboard_projects_dir,
            "--extract-corpora",
            "--wildebeest",
            "--collect-verse-counts",
            "--datestamp",
            "--stats",
            "--align",
            "--overwrite",
        ]
        onboard_project.main()
    finally:
        sys.argv = old_argv
except LoginError as e:
    print(e)
    print("Was not able to connect to ClearML to execute.  Stopping execution.")
    exit()

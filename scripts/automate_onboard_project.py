import argparse
import yaml

from clearml import Task
from clearml.backend_api.session.session import LoginError

parser = argparse.ArgumentParser()
parser.add_argument("project", type=str, help="The name of the Main Paratext project to onboard.")
parser.add_argument(
    "--draft-source",
    type=str,
    help="The name of the Drafting Source Paratext project to onboard.",
)
parser.add_argument(
    "--bt-project",
    type=str,
    help="The name of the Back Translation Paratext project to onboard.",
)
parser.add_argument(
    "--ref-projects",
    nargs="+",
    help="The names of the Reference Paratext projects to onboard.",
)
parser.add_argument("--completed-books", nargs="+", help="The ids of books that have been completed.")
parser.add_argument("--next-books", nargs="+", help="The ids of books planned for translation")
parser.add_argument(
    "--dir",
    type=str,
    help="The name of the directory to use as the source for onboarding.",
)
parser.add_argument("--task-name", type=str, help="The name of the ClearML task to create for this onboarding process.")
args = parser.parse_args()

try:
    task: Task = Task.init(
        project_name="Onboarding",
        task_name=args.task_name,
        tags=["silnlp-auto-onboarding"],
    )
    task.set_base_docker(
        docker_image="ghcr.io/sillsdev/silnlp:latest",
        docker_arguments=[
            "--env TOKENIZERS_PARALLELISM='false'",
            "--cap-add SYS_ADMIN",
            "--device /dev/fuse",
            "--security-opt apparmor=docker-apparmor",
            "--env CHECK_TRANSFERS=1",
            "--env SIL_NLP_DATA_PATH=/root/M",
            f"-v {args.dir}:/root/OnboardingProjects/{args.dir}",
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

    task.execute_remotely(queue_name="jobs_backlog.cpu_only")

    import sys

    from silnlp.common.onboard_project import main as onboard_project

    old_argv = sys.argv
    onboard_projects_dir = f"/root/OnboardingProjects/{args.dir}"
    onboarding_config = {}
    onboarding_config["onboarding"] = {
        "main_project": args.project,
        "draft_source": args.draft_source if args.draft_source else None,
        "bt_project": args.bt_project if args.bt_project else None,
        "ref_projects": args.ref_projects if args.ref_projects else [],
        "completed_books": args.completed_books if args.completed_books else [],
        "planned_books": args.planned_books if args.planned_bookes else [],
        "copy_from": str(onboard_projects_dir),
        "datestamp": True,
        "overwrite": True,
        "extract_corpora": True,
        "collect_verse_counts": True,
        "wildebeest": True,
        "stats": True,
        "align": True,
    }
    yaml.dump(onboarding_config, open(f"{onboard_projects_dir}/config.yml", "w"))
    try:
        sys.argv = ["", args.project, "--config" "", f"{onboard_projects_dir}/config.yml"]
        onboard_project()
    finally:
        sys.argv = old_argv
except LoginError as e:
    print(e)
    print("Was not able to connect to ClearML to execute.  Stopping execution.")
    exit()

import argparse
import os

from clearml import Task
from clearml.backend_api.session.session import LoginError

from silnlp.nmt.clearml_connection import setup_base_docker

parser = argparse.ArgumentParser()
parser.add_argument("main-project", type=str, required=True, help="The name of the main Paratext project to onboard.")
parser.add_argument(
    "--dir",
    type=str,
    required=True,
    help="The name of the directory in ONBOARDING_PATH to use as the source for onboarding.",
)
args = parser.parse_args()

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

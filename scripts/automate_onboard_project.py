# Add a command line argument for the task name so that we can create a new task for each onboarding request and track them separately in ClearML
import argparse
import os

from clearml import Task
from clearml.backend_api.session.session import LoginError

from silnlp.nmt.clearml_connection import set_base_docker_image

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task-name", type=str, required=True, help="The name of the ClearML task to create for this onboarding request."
)
args = parser.parse_args()

try:
    task: Task = Task.init(
        project_name="Onboarding",
        task_name=args.task_name,
        tags=["silnlp-auto-onboarding"],
    )

    task = set_base_docker_image(task)
    task.execute_remotely(queue_name="jobs_backlog.cpu_only")
    import sys

    from silnlp.common import onboard_project

    old_argv = sys.argv
    projects = os.listdir("/root/OnboardingProjects")
    try:
        sys.argv = [
            *projects,
            "--copy-from",
            "/root/OnboardingProjects",
            "--extract-corpora",
            "--wildebeest",
            "--extract-corpora",
            "--collect-verse-counts",
            "--datestamp",
            "--stats",
        ]
        onboard_project.main()
    finally:
        sys.argv = old_argv
except LoginError as e:
    print(e)
    print("Was not able to connect to ClearML to execute.  Stopping execution.")
    exit()

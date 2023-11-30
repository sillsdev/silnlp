import glob
import os

from clearml import Task

# Warning!  This does not work - it needs to be further debugged.

task = Task.init(
    project_name="clear_cache",
    task_name="clear_cache",
)

task.execute_remotely(queue_name="production")


def remove_files(path):
    print("deleting " + path)
    files = glob.glob(path + "/*")
    for f in files:
        os.remove(f)


# remove_files("/root/.cache/huggingface")
# remove_files("/var/cache/apt/archives")
remove_files("/root/.cache/pip")
remove_files("/root/.cache/pypoetry")
remove_files("/root/.clearml/pip-download-cache")
# remove_files("/clearml_agent_cache")
# remove_files("/root/.clearml/venvs-cache")

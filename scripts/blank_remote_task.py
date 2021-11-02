from clearml import Task
from silnlp.common.utils import get_git_revision_hash

task = Task.init(project_name="LangTech_Interactive", task_name=get_git_revision_hash())
task.set_base_docker(
    docker_cmd="silintlai/machine-silnlp:master-latest",
)
task.execute_remotely(queue_name="langtech_40gb")

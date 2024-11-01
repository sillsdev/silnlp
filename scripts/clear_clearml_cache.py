from clearml import Task

task = Task.init(
    project_name="clear_cache",
    task_name="clear_cache",
)
task.set_base_docker(
    docker_image="nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu22.04",
    docker_arguments="-v /home/clearml/.clearml/hf-cache:/root/.cache/huggingface",
    docker_setup_bash_script=[
        "apt install -y python3-venv",
        "python3 -m pip install --user pipx",
        "PATH=$PATH:/root/.local/bin",
        "pipx install poetry==1.7.1",
        "rm -rf /root/.cache/pip/{*,.*}",
        "rm -rf /root/.cache/pypoetry/{*,.*}",
        "rm -rf /root/.clearml/pip-download-cache/{*,.*}",
        # "rm -rf /clearml_agent_cache/{*,.*}",
        # "rm -rf /root/.clearml/venvs-cache/{*,.*}",
        # "rm -rf /root/.cache/vcs-cache/{*,.*}",
        # "rm -rf /root/.cache/huggingface/{*,.*}"
        # "rm -rf /var/cache/apt/archives/{*,.*}"
    ],
)
task.execute_remotely(queue_name="production")

print("Finished clearing caches.")

# SIL NLP

SIL NLP provides a set of pipelines for performing experiments on various NLP tasks with a focus on resource-poor and minority languages.

## Supported Pipelines

- Neural Machine Translation
- Statistical Machine Translation
- Word Alignment
---

## SILNLP Prerequisites
These are the main requirements for the SILNLP code to run on a local machine. Since there are many Python packages that need to be used with complex versioning requirements, we use a Python package called Poetry to mangage all of those. So here is a rough heirarchy of SILNLP with the major dependencies.

| Requirement           | Reason                                                            |
| --------------------- | ----------------------------------------------------------------- |
| Ubuntu-22.04          | Linux Distro that is officially supported                         |
| GIT                   | to get the repo from [github](https://github.com/sillsdev/silnlp) |
| Python                | to run the silnlp code                                            |
| Poetry                | to manage all the Python packages and versions                    |
| NVIDIA GPU            | Required to run on a local machine                                |
| Nvidia drivers        | Required for the GPU                                              |
| CUDA Toolkit          | Required for the Machine learning with the GPU                    |
| Environment variables | To tell SILNLP where to find the data, etc.                       |

## Using a local GPU
If using a local GPU, install the corresponding [NVIDIA driver](https://www.nvidia.com/download/index.aspx)
   
   On Ubuntu, the driver can alternatively be installed through the GUI by opening Software & Updates, navigating to Additional Drivers in the top menu, and selecting the newest NVIDIA driver with the labels proprietary and tested.

   After installing the driver, reboot your system.

## WSL Setup
Follow these steps if you plan to run silnlp on a Windows machine.

   1. Install Ubuntu-22.04 by running the following command: 
      ```
      wsl --install Ubuntu-22.04
      ```

      * Follow any prompts wsl provides, such as entering a UNIX username and password, this can be anything you would like.
      * To know that you are in wsl, the command line should be green with the following information <your_username>@<your_machine_name>:~$
      * To exit WSL, you can either close the command prompt or by running the "exit" command
      * To reenter WSL, you can open a command prompt and run the command "wsl ~"
      * To shutdown WSL when you are not using it, run the command "wsl --terminate Ubuntu-22.04"

   2. Exit WSL and run the following commands in a command prompt:

      * Run this command to set root as the default user:
         ```
         ubuntu2204 config --default-user root
         ```

   3. Run this command to set Ubuntu-22.04 as your default distro:
         ```
         wsl -s Ubuntu-22.04
         ```

### Note on WSL Paths: 
   * To access your Windows files from WSL:
   The paths are the same except with /mnt/ appended to them and the drive letter is lowercased with no colon following. For example, the Windows path may be "C:/Users/username/Desktop/silnlp", inside WSL it is "/mnt/c/Users/username/Desktop/silnlp"

   * To access WSL files from Windows, open File Explorer and either:
      - Scroll down and choose Linux, then Ubuntu-22.04
      - Or search "\\wsl.localhost\Ubuntu-22.04"

The rest of these instructions are assumed to be done in a WSL/Linux terminal as the root user in /root (or ~ when logged in as root).

### Note on IDEs with WSL
   * [VS Code Instructions](https://code.visualstudio.com/docs/remote/wsl)
   * [Pycharm Instructions](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html#create-wsl-interpreter)
   * For other IDEs, go to their official site and look for WSL instructions or, if it does not exist, look for Linux installation instructions and install the IDE within WSL.

## Environment Setup

1. Clone the silnlp repo:
      ```
      git clone https://github.com/sillsdev/silnlp.git
      ```

2. Navigate to the repo:
      ```
      cd silnlp
      ```

3. Create a env_vars.txt file with your credentials in this form (this can be done on Windows as well, but save the file at the silnlp repo, \\wsl.localhost\Ubuntu-22.04\root\silnlp):
   ```
   CLEARML_API_HOST="https://api.sil.hosted.allegro.ai"
   CLEARML_API_ACCESS_KEY=xxxxxxxxxxxxxxxx
   CLEARML_API_SECRET_KEY=xxxxxxxxxxxxxxxxxxx
   MINIO_ENDPOINT_URL=https://truenas.psonet.languagetechnology.org:9000
   MINIO_ENDPOINT_IP=xxxxxxxx
   MINIO_ACCESS_KEY=xxxxxxxxxx
   MINIO_SECRET_KEY=xxxxxxxxxxxxxx
   B2_ENDPOINT_URL=https://s3.us-east-005.backblazeb2.com
   B2_KEY_ID=xxxxxxxxxxxxxxx
   B2_APPLICATION_KEY=xxxxxxxxxxxxxxx
   ```

   * Include SIL_NLP_DATA_PATH="/silnlp" if you are not using MinIO or B2 and will be storing files locally.
   * If you do not intend to use SILNLP with ClearML, MinIO, and/or B2, you can leave out the respective variables. If you need to generate ClearML credentials, see ClearML setup.

4. Set your environment variables by running the following command:
   ```
   source ./setup_env_vars.sh env_vars.txt
   ```

5. Download [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2).
   Follow the instructions under the Quickstart install section, follow the Linux instructions.

6. Create the silnlp conda environment
   In a terminal run:
   ```
   conda env create --file "environment.yml"
   ```
   
   * Follow any prompts conda provides

7. Activate the silnlp conda environment
   ```
	conda activate silnlp
   ```
   * You will know the environment is active if the command line starts with "(silnlp)"

8. Install Poetry with the official installer
   ```
	curl -sSL https://install.python-poetry.org | python3 - --version 1.7.1
   ```
   * Follow Poetry's instructions, namely run:
      ```
      export PATH="/root/.local/bin:$PATH"
      ```

9. Configure Poetry to use the active Python
   ```
	poetry config virtualenvs.prefer-active-python true
   ```

10. Install the Python packages for the silnlp repo
   ```
	poetry install
   ```

11. If using MinIO or B2, you will need to set up rclone by running the following commands:
	```
   apt update
	source ./rclone_setup.sh minio
   ```

## Setting Up and Running Experiments

See the [wiki](../../wiki) for information on setting up and running experiments. The most important pages for getting started are the ones on [file structure](../../wiki/Folder-structure-and-file-naming-conventions), [model configuration](../../wiki/Configure-a-model), and [running experiments](../../wiki/NMT:-Usage). A lot of the instructions are specific to NMT, but are still helpful starting points for doing other things like [alignment](../../wiki/Alignment:-Usage).

See [this](../../wiki/Using-the-Python-Debugger) page for information on using the VS code debugger.

If you need to use a tool that is supported by SILNLP but is not installable as a Python library (which is probably the case if you get an error like "RuntimeError: eflomal is not installed."), follow the appropriate instructions [here](../../wiki/Installing-External-Libraries).


## Development Environment Setup

Follow the instructions below to set up a Dev Container in VS Code. This is the recommended way to develop in SILNLP. For manual setup, see [Manual Setup](manual_setup.md).

1. If using a local GPU, install the corresponding [NVIDIA driver](https://www.nvidia.com/download/index.aspx).
   * On Ubuntu, the driver can alternatively be installed through the GUI by opening Software & Updates, navigating to Additional Drivers in the top menu, and selecting the newest NVIDIA driver with the labels proprietary and tested.
   * After installing the driver, reboot your system.

2. Download and install [Docker Desktop](https://www.docker.com/get-started/).
   * Linux users (not including WSL) who want to use a local GPU should install Docker Engine rather than Docker Desktop.
   * Reboot after installing and completing the relevant steps below, confirm that all installation steps are complete before the next step.

   Windows (non-WSL) and macOS:
   * Open Settings in Docker Desktop and under the Resources tab, update File Sharing with any locations your source code is kept.
   
   WSL:
   * Enable WSL 2 backend:
      * Open Settings in Docker Desktop and check "Use WSL 2 based engine" under the General tab. It may already be checked.
      * To verify, check under the Resources tab in Settings for a message saying that you are using the WSL 2 backend.
   * If using a local GPU, double check that GPU support is enabled by following [these instructions](https://docs.docker.com/desktop/gpu/) from Docker.

   Linux:
   * Add your user to the docker group by using a terminal to run: `sudo usermod -aG docker $USER`
   * Sign out and back in again so your changes take effect
   * If using a local GPU, you'll also need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation) and configure Docker so that it can use the [NVIDIA Container Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).

3. Set up [ClearML](clear_ml_setup.md).

4. Define environment variables.

   Set the following environment variables with your respective credentials: CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, B2_KEY_ID, B2_APPLICATION_KEY. Also set MINIO_ENDPOINT_URL to https://truenas.psonet.languagetechnology.org:9000 and B2_ENDPOINT_URL to https://s3.us-east-005.backblazeb2.com with no quotations.
   * Linux / macOS users: To set environment variables permanently, add each variable as a new line to the `.bashrc` file (Linux) or `.profile` file (macOS) in your home directory with the format 
      ```
      export VAR="VAL"
      ```
      Close and reopen any open terminals for the changes to take effect.
   
   * Windows:
      1. Open Settings and go to the System tab.
      2. Under the "Device Specifications" section, in the "Related links", click "Advanced system settings".
      3. Click "Environment Variables".
      4. In the "System Variables" section, click "New".
      5. Enter the name and value of the variable and click "Ok". Repeat for as many variables as you need.
      6. Click "Ok" on the Environment Variables page to save your changes.
      7. Close and reopen any open command prompt terminals for the changes to take effect.

5. Install Visual Studio Code.

6. Clone the silnlp repo.

7. Open up silnlp folder in VS Code.

8. Install the Dev Containers extension for VS Code.

9. Build the dev container and open the silnlp folder in the container.
      * Click on the Remote Indicator in the bottom left corner.
      * Select "Reopen in Container" and choose the silnlp dev container if necessary. This will take a while the first time because the container has to build.
      * If it was successful, the window will refresh and it will say "Dev Container: SILNLP" in the bottom left corner.
      * Note: If you don't have a local GPU, you may need to comment out the `gpus --all` part of the `runArgs` field of the `.devcontainer/devcontainer.json` file.

10. Install and activate Poetry environment.
      * In the VS Code terminal, run `poetry install` to install the necessary Python libraries, and then run `poetry shell` to enter the environment in the terminal. 

11. (Optional) Locally mount the MinIO and/or B2 bucket(s). This will allow you to interact directly with the bucket(s) from your local terminal (outside of the dev container). See instructions [here](bucket_setup.md).

To get back into the dev container and poetry environment each subsequent time, open the silnlp folder in VS Code, select the "Reopen in Container" option from the Remote Connection menu (bottom left corner), and use the `poetry shell` command in the terminal.

## Setting Up and Running Experiments
See the [wiki](../../wiki) for information on setting up and running experiments. The most important pages for getting started are the ones on [file structure](../../wiki/File-conventions-and-cleanup), [model configuration](../../wiki/Configure-a-model), and [running experiments](../../wiki/NMT:-Usage). A lot of the instructions are specific to NMT, but are still helpful starting points for doing other things like [alignment](../../wiki/Alignment:-Usage).

See [this](../../wiki/Using-the-Python-Debugger) page for information on using the VS code debugger.

If you need to use a tool that is supported by SILNLP but is not installable as a Python library (which is probably the case if you get an error like "RuntimeError: eflomal is not installed."), follow the appropriate instructions [here](../../wiki/Installing-External-Libraries).

## .NET Machine alignment models

If you need to run the .NET versions of the Machine alignment models, you will need to install .NET Core SDK 8.0. After installing, run `dotnet tool restore`.
   * Windows: [.NET Core SDK](https://dotnet.microsoft.com/download)
   * Linux: Installation instructions can be found [here](https://learn.microsoft.com/en-us/dotnet/core/install/linux-ubuntu-2004).
   * macOS: Installation instructions can be found [here](https://learn.microsoft.com/en-us/dotnet/core/install/macos).

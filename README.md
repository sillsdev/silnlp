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
| GIT                   | to get the repo from [github](https://github.com/sillsdev/silnlp) |
| Python                | to run the silnlp code                                            |
| Poetry                | to manage all the Python packages and versions                    |
| NVIDIA GPU            | Required to run on a local machine                                |
| Nvidia drivers        | Required for the GPU                                              |
| CUDA Toolkit          | Required for the Machine learning with the GPU                    |
| Environment variables | To tell SILNLP where to find the data, etc.                       |

## Environment Setup
### Option 1: Docker
1. If using a local GPU, install the corresponding [NVIDIA driver](https://www.nvidia.com/download/index.aspx)
   
   On Ubuntu, the driver can alternatively be installed through the GUI by opening Software & Updates, navigating to Additional Drivers in the top menu, and selecting the newest NVIDIA driver with the labels proprietary and tested.

   After installing the driver, reboot your system.

2. Download and install [Docker Desktop](https://www.docker.com/get-started/)
   * If using Linux (not WSL), add your user to the docker group by using a terminal to run: `sudo usermod -aG docker $USER`
   * Reboot after installing, confirm that all installation steps are complete before the next step.
   
   If using a local GPU, you'll also need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation) and configure Docker so that it can use the [NVIDIA Container Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).

3. Pull Docker image
   
   In a terminal, run:
   ```
   docker pull ghcr.io/sillsdev/silnlp:latest
   ```
   * For Windows, use CMD Prompt
   * If there is an error like "request returned Internal Server Error for API route and version <url>, check if the server supports the requested API version" Check that the Docker Desktop installation steps are complete. Reopen CMD prompt and try again.

4. Create Docker container based on the image
   
   If you're using a local GPU, then in a terminal, run:
   ```
   docker create -it --gpus all --name silnlp ghcr.io/sillsdev/silnlp:latest
   ```
   Otherwise, run:
   ```
   docker create -it --name silnlp ghcr.io/sillsdev/silnlp:latest
   ```
   A docker container should be created. You should be able to see a container named 'silnlp' on the Containers page of Docker Desktop.

5. Create file for environment variables
   
   Create a text file with the following content and edit as necessary:
   ```
   CLEARML_API_HOST="https://api.sil.hosted.allegro.ai"
   CLEARML_API_ACCESS_KEY=xxxxxxx
   CLEARML_API_SECRET_KEY=xxxxxxx
   B2_ENDPOINT_URL=https://s3.us-east-005.backblazeb2.com
   B2_KEY_ID=xxxxxxxx
   B2_APPLICATION_KEY=xxxxxxxx
   MINIO_ENDPOINT_URL=https://truenas.psonet.languagetechnology.org:9000
   MINIO_ACCESS_KEY=xxxxxxxxx
   MINIO_SECRET_KEY=xxxxxxx
   ```
   * Include SIL_NLP_DATA_PATH="/silnlp" if you are not using B2 or MinIO and will be storing files locally.
   * If you do not intend to use SILNLP with ClearML and/or B2/MinIO, you can leave out the respective variables. If you need to generate ClearML credentials, see [ClearML setup](clear_ml_setup.md).
   * Note that this does not give you direct access to a B2 or MinIO bucket from within the Docker container, it only allows you to run scripts referencing files in the bucket.

6. Start container
   
   In a terminal, run:
   ```
      docker start silnlp
      docker exec -it --env-file path/to/env_vars_file silnlp bash
   ```

   * After this step, the terminal should change to say `root@xxxxx:~/silnlp#`, where `xxxxx` is a string of letters and numbers, instead of your current working directory. This is the command line for the docker container, and you're able to run SILNLP scripts from here.
   * To leave the container, run `exit`, and to stop it, run `docker stop silnlp`. It can be started again by repeating step 6. Stopping the container will not erase any changes made in the container environment, but removing  it will.

### Option 2: Conda
1. If using a local GPU, install the corresponding [NVIDIA driver](https://www.nvidia.com/download/index.aspx)
   
   On Ubuntu, the driver can alternatively be installed through the GUI by opening Software & Updates, navigating to Additional Drivers in the top menu, and selecting the newest NVIDIA driver with the labels proprietary and tested.

   After installing the driver, reboot your system.

2. Clone the silnlp repo
  
3. Install and initialize [Miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install)
   * If using Windows, run the next steps in the Anaconda Prompt (miniconda3) program rather than the command prompt unless stated otherwise.

4. Create the silnlp conda environment
   * In a terminal, navigate to the silnlp repo. Then inside the repo, run:
   ```
   conda env create --file "environment.yml"
   ```

5. Activate the silnlp conda environment
   * In a terminal, run:
   ```
   conda activate silnlp
   ```

6. Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) with the official installer
    * For Linux/macOS/WSL users, run:
    ```
    curl -sSL https://install.python-poetry.org | python3 - --version 1.7.1
    ```
   * For Windows users, in Powershell run:
    ```
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py - --version 1.7.1
    ```

8. Configure Poetry to use the active Python
   * In a terminal, run:
   ```
   poetry config virtualenvs.prefer-active-python true
   ```

9. Install the Python packages for the silnlp repo
   * In a terminal, run:
   ```
   poetry install
   ```

10. If using ClearML and/or B2/MinIO, set the following environment variables:
   ```
   CLEARML_API_HOST="https://api.sil.hosted.allegro.ai"
   CLEARML_API_ACCESS_KEY=xxxxxxx
   CLEARML_API_SECRET_KEY=xxxxxxx
   B2_ENDPOINT_URL=https://s3.us-east-005.backblazeb2.com
   B2_KEY_ID=xxxxxxxx
   B2_APPLICATION_KEY=xxxxxxxx
   MINIO_ENDPOINT_URL=https://truenas.psonet.languagetechnology.org:9000
   MINIO_ACCESS_KEY=xxxxxxxxx
   MINIO_SECRET_KEY=xxxxxxx
   ```
   * Include SIL_NLP_DATA_PATH="/silnlp" if you are not using B2 or MinIO and will be storing files locally.
   * If you need to generate ClearML credentials, see [ClearML setup](clear_ml_setup.md).
   * Note that this does not give you direct access to a B2 or MinIO bucket from within the Docker container, it only allows you to run scripts referencing files in the bucket.
   * For instructions on how to permanently set up environment variables for your operating system, see the corresponding section under the Development Environment Setup header below.

11. If using B2/MinIO, there are two options:
   * Option 1: Mount the bucket to your filesystem following the instructions under [Install and Configure Rclone](https://github.com/sillsdev/silnlp/blob/master/bucket_setup.md#install-and-configure-rclone).
   * Option 2: Create a local cache for the bucket following the instructions under [Create SILNLP cache](https://github.com/sillsdev/silnlp/blob/master/manual_setup.md#create-silnlp-cache).

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

   Set the following environment variables with your respective credentials: CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY, B2_KEY_ID, B2_APPLICATION_KEY, MINIO_ACCESS_KEY, MINIO_SECRET_KEY. Also set B2_ENDPOINT_URL to https://s3.us-east-005.backblazeb2.com and set MINIO_ENDPOINT_URL to https://truenas.psonet.languagetechnology.org:9000 with no quotations.
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

11. (Optional) Locally mount the B2 and/or MinIO bucket(s). This will allow you to interact directly with the bucket(s) from your local terminal (outside of the dev container). See instructions [here](bucket_setup.md).

To get back into the dev container and poetry environment each subsequent time, open the silnlp folder in VS Code, select the "Reopen in Container" option from the Remote Connection menu (bottom left corner), and use the `poetry shell` command in the terminal.

## Setting Up and Running Experiments
See the [wiki](../../wiki) for information on setting up and running experiments. The most important pages for getting started are the ones on [file structure](../../wiki/Folder-structure-and-file-naming-conventions), [model configuration](../../wiki/Configure-a-model), and [running experiments](../../wiki/NMT:-Usage). A lot of the instructions are specific to NMT, but are still helpful starting points for doing other things like [alignment](../../wiki/Alignment:-Usage).

See [this](../../wiki/Using-the-Python-Debugger) page for information on using the VS code debugger.

If you need to use a tool that is supported by SILNLP but is not installable as a Python library (which is probably the case if you get an error like "RuntimeError: eflomal is not installed."), follow the appropriate instructions [here](../../wiki/Installing-External-Libraries).

## .NET Machine alignment models

If you need to run the .NET versions of the Machine alignment models, you will need to install .NET Core SDK 8.0. After installing, run `dotnet tool restore`.
   * Windows: [.NET Core SDK](https://dotnet.microsoft.com/download)
   * Linux: Installation instructions can be found [here](https://learn.microsoft.com/en-us/dotnet/core/install/linux-ubuntu-2004).
   * macOS: Installation instructions can be found [here](https://learn.microsoft.com/en-us/dotnet/core/install/macos).

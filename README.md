# SIL NLP

SIL NLP provides a set of pipelines for performing experiments on various NLP tasks with a focus on resource-poor and minority languages.

## Supported Pipelines

- Neural Machine Translation
- Statistical Machine Translation
- Word Alignment
---

## SILNLP Prerequisites
These are the main requirements for the SILNLP code to run on a local machine. The SILNLP repo itself is hosted on Github, mainly written in Python and calls SIL.Machine.Tool. 'Machine' as we tend to call it, is a .NET application that has many functions for manipulating USFM data. Most of the language data we have for low resource languages in USFM format. Since Machine is a .Net application it depends upon the __.NET core SDK__ which works on Windows and Linux. Since there are many python packages that need to be used, with complex versioning requirements we use a Python package called Poetry to mangage all of those. So here is a rough heirarchy of SILNLP with the major dependencies.

| Requirement           | Reason                                                            |
| --------------------- | ----------------------------------------------------------------- |
| GIT                   | to get the repo from [github](https://github.com/sillsdev/silnlp) |
| Python                | to run the silnlp code                                            |
| Poetry                | to manage all the Python packages and versions                    |
| SIL.Machine.Tool      | to support many functions for data manipulation                   |
| .Net core SDK         | Required by SIL.Machine.Tool                                      |
| NVIDIA GPU            | Required to run on a local machine                                |
| Nvidia drivers        | Required for the GPU                                              |
| CUDA Toolkit          | Required for the Machine learning with the GPU                    |
| Environment variables | To tell SILNLP where to find the data, etc.                       |

## Environment Setup

1. If using a local GPU, install the corresponding [NVIDIA driver](https://www.nvidia.com/download/index.aspx).
   
   On Ubuntu, the driver can alternatively be installed through the GUI by opening Software & Updates, navigating to Additional Drivers in the top menu, and selecting the newest NVIDIA driver with the labels proprietary and tested.

   After installing the driver, reboot your system.

2. Download and install [Docker Desktop](https://www.docker.com/get-started/).
   * If using Linux (not WSL), add your user to the docker group by using a terminal to run: `sudo usermod -aG docker $USER`
   * Reboot after installing, confirm that all installation steps are complete before the next step.

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

   __If you do not intend to use SILNLP with ClearML and AWS, you can skip this step. If you need to generate ClearML credentials, see [ClearML setup](clear_ml_setup.md).__
   
   Create a text file with the following content and insert your credentials.
   ```
   CLEARML_API_ACCESS_KEY=xxxxx
   CLEARML_API_SECRET_KEY=xxxxx
   AWS_ACCESS_KEY_ID=xxxxx
   AWS_SECRET_ACCESS_KEY=xxxxx
   ```
   * Note that this does not give you direct access to an AWS S3 bucket from within the Docker container, it only allows you to run scripts referencing files in the bucket.

6. Start container
   
   If you completed step 5: \
   In a terminal, run:
   ```
      docker start silnlp
      docker exec -it --env-file path/to/env_vars_file silnlp bash
   ```
   If you did not complete step 5: \
   In a terminal, run:
   ```
   docker start silnlp
   docker exec -it silnlp bash
   ```
   * After this step, the terminal should change to say `root@xxxxx:~/silnlp#`, where `xxxxx` is a string of letters and numbers, instead of your current working directory. This is the command line for the docker container, and you're able to run SILNLP scripts from here.
   * To leave the container, run `exit`, and to stop it, run `docker stop silnlp`. It can be started again by repeating step 6. Stopping the container will not erase any changes made in the container environment, but removing  it will.

## Development Environment Setup

Follow the instructions below to set up a Dev Container in VS Code. This is the recommended way to develop in SILNLP. For manual setup, see [Manual Setup](manual_setup.md).

1. If using a local GPU, install the corresponding [NVIDIA driver](https://www.nvidia.com/download/index.aspx).
   * On Ubuntu, the driver can alternatively be installed through the GUI by opening Software & Updates, navigating to Additional Drivers in the top menu, and selecting the newest NVIDIA driver with the labels proprietary and tested.
   * After installing the driver, reboot your system.

2. Download and install [Docker Desktop](https://www.docker.com/get-started/).
   * Reboot after installing and completing the relevant steps below, confirm that all installation steps are complete before the next step.
   * Linux users should have no additional steps given that they follow Docker's distribution-specific setup.

   Windows (non-WSL) and macOS:
   * Open Settings in Docker Desktop and under the Resources tab, update File Sharing with any locations your source code is kept.
   
   WSL:
   * Enable WSL 2 backend:
      * Open Settings in Docker Desktop and check "Use WSL 2 based engine" under the General tab. It may already be checked.
      * To verify, check under the Resources tab in Settings for a message saying that you are using the WSL 2 backend.
   Linux:
   * Add your user to the docker group by using a terminal to run: `sudo usermod -aG docker $USER`
   * Sign out and back in again so your changes take effect

3. Set up the [S3 bucket](s3_bucket_setup.md).

4. Set up [ClearML](clear_ml_setup.md).

5. Define environment variables.

   Set the following environment variables with your respective credentials: CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY
   * Windows users: see [here](https://github.com/sillsdev/silnlp/wiki/Install-silnlp-on-Windows-10#permanently-set-environment-variables) for instructions on setting environment variables permanently
   * Linux users: To set environment variables permanently, add each variable as a new line to the `.bashrc` file in your home directory with the format 
      ```
      export VAR="VAL"
      ```

6. Install Visual Studio Code.

7. Clone the silnlp repo.

8. Open up silnlp folder in VS Code.

9. Install the Dev Containers extension for VS Code.

10. Build the dev container and open the silnlp folder in the container.
      * Click on the Remote Indicator in the bottom left corner.
      * Select "Reopen in Container" and choose the silnlp dev container if necessary. This will take a while the first time because the container has to build.
      * If it was successful, the window will refresh and it will say "Dev Container: SILNLP" in the bottom left corner.

11. Install and activate Poetry environment.
      * In the VS Code terminal, run `poetry install` to install the necessary Python libraries, and then run `poetry shell` to enter the environment in the terminal. 

To get back into the dev container and poetry environment each subsequent time, open the silnlp folder in VS Code, select the "Reopen in Container" option from the Remote Connection menu (bottom left corner), and use the `poetry shell` command in the terminal.

## Setting Up and Running Experiments
See the [wiki](https://github.com/sillsdev/silnlp/wiki) for information on setting up and running experiments. The most important pages for getting started are the ones on [file structure](https://github.com/sillsdev/silnlp/wiki/Folder-structure-and-file-naming-conventions), [model configuration](https://github.com/sillsdev/silnlp/wiki/Configure-a-model), and [running experiments](https://github.com/sillsdev/silnlp/wiki/NMT:-Usage). A lot of the instructions are specific to NMT, but are still helpful starting points for doing other things like [alignment](https://github.com/sillsdev/silnlp/wiki/Alignment:-Usage).

See [this](https://github.com/sillsdev/silnlp/wiki/Using-the-Python-Debugger) page for information on using the VS code debugger.

If you need to use a tool that is supported by SILNLP but is not installable as a Python library (which is probably the case if you get an error like "RuntimeError: eflomal is not installed."), follow the appropriate instructions [here](https://github.com/sillsdev/silnlp/wiki/Installing-External-Libraries).

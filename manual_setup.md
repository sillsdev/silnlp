# Manual Setup

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

## Setup

The SILNLP code can be run on either Windows or Linux operating systems. If using an Ubuntu distribution, the only compatible version is 20.04.

__Download and install__ the following before creating any projects or starting any code, preferably in this order to avoid most warnings:

1. If using a local GPU: [NVIDIA driver](https://www.nvidia.com/download/index.aspx)
   * On Ubuntu, the driver can alternatively be installed through the GUI by opening Software & Updates, navigating to Additional Drivers in the top menu, and selecting the newest NVIDIA driver with the labels proprietary and tested.
   * After installing the driver, reboot your system.
2. [Git](https://git-scm.com/downloads)
3. [Python 3.8](https://www.python.org/downloads/) (latest minor version, ie 3.8.19)
   * Can alternatively install Python using [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) if you're planning to use more than one version of Python. If following this method, activate your conda environment before installing Poetry.
4. [Poetry](https://python-poetry.org/docs/#installation)
   * Note that whether the command should call python or python3 depends on which is required on your machine.
   * It may (or may not) be possible to run the curl command within a VS Code terminal. If that causes permission errors close VS Code and try it in an elevated CMD prompt.

   Windows:
   At an administrator CMD prompt or a terminal within VS Code run:
      ```
      curl -sSL https://install.python-poetry.org | python - --version 1.7.1
      ```
      In Powershell, run:
      ```
      (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python
      ```

   Linux / macOS:
   In terminal, run:
      ```
      curl -sSL https://install.python-poetry.org | python3 - --version 1.7.1
      ```
      Add the following line to your .bashrc file (Linux) or .profile file (macOS) in your home directory:
      ```
      export PATH="$HOME/.local/bin:$PATH"
      ```
5. C++ Redistributable
   * Note - this may already be installed.  If it is not installed you may get cryptic errors such as "System.DllNotFoundException: Unable to load DLL 'thot' or one of its dependencies"
   * Windows: Download from https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0 and install
   * Linux: Instead of installing the redistributable, run the following commands:
      ```
      sudo apt-get update
      sudo apt-get install build-essential gdb
      ```

### Visual Studio Code setup

1. Install Visual Studio Code
2. Install Python extension for VS Code
3. Open up silnlp folder in VSC
4. In CMD window, type `poetry install` to create the virtual environment for silnlp
   * If using conda, activate your conda environment first before `poetry install`. Poetry will then install all the dependencies into the conda environment.
5. Choose the newly created virtual environment as the "Python Interpreter" in the command palette (ctrl+shift+P)
   * If using conda, choose the conda environment as the interpreter
6. Open the command palette and select "Preferences: Open User Settings (JSON)". In the `settings.json` file, add the following options:
   ``` json
      "python.formatting.provider": "black",
      "python.linting.pylintEnabled": true,
      "editor.formatOnSave": true,
   ```

### B2 and/or MinIO bucket(s) setup

See [Bucket setup](bucket_setup.md).

### ClearML setup

See [ClearML setup](clear_ml_setup.md).

### Create SILNLP cache
* Home directory ($HOME) on windows is usually C:\Users\<Username>\; on linux it is /home/username; and on macOS it is /Users/<Username>/. In your home directory, create the following directories.
* Create the directory "$HOME/.cache/silnlp"
* Create the directory "$HOME/.cache/silnlp/experiments" and set the environment variable SIL_NLP_CACHE_EXPERIMENT_DIR to that path.
* Create the directory "$HOME/.cache/silnlp/projects" and set the environment variable SIL_NLP_CACHE_PROJECT_DIR to that path.

### Additional Environment Variables
* Set the following environment variables with your respective credentials: CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY, B2_KEY_ID, B2_APPLICATION_KEY, MINIO_ACCESS_KEY, MINIO_SECRET_KEY.
* Set CLEARML_API_HOST to "https://api.sil.hosted.allegro.ai".
* Set B2_ENDPOINT_URL to https://s3.us-east-005.backblazeb2.com
* Set MINIO_ENDPOINT_URL to https://truenas.psonet.languagetechnology.org:9000

### Setting Up and Running Experiments

See the [wiki](../../wiki) for information on setting up and running experiments. The most important pages for getting started are the ones on [file structure](../../wiki/Folder-structure-and-file-naming-conventions), [model configuration](../../wiki/Configure-a-model), and [running experiments](../../wiki/NMT:-Usage). A lot of the instructions are specific to NMT, but are still helpful starting points for doing other things like [alignment](../../wiki/Alignment:-Usage).

See [this](../../wiki/Using-the-Python-Debugger) page for information on using the VS code debugger.

If you need to use a tool that is supported by SILNLP but is not installable as a Python library (which is probably the case if you get an error like "RuntimeError: eflomal is not installed."), follow the appropriate instructions [here](../../wiki/Installing-External-Libraries).

## Setting environment variables permanently

Linux / macOS users: To set environment variables permanently, add each variable as a new line to the `.bashrc` file (Linux) or `.profile` file (macOS) in your home directory with the format 
   ```
   export VAR="VAL"
   ```
   Close and reopen any open terminals for the changes to take effect.

Windows:
1. Open Settings and go to the System tab.
2. Under the "Device Specifications" section, in the "Related links", click "Advanced system settings".
3. Click "Environment Variables".
4. In the "System Variables" section, click "New".
5. Enter the name and value of the variable and click "Ok". Repeat for as many variables as you need.
6. Click "Ok" on the Environment Variables page to save your changes.
7. Close and reopen any open command prompt terminals for the changes to take effect.

## .NET Machine alignment models

If you need to run the .NET versions of the Machine alignment models, you will need to install .NET Core SDK 8.0. After installing, run `dotnet tool restore`.
   * Windows: [.NET Core SDK](https://dotnet.microsoft.com/download)
   * Linux: Installation instructions can be found [here](https://learn.microsoft.com/en-us/dotnet/core/install/linux-ubuntu-2004).
   * macOS: Installation instructions can be found [here](https://learn.microsoft.com/en-us/dotnet/core/install/macos).
# SIL NLP

SIL NLP provides a set of pipelines for performing experiments on various NLP tasks with a focus on resource-poor and minority languages.

## Supported Pipelines

- Neural Machine Translation
- Statistical Machine Translation
- Word Alignment
---

## SILNLP Prerequisites
These are the main requirements for the SILNLP code to run on a local machine. Using PyCharm is another way to configure the environment and instructions for that method are included later.
The SILNLP repo itself is hosted on Github, mainly written in Python and calls SIL.Machine.Tool. 'Machine' as we tend to call it, is a .NET application that has many functions for manipulating USFM data. Most of the language data we have for low resource languages in USFM format. Since Machine is a .Net application it depends upon the __.NET core SDK__ which works on Windows and Linux. Since there are many python packages that need to be used, with complex versioning requirements we use a Python package called Poetry to mangage all of those. So here is a rough heirarchy of SILNLP with the major dependencies.

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

### Option 1: Docker Container
1. If using a local GPU, install the corresponding [NVIDIA driver](https://www.nvidia.com/download/index.aspx).
   
   On Ubuntu, the driver can alternatively be installed through the GUI by opening Software & Updates, navigating to Additional Drivers in the top menu, and selecting the newest NVIDIA driver with the labels proprietary and tested.

   After installing the driver, reboot your system.

2. Download and install [Docker Desktop](https://www.docker.com/get-started/).
   * Reboot after installing, confirm that all installation steps are complete before the next step.
4. Pull Docker image
   
   In a terminal, run:
   ```
   docker pull ghcr.io/sillsdev/silnlp:latest
   ```
   * For Windows, use CMD Prompt
   * If there is an error like "request returned Internal Server Error for API route and version <url>, check if the server supports the requested API version" Check that the Docker Desktop installation steps are complete. Reopen CMD prompt and try again.

5. Create Docker container based on the image
   
   If you're using a local GPU, then in a terminal, run:
   ```
   docker create -it --gpus all --name silnlp ghcr.io/sillsdev/silnlp:latest
   ```
   Otherwise, run:
   ```
   docker create -it --name silnlp ghcr.io/sillsdev/silnlp:latest
   ```
   A docker container should be created. You should be able to see a container named 'silnlp' on the Containers page of Docker Desktop.

6. Create file for environment variables

   __If you do not intend to use SILNLP with ClearML and AWS, you can skip this step.__
   
   Create a text file with the following content and insert your credentials.
   ```
   CLEARML_API_ACCESS_KEY=xxxxx
   CLEARML_API_SECRET_KEY=xxxxx
   AWS_ACCESS_KEY_ID=xxxxx
   AWS_SECRET_ACCESS_KEY=xxxxx
   ```
   * Note that this does not give you direct access to an AWS S3 bucket from within the Docker container, it only allows you to run scripts referencing files in the bucket.

7. Start container
   
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

### Option 2: Manual Installation

The SILNLP code can be run on either Windows or Linux operating systems. If using an Ubuntu distribution, the only compatible version is 20.04.

__Download and install__ the following before creating any projects or starting any code, preferably in this order to avoid most warnings:

1. If using a local GPU: [NVIDIA driver](https://www.nvidia.com/download/index.aspx)
   * On Ubuntu, the driver can alternatively be installed through the GUI by opening Software & Updates, navigating to Additional Drivers in the top menu, and selecting the newest NVIDIA driver with the labels proprietary and tested.
   * After installing the driver, reboot your system.
2. [Git](https://git-scm.com/downloads)
3. [Python 3.7](https://www.python.org/downloads/) (latest minor version, ie 3.7.9)
   * Will also work with Python 3.8, but not Python 3.9 because of a [llvmlite incompatability](https://stackoverflow.com/questions/65798319/llvmlite-failed-to-install-error-building-llvmlite)
   * Can alternatively install Python using [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) if you're planning to use more than one version of Python. If following this method, activate your conda environment before installing Poetry.
4. [Poetry](https://python-poetry.org/docs/#installation)
   * Note that whether the command should call python or python3 depends on which is required on your machine.
   * It may (or may not) be possible to run the curl command within a VSCode terminal. If that causes permission errors close VS Code and try it in an elevated CMD prompt.


   Windows:

   At an administrator CMD prompt or a terminal within VSCode run:
      ```
      curl -sSL https://install.python-poetry.org | python - --version 1.2.2 
      ```
      
    
      
      In Powershell, run:
      ```
      (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python
      ```


   Linux:

   In terminal, run:
      ```
      curl -sSL https://install.python-poetry.org | python3 - 
      ```
      Add the following line to your .bashrc file in your home directory:
      ```
      export PATH="$HOME/.local/bin:$PATH"
      ```
      

5. .NET Core SDK
   * The necessary versions are 7.0 and 3.1. If your machine is only able to install version 7.0, you can set the DOTNET_ROLL_FORWARD environment variable to "LatestMajor", which will allow you to run anything that depends on dotnet 3.1.
   * Note - the .NET SDK is needed for [SIL.Machine.Tool](https://github.com/sillsdev/machine).  Many of the scripts in this repo require this .Net package.  The .Net package will be installed and updated when the silnlp is initialized in `__init__.py`.
   * Windows: [.NET Core SDK](https://dotnet.microsoft.com/download)
   * Linux: Installation instructions can be found [here](https://learn.microsoft.com/en-us/dotnet/core/install/linux-ubuntu-2004)
6. C++ Redistributable
   * Note - this may already be installed.  If it is not installed you may get cryptic errors such as "System.DllNotFoundException: Unable to load DLL 'thot' or one of its dependencies"
   * Windows: Download from https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0 and install
   * Linux: Instead of installing the redistributable, run the following commands:
      ```
      sudo apt-get update
      sudo apt-get install build-essential gdb
      ```

---
## Development Environment setup
### Option 1: PyCharm Setup
If you wish, you can use [PyCharm 2020.1](https://www.jetbrains.com/pycharm/) as your Python IDE.
First, you will need to install the Poetry plugin for PyCharm.

1. Go to `File -> Settings -> Plugins`.
2. Search for "Poetry" and install the plugin.

Once the Poetry plugin is installed, you can clone the the repo using PyCharm. If you have already cloned the repo, you can open the folder in PyCharm and skip these steps.

1. Go to `VCS -> Get from Version Control...`.
2. Enter `https://github.com/sillsdev/silnlp.git` in the URL field.
3. Click the `Clone`.
4. Enter your Github credentials if necessary.

Next, you will need to setup the interpreter for the project.

1. Go to `File -> Settings -> Project -> Project Interpreter`.
2. Click the gear button and select `Add...`.
3. Choose `Poetry Environment` and click `OK`.
4. PyCharm will setup the Poetry environment and install all dependencies.
5. Once PyCharm finishes the setup, click `OK`.

You will need to configure PyCharm to work properly with the project.

1. Go to `File -> Settings -> Editor -> Inspections`.
2. In the `Profile` dropdown, select `Project Default`.
3. Uncheck the `Python -> Package requirements` setting.
4. In the `Python -> PEP 8 coding style violation` setting, ignore the errors `E402` and `E203`.

Lastly, setup PyCharm to use the Black code formatter by following the instructions [here](https://black.readthedocs.io/en/stable/editor_integration.html#pycharm-intellij-idea).

### Option 2: Visual Studio Code setup
1. Install Visual Studio Code
2. Install Python extension for VSCode
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

## S3 bucket setup
We use Amazon S3 storage for storing our experiment data. Here is some workspace setup to enable a decent workflow.

### Install and configure AWS S3 storage
The following will allow the boto3 and S3Path libraries in Python correctly talk to the S3 bucket.
* Install the aws-cli from: https://aws.amazon.com/cli/
* In cmd, type: `aws configure` and enter your AWS access_key_id and secret_access_key and the region (we use region = us-east-1).
* The aws configure command will create a folder in your home directory named '.aws' it should contain two plain text files named 'config' and 'credentials'. The config file should contain the region and the credentials file should contain your access_key_id and your secret_access_key.
(Home directory on windows is usually C:\Users\<Username>\ and on linux it is /home/username)

### Install and configure rclone


**Windows**

The following will mount /aqua-ml-data on your S drive and allow you to explore, read and write.
* Install WinFsp: http://www.secfs.net/winfsp/rel/  (Click the button to "Download WinFsp Installer" not the "SSHFS-Win (x64)" installer)
* Download rclone from: https://rclone.org/downloads/
* Unzip to your desktop (or some convient location). 
* Add the folder that contains rclone.exe to your PATH environment variable.
* Take the `scripts/rclone/rclone.conf` file from this SILNLP repo and copy it to `~\AppData\Roaming\rclone` (creating folders if necessary) 
* Take the `scripts/rclone/mount_to_s.bat` file from this SILNLP repo and copy it to the folder that contains the unzipped rclone.
* Double-click the bat file. A command window should open and remain open. You should see something like:
```
C:\Users\David\Software\rclone>call rclone mount --vfs-cache-mode full --use-server-modtime s3aqua:aqua-ml-data S:
The service rclone has been started.
```

**Linux**

The following will mount /aqua-ml-data to an S folder in your home directory and allow you to explore, read and write.
* Download rclone from: https://rclone.org/install/
* Take the `scripts/rclone/rclone.conf` file from this SILNLP repo and copy it to `~/.config/rclone/rclone.conf` (creating folders if necessary)
* Add your credentials in the appropriate fields in `~/.config/rclone/rclone.conf`
* Create a folder called "S" in your user directory 
* Run the following command:
   ```
   rclone mount --vfs-cache-mode full --use-server-modtime s3aqua:aqua-ml-data ~/S
   ```
### To start S: drive on start up

**Windows**

Put a shortcut to the mount_to_s.bat file in the Startup folder.
* In Windows Explorer put `shell:startup` in the address bar or open `C:\Users\<Username>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup`
* Right click to add a new shortcut. Choose `mount_to_s.bat` as the target, you can leave the name as the default.  

Now your AWS S3 bucket should be mounted as S: drive when you start Windows.

**Linux**
* Run `crontab -e`
* Paste `@reboot rclone mount --vfs-cache-mode full --use-server-modtime s3aqua:aqua-ml-data ~/S` into the file, save and exit
* Reboot Linux

Now your AWS S3 bucket should be mounted as ~/S when you start Linux.


### Setup environment variable
The following will cause the SILNLP tools to select the S3 bucket for local silnlp operations. If you are using the Docker container, these variables will already be set and the cache will be located at `/root/.cache/silnlp`.

**Windows or Linux**
* Set the environment variable SIL_NLP_DATA_PATH to "/aqua-ml-data"
* Create the directory "/home/user/.cache/silnlp", replacing "user" with your username
* Set the environment variables SIL_NLP_CACHE_EXPERIMENT_DIR and SIL_NLP_CACHE_PROJECT_DIR to "/home/user/.cache/silnlp"

---

## Setup ClearML on local PC
To use Clear ML for managing experiments see the [ClearML Setup](clear_ml_windows_setup.md)

## Additional Information for Development Environments

### Additional Environment Variables
Set the following environment variables with your respective credentials: CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY
* Windows users: see [here](https://github.com/sillsdev/silnlp/wiki/Install-silnlp-on-Windows-10#permanently-set-environment-variables) for instructions on setting environment variables permanently
* Linux users: To set environment variables permanently, add each variable as a new line to the `.bashrc` file in your home directory with the format 
   ```
   export VAR="VAL"
   ```

### Setting Up and Running Experiments
See the [wiki](https://github.com/sillsdev/silnlp/wiki) for information on setting up and running experiments. The most important pages for getting started are the ones on [file structure](https://github.com/sillsdev/silnlp/wiki/Folder-structure-and-file-naming-conventions), [model configuration](https://github.com/sillsdev/silnlp/wiki/Configure-a-model), and [running experiments](https://github.com/sillsdev/silnlp/wiki/NMT:-Usage). A lot of the instructions are specific to NMT, but are still helpful starting points for doing other things like [alignment](https://github.com/sillsdev/silnlp/wiki/Alignment:-Usage).

If you are using VS Code, see [this](https://github.com/sillsdev/silnlp/wiki/Using-the-Python-Debugger) page for information on using the debugger.

If you need to use a tool that is supported by SILNLP but is not installable as a Python library (which is probably the case if you get an error like "RuntimeError: eflomal is not installed."), follow the appropriate instructions [here](https://github.com/sillsdev/silnlp/wiki/Installing-External-Libraries).

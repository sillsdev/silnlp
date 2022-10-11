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

__Download and install__ the following before creating any projects or starting any code, preferably in this order to avoid most warnings:

1. [Git](https://git-scm.com/downloads)
2. [Python 3.7](https://www.python.org/downloads/) (latest minor version, ie 3.7.9)
   * Will also work with Python 3.8, but not Python 3.9 because of a [llvmlite incompatability](https://stackoverflow.com/questions/65798319/llvmlite-failed-to-install-error-building-llvmlite)
3. Poetry via Powershell using the following command:
```
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python
```
4. Install [.NET Core SDK](https://dotnet.microsoft.com/download)
   * Note - the .NET SDK is needed for [SIL.Machine.Tool](https://github.com/sillsdev/machine).  Many of the scripts in this repo require this .Net package.  The .Net package will be installed and updated when the silnlp is initialized in `__init__.py`.
5. Install C++ Redistributable
   * Note - this may already be installed.  If it is not installed you may get cryptic errors such as "System.DllNotFoundException: Unable to load DLL 'thot' or one of its dependencies"
   * Download from https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0 and install

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
2. Open up silnlp folder in VSC
3. In CMD window, type `poetry install` to create the virual environment for silnlp
4. Choose the newly created virual environment as the "Python Interpreter"
5. In `settings.json`, add the following options:
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

### Windows: Install and configure rclone
The following will mount /aqua-ml-data on your S drive and allow you to explore, read and write.
* Install WinFsp: http://www.secfs.net/winfsp/rel/
* Download rclone from: https://rclone.org/downloads/
* Unzip to your desktop (or some convient location). Add the folder that contains rclone.exe to your PATH environment variable.
* Take the `scripts/rclone/rclone.conf` file from this SILNLP repo and copy it to `~\AppData\Roaming\rclone` (creating folders if necessary) 
* Take the `scripts/rclone/mount_to_s.bat` file from this SILNLP repo and copy it to the folder that contains the unzipped rclone.
* Double-click the bat file. A command window should open and remain open. You should see something like:
```
C:\Users\David\Software\rclone>call rclone mount --vfs-cache-mode full --use-server-modtime s3aqua:aqua-ml-data S:
The service rclone has been started.
```
### To start S: drive every time you open Windows
Put a shortcut to the mount_to_s.bat file in the Startup folder.
* In Windows Explorer put `shell:startup` in the address bar or open `C:\Users\<Username>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup`
* Right click to add a new shortcut. Choose `mount_to_s.bat` as the target, you can leave the name as the default.  

Now your AWS S3 bucket should be mounted as S: drive when you start Windows.

### Setup environment variable
The following will cause the SILNLP tools to select the S3 bucket for local silnlp operations
* Set the environment variable SIL_NLP_DATA_PATH to /aqua-ml-data
---
## Setup ClearML on local PC
Install the clearml python package:  
`pip install clearml`

### Create your Clear-ML credentials
Login to [Clear-ML](https://app.clear.ml/login) use a Google account with your work email address.  
Click the user icon in the top right corner and choose Settings from the menu.  

Connect your computer to the server by creating credentials, then run the below and follow the setup instructions:
`clearml-init`

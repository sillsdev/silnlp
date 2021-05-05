# SIL NLP

SIL NLP provides a set of pipelines for performing experiments on various NLP tasks with a focus on resource-poor and minority languages.

## Supported Pipelines

- Neural Machine Translation
- Statistical Machine Translation
- Word Alignment
---

## Environment Setup
These are the main requirements for the SILNLP code to run on a local machine.
1. Git
1. Python 
1. SILNLP repo 
1. .NET core SDK
1. SIL.Machine.Tool
1. Poetry
1. NVIDIA GPU
..* Nvidia driver
..* CUDA
..* Environment variables

#### Prep-Work

__Download and install__ the following before creating any projects or starting any code, preferably in this order to avoid most warnings:

1. [Git](https://git-scm.com/downloads)
2. [Python 3.7](https://www.python.org/downloads/) (latest minor version, ie 3.7.9)
3. Poetry via Powershell using the following command:
```
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python
```
4. [.NET Core SDK](https://dotnet.microsoft.com/download)
5. SIL.Machine.Tool
---
#### SIL.Machine.Tool

Many of the scripts in this repo require [SIL.Machine.Tool](https://github.com/sillsdev/machine). SIL.Machine.Tool is a dotnet program and it requires the __.NET core sdk__.
##### To install SIL.Machine.Tool
1. You'll need to choose the correct .NET core SDK according to your Operating system. 
1. Download and install the __.NET core sdk__ from [Microsoft](https://dotnet.microsoft.com/download)
1. To Install SIL.Machine.Tool:
   __Open the repo directory `silnlp`__ and execute the following command from that folder:
   ```
   dotnet tool restore
   ```
   When dotnet can't find the manifest file: `dotnet-tools.json` which is in the .config subdirectory of the silnlp repo it will report an error message:

   ```
   C:\Users\username>dotnet tool restore
   Cannot find a manifest file.
   For a list of locations searched, specify the "-d" option before the tool name.
   No tools were restored.
   ```
   Change the current working directory to the the repo and then dotnet should restore the sil Machine tool.

   The -d option is useful to show where dotnet is looking for the manifest file:
   ```
   D:\GitHub>dotnet -d tool restore
   Telemetry is: Enabled
   The list of searched paths:
           D:\GitHub\.config\dotnet-tools.json
           D:\GitHub\dotnet-tools.json
           D:\.config\dotnet-tools.json
           D:\dotnet-tools.json
   Cannot find a manifest file.
   For a list of locations searched, specify the "-d" option before the tool name.
   No tools were restored.

   D:\GitHub>
   ```
---
#### Optional: PyCharm Setup
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


# SIL NLP

SIL NLP provides a set of pipelines for performing experiments on various NLP tasks with a focus on resource-poor and minority languages.

## Supported Pipelines

- Neural Machine Translation
- Statistical Machine Translation
- Word Alignment

## Environment Setup

### Windows 10 + PyCharm + Poetry

#### Prep-Work

Download and install the following separately before creating any projects or
starting any code, preferably in this order to avoid most warnings:

1. [Git](https://git-scm.com/downloads)
1. [Python 3.7](https://www.python.org/downloads/) (latest minor version, ie 3.7.9)
1. Install [PyCharm 2020.1](https://www.jetbrains.com/pycharm/) or later
1. Poetry via Powershell using the following command:

```
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python
```

5. [.NET Core SDK](https://dotnet.microsoft.com/download)

#### PyCharm Setup

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

#### SIL.Machine.Tool

Many of the scripts in this repo use [SIL.Machine.Tool](https://github.com/sillsdev/machine).
Execute the following command from the repo directory to download it:

```
dotnet tool restore
```

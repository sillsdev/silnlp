# NLP Machine Translation Test

A broadly labeled Bible translation project for developing and experimenting
with NLP related machine translation tools and libraries.

## Status

**Private Development & Invite Only**

To avoid conflicts and confusion with other projects, this testing project is
privately shared. Locally modified libraries within this project should also
be kept private until vetted code can be officially released via a pull request
or other formal method of submission.

## Project Directory Structure

- `data/` - Individual small files (<100MB) of shared data to be used by the
  rest of the project. Do not forget to include an extension in the name to help
  tract data types. Larger files or sets of data should be kept local outside of
  the project directory, or utilize other cloud storage (like AWS S3).
- `docs/` - All document-style file types, including Python notebooks, various
  types of markdown, HTML output, etc.
- `nlp/` - The main Python package that includes all NLP tools.
- `R/` - R script files used by project are sourced from here.
- `vendors/` - Empty placeholder directory to store locally any third-party
  libraries, drivers, and other tools that will not be copied to the project
  repository.

## Environment Setup

### Windows 10 + PyCharm + Poetry

#### Prep-Work

Download and install the following separately before creating any projects or
starting any code, preferably in this order to avoid most warnings:

1. [Git](https://git-scm.com/downloads)
1. [Python 3.7](https://www.python.org/downloads/) (latest minor version, ie 3.7.6, and not 3.8 just yet, **not the
   Microsoft Store**)
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
2. Enter `https://github.com/andrewbulin/nlp_mt_testing.git` in the URL field.
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

#### SIL.Machine.Translator Tool

Many of the scripts in this repo use the [SIL.Machine.Translator](https://github.com/sillsdev/machine) tool. 

In order to install/update the tool you first need to have the dotnet SDK installed.

To install that on Linux issue this command:

```
$ sudo snap install dotnet-sdk --classic
```

Then execute the following command from the repo directory:

```
dotnet tool restore
```

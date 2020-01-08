# NLP Machine Translation Test

A broadly labeled Bible translation project for developing and experimenting
with NLP related machine translation tools and libraries.

## Status

**Private Development & Invite Only**

To avoid conflicts and confusion with other projects, this testing project is
privately shared.  Locally modified libraries within this project should also
be kept private until vetted code can be officially released via a pull request
or other formal method of submission.

## Project Directory Structure

* `data/` - Individual small files (<100MB) of shared data to be used by the
rest of the project.  Do not forget to include an extension in the name to help
tract data types. Larger files or sets of data should be kept local outside of
the project directory, or utilize other cloud storage (like AWS S3). 
* `docs/` - All document-style file types, including Python notebooks, various
types of markdown, HTML output, etc.
* `Python/` - Python script files used by project are sourced from here.
* `R/` - R script files used by project are sourced from here.
* `vendors/` - Empty placeholder directory to store locally any third-party
libraries, drivers, and other tools that will not be copied to the project
repository.

## Environment Setup Examples

### Windows 10 + PyCharm + pipenv

Microsoft recommends at the time of this writing to install a virtual Ubuntu
environment, and there are other conflicting methods out on the internet. This
is an attempt to document a particular setup that has worked so far.

#### Prep-Work

Download and install the following separately before creating any projects or
starting any code, preferably in this order to avoid most warnings:

1. [Git](https://git-scm.com/downloads)
1. [Python 3.7](https://www.python.org/downloads/) (latest minor version, ie 3.7.6, and not 3.8 just yet, **not the
Microsoft Store**)
1. Pipenv via Windows Console _(not PowerShell)_ using the following command
_(and do not use `--user` option for pip)_:

```
C:>pip install -e git+https://github.com/pypa/pipenv.git@master#egg=pipenv
```

_(Pip should work now if Python was properly installed.  If it does not work,
you will need to go back and check on the Python 3.7 installation.)_

#### Creating Project

This was the easiest option that automagically made use of the installation of
Python and Pipenv by auto-populating in PyCharm.  It also installed all
dependencies and popped open Github login windows to authenticate a private
repositories.

1. Start PyCharm and go through its basic initialization options before you
create a project.  If this was pre-installed, make sure it's up to date, and
save and close your current project.
1. From the welcome window, use the "Get from Version Control" option to
create a new project.
1. Enter Github URL and local path.  **Pay attention** that your local path is
correct if you change it after setting the URL, which may remove the
automatically created local directory name using the repository name.
1. Confirm open directory, which will open PyCharm and the related project.
Give it a couple of minutes to settle.
1. Go to File > Settings to fix the missing interpreter project settings.
1. For the interpreter, look for the option to "Add...".
1. In the new pop-up window, select Pipenv on the left, and give it a minute to
settle.
1. The interpreter and Pipenv paths should have begun to populate on the right.
If not, start with the drop-down to see if you can select it.  Also if you already
have multiple versions of Python and Pip installed, use the drop-downs to find
the latest 3.7 version installed.
1. Check the box to install packages from Pipfile.
1. Save with 'Okay', closing the pop-up, and taking you back to the settings
window.  Give it several minutes to install the packages settle down (this could
take some time).  If you have private Github package repos specified in your
Pipfile, be prepared to enter credentials in random Github pop-ups.
1. Once this is all done, hit 'Apply' and 'Okay', closing the settings window.
Give it a several minutes more to make the final changes and indexing (this
could take some time).

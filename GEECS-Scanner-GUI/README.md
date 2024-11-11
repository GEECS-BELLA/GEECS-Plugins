# Data Acquisition Module

This is a GUI wrapper for the Python-based data acquisition module in 
`GEECS-PythonAPI/geecs_python_api/controls/data_acquisition/`.  While it was
developed for HTU originally, it is designed to be generally usable by any
experiment using the GEECS/Master Control environment.

## Table of Contents
- [Installation](#installation)
- [Overview](#overview)
- [Core Classes](#core-classes)
  - [GEECSScanner](#geecs-scanner)
- [Basic Usage](#basic-usage)
- [Future Improvements](#future-improvements)

## Installation

Requires GEECS-Plugins installed somewhere, Python 3.10, and Poetry.  Can be run off of the Z: drive version, but it is not recommended to edit this version directly.  Beyond basic usage, we recommend cloning your own copy of the repo with `Github Desktop`.

Python 3.10 can be installed from the following link:
```link
https://www.python.org/downloads/release/python-31011/
```
Can download and use the `Windows installer (64-bit)` link.  Remember to select the "Add to Path" checkbox at the bottom of the installation window.

For Poetry, install using their official command-line
```link
https://python-poetry.org/docs/#installing-with-the-official-installer
```

Now, there are two installation paths:  one for just running the GUI and one for getting the environment set up for code development.

### To run the GUI

Open a terminal.  Change directory to this folder (the one with `pyproject.toml`).  Then run the following two commands:
```commandline
poetry install
poetry run GEECSScanner.py
```

Note1:  you can write a quick bash script and place on your desktop for these three steps.  In the future, we may have a standalone .exe which replaces these steps.

Note2:  this may not work if your default python version is not 3.10.  You can see what python version your system defaults to by typing `python --verison`.  If you are experiencing troubles with poetry here, might need to manually specify which python version to use in these two commands.  (See next section for an example)

Upon first running the GUI, it will prompt you for four things necessary for the GEECS-PythonAPI to function.  These are subsequently written to a config file located at `~\.config\geecs_python_api\`:
```config
[Paths]
geecs_data = C:\GEECS\user data\

[Experiment]
expt = Undulator
rep_rate_hz = 1
shot_control = U_DG645_ShotControl
```
These are examples for Undulator, but need to be defined for other experiments.  There is also a button on the GUI to re-set these values.

### Setting up the environment for development

Open a terminal.  Change directory to this folder (the one with `pyproject.toml`).  Then create the environment:
```commandline
poetry install
```
If this doesn't work, you have too many python versions installed on your computer.  To specify, go to `Edit the system environment variables` in your Windows search bar and look up where Python 3.10 lives in your Path variable.  Copy the entire path to the `python.exe` file and use that with poetry.

For example, if my Python 3.10 executable is in `C:\Users\loasis.LOASIS\AppData\Local\Programs\Python\Python310\python.exe` then I run the commands
```commandline
poetry env use C:\Users\loasis.LOASIS\AppData\Local\Programs\Python\Python310\
poetry install
```

This creates a virtual environment at a location that can be viewed with the following command:
```commandline
poetry env info --path
```

Depending on what code editor you are using, you can specify that the `python.exe` located within this folder is the virtual environment to use for code development.

Note:  If you want to add a package to the poetry file, there is an easy command you can use:
```commandline
poetry add <<python package>>
```

## Overview

To be written at a later date...

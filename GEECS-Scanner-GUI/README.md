# Data Acquisition Module

This is a GUI wrapper for the Python-based data acquisition module in 
`./geecs_scanner/data_acquisition/` that utilizes `GEECS-PythonAPI`.  It was originally
developed for HTU, but it is designed to be generally usable by any
experiment using the GEECS/Master Control environment.

While Master Control scans save everything, will crash or slowdown at any device encountering an error, and keep all devices explictly timed to one-another through designated shots; the python data acquisition operates instead in an "opt-in" framework.  Here, you are in control of what devices get saved in a given scan, if a device becomes unresponsive then the rest of the scan continues un-interrupted (so long as it is not vital to the triggering of devices), and data is collected instead by turing "Save" to "on" for devices and waiting a specified amount of time.  While you are not garunteed to get every device saved for exactly N number of shots, this framework allows for routine data-collection to be achieved very smoothly and reliably.  As an added bonus, being written in Python allows for much more flexibility in including additional features, such as

* Automated actions before and after scans
* Composite scan variables defined relative to current values
* Scan presets
* Scripted multi-scans

## Table of Contents
- [Installation](#installation)
- [Tutorial](#tutorial)
- [Core Classes](#core-classes)
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
poetry run main.py
```

Alternatively, you can use the included bash script `GEECS_Scanner.sh` to launch the version of the GUI on the Z drive.  This can be modified if needed.

Note:  this may not work if your default python version is not 3.10.  You can see what python version your system defaults to by typing `python --verison`.  If you are experiencing troubles with poetry here, might need to manually specify which python version to use in these two commands.

### Setting up the environment for development

Open a terminal.  Change directory to this folder (the one with `pyproject.toml`).  Then create the environment:
```commandline
poetry install
```
If this doesn't work, you have too many python versions installed on your computer.  To specify, go to `Edit the system environment variables` in your Windows search bar and look up where Python 3.10 lives in your Path variable.  Copy the entire path to the `python.exe` file and use that with poetry.

For example, if my Python 3.10 executable is in `C:\Users\loasis.LOASIS\AppData\Local\Programs\Python\Python310\python.exe` then I run the commands
```commandline
poetry env use C:\Users\loasis.LOASIS\AppData\Local\Programs\Python\Python310\python.exe
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

## Tutorial

This section serves as a guide to running the software once you are able to successfully launch it.  

### Experiment Config

Upon first running the GUI, it will prompt you for three things necessary for the GEECS-PythonAPI to function.  First is the path for GEECS user data.  Second is the name of the experiment, this should match the syntax and case sensitivity as shown on Master Control.  Next is the repetition rate of the experiment.  While GEECS Scanner does not operate on a shot-counting method for data acquisition, this number is instead used to estimate the acquisition time per step.  These are subsequently written to a config file located at `~\.config\geecs_python_api\`:
```config
[Paths]
geecs_data = C:\GEECS\user data\

[Experiment]
expt = Undulator
rep_rate_hz = 1
```
These are examples for Undulator, but need to be defined for other experiments.  There is also a button on the GUI to re-set these values.

### Timing Setup

This side-GUI opens up a dialog to specify what happens when the experiment goes into "Scan" mode.  This is composed of a single device to act as the shot controller and any number of associated variables.  You can then set the states of these variables for when the system goes into Off, Scan, and Standby modes.

### Element Editor

The next step is to add devices that can be saved during a scan.  This is done through the "Element Editor" and can be opened with the `New Element` button.  (Or `Open Element` if you have a previously-made element selected)  Each element can have multiple (or zero) devices, and for each device you add you must specify all scalars you wish to save.  One must-have variable for any synchronous device is the `timestamp` variable, which lets the data acquisition determine which saved images belong to each shot.

Next, you can optionally specify actions that can be automatically executed before or after the scan.  These includes `set` to set a device variable, `get` to check a device variable is a given value, `wait` to wait a specified amount of time, and `execute` to perform a action defined in the `Action Library`.  These actions are performed in the visible list in the element editor.

### Scan Setup

Next, you specify the type of scan.  Noscan (or statistics scan) is where no variables are changed and the settings are held constant for the number of shots specified.  1D Scan varies the specifed variable from the start value to the end value with the specified step size, acquiring a set number of shots each step.  The Scan Variables do not list every scan variable in the experiment, rather only the defined list in `geecs_python_api/controls/data_acquisition/configs/.../scan_devices/scan_devices.yaml`.  This file maps a nickname (ie: EMQ1) to a GEECS Variable name (ie: U_EMQTripletBipolar:Current_Limit.Ch1).  Additionally, composite variables can be defined in `composite_variables.yaml` and will show up in this list as well.

Note1:  Composite variables have a start and stop that are relative to the current values in Master Control.  Regular, single variables always are defined with an absolute start and stop values.

Note2:  Eventually there will be a GUI to edit these scan variables.

You also have the option to specify a text description for the scan.  Some info is auto-populated, such as if it is a scan/noscan, and what the scan variable is.

### Presets

You have the option to save and load presets for scans.  When you have configured the GUI to a particular arangement, click `Save` under the preset list and specify a name.  This preset will then appear in the list and you can double-click on it to load the exact same settings in the future.  This includes all of the scan options as well as the save elements.

### Running a scan

As with Master Control, `Start` will start the scan, `Stop` will stop it.  If a scan is stopped partway through, the data that was saved is still accessible through the data folder.

Notes:  Things seem to crash if you immediately try to stop a scan after starting one.  Also starting a scan with no save elements also seems to cause issues.  

### Multiscanner

Clicking the `Multi-Scan` button opens up the multiscanner gui.  Here, you can organize saved presets into a structured list for GEECS-Scanner to execute one after the other.  While the mutliscanner window is open, you cannot run scans or edit the experiment settings on the main window.  Once you are satisfied with the list, click `Start` to begin.

Note:  You can specify the position on the list from where to start the multiscan.  In case you are restarting halfway through.

Note2:  Like with the presets on the main window, you have the option to save and load presets of the multiscanner configuration itself.

To add more functionality, you can enable the `Scan Commands` checkbox to enable the second list.  This makes it so when the multiscan loads presets, it loads only the save elements from the left preset list and only the scan parameters from the right preset list.  This is beneficial if you want to adjust the scan parameters for a multiscan, but do not want to edit every single preset in the list.  You can instead a single preset containing the new scan parameters, and load that preset in for every slot on the rightmost list.

## Core Classes

TODO, right now files are separated by GUI window and each python file contains all functionality.  `RunControl.py` is the interface between the main window GUI `GEECSScanner.py` and the backend managed by `geecs_python_api/controls/data_acquisition/scan_manager.py`.  

## Future Improvements

Current improvements on the wishlist:

* Cleaning up and expanding THIS README.md file
* GUI for viewing/editing the 1D scan variables and composite variables
* GUI for viewing/editing the actions defined in `actions.yaml`, as well as executing actions without running a scan
* Checkbox to toggle if scan variables return to original position after the scan or not
* Long-term goal, implement an "optimization" scan with Xopt

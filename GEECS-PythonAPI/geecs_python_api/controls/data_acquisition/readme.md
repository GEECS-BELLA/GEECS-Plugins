# Data Acquisition Module

This submodule is part of the geecs_python_api. It is intended to provide a flexible and extensible framework to performing data aquisition in a similar manner to Master Control.

## Table of Contents
- [Overview](#overview)
- [Core Classes](#core-classes)
  - [ScanManager](#scanmanager)
  - [ScanDataManager](#scandatamanager)
  - [DeviceManager](#devicemanager)
  - [DataLogger](#datalogger)
  - [SoundPlayer](#soundplayer)
  - [ActionManager](#actionmanager)
- [Basic Usage](#basic-usage)
- [Future Improvements](#future-improvements)

## Overview

The `data_acquisition` module can be used to manage the acquisition of data from GEECS devices. It achieves data synchronicity through a slightly different approach to Master Control. Rather than shot-based acquisition, it relies on time-based acquisition. In particular, a GEECS devicetype driver must have a variable added called 'timestamp'. The timestamp can either be read directly from the hardware (e.g. from an NiImaq image object). Or, if not availble, systemtime from the computer defined in the aquire state of the driver can be used. 

An additional requirement is a GEECS device that can be used turn off/on the trigger to triggered devices. By turning off the trigger, devices can be put into a 'standby mode' in which the timestamp on the GEECS device becomes frozen. Once the trigger is re-enabled, the net timestamp for each device becomes the t0 which is subsequently used to establish relative timing between all synchronous devices.

In contrast to a Master Control scan, the naming of saved non-scalar data (e.g. images) does not natively follow the "shotnumber" naming convention. At the outset, saved images are arbitrarily named. The DataInterface class is used to read an s-file, extract the timestamp for each device, then it uses that information to find the corresponding saved file associated with that scan. The saved file must have the same timestamp embedded in the meta information and a method must be provided to extract that timestamp. See currently supported devices above. This information is collected and the saved files are renamed according the standard Master Control protocol. Not, due to some features of this general approach, a device which is not selected to be saving synchronously but is selected to be saving non-scalar data can produce "extra" images that don't have corresponding entries in the s-file.

Some neat features: 
- Composite or virtual variables can easily be scanned by defining them in the individual experiments' composite_variable.yaml file. A numerical relation between any set of variables can be defined in using normal numpy form.
- Predefined configuration files are used to select specifically which devices, variables and contents should be saved on each scan. Allows efficient way to add/remove which devices are saving (reduces data clutter)
- Serialized lists of actions can be defined in a the actions.yaml. These can be used as a prescan setup_action. For example, shutter the laser, insert a plunger, move a stage, then start a scan

A GUI has been developed to provide an easy to use interface for running scans in a similar fashion to Master Control. 

Much of the code is meant to be easily applicable across arbitrary BELLA Center labs. However, it has been developed while only testing on HTU. As such, some aspects may still be preferentially coded with HTU in mind. 

Currently supported GEECS device types:
- Point Grey Camera
- PicoscopeV2

Devices to add:
- Thorlabs Spectrometer
- Haso
- Other scopes.
- etc.

## Dependencies
This was primarily developed using the Xopt-GEECS anaconda environment (which exists in this repo). This is also uses the ImageAnalysis module in this same repo. ImageAnalysis is a depency of Xopt-GEECS but I find it useful to install it in editable mode (using pip install -e ImageAnalysis), which is not the default for the Xopt-GEECS env. 

The only additional dependency that is potentially needed is simpleaudio (pip install simpleaudio, not available via conda). This module is used for sound generation on Mac/Linux platforms. 

## Core Classes

### ScanManager

The `ScanManager` is the primary class intended to be interfaced with. There might be some standalone fucntionality to the other classes, but they were mostly written to support this core class. It coordinates and manages the execution of scans, including synchronous and asynchronous devices. It handles trigger control, pre-logging setup, scan step generation, and initiates the data_acquisition.

**Key Features:**
- Triggering device actions before, during, and after scans. (Specifics of triggering may need to change from experiment to experiment)
- Generating scan steps for composite and normal variables.
- Handling synchronous and asynchronous device data.
- Starting, stopping, and managing scan threads.

### ScanDataManager

The `ScanDataManager` class is responsible for managing the data paths, saving data in various formats, and converting logs to different formats like TDMS and CSV. It basically works as an interface between the acquire data and generating output in the exact same fashion as would be done from a Master Control Scan

**Key Features:**
- Managing file paths for saving data.
- Initializing and writing TDMS files.
- Saving "s-files" in text.

### DataInterface

The `DataInterface` class provides some basic tools to interface with generated scan data. It also has key methods for renaming images etc. saved during scans to conform with existing protocols. This could work as a standalone class for basic data operations.

**Key Features:**
- provides quick interface to data
- renames saved images and other to match the previously defined naming convention 


### DeviceManager

The `DeviceManager` class manages experimental devices, including subscribing to device variables and handling their states. It loads configurations for devices and parses and interprets composite variables.

**Key Features:**
- Subscribing devices for event-driven or asynchronous logging.
- Handling composite variables and their component mappings.
- Managing device initialization and reinitialization based on scan configurations.

### DataLogger

The `DataLogger` class handles the acquisition and collation of the scalar data into a timestamp synchronized manner. It manages event-driven TCP subscribtions and asynchronous polling and can trigger sound alerts during logging operations.

**Key Features:**
- Starting and stopping logging for synchronous and asynchronous devices.
- Polling asynchronous devices for data at regular intervals.
- Managing timestamp extraction and checking for duplicate entries.
- Sound notifications for key events (e.g., beeping on new log entries).

### SoundPlayer

The `SoundPlayer` class plays audio notifications during scans. It can play different types of sounds (e.g., beeps and toots) to signal key events, such as scan completion or new data entries.

**Key Features:**
- Playing beeps or toots in the background.
- Running in a separate thread to avoid interrupting the scanning or logging process.

### ActionManager

The `ActionManager` class handles predefined actions that can be executed before, during, or after scans. Actions can be nested, allowing complex sequences of device manipulations.

**Key Features:**
- Loading actions from configuration files.
- Executing actions or nested actions during the scan process.
- Managing device actions such as setting or getting values from devices.

## Basic Usage
Some example usage is available in a jupyter notebook in this repo. Code copied below. Also, please see GEECS-Scanner-GUI for implementation and use

### Running a scan

```python
from geecs_python_api.controls.interface import load_config
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.interface.geecs_errors import ErrorAPI
from geecs_python_api.controls.data_acquisition import ScanManager, SoundPlayer
from geecs_python_api.controls.data_acquisition import visa_config_generator

# Example usage for DataLogger
experiment_dir = 'Undulator'
scan_manager = ScanManager(experiment_dir=experiment_dir)

# Create specialize configuration or just load pre defined config
file1 = visa_config_generator('visa5','spectrometer')
scan_manager.reinitialize(file1)

#test config for a noscan below
scan_config = {'device_var': 'noscan', 'start': -1, 'end': 1, 'step': 2, 'wait_time': 5.5, 'additional_description':'Testing out new python data acquisition module'}

scan_manager.start_scan_thread(scan_config=scan_config)

```

## Future Improvements
this could contain quite lot of things
- **Enhanced Error Handling**: Improve exception handling for edge cases during device communication. In particular, enable more obvious warnings/errors when synchronous devices are not reporting regularly
- **Post scan actions**: Post scan actions, including bespoke data analysis would be excellent to integrate
- **Integrate optimization**: Some of the initial impetus for this was to improve optimization workflows
- **Develop framework for automating basic procedure**: For example, basic start up of HTU e-beam currently involves executing and analyzing a few basic jet position scans, picking optimal positions, then steering the beam to the center of a phosphor screen. Could be automated (or done through optimization)
- **Lots of small/specific code improvements**: Contact Chris/Sam to learn more and contribute

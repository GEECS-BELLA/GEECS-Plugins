# ScanAnalysis

Sub-repository hosting modules for analyzing full scans.  Often times this requires the use of the other sub-repo, "ImageAnalysis" to analyze an individual image.  Here, the main distinquishing functionalities are

* Averaging across bins in a single scan before sending to an image analyzer (and other basic functionality across the scan)
* Analyzers that require data from multiple devices
* Automatically finding an analyzing scans that fit a given criteria.

Designed to work for any GEECS experiment, given the save data is in a similar format to HTU.

## Installation

Requires GEECS-Plugins installed somewhere, Python 3.10, and Poetry.  Can be run off of the Z: drive version, but it is not recommended to edit this version directly if you are planning code development.  Beyond basic usage, we recommend cloning your own copy of the repo with `Github Desktop` onto your local PC.  

Python 3.10 can be installed from the following link:
```link
https://www.python.org/downloads/release/python-31011/
```
Can download and use the `Windows installer (64-bit)` link.  Remember to select the "Add to Path" checkbox at the bottom of the installation window.  Note: Python 3.10.11 is currently the only version of Python this has been developed and tested on, other version could potentially work but no garuntees.

For Poetry, install using their official command-line
```link
https://python-poetry.org/docs/#installing-with-the-official-installer
```

Now, there are two installation paths:  one for just running the GUI and one for getting the environment set up for code development.

### To run the GUI

Open a terminal.  Change directory to this folder (the one with `pyproject.toml`).  Then you should only need to run the bash script `./ScAnalyzer.sh`.  If this throws errors or doesn't work, you can check that the `BASE_PATH` within this bash script is pointing to the correct location of this repository.  (Currently it is hardcoded for HTW's Z drive)

Alternatively, you run the following commands:  (Note:  `main.py` is not in the same directory as `pyproject.toml`)
```commandline
poetry install
cd live_watch/scan_analysis_gui/
poetry run main.py
```

Note:  this may not work if your default python version is not 3.10.  You can see what python version your system defaults to by typing `python --verison`.  If you are experiencing troubles with poetry here, might need to manually specify which python version to use in these two commands.

GEECS-Python API will not work if you do not have a `config.ini` file configured on your local PC.  Currently, ScanAnalysis does not do this automatically.  There are two ways to create this.
1. You can launch the GEECSScanner GUI application (another sub-repo in this repository).  At first launch this will create the ini file if it does not exist.  (Eventually this functionality will be in ScAnalyzer!)
2. You can manually create the file.  Make the directory `~\.config\geecs_python_api\` and in this directory create the file `config.ini`.  In this file, have the following:
```config
[Paths]
geecs_data = C:\GEECS\user data\

[Experiment]
expt = ExperimentNameInMasterControl
```

For the `GEECS user data location` information, you will need to copy the existing user data folder from the server into the local path that you defined in the config.ini above.  For example, if you set your GEECS user data path to be in `C:\GEECS\user data`, then you will need to make the `GEECS` folder in `C:` and copy the user data from the server.  (For example, HTU's server data was located in `Z:\software\control-all-loasis\HTU\user data`)

Lastly, if this is the first time an experiment is being configured, you'll need to map the experiment name to the server address that data is saved at.  This can be done in the `GEECSPython_API` package (also within GEECS-Plugins)
1. Go to `GEECS-PythonAPI/geecs_python_api/controls/interface/geecs_paths_config.py`
2. Add your experiment and server address to `EXPERIMENT_TO_SERVER_DICT` on line 9, matching the formatting of the other entries in the dict.

This should at least get the GUI up and running, but to create and test analyzers in python we'll need to do a bit more work.

## Coding an Analyzer

This section aims to give a summary of what one would need for the bare minimum.  Feel free to use existing analyzers in `Undulator` as a template as well.  Here is a reference list of all the files you'll need to make/update for it to work 100%.
1. The analyzer itself:  (a) Located in `ScanAnalysis/scan_analysis/analyzers/YOUREXPERIMENT/` and (b) is at bare minimum a subclass of `ScanAnalysis` in `ScanAnalysis/scan_analysis/base.py`
2. A mapping of all analyzers in an experiment to the devices that they use:  (a) Located in `ScanAnalysis/scan_analysis/mapping`, (b) named `map_YOUREXPERIMENT.py`, and (c) contains a list of `AnalyzerInfo`'s (see `map_Undulator.py` for an example)
3. Map the experiment name to the list of analyzers in the `EXPERIMENT_TO_MAPPING` dict of `ScanAnalysis/live_watch/scan_analysis_gui/app/ScAnalyzer.py` (Also need an `import` statement before the dict)
4. Experiment name in `config.ini` file is set correctly
5. Experiment name maps to server address in `geecs_paths_config.py` (more info in previous section)

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

### Making an analyzer

For organization purposes, create your analyzer in the `ScanAnalysis/scan_analysis/analyzers/MyExperiment/` directory, replacing `MyExperiment` with your experiment.

At the very least, will need the following:

```python
from scan_analysis.base import ScanAnalyzer
from image_analysis.utils import read_imaq_png_image
from geecs_data_utils import ScanData


class MyCustomAnalyzer(ScanAnalyzer):
    def __init__(self, scan_tag: ScanTag, device_name: Optional[str] = None, skip_plt_show: bool = True,
                 image_analyzer=None):
        super().__init__(scan_tag, device_name=device_name, skip_plt_show=skip_plt_show, image_analyzer=None)

    def run_analysis(self, config_options: Optional[str] = None):
        << Insert
        any
        analysis
        code
        here >>
```

You may also want/need to include more stuff in the `__init__()` function of `MyCustomAnalyzer`.  That is completely fine.  Just remember to (a) have the four required arguments in the method signature (more is ok, but they won't be passed by the GUI and will go to their defaults) and (b) called the `super()` initialization.

**IMPORTANT NOTE**:  If you ever need to import a png that has been saved by labview, you *NEED* to use the `read_imaq_png_image` function.  Labview can save png images with a very strange bit-shift that will scale the image unexpectedly.  This utility function fixes this and loads all pngs images with equivalent scaling.

Remaining work is up to you.  There is a couple of useful features that you can make use of:
1. The `ScanData` module has methods to extract useful information about a given scan, and the base ScanAnalysis creates one by default in `self.scan_data`.  There are a lot of methods that generate folder/file paths and load sfile data.
2. The base initialization of `ScanAnalysis` generates many class variables you can use within `run_analysis`:
3. `self.noscan` is a boolean flag if the scan was a NoScan or not
4. `self.auxiliary_data` is a pandas dataframe of all information in the sfile.
5. `self.scan_parameter` and `self.binned_param_values` has information on the Scanned variable
6. `self.close_or_show_plot()` is a handy function that will show a matplotlib plot if you are running a test (with `skip_plt_show = False` in the init), but during automated analysis this function will just close the plot.
7. `self.append_to_sfile()` another handy function, appends a list to the sfile under a given column name!

Another useful tip is to make a quick block of code at the bottom of this file to allow for testing the analysis on a given scan.  Then you can simply run this file and perform all of the analysis as if it were automated.

```python
if __name__ == "__main__":
    from geecs_data_utils import ScanData
    tag = ScanData.get_scan_tag(year=2025, month=12, day=25, number=1, experiment='MyExperimentName')
    analyzer = MyCustomAnalyzer(scan_tag=tag, skip_plt_show=False)
    analyzer.run_analysis()
```

### Getting the analyzer into the GUI

Once you have the analyzer built and tested that it is working as intended on at least one scan, adding it to the GUI is easy!

First, need to add it to the list of all analyzers for a given experiment. This is located in `ScanAnalysis/scan_analysis/mapping`, named `map_YOUREXPERIMENT.py`, and contains a list of `AnalyzerInfo`'s (see `map_Undulator.py` for an example).  These can be as complex as you need them to be, and the `requirements` block allows for nests `AND` and `OR` lists.  `AND` lists will result in `True` if ALL of the named devices are present in a scan.  `OR` lists result in `True if at least ONE of the named devices is present.

Example 1:  HTU's magspec analysis only just needs that single camera to function
```python
    Info(analyzer_class=HiResMagCamAnalysis,
         requirements={'UC_HiResMagCam'}),
```

Example 2:  This camera can be put through a generic image analyzer, but we still need to pass its device name
```python
    Info(analyzer_class=CameraImageAnalysis,
         requirements={'UC_ALineEBeam3'},
         device_name='UC_ALineEBeam3'),
```

Example 3:  This analyzer works if just one of the cameras is saved.
```python
    Info(analyzer_class=VisaEBeamAnalysis,
         requirements={'OR': ['UC_VisaEBeam1', 'UC_VisaEBeam2', 'UC_VisaEBeam3', 'UC_VisaEBeam4',
                              'UC_VisaEBeam5', 'UC_VisaEBeam6', 'UC_VisaEBeam7', 'UC_VisaEBeam8']}),
```

Example 4:  This analyzer needs the given camera AND at least one of the two ICT devices named.
```python
    Info(analyzer_class=Rad2SpecAnalysis,
         requirements={'AND': ['UC_UndulatorRad2', {'OR': ['U_BCaveICT', 'U_UndulatorExitICT']}]}),
```

Secondly, you'll need to map the experiment name to the list of analyzers in the `EXPERIMENT_TO_MAPPING` dict of `ScanAnalysis/live_watch/scan_analysis_gui/app/ScAnalyzer.py` (Also need an `import` statement before the dict)

### Uploading to google doc automatically

This is an advanced step.  Basically, there are two things you'll need to do:
1. Ensure that all important saved plots have their filenames appended to the `self.display_contents` list in ScanAnalysis.  This list should be the `return` of `run_analysis()`.
2. You'll need to make a custom version of the uploading scripts to match the formatting and logistics of the scan log you use.  This is a bit on the frontier of documentation, but the place to start is at `insert_display_content_to_doc()` function in `ScanAnalysis/scan_analysis/execute_scan_analysis.py` and follow the logic into the `LogMaker4GoogleDocs` package.

### Tips and notes for future changes to ScanAnalysis

Our general vision is for ImageAnalysis to be where a *single* image is analyzed, and for ScanAnalysis to be where anything more complex than that is hosted.  For hardcore image analysis, the correct place to put that is in an ImageAnalyzer and not the ScanAnalysis analyzer, although both will work just fine.

We are currently working on a generalized `2DScanAnalysis` analyzer that will streamline the way images are analyzed.  Rather than defining a `ScanAnalysis`, you instead define an `ImageAnalyzer` in the other package, and just map a device name to the image analyzer through the `2DScanAnalysis` analyzer, which results in not needing to code up a ScanAnalysis class at all.  If this sounds confusing, don't worry.  By the time this functionality is done it should hopefully be very easy to follow an example for this.

## Editing the gui files

The gui's themselves are developed using `pyqt5`, which is a python package for gui development.  To launch the designer tool:

1. Be in an active terminal within the poetry virtual environment
2. `pyqt5-tools designer`  (*If you aren't in the virtual environment, use `poetry run pyqt5-tools designer`)
3. Do whatever you want, saving and closing the designer after you are done.  (GUI files are located in `geecs_scanner\app\gui\`)
4. For any changed file, you'll need to update the python backend code.  To do so, first `cd` into the `gui\` directory.  Then, for anything you changed execute the following:  `pyuic5 -o .\<GUIFile>_ui.py .\<GUIFile>.ui`  (*replacing `<GUIFile>` with the `.ui` file you created/edited)  (**as with step 2, if not in the virtual environment type `poetry run` before this command)

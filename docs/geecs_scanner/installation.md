# GEECS Scanner GUI - Installation & Setup

This guide covers the complete installation and setup process for the GEECS Scanner GUI, including both basic usage and development environment setup.


## Prerequisites

### Required Software

**Python 3.10**:
The scanner requires Python 3.10 specifically. Download from: [Python 3.10.11](https://www.python.org/downloads/release/python-31011/). Use the "Windows installer (64-bit)" link. **Important**: Check "Add Python 3.10 to PATH" during installation. 



**Poetry**:
For dependency management install using the official installer: [Poetry Installation](https://python-poetry.org/docs/#installing-with-the-official-installer). Note, there is a common issue when installing on a windows computer when using the powershell command from the official installer. You may need to change the powershell command to use 'python' rather than 'py' (as noted on the website) if there is an error in installation. After installing, add poetry to PATH. 

**GEECS-Plugins and GEECS-Plugins-Configs Repositories**:
Clone or ensure access to the GEECS Plugings and GEECS Plugins Configuration repositories. For basic usage (run only) it is easiest to map to the 'network' repos (i.e. 'z' or 'n' drive). For Development, it is recommended to clone the repos with GitHub Desktop. The HTU workflow for development is to clone the repositories to the 'source' folder located at C:\GEECS\Developers Version\source. 




### Verify Installation

Check your Python installation:
```bash
python --version
# Should show: Python 3.10.11
```

Check your Poetry installation:
```bash
poetry --version
# Should show: Poetry (version 2.10.x)
# Note: If poetry version is not shown, make sure poetry was added to PATH.
```


## Installation Options

### Option 1: Basic Usage (Run Only)
For users who only want to run the GUI:

 <!-- 1.  #### If you have multiple Python versions, specify Python 3.10 explicitly:
   ```bash
   # Find your Python 3.10 path (example path shown)
   poetry env use C:\Users\username\AppData\Local\Programs\Python\Python310\python.exe
   poetry install
   ```

 2.  #### Verify Environment:
   ```bash
   poetry env info --path
   # This shows the virtual environment location for your IDE configuration. -->
   <!-- ``` -->


 3.  #### Launch the GUI:

    a. **Open Terminal**: Navigate to the GEECS-Scanner-GUI folder in Z: drive
      ```bash
      cd Z:\path\to\control-all-loasis\HTU\Active Version\GEECS-Plugins\GEECS-Scanner-GUI
      ```

    b. **Install Dependencies**:
      ```bash
      poetry install
      ```

    c. **Run the GUI**:
      ```bash
      poetry run python main.py
      ```

 4. #### Experiment Configuration

    When you first run the GUI, a pop-up window will prompt for three essential settings (if not prompted, see the 'Create Configuration File' section):

    a. **GEECS User Data Path**: path to the user data in network drive used for configs, database access etc.

         - Example: `Z:\path\to\control-all-loasis\HTU\user data`

    b. **Experiment Name**: Must match the name shown in Master Control

         - Example: `Undulator`, `Thomson`, etc.
         - Case-sensitive!

    c. **Repetition Rate**: Experiment repetition rate in Hz

         - Used for timing estimates
         - Example: `1` for 1 Hz operation

      These settings are saved to the path defined in the pop-up window. To populate existing scan elements etc, edit this new config file to contain all the paths listed in the `Create Config File` section below.


      Note: If the GUI configuration dialog doesn't work when you first run the GUI, create the config file manually:

      a. **Create Directory**:
         ```bash
         mkdir ~Users/loasis.LOASIS/.config/geecs_python_api
         ```

       b. **Create Config File** (`~Users/loasis.LOASIS/.config/geecs_python_api/config.ini`):
         ```ini
         [Paths]
         geecs_data = Z:\path\to\control-all-loasis\HTU\user data
         scanner_config_root_path = Z:\path\to\control-all-loasis\HTU\Active Version\GEECS-Plugins-Configs
         image_analysis_configs_path = Z:\path\to\control-all-loasis\HTU\Active Version\GEECS-Plugins-Configs\image_analysis_configs
         scan_analysis_configs_path = Z:\path\to\control-all-loasis\HTU\Active Version\GEECS-Plugins-Configs\scan_analysis_configs

         [Experiment]
         expt = YourExperimentName
         rep_rate_hz = 1
         ```

 5. #### Create Shortcut to launch GUI day-to-day:

      Navigate to `Z:\path\to\control-all-loasis\HTU\Active Version\GEECS-Plugins\GEECS-Scanner-GUI` and create a desktop shortcut to the following file:
      ```bash
      ./GEECS_Scanner.sh
      ```
      Double-click this script every time you want to launch the GUI. If needed, edit this script and add `poetry lock` before `poetry install`. 


### Option 2: Develop + Basic Usage 

For users who want to run and develop/customize the GUI:

 1.  #### If you have multiple Python versions, specify Python 3.10 explicitly:
   ```bash
   # Find your Python 3.10 path (example path shown)
   poetry env use C:\Users\username\AppData\Local\Programs\Python\Python310\python.exe
   poetry install
   ```

 2.  #### Verify Environment:
   ```bash
   poetry env info --path
   # This shows the virtual environment location for your IDE configuration.
   ```
   
 3.  #### Setup Data Directory:

    You must copy the experiment's user data from the server to your local path if not already done:

    a. **Create Local Directory**: Make the directory specified in your config
         ```bash
         mkdir "C:\GEECS\user data"
         ```

    b. **Copy Server Data**: Copy existing user data from the server into the local directory specified in previous step.

         - HTU example: Copy from `Z:\software\control-all-loasis\HTU\user data`
         - Copy to your configured local path

 4.  #### Launch the GUI:

    a. **Open Terminal**: Navigate to the GEECS-Scanner-GUI folder
      ```bash
      cd path/to/GEECS-Plugins/GEECS-Scanner-GUI
      ```

    b. **Install Dependencies**:
      ```bash
      poetry install
      ```

    c. **Run the GUI**:
      ```bash
      poetry run python main.py
      ```

 5. #### Experiment Configuration

    When you first run the GUI, a pop-up window will prompt for three essential settings (if not prompted, see the 'Create Configuration File' section):

    a. **GEECS User Data Path**: Local path to the user data used for configs, database access etc.

      - Example: `C:\GEECS\user data\`

    b. **Experiment Name**: Must match the name shown in Master Control

      - Example: `Undulator`, `Thomson`, etc.
      - Case-sensitive!

    c. **Repetition Rate**: Experiment repetition rate in Hz

      - Used for timing estimates
      - Example: `1` for 1 Hz operation

    These settings are saved to: `~\.config\geecs_python_api\config.ini`


    Note: If the GUI configuration dialog doesn't work when you first run the GUI, create the config file manually:

    a. **Create Directory**:
      ```bash
      mkdir ~/.config/geecs_python_api
      ```

    b. **Create Config File** (`~/.config/geecs_python_api/config.ini`):
      ```ini
      [Paths]
      geecs_data = C:\GEECS\user data\
      scanner_config_root_path = C:\GEECS\Developers Version\source\GEECS-Plugins-Configs
      image_analysis_configs_path = C:\GEECS\Developers Version\source\GEECS-Plugins-Configs\image_analysis_configs
      scan_analysis_configs_path = C:\GEECS\Developers Version\source\GEECS-Plugins-Configs\scan_analysis_configs

      [Experiment]
      expt = YourExperimentName
      rep_rate_hz = 1
      ```
    Modify the script if your installation path differs from the default. 


 5. #### Create Shortcut to launch GUI day-to-day

      Navigate to `C:\GEECS\Developers Version\source\GEECS-Plugins\GEECS-Scanner-GUI` and create a desktop shortcut to the following file:
      ```bash
      ./GEECS_Scanner.sh
      ```
      Double-click this script every time you want to launch the GUI. Note: you can modify this script to remove `poetry install` to remove redundancy.



### Option 2: Development Environment

For code development and customization:

1. **Navigate to Project Directory**:
   ```bash
   cd path/to/GEECS-Plugins/GEECS-Scanner-GUI
   ```

2. **Create Virtual Environment**:
   ```bash
   poetry install
   ```

   **If you have multiple Python versions**, specify Python 3.10 explicitly:
   ```bash
   # Find your Python 3.10 path (example path shown)
   poetry env use C:\Users\username\AppData\Local\Programs\Python\Python310\python.exe
   poetry install
   ```

3. **Verify Environment**:
   ```bash
   poetry env info --path
   ```
   This shows the virtual environment location for your IDE configuration.

## First-Time Configuration







## Development Setup

### GUI Development

The GUI uses PyQt5. To edit GUI files:

1. **Launch Designer**:
   ```bash
   # From within the poetry environment
   pyqt5-tools designer
   ```

2. **Edit GUI Files**: Located in `geecs_scanner/app/gui/`

3. **Compile Changes**: After editing any `.ui` file:
   ```bash
   cd geecs_scanner/app/gui/
   pyuic5 -o ./YourFile_ui.py ./YourFile.ui
   ```

### Adding Dependencies

To add new Python packages:
```bash
poetry add package_name
```

## Troubleshooting

### Common Issues

#### Python Version Problems
**Error**: Poetry can't find Python 3.10
**Solution**: Explicitly specify the Python path:
```bash
poetry env use /path/to/python3.10/python.exe
```

#### First Launch Issues
**Problem**: GUI freezes or doesn't enable scanning after first configuration
**Solution**: Restart the GUI after initial configuration

#### No Save Elements Error
**Problem**: Starting scan with no save elements causes freeze
**Solution**: Always add at least one save element before starting a scan

#### Immediate Stop Issues
**Problem**: Stopping scan immediately after start causes problems
**Solution**: Wait a few seconds after starting before stopping

### Configuration Reset

To reconfigure experiment settings, use the "Reset Config" button in the GUI, or delete the config file:
```bash
rm ~/.config/geecs_python_api/config.ini
```

## Verification

### Test Installation

1. **Launch GUI**: Should open without errors
2. **Check Configuration**: Experiment settings should be loaded
3. **Test Elements**: Try creating a save element
4. **Test Scan**: Run a simple NoScan to verify functionality

### Expected File Structure

After successful setup, you should have:
```
~/.config/geecs_python_api/
├── config.ini

C:/GEECS/user data/
├── [experiment-specific directories]
```

## Next Steps

Once installation is complete:

1. **Tutorial**: Work through the [Tutorial](tutorial.md) for hands-on learning
2. **Configuration**: Set up timing and scan variables for your experiment
3. **Elements**: Create save elements for your devices
4. **Scanning**: Start with simple NoScans before attempting parameter scans

---

*For additional help, consult the [Tutorial](tutorial.md).*

# Xopt-GEECS
GUI for using Xopt in BELLA GEECS through GEECS-PythonAPI and Badger. The "geecs" interface handles all set/get commands that originate from badger. To use, an enviroment for the optimization problem must be created. See badger documentation and the bella_test environment on how to implement. 

## Windows Specific Instructions
Badger is only explicitly supported on Linux, but the installation procedure is also verified on Mac. Badger was made to work on windows using a slightly different conda environment. Make sure to use environment_win.yml to create the Windows-specifc environment.

1. Make sure you have `conda` installed ([Conda installation](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)) and open up the Anaconda Prompt
2. Navigate to the `Xopt-GEECS` directory
3. Run `conda env create -f environment_win.yml`.
	1. Sometimes there may be errors with installing the `geecs-pythonapi` package from GEECS-Plugins.  Can check if the install worked properly by typing the following in the Anaconda Prompt:
 	2. `conda activate Xopt-GEECS`
  	3. `pip list`
   	4. Look for `geecs-pythonapi` in the list.  If it exists, then should be good to go!  Otherwise, might need to track down package version issues between the `poetry.toml` files in `GEECS-Plugins/GEECS-PythonAPI/` and `GEECS-Plugins/ImageAnalysis/`.
4. Run (in Powershell with Admin Privileges):
	`.\initialize_config_file.bat`
	This will generate a config file and a shortcut to the config file in the current directory if either does not exist.
5. If needed, edit the config file for your experiment and/or virtual environment location:
	1. To use the geecs-python-api necessary for communicating with hardware from within badger, the api needs to know where the GEECS "user data" folder in addition to the experiment name and conda environment path. This should be done using the config file option because Badger runs from outside the normal GEECS file structure. In other words, a config file should be created in this path: `<username>/.config/geecs_python_api/config.ini` and it should have entries like this:
```config.ini
[Paths] 
geecs_data = C:\GEECS\user data\ 
conda_xopt = C:\Users\loasis.LOASIS\AppData\Local\anaconda3\envs\Xopt-GEECS 
 
[Experiment] 
expt = Undulator	# Name of the experiment as seen in Master Control.
examples = True		# Flag for whether to include symlinks between Badger and the example XOpt Environments
legacy = False 		# Flag for whether to incldue symlinks between Badger and old environments (not suggested)

```
6. Run (in Powershell with Admin Privileges):
	`.\setup_Xopt-GEECS.bat`
	1. Keep this window open!  It is helpful to rerun this script to refresh the available symlinks to optimization environments.

## Creating an Optimization Environment
Most of this is currently not automated.  Best practice is to follow the organization of environments located in the `examples\` experimental subfolder in `environments\`.  If creating a new experimental subfolder, do remember to create a blank `__init__.py` file in the experimental subfolder.  Also, when creating a new environment, do remember that the `name` parameter in the environment's `__init__.py` file matches the environment's folder name.

After creating a new environment, you will need to run `.\setup_Xopt-GEECS.bat` again to symlink the new environment from the conda virtual environment.

Note:  By default the `generate_symlink_bat.py` script will assume that the experimental subfolder within `environments/` is the lowercase version of whatever `expt` you specify in the config file above.  If you want to customize this (ie: an expt of "Bella" refers to the "pw" exprimental subfolder) then you need to include a dictionary item at the top of `generate_symlink_bat.py`.

## Running Badger - Quick Startup Guide
1. Open Anaconda Prompt window
2. `conda activate Xopt-GEECS`
3. `badger -g`
4. For the first time launching Badger:
	1. When the window opens click the "Badger settings" cog button at the bottom-right hand corner.
	2. In the four "Root" paths, enter in the location of the `Badger-Plugins` folder within the `Xopt-GEECS` conda environment.
	3. Ex: `C:\Users\loasis.LOASIS\AppData\Local\anaconda3\envs\Xopt-GEECS\Lib\Badger-Plugins`
5. Add a routine by clicking the `+` button in the top-left next to the Search bar
	1. give the routine a name or can leave as the randomly-generated name
	2. specify the algorithm (can use `upper_confidence_bound` and set `use_low_noise_prior: false` for experimental data)
 	3. select the environment from the list.  If it does not show up make sure the symlink was created correctly and/or try running `./setup_Xopt-GEECS.bat` again from a Admin Powershell terminal
  	4. adjust input variables as needed
   	5. check the boxes to add variables to the routine
   	6. click `Add Current` to create start points for your variables
   	7. check the box to add the objective(s) and select if they should be maximized or minizimized (most often want maximize)
6. On the run monitor for the routine, click the big green button

## OUTDATED Linux Install
Badger core is maintained by the SLAC-ML group in their slaclab/badger github repo. 

1. Make sure you have `conda` installed ([Conda installation](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)) and open up the Anaconda Prompt window
2. Navigate to the `Xopt-GEECS` directory
3. Run `conda env create -f environment.yml`.
4. To use the geecs-python-api necessary for communicating with hardware from within badger, the api needs to know where the GEECS "user data" folder is and what the default experiment name is. This should be done using the config file option because Badger runs from outside the normal GEECS file structure. In other words, a config file should be created in this path: `<username>/.config/geecs_python_api/config.ini` and it should have two entries like this:
```config.ini
[Paths]
geecs_data = C:\GEECS\user data\

[Experiment]
expt = Undulator
```
6. Copy the Badger-Plugins into the environment[^1]:
    1. Find the path to the `Xopt-GEECS` environment with `conda info --envs`
    2. Run `git clone https://github.com/slaclab/Badger-Plugins.git <path to Xopt-GEECS env>\Lib\Badger-Plugins` filling in the path you found in the previous step.
7. Create symlinks in the `environment` and `interface` folders in `<path to Xopt-GEECS env>\Lib\Badger-Plugins` called `geecs` to the `badger-plugins/environments/geecs` and  `badger-plugins/interfaces/geecs` folders respectively (see `badger-plugins/README.md`).  For windows there is an automated script for this.  An example of the old way is located at the bottom of this README.

[^1] If anyone knows how to automate the Badger-Plugins copy, either by:
* Adding the clone-repo instruction from within the `environment.yml` file
* Setting the Xopt-GEECS environment path to a variable

that would be appreciated!

### Old symlink .bat file
```setup_Xopt-GEECS.bat
SET "ANACONDA_Xopt_GEECS=C:\Users\loasis.LOASIS\AppData\Local\anaconda3\envs\Xopt-GEECS"

git clone https://github.com/slaclab/Badger-Plugins.git "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins"

mklink %ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\environments\geecs "C:\GEECS\Developers Version\source\GEECS-Plugins\Xopt-GEECS\badger-plugins\environments\geecs"
mklink "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\environments\camera_exposure_time_test" "C:\GEECS\Developers Version\source\GEECS-Plugins\Xopt-GEECS\badger-plugins\environments\camera_exposure_time_test"
mklink "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\environments\HTU_hex_alignment_sim" "C:\GEECS\Developers Version\source\GEECS-Plugins\Xopt-GEECS\badger-plugins\environments\HTU_hex_alignment_sim"
mklink "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\interfaces\geecs" "C:\GEECS\Developers Version\source\GEECS-Plugins\Xopt-GEECS\badger-plugins\interfaces\geecs"
pause

```

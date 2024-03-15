# Xopt-GEECS
GUI for using Xopt in BELLA GEECS through GEECS-PythonAPI and Badger. The "geecs" interface handles all set/get commands that originate from badger. To use, an enviroment for the optimization problem must be created. See badger documentation and the bella_test environment on how to implement. 

## Install
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

## Windows Specific Instructions
Badger is only explicitly supported on Linux, but the above installation procedure is also verified on Mac. Badger was made to work on windows using a slightly different conda environment. Instead use environment_win.yml to create to windows specifc environment.

After creating the Xopt-GEECS conda env using the correct environment file, you can continue the above instructions. Or to expedite steps 5 and 6, the batch script `setup_Xopt-GEECS.bat` can be executed in a cmd prompt run as administrator to finalize the installation. Notes: (1) ensure the path to the Xopt-GEECS in the first line is correct (verified by using: `conda info --envs`) and (2) make sure the config file is created, as described below.

1. Make sure you have `conda` installed ([Conda installation](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)) and open up the Anaconda Prompt
2. Navigate to the `Xopt-GEECS` directory
3. Run `conda env create -f environment_win.yml`.
4. To use the geecs-python-api necessary for communicating with hardware from within badger, the api needs to know where the GEECS "user data" folder is and what the default experiment name is. This should be done using the config file option because Badger runs from outside the normal GEECS file structure. In other words, a config file should be created in this path: `<username>/.config/geecs_python_api/config.ini` and it should have two entries like this:
```config.ini
[Paths]
geecs_data = C:\GEECS\user data\

[Experiment]
expt = Undulator
```
6. Run (in Powershell with Admin Privileges):
	`Start-Process -FilePath .\setup_Xopt-GEECS.bat`

## Running Badger - Quick Startup Guide
1. Open Anaconda Prompt window
2. `conda activate Xopt-GEECS`
3. `badger -g`
4. Add a routine
	1. specify the algorithm (can use 'upper_confidence_bound' and set 'use_low_noise_prior' to false for experimental data)
 	2. select the environment from the list.  If it does not show up make sure the symlink was created correctly and/or try running "setup_Xopt-GEECS.bat" again
  	3. adjust input variables as needed
   	4. check the boxes to add variables to the routine
   	5. click 'Add Current' to create start points for your variables
   	6. check the box to add the objective(s) and select if they should be maximized or minizimized (most often want maximize)
5. On the run monitor for the routine, click the big green button

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

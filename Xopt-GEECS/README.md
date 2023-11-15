# Xopt-GEECS

GUI for using Xopt in BELLA GEECS through GEECS-PythonAPI and Badger

## Install

Badger is in the process of refactoring some items and will release a version 
with refactored names soon. This project will use the refactored names using
Badger's `plugin-refactoring` branch ahead of that release. 

1. Make sure you have `conda` installed ([Conda installation](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html))
1. Navigate to the `Xopt-GEECS` directory
1. Run `conda env create -f environment.yml`.
1. Copy the Badger-Plugins into the environment[^1]:
    1. Find the path to the `Xopt-GEECS` environment with `conda info --envs`
    1. Run `git clone https://github.com/SLAC-ML/Badger-Plugins.git <path to Xopt-GEECS env>\Lib\Badger-Plugins`  filling in the path you found in the previous step.
    1. Navigate to the new Badger-Plugins folder, and switch to the 
    `plugin-refactoring` branch using `git checkout plugin-refactoring`

[^1] If anyone knows how to automate the Badger-Plugins copy, either by:
* Adding the clone-repo instruction from within the `environment.yml` file
* Setting the Xopt-GEECS environment path to a variable

that would be appreciated!


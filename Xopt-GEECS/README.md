# Xopt-GEECS

GUI for using Xopt in BELLA GEECS through GEECS-PythonAPI and Badger. The "geecs" interface handles all set/get commands that originate from badger. To use, an enviroment for the optimization problem must be created. See badger documentation and the bella_test environment on how to implement. 

## Install

Badger core is maintained by the SLAC-ML group in their slaclab/badger github repo. 

1. Make sure you have `conda` installed ([Conda installation](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html))
1. Navigate to the `Xopt-GEECS` directory
1. Run `conda env create -f environment.yml`.
1. Copy the Badger-Plugins into the environment[^1]:
    1. Find the path to the `Xopt-GEECS` environment with `conda info --envs`
    1. Run `git clone https://github.com/slaclab/Badger-Plugins.git <path to Xopt-GEECS env>\Lib\Badger-Plugins` 
       filling in the path you found in the previous step.
1. Create symlinks in the `environment` and `interface` folders in `<path to Xopt-GEECS env>\Lib\Badger-Plugins`
   called `geecs` to the `badger-plugins/environments/geecs` and 
   `badger-plugins/interfaces/geecs` folders respectively (see `badger-plugins/README.md`)

[^1] If anyone knows how to automate the Badger-Plugins copy, either by:
* Adding the clone-repo instruction from within the `environment.yml` file
* Setting the Xopt-GEECS environment path to a variable

that would be appreciated!


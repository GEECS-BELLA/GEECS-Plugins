"""
Writes a .bat file to symlink all of the folders within badger-plugins to the conda virtual environment location for
badger.  Make sure that your conda environment is specified in setup_Xopt-GEECS.bat
(the default is 'C:/Users/loasis.LOASIS/AppData/Local/anaconda3/env/Xopt-GEECS')

-Chris
"""

import os
import sys
from geecs_python_api.controls.interface import load_config

if len(sys.argv) > 1:
    conda_env_directory = sys.argv[1]
else:
    print("No conda directory given.  Assuming default in py script:")
    conda_env_directory = r'C:\Users\loasis.LOASIS\AppData\Local\anaconda3\envs\Xopt-GEECS'

config = load_config()
if config and 'Experiment' in config and 'expt' in config['Experiment']:
    default_experiment = config['Experiment']['expt']
    print(f"default experiment is: {default_experiment}")
else:
    print("Configuration file not found or default experiment not defined. While use Undulator as experiment. Could "
          "be a problem for you.")
    default_experiment = 'Undulator'

environment_experiment_dictionary = {
    'Undulator': 'undulator',
    'WhatIsThePWExperimentName?': 'pw'
}
environment_subfolder_list = ['examples',
                              'geecs_general',
                              'legacy',
                              environment_experiment_dictionary[default_experiment]]

environment_folder = r'\badger-plugins\environments'
current_directory = os.getcwd()
batch_filename = current_directory + r'\autosetup_Xopt-GEECS.bat'

with open(batch_filename, 'w') as file:
    file.write(r'SET "ANACONDA_Xopt_GEECS=' + conda_env_directory + '"\n')
    file.write(r'SET "Xopt_GEECS_Directory=' + str(current_directory) + '"\n\n')
    file.write(r'git clone https://github.com/slaclab/Badger-Plugins.git "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins"'+'\n\n')
    file.write(r'mklink "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\interfaces\geecs" "%Xopt_GEECS_Directory%\badger-plugins\interfaces\geecs"'+'\n\n')

    print("Writing links to the following environments:")
    for sub_experiment in environment_subfolder_list:
        print("--", sub_experiment, "--")
        badger_env_dir = r'%Xopt_GEECS_Directory%'+environment_folder
        subfolder = current_directory + environment_folder + '\\' + sub_experiment
        entries = os.listdir(subfolder)
        environments = [entry for entry in entries if os.path.isdir(os.path.join(subfolder, entry))]
        print(environments)

        conda_env = r'%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\environments'
        for environment in environments:
            env_folder = '\\' + str(environment)
            file.write(r'mklink "' + conda_env + env_folder + '" "' + badger_env_dir + '\\' + sub_experiment + env_folder + '"\n')

    file.write('\n' + r'ECHO Finished symlinks, deleting auto-script...' + '\n')
    file.write(r'pause' + '\n')
    file.write(r'del "%~f0"' + '\n')

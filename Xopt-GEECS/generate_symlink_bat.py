"""
Writes a .bat file to symlink all the folders within badger-plugins to the conda virtual environment location for
badger.  Make sure that your conda environment is specified in setup_Xopt-GEECS.bat
(the default is 'C:/Users/loasis.LOASIS/AppData/Local/anaconda3/env/Xopt-GEECS')

-Chris
"""

import os
import sys
import configparser


default_config_file_location = '~/.config/geecs_python_api/config.ini'
environment_experiment_dictionary = {
    'Undulator': 'undulator',
    'Bella': 'pw',
    'Thomson': 'thomson'
}


def load_config():
    config_file = configparser.ConfigParser()
    config_path = os.path.expanduser(default_config_file_location)
    if os.path.exists(config_path):
        config_file.read(config_path)
        return config_file
    else:
        return None


environment_subfolder_list = ['geecs_general']
missing_information = False
conda_env_directory = None

config = load_config()
if config:
    if 'Experiment' in config and all(key in config['Experiment'] for key in ['expt', 'examples', 'legacy',]):
        default_experiment = config['Experiment']['expt']
        print(f"Configured experiment is: {default_experiment}")
        if default_experiment in environment_experiment_dictionary:
            environment_subfolder_list.append(environment_experiment_dictionary[default_experiment])
        else:
            assumed_directory = default_experiment.lower()
            print(f"Unknown experiment for generating symlinks, assuming {assumed_directory}")
            environment_subfolder_list.append(assumed_directory)
        if config['Experiment']['examples'].lower() == "true":
            environment_subfolder_list.append('examples')
        if config['Experiment']['legacy'].lower() == "true":
            environment_subfolder_list.append('legacy')
    else:
        missing_information = True

    if 'Paths' in config and 'conda_xopt' in config['Paths']:
        conda_env_directory = config['Paths']['conda_xopt']
    else:
        missing_information = True
else:
    missing_information = True
if missing_information:
    print("ERROR: Missing information from configuration file, consider deleting the existing config file and/or "
          "running 'initialize_config_file.bat'")
    sys.exit()


environment_folder = r'\badger-plugins\environments'
current_directory = os.getcwd()
batch_filename = current_directory + r'\autosetup_Xopt-GEECS.bat'

with open(batch_filename, 'w') as file:
    file.write(r'ECHO Attempting to clone Badger-Plugins git...' + '\n')
    file.write(r'SET "ANACONDA_Xopt_GEECS=' + conda_env_directory + '"\n')
    file.write(r'SET "Xopt_GEECS_Directory=' + str(current_directory) + '"\n\n')
    file.write(r'git clone https://github.com/slaclab/Badger-Plugins.git "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins"' +
               '\n\n')
    file.write(r'ECHO.' + '\n')

    file.write(r'ECHO Attempting to symlink the geecs interface in Badger-Plugins...' + '\n')
    file.write(r'mklink "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\interfaces\geecs" '
               r'"%Xopt_GEECS_Directory%\badger-plugins\interfaces\geecs"'+'\n\n')
    file.write(r'ECHO.' + '\n')

    conda_env = r'%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\environments'
    file.write(r'CALL remove_existing_symlinks.bat "' + conda_env + '"\n')
    file.write(r'ECHO.' + '\n')

    file.write(r'ECHO Generating new symbolic links...' + '\n')
    print("Writing links to the following environments:")
    for sub_experiment in environment_subfolder_list:
        print("--", sub_experiment, "--")
        badger_env_dir = r'%Xopt_GEECS_Directory%'+environment_folder
        subfolder = current_directory + environment_folder + '\\' + sub_experiment
        if not os.path.exists(subfolder):
            print("Experiment subfolder", sub_experiment, "does not exist!  Creating an empty subfolder:")
            os.makedirs(subfolder)
            init_file_path = os.path.join(subfolder, "__init__.py")
            with open(init_file_path, "w") as temporary_generation:
                pass
        else:
            entries = os.listdir(subfolder)
            environments = [entry for entry in entries if os.path.isdir(os.path.join(subfolder, entry))]
            print(environments)

            for environment in environments:
                env_folder = '\\' + str(environment)
                file.write(r'mklink "' + conda_env + env_folder + '" "' +
                           badger_env_dir + '\\' + sub_experiment + env_folder + '"\n')

    print()
    file.write(r'ECHO.' + '\n')
    file.write('\n' + r'ECHO Finished symlinks, deleting auto-script...' + '\n')
    file.write(r'pause' + '\n')
    file.write(r'start cmd /c del "%~f0"' + '\n')
    #file.write(r'exit' + '\n')

    #file.write(r'del "%~f0"' + '\n')

"""
Writes a .bat file to symlink all of the folders within badger-plugins to the conda virtual environment location for
badger.  Make sure that your conda environment is specified in setup_Xopt-GEECS.bat
(the default is 'C:/Users/loasis.LOASIS/AppData/Local/anaconda3/env/Xopt-GEECS')

-Chris
"""

import os
import sys

if len(sys.argv) > 1:
    conda_env_directory = sys.argv[1]
else:
    print("No conda directory given.  Assuming default in py script:")
    conda_env_directory = r'C:\Users\loasis.LOASIS\AppData\Local\anaconda3\envs\Xopt-GEECS'

environment_folder = r'\badger-plugins\environments'
current_directory = os.getcwd()
batch_filename = current_directory + r'\autosetup_Xopt-GEECS.bat'

with open(batch_filename, 'w') as file:
    file.write(r'SET "ANACONDA_Xopt_GEECS=' + conda_env_directory + '"\n')
    file.write(r'SET "Xopt_GEECS_Directory=' + str(current_directory) + '"\n\n')
    file.write(r'git clone https://github.com/slaclab/Badger-Plugins.git "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins"'+'\n\n')

    badger_env_dir = r'%Xopt_GEECS_Directory%'+environment_folder
    entries = os.listdir(current_directory + environment_folder)
    environments = [entry for entry in entries if os.path.isdir(os.path.join(current_directory+environment_folder, entry))]

    file.write(r'mklink "%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\interfaces\geecs" "%Xopt_GEECS_Directory%\badger-plugins\interfaces\geecs"'+'\n\n')

    conda_env = r'%ANACONDA_Xopt_GEECS%\Lib\Badger-Plugins\environments'
    for environment in environments:
        env_folder = '\\' + str(environment)
        file.write(r'mklink "' + conda_env + env_folder + '" "' + badger_env_dir + env_folder + '"\n')

    file.write('\n' + r'ECHO Finished symlinks, deleting auto-script...' + '\n')
    file.write(r'pause' + '\n')
    file.write(r'del "%~f0"' + '\n')

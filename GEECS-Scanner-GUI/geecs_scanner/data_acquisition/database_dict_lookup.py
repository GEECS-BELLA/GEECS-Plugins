import logging
from typing import Optional
from geecs_python_api.controls.interface import load_config, GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice


class DatabaseDictLookup:
    """ Stores the current database dictionary. Reloads upon new Scan Manager instance, but only if the experiment
     name has changed.  """
    def __init__(self):
        self.experiment_name = None
        self.load_config_flag = False
        self.database_dict: dict = {}

    def get_database(self) -> dict:
        return self.database_dict

    def reload(self, experiment_name: Optional[str] = None):
        if self.experiment_name == experiment_name and self.load_config_flag:
            return
        self.load_config_flag = True  # Flag ensures config file is read at least once if no experiment name given

        if experiment_name is None:
            config = load_config()

            if config and 'Experiment' in config and 'expt' in config['Experiment']:
                experiment_name = config['Experiment']['expt']
                logging.info(f"default experiment is: {experiment_name}")
            else:
                logging.warning("Configuration file not found or default experiment not defined.")

        self.experiment_name = experiment_name
        try:
            GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(experiment_name)
            self.database_dict = GeecsDevice.exp_info['devices']
        except AttributeError:
            logging.warning("Could not retrieve dictionary of GEECS Devices")
            self.database_dict = {}

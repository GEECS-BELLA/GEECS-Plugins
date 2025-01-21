import sys
import traceback
import configparser
from pathlib import Path


def exception_hook(exctype, value, tb):
    """
    Global wrapper to print out tracebacks of python errors during the execution of a PyQT window.

    :param exctype: Exception Type
    :param value: Value of the exception
    :param tb: Traceback
    """
    print("An error occurred:")
    traceback.print_exception(exctype, value, tb)
    sys.__excepthook__(exctype, value, tb)
    sys.exit(1)


class ApplicationPaths:
    """
    Contains paths and folder names used by GEECS Scanner.  Upon initialization, creates these folders if non-existent
    """

    CONFIG_PATH = Path('~/.config/geecs_python_api/config.ini').expanduser()
    BASE_PATH = Path(__file__).parents[1] / "scanner_configs" / "experiments"
    SAVE_DEVICES_FOLDER = "save_devices"
    SCAN_DEVICES_FOLDER = "scan_devices"
    PRESET_FOLDER = "scan_presets"
    MULTISCAN_FOLDER = "multiscan_presets"
    SHOT_CONTROL_FOLDER = "shot_control_configurations"

    def __init__(self, experiment: str, create_new: bool = True):
        if experiment == "":
            raise ValueError("Cannot set empty experiment in Application Paths")

        self.exp_path = self.BASE_PATH / experiment

        self.exp_save_devices = self.exp_path / self.SAVE_DEVICES_FOLDER
        self.exp_scan_devices = self.exp_path / self.SCAN_DEVICES_FOLDER
        self.exp_presets = self.exp_path / self.PRESET_FOLDER
        self.exp_multiscan = self.exp_path / self.MULTISCAN_FOLDER
        self.exp_shot_control = self.exp_path / self.SHOT_CONTROL_FOLDER

        if create_new:
            for attr_name in dir(self):
                attr_val = getattr(self, attr_name)
                if isinstance(attr_val, Path):
                    if not attr_val.exists():
                        print(f"Creating folder: '{attr_val}'")
                        attr_val.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def create_config_if_missing() -> bool:
        """
        Creates a new config file with default values if one doesn't already exist
        :return: True if new file was created.  Otherwise, False
        """
        file = ApplicationPaths.config_file()
        if file.exists():
            return False
        file.parent.mkdir(parents=True, exist_ok=True)
        default_content = configparser.ConfigParser()
        default_content['Paths'] = {
            'geecs_data': 'C:\\GEECS\\user data\\'
        }
        default_content['Experiment'] = {
            'expt': '',
            'rep_rate_hz': '1',
        }
        with open(file, 'w') as config_file:
            default_content.write(config_file)

        return True

    @staticmethod
    def config_file() -> Path:
        """ :return: folder of user .ini config file """
        return ApplicationPaths.CONFIG_PATH

    @staticmethod
    def base_path() -> Path:
        """ :return: root folder for all experiment config files """
        return ApplicationPaths.BASE_PATH

    def experiment(self) -> Path:
        """ :return: root folder for all config files in given experiment """
        return self.exp_path

    def save_devices(self) -> Path:
        """ :return: folder for the save device yaml's """
        return self.exp_save_devices

    def scan_devices(self) -> Path:
        """ :return: folder for the scan device yaml's """
        return self.exp_scan_devices

    def presets(self) -> Path:
        """ :return: folder for the scan preset yaml's """
        return self.exp_presets

    def multiscan_presets(self) -> Path:
        """ :return: folder for the multiscan preset yaml's """
        return self.exp_multiscan

    def shot_control(self) -> Path:
        """ :return: folder for the timing configuration yaml's """
        return self.exp_shot_control

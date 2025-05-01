from __future__ import annotations

import os
import logging

from pathlib import Path
from dateutil.parser import parse as dateparse
from configparser import ConfigParser
from typing import Optional, Union, NamedTuple

# module‐level logger
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

# support both strings and real Path objects
SysPath = Union[str, bytes, os.PathLike, Path]

EXPERIMENT_TO_SERVER_DICT: dict[str, Path] = {
    'Undulator': Path('Z:/data'),
    'Thomson': Path('Z:/data'),
    'DataLogging': Path('N:/data/PWlaserData')
}

class ScanTag(NamedTuple):
    year: int
    month: int
    day: int
    number: int
    experiment: Optional[str] = None

def month_to_int(month: Union[str, int]) -> int:
    """ :return: an integer representing the given month """
    try:
        month_int = int(month)
        if 1 <= month_int <= 12:
            return month_int
    except ValueError:
        pass

    if isinstance(month, str):
        return dateparse(month).month
    else:
        raise ValueError(f"'{month}' is not a valid month")

class ConfigurationError(Exception):
    """Exception raised for errors in the configuration file."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class GeecsPathsConfig:
    """
    Manages configuration for GEECS-related paths and experiment settings.

    Attributes:
    ----------
    base_path : Path
        The base directory for storing GEECS data locally.
    experiment : str
        The default experiment name.
    """

    def __init__(self,
                 config_path: Path = Path('~/.config/geecs_python_api/config.ini').expanduser(),
                 default_experiment: Optional[str] = None,
                 set_base_path: Optional[Union[Path, str]] = None):
        """
        Loads paths and experiment settings from a configuration file.  Will first try the given base path from the
        initialization argument.  Then will try the default server address in the above dictionary.  Lastly, if still
        not located, will try loading the config file and using the defined base path there.

        Parameters:
        ----------
        config_path : Path, optional
            Path to the configuration file (default: ~/.config/geecs_python_api/config.ini).
        default_experiment : str, optional
            Default experiment name (default: "Undulator").
        set_base_path : Path, optional
            Default path for locating GEECS data (default: Z:/data).

        Raises:
        ------
        ValueError
            If either the experiment name or base path was not defined by the end of initialization
        """

        # First try using the explicit options given as options
        if set_base_path is None or not Path(set_base_path).exists():
            base_path = None
        else:
            base_path = Path(set_base_path)
        experiment = default_experiment

        # See if the server address can be extracted from the experiment name without loading config file
        if experiment is not None and base_path is None:
            base_path = self._validate_and_set_base_path(self.get_default_server_address(experiment))

        # If either was not set in arguments, then open up the config file and read its contents
        if experiment is None or base_path is None:
            config = ConfigParser()
            if config_path.exists():
                try:
                    config.read(config_path)

                    # Get the experiment name if it is not provided
                    if experiment is None:
                        experiment = config['Experiment'].get('expt')

                    # Then, if no base path specified first try the default server path for the given experiment
                    if base_path is None:
                        base_path = self._validate_and_set_base_path(self.get_default_server_address(experiment))

                    # If not connected to server path, then default to the local base path defined in the config file
                    if base_path is None:
                        local_path = Path(config['Paths'].get('GEECS_DATA_LOCAL_BASE_PATH', None))
                        base_path = self._validate_and_set_base_path(local_path)

                except Exception as e:
                    logger.error(f"Error reading config file {config_path}: {e}")
            else:
                logger.warning(f"Config file {config_path} not found. Using default paths.")

        if experiment is None or base_path is None:
            raise ConfigurationError(f"Could not set experiment name and base path. Check config file {config_path}")

        self.base_path = base_path  # .resolve()
        self.experiment = experiment

    def is_default_server_address(self) -> bool:
        """ Returns True if the base directory is equivalent to the default server address """
        default_path = self.get_default_server_address(self.experiment)  # .resolve()
        return self.base_path == default_path

    @staticmethod
    def get_default_server_address(experiment_name: str) -> Optional[Path]:
        """ Returns the corresponding base path on the server for the given experiment, defaults to None if unknown """
        return EXPERIMENT_TO_SERVER_DICT.get(experiment_name, None)

    @staticmethod
    def _validate_and_set_base_path(input_path) -> Optional[Path]:
        """ If the given path is a path and exists, return it.  Otherwise, return None """
        if input_path is not None and input_path.exists():
            return input_path
        return None
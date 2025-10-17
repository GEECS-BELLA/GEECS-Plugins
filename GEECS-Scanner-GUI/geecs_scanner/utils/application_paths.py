"""Basic functionality for finding and defining paths to necessary config files."""

import configparser
import os
from functools import lru_cache
from pathlib import Path
from typing import ClassVar


class ApplicationPaths:
    """
    Contains paths and folder names used by GEECS Scanner.

    Config resolution order:
        1. GEECS_SCANNER_CONFIG_DIR environment variable
        2. ~/.config/geecs_python_api/config.ini [Paths] config_root
        3. Local scanner_configs/experiments (fallback)
    """

    CONFIG_PATH: ClassVar[Path] = Path(
        "~/.config/geecs_python_api/config.ini"
    ).expanduser()
    _DEFAULT_BASE_PATH: ClassVar[Path] = (
        Path(__file__).parents[2] / "scanner_configs" / "experiments"
    )

    @classmethod
    @lru_cache(maxsize=1)
    def BASE_PATH(cls) -> Path:
        """
        Resolve and cache the scanner config base path.

        Cached after first access. To force re-resolution, delete the cached value.

        Returns
        -------
        Path
            The resolved base path for scanner configurations.
        """
        # 1. Try environment variable
        if env_path := os.getenv("GEECS_SCANNER_CONFIG_DIR"):
            if (path := Path(env_path).expanduser().resolve()).exists():
                return path

        # 2. Try config file
        if cls.CONFIG_PATH.exists():
            config = configparser.ConfigParser()
            config.read(cls.CONFIG_PATH)
            if scanner_config_root_path := config.get(
                "Paths", "scanner_config_root_path", fallback=None
            ):
                scanner_path = (
                    Path(scanner_config_root_path).expanduser().resolve()
                    / "scanner_configs"
                    / "experiments"
                )
                if scanner_path.exists():
                    return scanner_path

        # 3. Fallback to local
        return cls._DEFAULT_BASE_PATH

    SAVE_DEVICES_FOLDER = "save_devices"
    SCAN_DEVICES_FOLDER = "scan_devices"
    PRESET_FOLDER = "scan_presets"
    MULTISCAN_FOLDER = "multiscan_presets"
    SHOT_CONTROL_FOLDER = "shot_control_configurations"
    ACTION_LIBRARY_FOLDER = "action_library"
    OPTIMIZATION_CONFIGS = "optimizer_configs"

    def __init__(self, experiment: str, create_new: bool = True):
        """
        Initialize paths for a specific experiment.

        Parameters
        ----------
        experiment : str
            Name of the experiment
        create_new : bool, default=True
            If True, creates folders if they don't exist
        """
        if not experiment:
            raise ValueError("Cannot set empty experiment in Application Paths")

        self.exp_path = self.base_path() / experiment
        self.exp_save_devices = self.exp_path / self.SAVE_DEVICES_FOLDER
        self.exp_scan_devices = self.exp_path / self.SCAN_DEVICES_FOLDER
        self.exp_presets = self.exp_path / self.PRESET_FOLDER
        self.exp_multiscan = self.exp_path / self.MULTISCAN_FOLDER
        self.exp_shot_control = self.exp_path / self.SHOT_CONTROL_FOLDER
        self.exp_action_library = self.exp_path / self.ACTION_LIBRARY_FOLDER
        self.exp_optimization_routines = self.exp_path / self.OPTIMIZATION_CONFIGS

        if create_new:
            self._create_directories()

    def _create_directories(self) -> None:
        """Create all experiment directories if they don't exist."""
        for attr_val in (getattr(self, attr) for attr in dir(self)):
            if isinstance(attr_val, Path) and not attr_val.exists():
                print(f"Creating folder: '{attr_val}'")
                attr_val.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def create_config_if_missing() -> bool:
        """
        Create config file with default values if missing.

        Returns
        -------
        bool
            True if new file was created, False if already exists.
        """
        if ApplicationPaths.CONFIG_PATH.exists():
            return False

        ApplicationPaths.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        config = configparser.ConfigParser()
        config["Paths"] = {
            "geecs_data": "C:\\GEECS\\user data\\",
            "config_root": "",  # User sets this to point to GEECS-Configs
        }
        config["Experiment"] = {
            "expt": "",
            "rep_rate_hz": "1",
        }

        with open(ApplicationPaths.CONFIG_PATH, "w") as f:
            config.write(f)

        return True

    @staticmethod
    def config_file() -> Path:
        """Return path to user config file."""
        return ApplicationPaths.CONFIG_PATH

    @classmethod
    def base_path(cls) -> Path:
        """Return root folder for all experiment config files."""
        return cls.BASE_PATH()

    def experiment(self) -> Path:
        """Return root folder for all config files in given experiment."""
        return self.exp_path

    def save_devices(self) -> Path:
        """Return folder for the save device yaml's."""
        return self.exp_save_devices

    def scan_devices(self) -> Path:
        """Return folder for the scan device yaml's."""
        return self.exp_scan_devices

    def presets(self) -> Path:
        """Return folder for the scan preset yaml's."""
        return self.exp_presets

    def multiscan_presets(self) -> Path:
        """Return folder for the multiscan preset yaml's."""
        return self.exp_multiscan

    def shot_control(self) -> Path:
        """Return folder for the timing configuration yaml's."""
        return self.exp_shot_control

    def action_library(self) -> Path:
        """Return folder for the action library yaml's."""
        return self.exp_action_library

    def optimizer_configs(self) -> Path:
        """Return folder for the optimizer yaml's."""
        return self.exp_optimization_routines

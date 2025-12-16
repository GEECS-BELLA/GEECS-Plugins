"""Basic functionality for finding and defining paths to necessary config files."""

import configparser
import logging
from pathlib import Path
from typing import ClassVar

from geecs_data_utils.config_base import ConfigDirManager

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("~/.config/geecs_python_api/config.ini").expanduser()
DEFAULT_BASE_PATH = Path(__file__).parents[2] / "scanner_configs" / "experiments"


def _resolve_from_config_ini(config_path: Path) -> Path | None:
    """
    Resolve scanner config base from config.ini.

    Returns resolved path if it exists, otherwise None.
    """
    if not config_path.exists():
        return None

    config = configparser.ConfigParser()
    config.read(config_path)
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
    return None


_CONFIG_MANAGER = ConfigDirManager(
    env_var="GEECS_SCANNER_CONFIG_DIR",
    logger=logger,
    name="Scanner config",
    fallback_resolver=lambda: _resolve_from_config_ini(CONFIG_PATH),
    fallback_name="config.ini Paths.scanner_config_root_path",
)

_CONFIG_MANAGER.bootstrap_from_env_or_fallback()
if _CONFIG_MANAGER.base_dir is None:
    try:
        _CONFIG_MANAGER.set_base_dir(DEFAULT_BASE_PATH)
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning("Unable to set default scanner config base: %s", exc)


class ApplicationPaths:
    """
    Contains paths and folder names used by GEECS Scanner.

    Config resolution order:
        1. GEECS_SCANNER_CONFIG_DIR environment variable
        2. ~/.config/geecs_python_api/config.ini [Paths] scanner_config_root_path
        3. Local scanner_configs/experiments (fallback)
    """

    CONFIG_PATH: ClassVar[Path] = CONFIG_PATH
    _DEFAULT_BASE_PATH: ClassVar[Path] = DEFAULT_BASE_PATH
    _CONFIG_MANAGER: ClassVar[ConfigDirManager] = _CONFIG_MANAGER

    @classmethod
    def BASE_PATH(cls) -> Path:
        """Backward-compatible alias for the scanner config base path."""
        return cls.base_path()

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
        """
        Return root folder for all experiment config files.

        If no base dir is set (e.g., invalid env var and missing defaults),
        fall back to the local default path.
        """
        if cls._CONFIG_MANAGER.base_dir is None:
            return cls._DEFAULT_BASE_PATH
        return cls._CONFIG_MANAGER.base_dir

    @classmethod
    def set_base_path(cls, path: Path) -> Path:
        """Set the scanner config base path (clears cache)."""
        return cls._CONFIG_MANAGER.set_base_dir(path)

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

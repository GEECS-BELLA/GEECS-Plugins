"""
Configuration management for GEECS data paths for various experiments.

This module provides the GeecsPathsConfig class for managing file system
paths and experiment configurations used throughout the GEECS plugin suite.

The module handles automatic path detection, configuration file loading,
and provides fallback mechanisms for different deployment scenarios.
"""

from __future__ import annotations

from typing import Optional, Union

import logging
import os
from configparser import ConfigParser
from configparser import Error as ConfigParserError
from pathlib import Path
from geecs_data_utils.utils import ConfigurationError

# Module-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.propagate = False


class GeecsPathsConfig:
    """
    Manages configuration for GEECS-related paths and experiment settings.

    Attributes
    ----------
    base_path : Path
        The base directory for storing GEECS data locally.
    experiment : str, optional
        The default experiment name. May be None on analysis-only machines where
        the experiment is always supplied at runtime (e.g. via the data portal).
    scan_analysis_configs_path : Path
        path to directory containing scan analysis configs
    """

    def __init__(  # noqa: C901, PLR0912, PLR0915
        self,
        config_path: Path = Path("~/.config/geecs_python_api/config.ini").expanduser(),
        default_experiment: Optional[str] = None,
        set_base_path: Optional[Union[Path, str]] = None,
        scan_analysis_configs_path: Optional[Union[Path, str]] = None,
    ):
        """
        Initialize GEECS paths configuration.

        Load paths and experiment settings from a configuration file. Will first try the given base path from the
        initialization argument. Then will try the default server address in the above dictionary. Lastly, if still
        not located, will try loading the config file and using the defined base path there.

        Parameters
        ----------
        config_path : Path, optional
            Path to the configuration file (default: ~/.config/geecs_python_api/config.ini).
        default_experiment : str, optional
            Experiment name; when omitted, read from ``[Experiment] expt`` in the
            config file (may stay None on analysis-only machines).
        set_base_path : Path, optional
            Default path for locating GEECS data (default: Z:/data).
        scan_analysis_configs_path : Path, optional
            default path to folder containing scan_analysis_configs

        Raises
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

        # If base path or experiment not yet resolved, read the config file
        if experiment is None or base_path is None:
            config = ConfigParser()
            if config_path.exists():
                try:
                    config.read(config_path)

                    # Get the experiment name if it is not provided
                    if experiment is None:
                        experiment = config["Experiment"].get("expt")

                    if base_path is None:
                        local_path_str = config["Paths"].get(
                            "GEECS_DATA_LOCAL_BASE_PATH", None
                        )
                        if local_path_str is not None:
                            base_path = self._validate_path(Path(local_path_str))

                    if scan_analysis_configs_path is None:
                        local_path = Path(
                            config["Paths"].get("scan_analysis_configs_path", None)
                        )
                        scan_analysis_configs_path = self._validate_path(local_path)

                except Exception as e:
                    logger.exception("Error reading config file %s: %s", config_path, e)
            else:
                logger.warning(
                    "Config file %s not found. Using default paths.", config_path
                )

        # Experiment is optional — base_path alone is sufficient for path-only usage.
        # Callers that need the experiment (e.g. GDoc integration) supply it via ScanTag.
        if base_path is None:
            raise ConfigurationError(
                f"Could not determine base data path. "
                f"Set GEECS_DATA_LOCAL_BASE_PATH = <path> under [Paths] in {config_path}."
            )

        self.base_path = base_path  # .resolve()
        self.experiment = experiment
        self.scan_analysis_configs_path = scan_analysis_configs_path

        # Optional tool-specific paths loaded from config.ini
        self.frog_dll_path: Optional[Path] = None
        self.frog_python32_path: Optional[Path] = None
        #: Command prefix that runs the 32-bit Windows Python (e.g. ``wine``
        #: on a Linux host); unset runs it directly, as on Windows.
        self.frog_launcher: Optional[str] = None
        #: HASO WaveKit (optional, the ``haso`` measure): the SDK directory
        #: (``wavekit_py/`` + ``dlls/x64/``), the 64-bit Windows Python that
        #: runs it, the command that runs that Python on Linux (``env
        #: WINEPREFIX=... wine``; unset runs it directly, as on Windows), and
        #: the directory of per-sensor ``.dat`` / ``.lift`` files a recipe's
        #: ``sensor_config`` names.
        self.wavekit_sdk_path: Optional[Path] = None
        self.wavekit_python_path: Optional[Path] = None
        self.wavekit_launcher: Optional[str] = None
        self.wavekit_configs_path: Optional[Path] = None
        self.data_folders: dict[str, str] = (
            self._read_data_folders(config_path) if config_path.exists() else {}
        )

        if config_path.exists():
            try:
                _config = ConfigParser()
                _config.read(config_path)
                if _config.has_option("Paths", "frog_dll_path"):
                    self.frog_dll_path = self._validate_path(
                        Path(_config["Paths"]["frog_dll_path"])
                    )
                if _config.has_option("Paths", "frog_python32_path"):
                    self.frog_python32_path = self._validate_path(
                        Path(_config["Paths"]["frog_python32_path"])
                    )
                if _config.has_option("Paths", "frog_launcher"):
                    self.frog_launcher = (
                        _config["Paths"]["frog_launcher"].strip() or None
                    )
                for key in (
                    "wavekit_sdk_path",
                    "wavekit_python_path",
                    "wavekit_configs_path",
                ):
                    if _config.has_option("Paths", key):
                        setattr(
                            self, key, self._validate_path(Path(_config["Paths"][key]))
                        )
                if _config.has_option("Paths", "wavekit_launcher"):
                    self.wavekit_launcher = (
                        _config["Paths"]["wavekit_launcher"].strip() or None
                    )
            except Exception as e:  # noqa: BLE001 — config read fails soft
                logger.debug("Could not read tool paths from config: %s", e)

    @staticmethod
    def _read_data_folders(config_path: Path) -> dict[str, str]:
        """Read ``[data_folders]``, keeping the experiment names' case.

        A value that is not a single folder name (a separator, ``.`` or
        ``..``) is logged and ignored, so the experiment falls back to its
        own name rather than reaching outside the data root.
        """
        config = ConfigParser()
        config.optionxform = str  # experiment names are case-sensitive folders
        try:
            config.read(config_path)
        except (OSError, ConfigParserError) as e:
            logger.debug("Could not read [data_folders] from config: %s", e)
            return {}
        if not config.has_section("data_folders"):
            return {}
        folders: dict[str, str] = {}
        for experiment, raw in config.items("data_folders"):
            folder = raw.strip()
            if folder in (".", "..") or any(sep in folder for sep in "/\\"):
                logger.warning(
                    "[data_folders] %s = %r is not a single folder name; ignored",
                    experiment,
                    raw,
                )
                continue
            folders[experiment] = folder
        return folders

    def data_folder_for(self, experiment: Optional[str]) -> str:
        """The folder under ``base_path`` that holds *experiment*'s scans.

        Parameters
        ----------
        experiment : str or None
            GEECS experiment name.

        Returns
        -------
        str
            The ``[data_folders]`` entry when there is one (empty: the
            ``Y{YYYY}`` folders sit directly in ``base_path``), else the
            experiment name itself; empty for no experiment.
        """
        if not experiment:
            return ""
        return self.data_folders.get(experiment, experiment)

    def experiment_for_data_folder(
        self, folder: Path, base_path: Optional[Path] = None
    ) -> str:
        """The experiment whose scans live in *folder*, the inverse of :meth:`data_folder_for`.

        Parameters
        ----------
        folder : Path
            The folder above a scan's ``Y{YYYY}`` folder.
        base_path : Path, optional
            The data root *folder* was found under; defaults to
            ``self.base_path``.

        Returns
        -------
        str
            The experiment mapped to an empty folder when *folder* is the
            data root itself, else the experiment mapped to *folder*'s
            name, else *folder*'s name (the GEECS convention).
        """
        root = base_path if base_path is not None else self.base_path
        if root is not None and _same_path(folder, Path(root)):
            flat = [e for e, f in self.data_folders.items() if f == ""]
            if len(flat) == 1:
                return flat[0]
            if len(flat) > 1:
                logger.warning(
                    "[data_folders] maps several experiments to the data root "
                    "(%s); a scan there cannot be attributed to one",
                    ", ".join(flat),
                )
        renamed = [e for e, f in self.data_folders.items() if f and f == folder.name]
        if len(renamed) == 1:
            return renamed[0]
        return folder.name

    @staticmethod
    def _validate_path(input_path) -> Optional[Path]:
        """
        Validate and return path if it exists.

        Parameters
        ----------
        input_path : Path or None
            Path to validate

        Returns
        -------
        Optional[Path]
            The input path if it exists, None otherwise
        """
        if input_path is not None and input_path.exists():
            return input_path
        logger.warning("%s path was not found", input_path)
        return None


def _same_path(a: Path, b: Path) -> bool:
    """Whether two paths name the same folder, lexically (no filesystem access)."""
    return os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(
        os.path.normpath(str(b))
    )


if __name__ == "__main__":
    pass

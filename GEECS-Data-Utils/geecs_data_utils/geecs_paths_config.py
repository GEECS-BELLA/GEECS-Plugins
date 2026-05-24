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
from configparser import ConfigParser
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
        the experiment is always supplied at runtime (e.g. via LiveWatch GUI).
    image_analysis_configs_path : Path
        path to directory containing image analysis configs
    scan_analysis_configs_path : Path
        path to directory containing scan analysis configs
    """

    def __init__(
        self,
        config_path: Path = Path("~/.config/geecs_python_api/config.ini").expanduser(),
        default_experiment: Optional[str] = None,
        set_base_path: Optional[Union[Path, str]] = None,
        image_analysis_configs_path: Optional[Union[Path, str]] = None,
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
            Default experiment name (default: "Undulator").
        set_base_path : Path, optional
            Default path for locating GEECS data (default: Z:/data).
        image_analysis_configs_path : Path, optional
            default path to folder containing image_analysis_configs
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

                    # Resolve scan_analysis_configs_path FIRST so the
                    # image-side path can derive from it.
                    if scan_analysis_configs_path is None:
                        local_path = Path(
                            config["Paths"].get("scan_analysis_configs_path", None)
                        )
                        scan_analysis_configs_path = self._validate_path(local_path)

                    if image_analysis_configs_path is None:
                        image_analysis_configs_path = self._resolve_image_path(
                            config, scan_analysis_configs_path
                        )

                except Exception as e:
                    logger.error(f"Error reading config file {config_path}: {e}")
            else:
                logger.warning(
                    f"Config file {config_path} not found. Using default paths."
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
        self.image_analysis_configs_path = image_analysis_configs_path
        self.scan_analysis_configs_path = scan_analysis_configs_path

        # Optional tool-specific paths loaded from config.ini
        self.frog_dll_path: Optional[Path] = None
        self.frog_python32_path: Optional[Path] = None
        self.wavekit_config_path: Optional[Path] = None

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
                if _config.has_option("Paths", "wavekit_config_path"):
                    self.wavekit_config_path = self._validate_path(
                        Path(_config["Paths"]["wavekit_config_path"])
                    )
            except Exception as e:
                logger.debug(f"Could not read tool paths from config: {e}")

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
        else:
            logger.warning("%s path was not found", input_path)
        return None

    @classmethod
    def _resolve_image_path(
        cls, config: ConfigParser, scan_analysis_configs_path: Optional[Path]
    ) -> Optional[Path]:
        """Resolve the image-analysis config path with unified-configs awareness.

        After the unified-configs migration, image and scan analysis
        share one config tree: the per-diagnostic YAMLs under
        ``scan_analysis_configs_path/analyzers`` carry both an
        ``image:`` section and a ``scan:`` section. ImageAnalysis
        searches that subtree recursively to find a camera or line
        config by name.

        The legacy ``image_analysis_configs_path`` key in ``config.ini``
        is honoured (with a deprecation warning) for users who haven't
        yet updated their config file. When the legacy key is absent,
        the image path derives from ``scan_analysis_configs_path /
        "analyzers"``.

        Parameters
        ----------
        config : ConfigParser
            Loaded ``config.ini``.
        scan_analysis_configs_path : Path, optional
            Already-resolved scan-analysis configs root.

        Returns
        -------
        Path or None
            Resolved image-analysis configs path, or ``None`` if neither
            the legacy key nor the derived path exists.
        """
        legacy = config["Paths"].get("image_analysis_configs_path", None)
        if legacy is not None:
            logger.warning(
                "image_analysis_configs_path in config.ini is deprecated. "
                "After the unified-configs migration, ImageAnalysis reads "
                "from scan_analysis_configs_path/analyzers (which holds "
                "the embedded image: section of each diagnostic). Remove "
                "the line from config.ini to use auto-derivation."
            )
            resolved = cls._validate_path(Path(legacy))
            if resolved is not None:
                return resolved

        if scan_analysis_configs_path is not None:
            return cls._validate_path(Path(scan_analysis_configs_path) / "analyzers")

        return None


if __name__ == "__main__":
    pass

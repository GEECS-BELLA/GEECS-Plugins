"""Locate and load scanner configuration files from the configs repository.

Mirrors GEECS-Scanner-GUI's ``ApplicationPaths`` resolution without importing
it (GeecsBluesky does not depend on ``geecs_scanner``): the
``GEECS_SCANNER_CONFIG_DIR`` env var is used as the experiments root directly,
else config.ini ``[Paths] scanner_config_root_path`` +
``scanner_configs/experiments``.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

import yaml

from geecs_bluesky.models.shot_control import ShotControlConfig

SHOT_CONTROL_FOLDER = "shot_control_configurations"


def scanner_configs_base() -> Path:
    """Resolve the scanner-configs ``experiments`` base the production way.

    Raises
    ------
    RuntimeError
        If neither the env var nor the config.ini entry resolves.
    """
    env = os.environ.get("GEECS_SCANNER_CONFIG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    config_ini = Path("~/.config/geecs_python_api/config.ini").expanduser()
    if config_ini.exists():
        parser = configparser.ConfigParser()
        parser.read(config_ini)
        root = parser.get("Paths", "scanner_config_root_path", fallback=None)
        if root:
            return Path(root).expanduser().resolve() / "scanner_configs" / "experiments"
    raise RuntimeError(
        "Cannot resolve the scanner configs base. Set GEECS_SCANNER_CONFIG_DIR, or "
        "config.ini [Paths] scanner_config_root_path pointing at GEECS-Plugins-Configs."
    )


def load_shot_control_config(name: str, experiment: str) -> ShotControlConfig:
    """Load and validate one shot-control YAML from the configs repository.

    Parameters
    ----------
    name : str
        Config file name with or without ``.yaml`` (e.g. ``"HTU-LaserOFF"``).
    experiment : str
        Experiment folder under ``scanner_configs/experiments``.

    Returns
    -------
    ShotControlConfig
        Validated config — an unparseable or empty file fails loudly here
        rather than mid-scan against live hardware.
    """
    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"
    path = scanner_configs_base() / experiment / SHOT_CONTROL_FOLDER / name
    if not path.exists():
        raise RuntimeError(f"Shot-control config not found: {path}")
    with open(path) as handle:
        info = yaml.safe_load(handle)
    config = ShotControlConfig.from_information(info)
    if config is None:
        raise RuntimeError(f"{path} is empty / not a valid shot-control config.")
    return config

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

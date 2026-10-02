"""Where the GEECS-Plugins-Configs repository is, and the per-experiment files it holds.

One resolver for every consumer — the CA gateway's derived channels, the
archiver's policy, the scanner's configs — so the three-step lookup cannot
drift between packages (it had, before this module):

1. ``GEECS_SCANNER_CONFIG_DIR`` points directly at ``scanner_configs/experiments``.
2. ``GEECS_PLUGINS_CONFIGS`` points at the configs repository root.
3. ``~/.config/geecs_python_api/config.ini`` ``[Paths] scanner_config_root_path``
   points at the repository root.

The client half of the site profile (``docs/platform/site_profile.md``) is
that ``config.ini``; :func:`read_config_entry` is the one-value reader the
callers here need (the DB client keeps its own, for the ``Configurations.INI``
hop it alone makes).
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

#: The client-side site profile.
CONFIG_PATH = Path.home() / ".config" / "geecs_python_api" / "config.ini"
#: Where the per-experiment folders live inside the configs repository.
EXPERIMENTS_SUBDIR = ("scanner_configs", "experiments")


def read_config_entry(
    section: str, key: str, config_path: Path | None = None
) -> str | None:
    """One ``config.ini`` value, stripped; ``None`` when the file, section or key is absent or blank."""
    path = config_path or CONFIG_PATH
    if not path.exists():
        return None
    parser = configparser.ConfigParser()
    parser.read(path)
    if section not in parser:
        return None
    value = parser[section].get(key)
    return value.strip() if value and value.strip() else None


def scanner_configs_base(config_path: Path | None = None) -> Path | None:
    """Resolve the configs repository's ``scanner_configs/experiments`` directory, or ``None``.

    Parameters
    ----------
    config_path : Path, optional
        The ``config.ini`` to consult for step 3 (default: the user's).
    """
    env = os.environ.get("GEECS_SCANNER_CONFIG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    repo_env = os.environ.get("GEECS_PLUGINS_CONFIGS")
    if repo_env:
        return Path(repo_env).expanduser().resolve().joinpath(*EXPERIMENTS_SUBDIR)
    root = read_config_entry("Paths", "scanner_config_root_path", config_path)
    if root:
        return Path(root).expanduser().resolve().joinpath(*EXPERIMENTS_SUBDIR)
    return None


def experiment_config_path(
    experiment: str,
    folder: str,
    filename: str,
    *,
    base: Path | None = None,
    config_path: Path | None = None,
) -> Path | None:
    """The conventional per-experiment file ``<base>/<experiment>/<folder>/<filename>``, if it exists.

    ``None`` when the repository cannot be resolved or the file is absent —
    every consumer treats a missing overlay as "the defaults".
    """
    base = base if base is not None else scanner_configs_base(config_path)
    if base is None:
        return None
    path = base / experiment / folder / filename
    return path if path.exists() else None

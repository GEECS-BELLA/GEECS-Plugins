"""Where the server listens and which interlocks it watches.

Facility values live in ``~/.config/geecs_python_api/config.ini`` (the
client half of the site profile)::

    [Experiment]
    expt = <experiment>

    [interlocks]
    ; where Master Control's stream listens
    status_host = 0.0.0.0
    status_port = 5001
    ; where the web client listens
    web_host = 0.0.0.0
    web_port = 8600
    ; where external (UDP) signals are received
    external_host = 0.0.0.0
    ; the shot gate's port, on status_host (absent: no gate)
    gate_port = 5002

A comment goes on its own line: ``configparser`` keeps one written after a
value as part of the value.

Every key is optional. Without them every server listens on ``127.0.0.1``
only — a laptop running the demo is never reachable from the network until
its config says so. Command-line flags override the file.

The interlocks are one of the experiment's named configurations,
``interlocks/<name>.yaml`` in the configs repo (``geecs_core.configs_repo``),
or a file given on the command line.
"""

from __future__ import annotations

import logging
from pathlib import Path

from geecs_core.configs_repo import (
    read_config_entry,
    scanner_configs_base,
    settled_exists,
)
from geecs_schemas import INTERLOCKS_FOLDER, InterlockSet
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

#: The ``config.ini`` section this package reads.
SECTION = "interlocks"
#: Where both servers listen when the config says nothing: this machine only.
LOCAL_HOST = "127.0.0.1"
DEFAULT_STATUS_PORT = 5001
DEFAULT_WEB_PORT = 8600


class Settings(BaseModel):
    """Where the servers listen, and for which experiment."""

    model_config = ConfigDict(frozen=True)

    experiment: str | None = Field(None, description="The experiment's name.")
    status_host: str = LOCAL_HOST
    status_port: int = Field(DEFAULT_STATUS_PORT, ge=0, le=65535)
    web_host: str = LOCAL_HOST
    web_port: int = Field(DEFAULT_WEB_PORT, ge=0, le=65535)
    external_host: str = LOCAL_HOST
    #: The shot gate's port (on ``status_host``); ``None``: no gate is served.
    gate_port: int | None = Field(None, ge=0, le=65535)

    @classmethod
    def from_config_ini(cls, config_path: Path | None = None) -> Settings:
        """Read ``config.ini`` (every key optional; a malformed port is an error)."""

        def entry(key: str) -> str | None:
            return read_config_entry(SECTION, key, config_path)

        values: dict[str, str] = {
            key: value
            for key in (
                "status_host",
                "status_port",
                "web_host",
                "web_port",
                "external_host",
                "gate_port",
            )
            if (value := entry(key)) not in (None, "")
        }
        experiment = read_config_entry("Experiment", "expt", config_path)
        return cls.model_validate({"experiment": experiment, **values})


def interlocks_folder(experiment: str, config_path: Path | None = None) -> Path | None:
    """The experiment's ``interlocks/`` folder in the configs repo (``None``: no repo found)."""
    base = scanner_configs_base(config_path)
    return None if base is None else base / experiment / INTERLOCKS_FOLDER


def interlocks_path(
    experiment: str, name: str, config_path: Path | None = None
) -> Path | None:
    """The named configuration ``interlocks/<name>.yaml`` (``None``: no repo found)."""
    folder = interlocks_folder(experiment, config_path)
    return None if folder is None else folder / f"{name}.yaml"


def configuration_names(experiment: str, config_path: Path | None = None) -> list[str]:
    """The experiment's saved configurations, by name (empty: none, or no repo)."""
    folder = interlocks_folder(experiment, config_path)
    if folder is None or not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.yaml"))


def load_interlocks(path: Path | None) -> tuple[InterlockSet, str]:
    """The document at *path* and a line saying where it came from.

    A missing file (or no configs repo) is the empty document — every
    reader then reports ``NO INTERLOCKS``, never safe. A file that does not
    validate raises: the caller decides whether that is fatal.

    Raises
    ------
    pydantic.ValidationError, yaml.YAMLError
        *path* exists but is not a valid interlocks document.
    """
    if path is None:
        return InterlockSet(), "no configs repo found (config.ini [Paths])"
    # The configurator fast-forwards this checkout: wait out a rewrite in
    # progress so a file is never read missing or half written.
    if not settled_exists(path):
        logger.warning("no interlocks file at %s", path)
        return InterlockSet(), f"no interlocks file at {path}"
    return InterlockSet.from_path(path), str(path)


__all__ = [
    "DEFAULT_STATUS_PORT",
    "DEFAULT_WEB_PORT",
    "LOCAL_HOST",
    "SECTION",
    "Settings",
    "configuration_names",
    "interlocks_folder",
    "interlocks_path",
    "load_interlocks",
]

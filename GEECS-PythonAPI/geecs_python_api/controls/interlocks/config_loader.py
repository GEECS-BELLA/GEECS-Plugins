"""YAML config loader for :class:`InterlockServer`.

Turns a YAML file like

.. code-block:: yaml

    server:
      host: 0.0.0.0
      port: 9999

    interlocks:
      - type: camera_threshold
        device: CAM-PL1-LC_Film
        variable: MeanCounts
        threshold: 2.0

into a fully-configured ``InterlockServer`` whose interlocks are
registered (but not yet connected to devices — that happens when
``run_forever`` / ``start`` is called).

The ``type`` field selects a class from :data:`INTERLOCK_REGISTRY`.
Adding a new interlock type means writing a ``BaseInterlock`` subclass
and registering it here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Type, Union

import yaml
from pydantic import BaseModel, Field

from geecs_data_utils import GeecsPathsConfig

from .base_interlock import BaseInterlock
from .camera_threshold import CameraThresholdInterlock
from .geecs_interlock_server import InterlockServer

logger = logging.getLogger(__name__)


INTERLOCK_REGISTRY: Dict[str, Type[BaseInterlock]] = {
    "camera_threshold": CameraThresholdInterlock,
}


class ServerConfig(BaseModel):
    """Bind address for the TCP server."""

    host: str = "0.0.0.0"
    port: int = 9999


class InterlockServerConfig(BaseModel):
    """Top-level shape of the YAML file."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    interlocks: List[Dict[str, Any]] = Field(default_factory=list)


def _resolve_config_path(path: Union[str, Path]) -> Path:
    """Resolve ``path`` against the GEECS interlock-configs directory if relative."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p

    paths = GeecsPathsConfig()
    base = paths.interlock_configs_path
    if base is None:
        raise FileNotFoundError(
            f"Interlock config {p!s} not found and "
            "interlock_configs_path is not configured. Set it under "
            "[Paths] in ~/.config/geecs_python_api/config.ini."
        )
    return Path(base) / p


def load_interlock_server(config_path: Union[str, Path]) -> InterlockServer:
    """Build an :class:`InterlockServer` from a YAML config file.

    Parameters
    ----------
    config_path : str or Path
        Either an absolute path or a filename relative to
        ``GeecsPathsConfig().interlock_configs_path``.

    Returns
    -------
    InterlockServer
        Server with all interlocks registered.  Call
        :meth:`InterlockServer.run_forever` (or :meth:`start`) to begin
        serving.
    """
    resolved = _resolve_config_path(config_path)
    logger.info(f"Loading interlock config from {resolved}")
    with open(resolved) as f:
        raw = yaml.safe_load(f) or {}

    config = InterlockServerConfig.model_validate(raw)

    server = InterlockServer(host=config.server.host, port=config.server.port)

    for entry in config.interlocks:
        interlock_type = entry.pop("type", None)
        if interlock_type is None:
            raise ValueError(f"Interlock entry missing 'type': {entry}")
        cls = INTERLOCK_REGISTRY.get(interlock_type)
        if cls is None:
            known = ", ".join(sorted(INTERLOCK_REGISTRY))
            raise ValueError(
                f"Unknown interlock type {interlock_type!r}. Known: {known}"
            )
        interlock = cls(**entry)
        server.register_interlock(interlock)
        logger.info(f"Registered interlock '{interlock.name}' ({interlock_type})")

    return server

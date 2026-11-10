"""Settings, from the environment: ``GEECS_CONFIGURATOR_*``, the shared site keys beneath.

No facility value has a default here. The configs root is
``GEECS_CONFIGURATOR_CONFIGS_ROOT``, else the shared ``GEECS_CONFIGS_ROOT``,
else what ``geecs_core.configs_repo`` resolves from ``config.ini``; the
experiment shown first is ``GEECS_CONFIGURATOR_EXPERIMENT`` or
``GEECS_EXPERIMENT`` (any experiment in the root can be browsed). The CORS
origins are the surfaces that embed the editor, a site value.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from geecs_core.configs_repo import EXPERIMENTS_SUBDIR, scanner_configs_base

_P = "GEECS_CONFIGURATOR_"


class SettingsError(ValueError):
    """A required setting is missing or malformed."""


@dataclass(frozen=True)
class Settings:
    """Where the configs are, which experiment opens first, how the process serves."""

    configs_root: Path
    experiment: str = ""
    title: str = "Configurator"
    bind: str = "127.0.0.1"
    port: int = 8800
    cors_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        """Read the settings; refuse a configs root that cannot be found, or a bad port."""

        def get(key: str, shared: str = "") -> str:
            return (env.get(_P + key) or (shared and env.get(shared)) or "").strip()

        root = get("CONFIGS_ROOT", "GEECS_CONFIGS_ROOT")
        if not root:
            base = scanner_configs_base()
            if base is None:
                raise SettingsError(f"{_P}CONFIGS_ROOT (or GEECS_CONFIGS_ROOT, or config.ini's scanner_config_root_path) is not set")  # fmt: skip
            root = str(base.parents[len(EXPERIMENTS_SUBDIR) - 1])
        port = get("PORT") or "8800"
        if not port.isdigit():
            raise SettingsError(f"{_P}PORT is not a port: {port!r}")
        origins = tuple(o.strip().rstrip("/") for o in get("CORS_ORIGINS").split(",") if o.strip())  # fmt: skip
        return cls(
            configs_root=Path(root),
            experiment=get("EXPERIMENT", "GEECS_EXPERIMENT"),
            title=get("TITLE") or "Configurator",
            bind=get("BIND") or "127.0.0.1",
            port=int(port),
            cors_origins=origins,
        )

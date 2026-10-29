"""Settings, from the environment: ``GEECS_CONTROL_ROOM_*``, the shared site keys beneath.

Non-secret values are ``site.env`` keys; the read key is a secret and comes
from the unit's secrets file (``/etc/geecs/control-room/secrets.env``). No
facility value has a default here: the experiment and configs root fall back
to the shared ``GEECS_EXPERIMENT`` and ``GEECS_CONFIGS_ROOT``, and nothing
falls back to a name.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

_P = "GEECS_CONTROL_ROOM_"
#: The launcher's link words (GEECS-Schemas ``HomeLink``), each read from ``<P><WORD>_URL``.
_LINKS = ("scanner", "portal", "logbook", "config_editor")


class SettingsError(ValueError):
    """A required setting is missing or malformed."""


@dataclass(frozen=True)
class Settings:
    """What the process needs: where Canopy is, which experiment, which configs, the page title."""

    experiment: str
    configs_root: Path
    canopy_url: str
    read_key: str
    #: Canopy as the browser reaches it (the front door's ``/canopy``); default ``canopy_url``.
    canopy_browser_url: str = ""
    title: str = "Control Room"
    bind: str = "127.0.0.1"
    port: int = 8700
    #: The scanner as this process reaches it (the home's run state); empty: not configured.
    scanner_url: str = ""
    #: The launcher's links as the browser reaches them, by ``HomeLink`` word; empty: not configured.
    links: Mapping[str, str] = field(default_factory=dict)

    @property
    def experiment_dir(self) -> Path:
        """The experiment's folder in the configs repository.

        The layout is ``geecs_core.configs_repo.EXPERIMENTS_SUBDIR``'s, copied:
        this package depends on GEECS-Schemas and the theme only.
        """
        return self.configs_root / "scanner_configs" / "experiments" / self.experiment

    @property
    def browser_url(self) -> str:
        """Canopy's base URL for the page, without a trailing slash."""
        return (self.canopy_browser_url or self.canopy_url).rstrip("/")

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        """Read the settings; refuse a missing experiment, configs root, Canopy URL or key."""

        def need(key: str, shared: str | None = None) -> str:
            value = env.get(_P + key) or (shared and env.get(shared)) or ""
            if not value.strip():
                also = f" (or {shared})" if shared else ""
                raise SettingsError(f"{_P}{key}{also} is not set")
            return value.strip()

        port = env.get(_P + "PORT", "8700")
        if not port.isdigit():
            raise SettingsError(f"{_P}PORT is not a port: {port!r}")
        links = {w: v for w in _LINKS if (v := env.get(f"{_P}{w.upper()}_URL", "").strip())}  # fmt: skip
        if browser := env.get(_P + "SCANNER_BROWSER_URL", "").strip():
            links["scanner"] = browser  # the scanner as the browser reaches it
        return cls(
            experiment=need("EXPERIMENT", "GEECS_EXPERIMENT"),
            configs_root=Path(need("CONFIGS_ROOT", "GEECS_CONFIGS_ROOT")),
            canopy_url=need("CANOPY_URL").rstrip("/"),
            read_key=need("READ_KEY"),
            canopy_browser_url=env.get(_P + "CANOPY_BROWSER_URL", "").strip(),
            title=env.get(_P + "TITLE", "").strip() or "Control Room",
            bind=env.get(_P + "BIND", "").strip() or "127.0.0.1",
            port=int(port),
            scanner_url=env.get(_P + "SCANNER_URL", "").strip().rstrip("/"),
            links=links,
        )

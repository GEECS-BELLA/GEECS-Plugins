"""Settings, from the environment: ``GEECS_CONFIGURATOR_*``, the shared site keys beneath.

No facility value has a default here. The configs root is
``GEECS_CONFIGURATOR_CONFIGS_ROOT``, else the shared ``GEECS_CONFIGS_ROOT``,
else what ``geecs_core.configs_repo`` resolves from ``config.ini``; the
experiment shown first is ``GEECS_CONFIGURATOR_EXPERIMENT`` or
``GEECS_EXPERIMENT`` (any experiment in the root can be browsed). The CORS
origins are the surfaces that embed the editor, a site value.

**The commit path** is on when ``GEECS_CONFIGURATOR_STATE_DIR`` and
``GEECS_CONFIGURATOR_ORIGIN`` (the configs repo's git URL) are both set:
the service keeps its bare repository, live checkout, staging worktrees and
inbox under the state directory, and the configs root it serves is the live
checkout (the configs-root keys are then not read). Unset, it is read-only
over the configs root, as on a laptop. The commit identity
(``_COMMITTER_NAME``, ``_COMMITTER_EMAIL``) and the inbox's author
(``_INBOX_AUTHOR``) have neutral defaults; ``_GITHUB_REPO`` (``owner/name``)
gives a proposed change its compare link.
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
    state_dir: Path | None = None
    origin: str = ""
    branch: str = "main"
    committer_name: str = "GEECS configurator"
    committer_email: str = "configurator@geecs.invalid"
    inbox_author: str = "GEECS worker via configurator"
    reconcile_seconds: float = 5.0
    github_repo: str = ""

    @property
    def commits(self) -> bool:
        """Whether the commit path is on (a state directory and an origin)."""
        return self.state_dir is not None and bool(self.origin)

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        """Read the settings; refuse a configs root that cannot be found, or a bad port."""

        def get(key: str, shared: str = "") -> str:
            return (env.get(_P + key) or (shared and env.get(shared)) or "").strip()

        state, origin = get("STATE_DIR"), get("ORIGIN")
        if bool(state) != bool(origin):
            raise SettingsError(f"{_P}STATE_DIR and {_P}ORIGIN are set together (the commit path) or not at all")  # fmt: skip
        root = str(Path(state) / "live") if state else get("CONFIGS_ROOT", "GEECS_CONFIGS_ROOT")  # fmt: skip
        if not root:
            base = scanner_configs_base()
            if base is None:
                raise SettingsError(f"{_P}CONFIGS_ROOT (or GEECS_CONFIGS_ROOT, or config.ini's scanner_config_root_path) is not set")  # fmt: skip
            root = str(base.parents[len(EXPERIMENTS_SUBDIR) - 1])
        port = get("PORT") or "8800"
        if not port.isdigit():
            raise SettingsError(f"{_P}PORT is not a port: {port!r}")
        try:
            every = float(get("RECONCILE_SECONDS") or "5")
        except ValueError:
            raise SettingsError(f"{_P}RECONCILE_SECONDS is not a number") from None
        origins = tuple(o.strip().rstrip("/") for o in get("CORS_ORIGINS").split(",") if o.strip())  # fmt: skip
        return cls(
            configs_root=Path(root),
            experiment=get("EXPERIMENT", "GEECS_EXPERIMENT"),
            title=get("TITLE") or "Configurator",
            bind=get("BIND") or "127.0.0.1",
            port=int(port),
            cors_origins=origins,
            state_dir=Path(state) if state else None,
            origin=origin,
            branch=get("BRANCH") or "main",
            committer_name=get("COMMITTER_NAME") or cls.committer_name,
            committer_email=get("COMMITTER_EMAIL") or cls.committer_email,
            inbox_author=get("INBOX_AUTHOR") or cls.inbox_author,
            reconcile_seconds=every,
            github_repo=get("GITHUB_REPO"),
        )

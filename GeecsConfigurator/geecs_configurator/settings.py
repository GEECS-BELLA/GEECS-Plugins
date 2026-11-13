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

The **GitHub App** is on when ``_GITHUB_APP_ID``, ``_GITHUB_INSTALLATION_ID``
and ``_GITHUB_APP_KEY_FILE`` (the private key, a secrets file beside
site.env readable by the service account only) are all set; then
``ORIGIN`` is the configs repo's HTTPS URL and the App pushes and opens the
maintainer tier's pull requests. Unset, git's own credential (a deploy key)
pushes and a proposal gives its compare link. ``_SHARE_CLONE`` is the
share's clone, fast-forwarded from the service's repository after every
accepted change; ``_FETCH_SECONDS`` (default 60; 0 = off) is how often
main is fetched so a merge on GitHub takes effect.

**The applies and dry runs** reach the consumers: ``_SCANNER_URL`` (the
scanner as this process reaches it: the worker's environment reopen, the
trigger-profile dry run, readbacks; one URL is the served experiment's,
``Exp=url,Exp2=url`` names each experiment instance's scanner) and ``_ARCHIVER_COMMAND`` (the
archiver's CLI in its own environment, whitespace-split: onboard and its
dry run). Unset, those applies are handed to the maintainer (pending) and
those dry runs say they did not run. ``_APPLY_SECONDS`` (default 30; 0 =
off) is how often the queued applies are tried.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from geecs_core.configs_repo import EXPERIMENTS_SUBDIR, scanner_configs_base

_P = "GEECS_CONFIGURATOR_"


def _named_scanners(raw: str) -> dict[str, str] | None:
    """``Exp=url,Exp2=url`` as a mapping, or ``None`` when no part names an experiment (one plain URL)."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    named = [re.fullmatch(r"([A-Za-z0-9_.-]+)=(https?://\S+)", p) for p in parts]
    if not any(named):
        return None
    if not all(named):
        raise SettingsError(f"{_P}SCANNER_URL names an experiment for some parts but not every one: {raw!r}")  # fmt: skip
    return {m.group(1): m.group(2).rstrip("/") for m in named if m}


def _check_scanner_url(scanner: str, experiment: str) -> None:
    """Refuse a ``_SCANNER_URL`` that names some experiments but not all, or one URL with no served experiment."""
    if not scanner:
        return
    named = _named_scanners(scanner)
    if named is None and not experiment:
        raise SettingsError(f"{_P}SCANNER_URL as one URL is the served experiment's: set GEECS_EXPERIMENT, or name each (Exp=url,Exp2=url)")  # fmt: skip
    if named is None and "," in scanner:
        raise SettingsError(f"{_P}SCANNER_URL is one URL, or Exp=url for every part: {scanner!r}")  # fmt: skip


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
    fetch_seconds: float = 60.0
    share_clone: Path | None = None
    github_app_id: str = ""
    github_installation_id: str = ""
    github_app_key_file: Path | None = None
    scanner_url: str = ""
    archiver_command: str = ""
    apply_seconds: float = 30.0

    def scanner_urls(self) -> dict[str, str]:
        """Each experiment's scanner: ``_SCANNER_URL`` is one URL (the served experiment's) or ``Exp=url,Exp2=url``."""
        raw = self.scanner_url
        if not raw:
            return {}
        named = _named_scanners(raw)
        if named is not None:
            return named
        return {self.experiment: raw.rstrip("/")} if self.experiment else {}

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
            fetch_every = float(get("FETCH_SECONDS") or "60")
            apply_every = float(get("APPLY_SECONDS") or "30")
        except ValueError:
            raise SettingsError(
                f"{_P}RECONCILE_SECONDS, _FETCH_SECONDS and _APPLY_SECONDS are seconds"
            ) from None
        app = (
            get("GITHUB_APP_ID"),
            get("GITHUB_INSTALLATION_ID"),
            get("GITHUB_APP_KEY_FILE"),
        )
        if any(app) and not all(app):
            raise SettingsError(f"{_P}GITHUB_APP_ID, _GITHUB_INSTALLATION_ID and _GITHUB_APP_KEY_FILE are set together or not at all")  # fmt: skip
        if all(app) and not (state and get("GITHUB_REPO")):
            raise SettingsError(f"the GitHub App needs the commit path and {_P}GITHUB_REPO (owner/name)")  # fmt: skip
        share = get("SHARE_CLONE")
        _check_scanner_url(get("SCANNER_URL"), get("EXPERIMENT", "GEECS_EXPERIMENT"))
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
            fetch_seconds=fetch_every,
            share_clone=Path(share) if share else None,
            github_app_id=app[0],
            github_installation_id=app[1],
            github_app_key_file=Path(app[2]) if app[2] else None,
            scanner_url=get("SCANNER_URL").rstrip("/"),
            archiver_command=get("ARCHIVER_COMMAND"),
            apply_seconds=apply_every,
        )

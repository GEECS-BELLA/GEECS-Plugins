"""Where the GEECS-Plugins-Configs repository is, and the per-experiment files it holds.

One resolver for the consumers of the configs repository — the CA gateway's
derived channels, the archiver's policy and GeecsBluesky's config resolver —
so the three-step lookup cannot drift between packages:

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
import time
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


#: The site key naming where a process WRITES configs (a configs root: the parent of ``scanner_configs/``).
WRITE_ROOT_ENV = "GEECS_CONFIGS_WRITE_ROOT"


def scanner_configs_write_base(config_path: Path | None = None) -> Path | None:
    """Where a process writes into the configs tree: ``scanner_configs/experiments`` under the write root.

    ``GEECS_CONFIGS_WRITE_ROOT`` (a configs root, the parent of
    ``scanner_configs/``) when set, else the read root
    (:func:`scanner_configs_base`), so a laptop writes where it reads. On
    the services box it is the configurator's inbox: the live checkout the
    services read is never written, because git refuses to fast-forward
    over an uncommitted file.
    """
    env = os.environ.get(WRITE_ROOT_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve().joinpath(*EXPERIMENTS_SUBDIR)
    return scanner_configs_base(config_path)


#: How long a reader waits out a checkout rewriting a file (seconds), and how often it looks.
SETTLE_WAIT_S, SETTLE_POLL_S = 2.0, 0.05


def _index_lock(path: Path) -> Path | None:
    """The ``index.lock`` of the checkout holding *path* (a worktree's own), or ``None`` outside one."""
    for parent in path.parents:
        dot = parent / ".git"
        if dot.is_dir():
            return dot / "index.lock"
        if dot.is_file():
            text = dot.read_text(encoding="utf-8", errors="replace").strip()
            if text.startswith("gitdir:"):
                gitdir = Path(text.removeprefix("gitdir:").strip())
                return (gitdir if gitdir.is_absolute() else parent / gitdir) / "index.lock"  # fmt: skip
            return None
    return None


def settled_exists(path: Path, *, wait_s: float = SETTLE_WAIT_S) -> bool:
    """Whether *path* exists, a checkout's rewrite of it waited out.

    A fast-forward rewrites a changed file by unlink and create, so a reader
    of a checkout the configurator moves can find a file missing for a
    moment. A missing file is believed only once no git operation holds the
    checkout's ``index.lock`` (or after *wait_s*); a missing file outside
    any rewrite costs a few ``stat`` calls, never a wait.
    """
    path = Path(path)
    if path.exists():
        return True
    try:
        lock = _index_lock(path)
    except OSError:
        return False
    deadline = time.monotonic() + wait_s
    while lock is not None and lock.exists() and time.monotonic() < deadline:
        time.sleep(SETTLE_POLL_S)
        if path.exists():
            return True
    return path.exists()


def experiment_config_path(
    experiment: str,
    folder: str,
    filename: str,
    *,
    base: Path | None = None,
    config_path: Path | None = None,
) -> Path | None:
    """The conventional per-experiment file ``<base>/<experiment>/<folder>/<filename>``, if it exists.

    An empty *folder* is the experiment folder's root (``native_devices.yaml``).

    ``None`` when the repository cannot be resolved or the file is absent —
    every consumer treats a missing overlay as "the defaults".
    """
    base = base if base is not None else scanner_configs_base(config_path)
    if base is None:
        return None
    path = base / experiment / folder / filename
    return path if settled_exists(path) else None

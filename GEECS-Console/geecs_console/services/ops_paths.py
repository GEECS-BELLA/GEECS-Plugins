"""Read-only path resolution for the Ops menu.

Small pure functions returning ``Path | None`` so the menu handlers stay
trivial (resolve, then ``QDesktopServices.openUrl``) and every resolution
rule is unit-testable without launching Finder/Explorer.

The today's-scans resolver is **strictly read-only** — it never creates
directories.  The repo-wide scan-folder invariant applies: GUI/analysis code
is a consumer of scan folders, never a producer; only the scanner side
brings ``scans/`` trees into existence.  A missing daily folder is reported
by the caller ("no scans today"), never created here.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: ``ScanNNN`` daily-folder entries (three or more digits, per convention).
_SCAN_DIR_RE = re.compile(r"^Scan(\d{3,})$")

GITHUB_URL = "https://github.com/GEECS-BELLA/GEECS-Plugins"

#: The shared config file every GEECS package reads.  A path literal only —
#: the console never imports the legacy API package that owns this file
#: (the no-import pin blesses exactly this one string).
USER_CONFIG_PATH = Path("~/.config/geecs_python_api/config.ini")


def experiment_configs_folder(
    experiment: str, base: Optional[Path] = None
) -> Optional[Path]:
    """Resolve the current experiment's configs directory, if it exists.

    Parameters
    ----------
    experiment : str
        The selected experiment name; when empty the configs-repo
        experiments root itself is the target.
    base : Path, optional
        The configs-repo experiments root.  Defaults to the same root
        ``ConsoleConfigs`` scans (lazy ``geecs_bluesky`` import inside,
        offline-safe); tests pass a tmp path.

    Returns
    -------
    Path or None
        The existing directory to open, or ``None`` when the configs root
        is unresolvable or the folder does not exist.
    """
    if base is None:
        from geecs_console.services.configs import _configs_base

        base = _configs_base()
    if base is None:
        return None
    target = base / experiment if experiment else base
    return target if target.is_dir() else None


def user_config_target(config_path: Optional[Path] = None) -> Optional[Path]:
    """Resolve what "open the user config" should open.

    Parameters
    ----------
    config_path : Path, optional
        The config file location; defaults to :data:`USER_CONFIG_PATH`
        (``~`` expanded).  Tests pass a tmp path.

    Returns
    -------
    Path or None
        The ``config.ini`` file when it exists, its parent directory when
        only the directory exists (the caller reports the missing file), or
        ``None`` when neither exists.
    """
    path = (config_path if config_path is not None else USER_CONFIG_PATH).expanduser()
    if path.is_file():
        return path
    if path.parent.is_dir():
        return path.parent
    return None


def todays_scan_folder(
    experiment: str = "",
    base_path: Optional[Path] = None,
    today: Optional[date] = None,
) -> Optional[Path]:
    """Build today's daily ``scans/`` folder path — read-only, never creates.

    Uses ``geecs_data_utils.ScanPaths.get_daily_scan_folder`` (imported
    lazily), which is pure path construction: no directory is created or
    touched here, and the returned path may not exist yet — the caller
    checks ``is_dir()`` and reports "no scans today" instead of creating
    anything (repo scan-folder invariant).

    Parameters
    ----------
    experiment : str, optional
        The selected experiment; falls back to the ``config.ini`` default
        experiment when empty.
    base_path : Path, optional
        The data root; defaults to the ``GeecsPathsConfig`` base path.
        Tests pass a tmp path.
    today : datetime.date, optional
        The date to resolve (tests pin it); defaults to today.

    Returns
    -------
    Path or None
        The candidate ``.../{YY_MMDD}/scans`` path (existing or not), or
        ``None`` when ``geecs_data_utils`` / the data root / the experiment
        is unresolvable.
    """
    try:
        from geecs_data_utils import ScanPaths
    except Exception as exc:  # noqa: BLE001 — offline-first: report, don't raise
        logger.info("geecs_data_utils unavailable: %s", exc)
        return None
    paths_config = ScanPaths.paths_config
    if base_path is None:
        if paths_config is None:
            return None
        base_path = Path(paths_config.base_path)
    resolved_experiment = experiment or (
        getattr(paths_config, "experiment", None) or ""
    )
    if not resolved_experiment:
        return None
    day = today if today is not None else date.today()
    try:
        tag = ScanPaths.get_scan_tag(
            day.year, day.month, day.day, 0, experiment=resolved_experiment
        )
        return ScanPaths.get_daily_scan_folder(tag=tag, base_directory=base_path)
    except Exception as exc:  # noqa: BLE001 — path building must not crash the GUI
        logger.info("today's scan folder unresolvable: %s", exc)
        return None


def highest_scan_number(scans_dir: Optional[Path]) -> Optional[int]:
    """Return the highest existing ``ScanNNN`` number — strictly read-only.

    Peeks at an existing daily ``scans/`` folder for the R6 idle display
    ("Scan NNN (previous)").  Resolution and listing only: nothing on this
    path is ever created or modified (repo scan-folder invariant — the
    scanner side is the only producer of scan folders).

    Parameters
    ----------
    scans_dir : Path or None
        The daily ``scans/`` folder (usually from
        :func:`todays_scan_folder`); ``None`` or a missing/unreadable
        directory yields ``None``.

    Returns
    -------
    int or None
        The largest ``NNN`` among ``ScanNNN`` subdirectories, or ``None``
        when the folder is absent, unreadable, or holds no scan folders.
    """
    if scans_dir is None:
        return None
    try:
        if not scans_dir.is_dir():
            return None
        numbers = [
            int(match.group(1))
            for entry in scans_dir.iterdir()
            if entry.is_dir() and (match := _SCAN_DIR_RE.match(entry.name))
        ]
    except OSError as exc:  # a flaky network mount must not crash the GUI
        logger.info("cannot list %s: %s", scans_dir, exc)
        return None
    return max(numbers) if numbers else None

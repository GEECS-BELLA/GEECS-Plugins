"""Claim the next day-scoped scan number and folder — the scanner-side act.

Day-scoped scan numbering with a multi-writer claim protocol is one of the
GEECS things with no native Bluesky home
(``Planning/native_bluesky/03_clean_room_rebuild.md`` §6).  This module is
the **only** place in GeecsBluesky allowed to bring a ``scans/ScanNNN/``
folder into existence (the cross-package invariant in the root
``CLAUDE.md``): every analysis-side consumer treats the folder as
pre-existing.

Three pieces, one job each (§4.C):

- :func:`claim_scan` — the claim itself (``geecs_data_utils.ScanPaths``).
- :class:`GeecsScanPathProvider` — the ophyd-async ``PathProvider`` every
  native-saving detector holds: ``ScanNNN/<GEECS device>/`` for the run
  currently claimed, nothing outside a run.
- :func:`claim_scan_preprocessor` — the RunEngine preprocessor: on
  ``open_run`` it claims, injects ``scan_number`` / ``scan_id`` /
  ``scan_folder`` / ``experiment`` into the run's metadata and points the
  provider at the folder; on ``close_run`` it releases the provider.  It
  does exactly this and must never grow a second job (it is not the #809
  preamble).

**Every run claims.**  There is no per-run opt-out: a run the RunEngine
opens on the worker is a GEECS scan, numbered and foldered, whatever its
detectors are (a scalar-only magnet scan still gets its s-file).  A
RunEngine that must not claim (hermetic tests, a box without the data
share) is built without the preprocessor (``make_run_engine(claim=False)``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from pathlib import Path, PurePath
from typing import Any

from bluesky.preprocessors import msg_mutator
from bluesky.utils import Msg
from ophyd_async.core import PathInfo, PathProvider

from geecs_bluesky.exceptions import GeecsConfigurationError

logger = logging.getLogger(__name__)


def claim_scan(experiment: str = "") -> tuple[Any | None, str | None]:
    """Claim the next day-scoped scan via ``geecs_data_utils``; return (ScanTag, folder).

    Scanner-side operation — the one place allowed to bring a
    ``scans/ScanNNN/`` folder into existence.  Returns ``(None, None)`` if ``geecs_data_utils`` is unavailable,
    the NetApp is unreachable, or the claim fails.  The full ``ScanTag`` is
    returned for callers that need it (e.g. ScanAnalysis analyzers load files
    by tag); use :func:`claim_scan_number` when only the number matters.

    Parameters
    ----------
    experiment:
        GEECS experiment name (e.g. ``"Undulator"``).
    """
    try:
        from geecs_data_utils import ScanPaths
    except Exception:
        logger.debug("geecs_data_utils not available; scan numbering disabled")
        return None, None

    try:
        if ScanPaths.paths_config is None:
            ScanPaths.reload_paths_config(default_experiment=experiment or None)
        tag = ScanPaths.get_next_scan_tag(experiment=experiment or None)
        scan_data = ScanPaths(tag=tag, read_mode=False)
        folder = scan_data.get_folder()
        logger.info("Claimed scan number %d -> %s", tag.number, folder)
        return tag, str(folder) if folder else None
    except Exception:
        logger.warning("Could not claim scan number", exc_info=True)
        return None, None


def claim_scan_number(experiment: str = "") -> tuple[int | None, str | None]:
    """Claim the next day-scoped scan number and folder (see :func:`claim_scan`)."""
    tag, folder = claim_scan(experiment)
    return (tag.number if tag is not None else None), folder


class GeecsScanPathProvider(PathProvider):
    """``ScanNNN/<device>/`` for the run currently claimed.

    One instance per worker, shared by every native-saving detector the
    namespace builds; :func:`claim_scan_preprocessor` points it at each
    run's folder at ``open_run`` and releases it at ``close_run``.  Called
    with the **GEECS device name** (the directory analysis readers look
    for), it returns that sub-directory of the run folder — the detector's
    data logic creates the leaf, never the run folder.  Outside a run it
    refuses: a detector prepared with no scan claimed has nowhere to write.
    """

    def __init__(self) -> None:
        self._folder: Path | None = None

    @property
    def folder(self) -> Path | None:
        """The claimed run folder, ``None`` between runs."""
        return self._folder

    def point_at(self, folder: Path | str | None) -> None:
        """Set (or, with ``None``, release) the run folder."""
        self._folder = None if folder is None else Path(folder)

    def __call__(self, datakey_name: str | None = None) -> PathInfo:
        """The device directory for *datakey_name* (a GEECS device name) this run."""
        if self._folder is None:
            raise GeecsConfigurationError(
                "no scan claimed: a native-saving detector was prepared outside "
                "a claimed run (is the claim_scan preprocessor installed?)"
            )
        if not datakey_name:
            raise ValueError("GeecsScanPathProvider needs the device directory name")
        return PathInfo(
            directory_path=PurePath(self._folder / datakey_name),
            filename=datakey_name,
        )


def claim_scan_preprocessor(
    plan: Generator[Msg, Any, Any],
    *,
    experiment: str,
    path_provider: GeecsScanPathProvider | None = None,
    claim: Callable[[str], tuple[Any | None, str | None]] = claim_scan,
) -> Generator[Msg, Any, Any]:
    """Claim a scan number for every run *plan* opens; release it when it closes.

    A :func:`~bluesky.preprocessors.msg_mutator` with one job: on
    ``open_run`` it calls *claim*, injects ``scan_number``, ``scan_id`` (the
    Bluesky display field), ``scan_folder``, ``experiment`` and
    ``scan_tag`` into the run's metadata and points *path_provider* at the
    folder; on ``close_run`` it releases the provider.  A failed claim
    raises :class:`GeecsConfigurationError` before the run opens — a scan
    the data share cannot number does not run.

    Parameters
    ----------
    plan :
        The plan to wrap.
    experiment :
        GEECS experiment name (the day folder's root).
    path_provider :
        The shared provider the namespace's detectors hold.
    claim :
        The claim function (:func:`claim_scan`); a test seam.
    """

    def _mutate(msg: Msg) -> Msg:
        if msg.command == "open_run":
            tag, folder = claim(experiment)
            if tag is None or folder is None:
                raise GeecsConfigurationError(
                    f"could not claim a scan number for {experiment!r}: the "
                    "data share is unreachable or geecs_data_utils cannot "
                    "resolve it (see the warning above) — the scan does not run"
                )
            md = dict(msg.kwargs)
            md.update(
                scan_number=tag.number,
                scan_id=tag.number,
                scan_folder=str(folder),
                experiment=experiment,
                scan_tag={
                    "year": tag.year,
                    "month": tag.month,
                    "day": tag.day,
                    "number": tag.number,
                    "experiment": experiment,
                },
            )
            if path_provider is not None:
                path_provider.point_at(folder)
            logger.info("scan %d claimed: %s", tag.number, folder)
            return Msg("open_run", msg.obj, *msg.args, run=msg.run, **md)
        if msg.command == "close_run" and path_provider is not None:
            # Released as the run closes: nothing prepares a detector between
            # close_run and unstage, and the callbacks read scan_folder from
            # the start document, not from the provider.
            path_provider.point_at(None)
        return msg

    return (yield from msg_mutator(plan, _mutate))


__all__ = [
    "GeecsScanPathProvider",
    "claim_scan",
    "claim_scan_number",
    "claim_scan_preprocessor",
]

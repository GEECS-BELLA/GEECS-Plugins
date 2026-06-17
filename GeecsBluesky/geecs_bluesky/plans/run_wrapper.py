"""geecs_run_wrapper — GEECS run bookkeeping shared by the scanner and notebooks.

Wraps any plan so it inherits the GEECS event-schema run metadata (scan number,
folder, ``scan_id``, experiment) and per-detector native file saving — the same
bookkeeping :class:`~geecs_bluesky.scanner_bridge.BlueskyScanner` applies — so a
custom notebook plan lands data in the same place, numbered the same way, with
the same start-document shape as a GUI scan.

The wrapper does **not** itself create scan folders: the caller claims the
number with :func:`claim_scan_number` (an explicit scanner-side action) and
passes it in.  This keeps the "analysis code never creates scan folders"
cross-package invariant intact — only the deliberate claim step does.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp

logger = logging.getLogger(__name__)


def claim_scan_number(experiment: str = "") -> tuple[int | None, str | None]:
    """Claim the next day-scoped scan number and folder via ``geecs_data_utils``.

    Scanner-side operation — this is the one place (outside the GUI's own
    ``ScanDataManager``) allowed to bring a ``scans/ScanNNN/`` folder into
    existence.  Returns ``(None, None)`` if ``geecs_data_utils`` is unavailable,
    the NetApp is unreachable, or the claim fails.

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
        return tag.number, str(folder) if folder else None
    except Exception:
        logger.warning("Could not claim scan number", exc_info=True)
        return None, None


def _save_cleanup_plan(saving_detectors: list[tuple]):
    """Turn saving off for all saving detectors (runs even on abort)."""
    if not saving_detectors:
        return
    mv_args: list = []
    for det, _path in saving_detectors:
        mv_args.extend([det.save, "off"])
        logger.debug("Saving disabled for %s", det.name)
    yield from bps.mv(*mv_args)


def _collect_scalar_headers(devices: list) -> dict[str, str]:
    """Merge each device's ``_column_headers`` into one event-key → header map.

    The map lets the Tiled→s-file exporter rename Bluesky's irreversibly
    mangled ``<ophyd>-<safe_var>`` columns back to the legacy
    ``Device Variable`` headers.  Devices without ``_column_headers`` (or a
    falsy one) contribute nothing.
    """
    headers: dict[str, str] = {}
    for dev in devices:
        dev_headers = getattr(dev, "_column_headers", None)
        if dev_headers:
            headers.update(dev_headers)
    return headers


def geecs_run_wrapper(
    plan,
    *,
    experiment: str = "",
    scan_number: int | None = None,
    scan_folder: str | None = None,
    saving_detectors: list[tuple] | None = None,
    devices: list | None = None,
    extra_md: dict[str, Any] | None = None,
):
    """Wrap *plan* with GEECS scan-number metadata and native file saving.

    Parameters
    ----------
    plan:
        Any Bluesky plan (typically a ``geecs_step_scan`` /
        ``geecs_free_run_step_scan`` generator).
    experiment:
        GEECS experiment name, recorded in the start document.
    scan_number, scan_folder:
        The claimed day-scoped number and folder (see :func:`claim_scan_number`).
        When ``scan_number`` is given it is also set as the Bluesky ``scan_id``
        (the display field) — see ``EVENT_SCHEMA.md``.
    saving_detectors:
        ``(detector, save_path)`` tuples for devices that write native files.
        Each gets ``localsavingpath`` + ``save="on"`` before the run and
        ``save="off"`` in a finalize wrapper (runs even on abort).
    devices:
        All devices contributing scalar columns (detectors + scan motor).
        Their ``_column_headers`` are merged into a ``geecs_scalar_headers``
        start-doc key so the Tiled→s-file exporter can recover legacy
        ``Device Variable`` headers (see ``EVENT_SCHEMA.md``).
    extra_md:
        Additional run metadata to inject (e.g. ``device_var``, ``description``).

    Yields
    ------
    Bluesky messages.

    Notes
    -----
    Injected metadata takes precedence over the plan's own (``inject_md_wrapper``
    uses a ``ChainMap``), so ``scan_id`` and scan bookkeeping always win; the
    plan keeps ownership of its intrinsic keys (``plan_name``,
    ``acquisition_mode``, ``geecs_event_schema``, positions, …).
    """
    saving = list(saving_detectors or [])

    md: dict[str, Any] = {"bluesky_backend": True}
    if experiment:
        md["experiment"] = experiment
    if scan_number is not None:
        md["scan_number"] = scan_number
        md["scan_id"] = scan_number  # Bluesky display field = GEECS scan number
    if scan_folder is not None:
        md["scan_folder"] = scan_folder
    if saving:
        md["nonscalar_save_paths"] = {
            getattr(det, "_geecs_device_name", det.name): path for det, path in saving
        }
    scalar_headers = _collect_scalar_headers(devices or [])
    if scalar_headers:
        md["geecs_scalar_headers"] = scalar_headers
    md.update(extra_md or {})

    wrapped = bpp.inject_md_wrapper(plan, md)

    if not saving:
        yield from wrapped
        return

    mv_args: list = []
    for det, path in saving:
        os.makedirs(path, exist_ok=True)
        logger.info("Save path for %s: %s", det.name, path)
        mv_args.extend([det.localsavingpath, path, det.save, "on"])
    yield from bps.mv(*mv_args)
    yield from bpp.finalize_wrapper(wrapped, _save_cleanup_plan(saving))

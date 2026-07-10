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
from collections.abc import Sequence
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp

logger = logging.getLogger(__name__)


def _saving_detector_paths(item: Sequence) -> tuple[Any, str, str]:
    """Return ``(detector, event_path, device_command_path)`` for one save item."""
    if len(item) == 2:
        det, event_path = item
        device_command_path = event_path
    elif len(item) == 3:
        det, event_path, device_command_path = item
    else:
        raise ValueError(
            "saving_detectors entries must be (detector, event_path) or "
            "(detector, event_path, device_command_path)"
        )
    return det, str(event_path), str(device_command_path)


def claim_scan(experiment: str = "") -> tuple[Any | None, str | None]:
    """Claim the next day-scoped scan via ``geecs_data_utils``; return (ScanTag, folder).

    Scanner-side operation — this is the one place (outside the GUI's own
    ``ScanDataManager``) allowed to bring a ``scans/ScanNNN/`` folder into
    existence.  Returns ``(None, None)`` if ``geecs_data_utils`` is unavailable,
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


def _save_cleanup_plan(saving_detectors: list[tuple]):
    """Turn saving off for all saving detectors (runs even on abort)."""
    if not saving_detectors:
        return
    mv_args: list = []
    for det, _event_path, _device_command_path in map(
        _saving_detector_paths, saving_detectors
    ):
        mv_args.extend([det.save, "off"])
        logger.debug("Saving disabled for %s", det.name)
    yield from bps.mv(*mv_args)


def save_enable_plan(saving_detectors: list[tuple]):
    """Plan stub: create save dirs, set ``localsavingpath`` + ``save='on'``.

    The device puts (``:SP`` writes on connected devices) that start native
    file saving.  Callers that window saving to the trigger-stopped part of
    the scan (Gate-2 save windowing: ``GeecsBluesky/CLAUDE.md``) pass this
    as the step plans' ``enable_saving`` hook and give
    :func:`geecs_run_wrapper` ``defer_save_on=True``; direct
    ``geecs_run_wrapper`` users keep the eager default.

    Parameters
    ----------
    saving_detectors:
        ``(detector, event_path[, device_command_path])`` tuples, as in
        :func:`geecs_run_wrapper`.

    Yields
    ------
    Bluesky messages (one concurrent ``mv`` over all detectors).
    """
    saving = list(saving_detectors or [])
    if not saving:
        return
    mv_args: list = []
    for det, event_path, device_command_path in map(_saving_detector_paths, saving):
        os.makedirs(event_path, exist_ok=True)
        logger.info(
            "Save path for %s: event=%s, device=%s",
            det.name,
            event_path,
            device_command_path,
        )
        mv_args.extend([det.localsavingpath, device_command_path, det.save, "on"])
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
    defer_save_on: bool = False,
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
    defer_save_on:
        When true, the wrapper does **not** enable saving itself — the inner
        plan yields :func:`save_enable_plan` once the trigger can no longer
        free-run (Gate-2 save windowing: ``GeecsBluesky/CLAUDE.md``).  The
        finalize ``save="off"`` and the ``nonscalar_save_paths`` metadata
        are emitted either way.
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
            getattr(det, "_geecs_device_name", det.name): event_path
            for det, event_path, _device_command_path in map(
                _saving_detector_paths, saving
            )
        }
    scalar_headers = _collect_scalar_headers(devices or [])
    if scalar_headers:
        md["geecs_scalar_headers"] = scalar_headers
    md.update(extra_md or {})

    wrapped = bpp.inject_md_wrapper(plan, md)

    if not saving:
        yield from wrapped
        return

    if not defer_save_on:
        yield from save_enable_plan(saving)
    # Save-off stays the innermost finalize: it runs (even on abort) BEFORE
    # the caller's disarm/closeout finalizes, so saving is always stopped
    # while the trigger is still unable to free-run.
    yield from bpp.finalize_wrapper(wrapped, _save_cleanup_plan(saving))

"""Claim the next day-scoped scan number and folder — the scanner-side act.

Day-scoped scan numbering with a multi-writer claim protocol is one of the
six GEECS things with no native Bluesky home
(``Planning/native_bluesky/03_clean_room_rebuild.md`` §6).  This module is
the **only** place in GeecsBluesky allowed to bring a ``scans/ScanNNN/``
folder into existence (the cross-package invariant in the root
``CLAUDE.md``): every analysis-side consumer treats the folder as
pre-existing.  Phase 2 of the rebuild wraps :func:`claim_scan` in the
``claim_scan`` preprocessor that claims on ``open_run``, injects the number
into ``md`` and points the path provider at the run (§4.C).
"""

from __future__ import annotations

import logging
from typing import Any

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

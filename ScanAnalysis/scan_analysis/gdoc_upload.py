"""Google Doc upload integration for scan analysis results.

Provides a thin wrapper around ``logmaker_4_googledocs.docgen`` to upload
summary figures produced by completed analyzers into the experiment scan log.

The ``logmaker_4_googledocs`` package is optional: if it is not installed
(or fails to import), all upload calls are silently skipped.

Slot mapping
------------
Each analyzer that opts in to gdoc upload is assigned a ``gdoc_slot`` (0–3)
in its config.  Slots map to cells in the 2×2 display table that the scan-log
template inserts for every scan entry:

    slot 0 → row 1, col 0   |  slot 1 → row 1, col 1
    slot 2 → row 2, col 0   |  slot 3 → row 2, col 1

Usage
-----
In ``task_queue.run_worklist``, after a successful analyzer run::

    from scan_analysis.gdoc_upload import upload_summary_to_gdoc

    if getattr(analyzer, "upload_to_gdoc", False) and display_files:
        upload_summary_to_gdoc(
            scan_tag=tag,
            display_files=display_files,
            gdoc_slot=analyzer.gdoc_slot,
        )
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import List, Optional

from geecs_data_utils import ScanTag

logger = logging.getLogger(__name__)

# Slot index → (row, col) in the 2×2 display table embedded in each scan entry.
SLOT_TO_ROW_COL: dict[int, tuple[int, int]] = {
    0: (1, 0),
    1: (1, 1),
    2: (2, 0),
    3: (2, 1),
}


def _try_load_docgen():
    """Attempt to import docgen without breaking callers if unavailable."""
    if not find_spec("logmaker_4_googledocs"):
        logger.debug("logmaker_4_googledocs not installed; gdoc upload disabled.")
        return None
    try:
        from logmaker_4_googledocs import docgen

        logger.info("Loaded logmaker_4_googledocs.docgen successfully.")
        return docgen
    except Exception as exc:
        logger.warning("docgen import failed (%s); gdoc upload disabled.", exc)
        return None


DOCGEN = _try_load_docgen()


def upload_summary_to_gdoc(
    scan_tag: ScanTag,
    display_files: List[str],
    gdoc_slot: int,
    document_id: Optional[str] = None,
) -> bool:
    """Upload the last summary figure to a table cell in the scan log.

    Parameters
    ----------
    scan_tag : ScanTag
        Identifies the scan (year/month/day/number/experiment).  The
        ``experiment`` field is used to look up the correct experiment INI and
        Google Doc ID.
    display_files : list of str
        Paths to summary figures produced by the analyzer.  The **last** entry
        is uploaded (typically the composite summary figure).
    gdoc_slot : int
        Target cell index (0–3) in the 2×2 display table.
    document_id : str, optional
        Google Doc ID.  If ``None``, the ID is read from the experiment INI
        via ``logmaker_4_googledocs``.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if the upload was skipped or failed.
    """
    if not DOCGEN:
        logger.debug(
            "docgen not available; skipping gdoc upload for scan %s.", scan_tag
        )
        return False

    if not display_files:
        logger.warning(
            "No display files to upload for scan %s slot %d.", scan_tag, gdoc_slot
        )
        return False

    row_col = SLOT_TO_ROW_COL.get(gdoc_slot)
    if row_col is None:
        logger.error(
            "Invalid gdoc_slot %d for scan %s; must be 0–3.", gdoc_slot, scan_tag
        )
        return False

    image_path = str(display_files[-1])
    row, col = row_col

    try:
        success = DOCGEN.insertImageToExperimentLog(
            scanNumber=scan_tag.number,
            row=row,
            column=col,
            image_path=image_path,
            documentID=document_id,
            experiment=scan_tag.experiment,
        )
        if success:
            logger.info(
                "Uploaded '%s' to scan log slot %d (row=%d, col=%d) for scan %s.",
                image_path,
                gdoc_slot,
                row,
                col,
                scan_tag,
            )
        else:
            logger.warning(
                "Upload call completed but Apps Script reported failure for scan %s slot %d.",
                scan_tag,
                gdoc_slot,
            )
        return bool(success)
    except Exception as exc:
        logger.error(
            "Failed to upload display file to gdoc for scan %s slot %d: %s",
            scan_tag,
            gdoc_slot,
            exc,
            exc_info=True,
        )
        return False

"""Google Doc upload integration for scan analysis results.

Provides a thin wrapper around ``logmaker_4_googledocs.docgen`` to upload
summary figures produced by completed analyzers into the experiment scan log.

The ``logmaker_4_googledocs`` package is optional: if it is not installed
(or fails to import), all upload calls are silently skipped.

Slot mapping
------------
Analyzers with ``gdoc_slot`` set (0–3) have their last summary figure inserted
into the corresponding cell of the 2×2 display table in the scan-log entry:

    slot 0 → row 1, col 0   |  slot 1 → row 1, col 1
    slot 2 → row 2, col 0   |  slot 3 → row 2, col 1

Analyzers without ``gdoc_slot`` will have their display files uploaded to the
per-day Drive folder and appended as hyperlinks (implemented in the follow-on
PR on branch ``gdoc-hyperlinks``).
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

# Mirrors the mapping in docgen so we can read the INI without importing docgen directly.
_EXPERIMENT_INI_MAP = {
    "Undulator": "HTUparameters.ini",
    "Thomson": "HTTparaeters.ini",
}


def resolve_document_id(experiment: str) -> Optional[str]:
    """Read the current LogID from the experiment INI.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g. ``'Undulator'``).

    Returns
    -------
    str or None
        The Google Doc ID, or ``None`` if the INI is missing or has no ``LogID``.
    """
    if not DOCGEN:
        return None
    import configparser
    from pathlib import Path

    config_file = _EXPERIMENT_INI_MAP.get(experiment)
    if not config_file:
        logger.warning("No INI mapping for experiment '%s'.", experiment)
        return None
    config_path = Path(DOCGEN.__file__).parent / config_file
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    doc_id = cfg["DEFAULT"].get("logid")
    if not doc_id:
        logger.warning(
            "LogID not found in %s — run createExperimentLog() first.", config_path
        )
        return None
    return doc_id


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


def upload_links_to_gdoc(
    scan_tag: ScanTag,
    analyzer_id: str,
    display_files: List[str],
    document_id: Optional[str] = None,
) -> bool:
    """Upload all display files to Drive and append hyperlinks in the scan log.

    Calls the ``upload_display_files_and_link`` helper in ``docgen``, which
    uploads each file to the per-day Drive folder and appends a clickable link
    inside the ``"Additional diagnostics:"`` cell of the scan entry.

    Parameters
    ----------
    scan_tag : ScanTag
        Identifies the scan (year/month/day/number/experiment).
    analyzer_id : str
        Analyzer identifier; used as a label prefix for each link.
    display_files : list of str
        Local paths to the files to upload and link.
    document_id : str, optional
        Google Doc ID. If ``None``, the ID is read from the experiment INI.

    Returns
    -------
    bool
        ``True`` if at least one file was successfully linked, ``False`` otherwise.
    """
    if not DOCGEN:
        logger.debug(
            "docgen not available; skipping hyperlink upload for scan %s.", scan_tag
        )
        return False

    if not display_files:
        logger.warning("No display files to link for scan %s.", scan_tag)
        return False

    try:
        linked = DOCGEN.upload_display_files_and_link(
            scan_number=scan_tag.number,
            analyzer_id=analyzer_id,
            display_files=display_files,
            document_id=document_id,
            experiment=scan_tag.experiment,
        )
        logger.info(
            "Linked %d/%d files in scan log for scan %s.",
            linked,
            len(display_files),
            scan_tag,
        )
        return linked > 0
    except Exception as exc:
        logger.error(
            "Failed to upload links for scan %s analyzer %s: %s",
            scan_tag,
            analyzer_id,
            exc,
            exc_info=True,
        )
        return False

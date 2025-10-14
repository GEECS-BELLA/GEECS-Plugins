"""Dispatch and orchestration for scan analyzers.

Maps analyzer descriptors to instances and runs them for a given scan tag.
Optionally inserts selected outputs into a Google Doc scan log when available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from scan_analysis.base import ScanAnalyzerInfo
    from geecs_data_utils import ScanTag

import itertools
import logging
from importlib.util import find_spec

from scan_analysis.base import ScanAnalyzer  # required import

logger = logging.getLogger(__name__)


def try_load_docgen():
    """Attempt to import docgen safely without breaking the main module."""
    if not find_spec("logmaker_4_googledocs.docgen"):
        logger.debug("logmaker_4_googledocs not installed.")
        return None
    try:
        from logmaker_4_googledocs import docgen  # noqa: F401

        logger.info("Loaded docgen successfully.")
        return docgen
    except Exception as e:
        logger.warning("Docgen import failed (%s); disabling.", e)
        return None


DOCGEN = try_load_docgen()
LOADED_DOCGEN = DOCGEN is not None


def instantiate_scan_analyzer(scan_analyzer_info: ScanAnalyzerInfo) -> ScanAnalyzer:
    """Instantiate an analyzer from a :class:`ScanAnalyzerInfo`.

    Parameters
    ----------
    scan_analyzer_info : ScanAnalyzerInfo
        Descriptor containing the target analyzer class, device name, and
        constructor kwargs (if any).

    Returns
    -------
    ScanAnalyzer
        A concrete analyzer instance (subclass of :class:`ScanAnalyzer`), ready to run.
    """
    return scan_analyzer_info.scan_analyzer_class(
        device_name=scan_analyzer_info.device_name,
        skip_plt_show=True,
        **scan_analyzer_info.scan_analyzer_kwargs,
    )


def analyze_scan(
    tag: ScanTag,
    scan_analyzer_list: list[ScanAnalyzer],
    upload_to_scanlog: bool = True,
    documentID: Optional[str] = None,
    debug_mode: bool = False,
) -> None:
    """Run all analyzers for a given scan and optionally update the scan log.

    Parameters
    ----------
    tag : ScanTag
        Identifies the scan (date/experiment/number).
    scan_analyzer_list : list[ScanAnalyzer]
        Analyzer instances to execute.
    upload_to_scanlog : bool, default=True
        If True and Google Doc integration is available, insert selected outputs
        into the scan log document.
    documentID : str, optional
        Google Doc ID. If omitted, downstream logic may choose a default (e.g., today's log).
    debug_mode : bool, default=False
        If True, analysis is skipped (helpful for dry runs of the pipeline).

    Returns
    -------
    None
        Results are handled by each analyzer; when enabled, selected artifacts
        are uploaded to the scan log.
    """
    all_display_files: list[list[str]] = []

    for analyzer in scan_analyzer_list:
        if debug_mode:
            logger.info("Debug mode: skipping analysis for %s", analyzer)
            continue
        try:
            index_of_files = analyzer.run_analysis(scan_tag=tag)
            logger.debug("Analyzer %s produced: %s", analyzer, index_of_files)
            if index_of_files:
                all_display_files.append(index_of_files)
        except Exception as err:
            logger.error(
                "Error in analyze_scan %02d/%02d/%04d:Scan%03d: %s",
                tag.month,
                tag.day,
                tag.year,
                tag.number,
                err,
                exc_info=True,
            )

    if LOADED_DOCGEN and upload_to_scanlog and all_display_files:
        flattened = list(itertools.chain.from_iterable(all_display_files))
        logger.info("Uploading %d artifacts to scan log.", len(flattened))
        insert_display_content_to_doc(tag, flattened, documentID=documentID)
    else:
        if upload_to_scanlog and not LOADED_DOCGEN:
            logger.debug("Docgen unavailable—skipping scan log upload.")
        elif not all_display_files:
            logger.debug("No artifacts to upload.")


def insert_display_content_to_doc(
    scan_tag: ScanTag, path_list: list[str], documentID: Optional[str] = None
) -> None:
    """Insert selected artifacts into the Google Doc scan log.

    Parameters
    ----------
    scan_tag : ScanTag
        Tag containing metadata (year, month, day, number, experiment).
    path_list : list of str
        File paths to insert (e.g., images). Only the first four entries are used,
        placed into a 2×2 table in row-major order.
    documentID : str, optional
        Target Google Doc ID. If omitted, the integration layer may select a default.

    Returns
    -------
    None
        Side effect: updates the Google Doc when integration is available.

    Notes
    -----
    - Requires `logmaker_4_googledocs.docgen`. If unavailable, this function logs an error.
    - Paths are passed as strings for compatibility with Apps Script.
    """
    if not DOCGEN:
        logger.error("Docgen not loaded; cannot insert display content.")
        return

    # 2x2 table mapping (row, col)
    table_mapping = [(1, 0), (1, 1), (2, 0), (2, 1)]

    try:
        for i, image_path in enumerate(path_list[: len(table_mapping)]):
            row, col = table_mapping[i]
            DOCGEN.insertImageToExperimentLog(
                scanNumber=scan_tag.number,
                row=row,
                column=col,
                image_path=str(image_path),
                documentID=documentID,
                experiment=scan_tag.experiment,
            )
        logger.info("Inserted display content for scan %03d.", scan_tag.number)
    except Exception as e:
        logger.error(
            "Error inserting display content for scan %03d: %s",
            scan_tag.number,
            e,
            exc_info=True,
        )


if __name__ == "__main__":
    from geecs_data_utils import ScanPaths
    from scan_analysis.mapping.map_Undulator import undulator_analyzers

    test_tag = ScanPaths.get_scan_tag(2025, 4, 3, number=2, experiment="Undulator")
    test_analyzer = undulator_analyzers[0]
    analyze_scan(test_tag, scan_analyzer_list=[test_analyzer], upload_to_scanlog=False)

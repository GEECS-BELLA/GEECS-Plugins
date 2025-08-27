"""Dispatch and orchestration for scan analyzers.

This module maps lightweight analyzer descriptors to concrete analyzer instances
and runs them for a given scan tag. Optionally, it inserts selected output
artifacts (images/paths) into a Google Doc scan log when available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from scan_analysis.base import ScanAnalyzerInfo
    from geecs_data_utils import ScanTag

import logging
import itertools

try:
    from logmaker_4_googledocs import docgen

    loaded_docgen = True
except Exception:
    logging.warning(
        "Could not properly load docgen, results will not auto-populate scan log."
    )
    loaded_docgen = False

from scan_analysis.base import ScanAnalyzer

try:
    # Optional imports used only by certain analyzers/configs; safe to fail on machines without these deps.
    pass  # noqa: F401
except Exception:
    logging.warning("From execute_analysis: could not load HASO components.")


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
):
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
    all_display_files = []

    for analyzer in scan_analyzer_list:
        if not debug_mode:
            try:
                index_of_files = analyzer.run_analysis(scan_tag=tag)
                print(f"index of files: {index_of_files}")

                # If analysis produces files, add them to the list.
                if index_of_files is not None:
                    all_display_files.append(index_of_files)
            except Exception as err:
                logging.error(
                    f"Error in analyze_scan {tag.month}/{tag.day}/{tag.year}:Scan{tag.number:03d}): {err}"
                )

    if loaded_docgen and upload_to_scanlog and len(all_display_files) > 0:
        flattened_file_paths = list(itertools.chain.from_iterable(all_display_files))
        print(f"flatten file list: {flattened_file_paths}")
        insert_display_content_to_doc(tag, flattened_file_paths, documentID=documentID)


def insert_display_content_to_doc(
    scan_tag: ScanTag, path_list: list[str], documentID: Optional[str] = None
):
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
    try:
        # Map content entries to table cells
        table_mapping = [
            (1, 0),
            (1, 1),  # Row 1: Col 1, Col 2
            (2, 0),
            (2, 1),  # Row 2: Col 1, Col 2
        ]

        # Iterate over the paths and insert them into the Google Doc
        for i, image_path in enumerate(path_list):
            if i >= len(table_mapping):
                print(
                    f"Ignoring extra entry: {image_path} as the table is limited to 2x2."
                )
                break

            row, col = table_mapping[i]

            # Insert the image into the Google Doc
            docgen.insertImageToExperimentLog(
                scanNumber=scan_tag.number,
                row=row,
                column=col,
                image_path=image_path,  # Pass as string for use with Google scripts
                documentID=documentID,
                experiment=scan_tag.experiment,
            )

        print(f"Successfully inserted display content for scan {scan_tag.number}.")

    except Exception as display_err:
        logging.error(
            f"Error processing display content for scan {scan_tag.number}: {display_err}"
        )


if __name__ == "__main__":
    from geecs_data_utils import ScanPaths
    from scan_analysis.mapping.map_Undulator import undulator_analyzers

    test_tag = ScanPaths.get_scan_tag(2025, 4, 3, number=2, experiment="Undulator")
    test_analyzer = undulator_analyzers[0]

    analyze_scan(test_tag, scan_analyzer_list=[test_analyzer], upload_to_scanlog=False)

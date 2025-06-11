"""
Module containing the mapping of specific analyzers to their respective classes.  Gives an analysis command to the
specified analyzer with the scan folder location
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
except:
    logging.warning(f'could not properly load docgen, results will not auto populate scan log')
    loaded_docgen = False

from scan_analysis.base import ScanAnalyzer

try:
    from image_analysis.offline_analyzers.density_from_phase_analysis import PhaseAnalysisConfig  # import your config class
    from image_analysis.offline_analyzers.HASO_himg_has_processor import HasoHimgHasConfig  # import your config class
except:
    logging.warning(f'from execute_analysis: could not load haso components')



def instantiate_scan_analyzer(scan_analyzer_info: ScanAnalyzerInfo) -> ScanAnalyzer:
    """
    Instantiate a ScanAnalysis (or subclass) using the provided ScanAnalyzerInfo.

    This function unpacks the analyzer class and keyword arguments from the ScanAnalyzerInfo object
    and constructs an instance, injecting the provided scan_tag, device_name, and any analyzer-specific
    configuration from scan_analyzer_kwargs.

    Args:
        scan_analyzer_info (ScanAnalyzerInfo): Metadata describing which analyzer to construct and how.

    Returns:
        ScanAnalyzer: A fully initialized ScanAnalysis or subclass instance ready for use.
    """
    return scan_analyzer_info.scan_analyzer_class(
        device_name=scan_analyzer_info.device_name,
        skip_plt_show=True,
        **scan_analyzer_info.scan_analyzer_kwargs
    )


def analyze_scan(tag: ScanTag, scan_analyzer_list: list[ScanAnalyzerInfo], upload_to_scanlog: bool = True,
                 documentID: Optional[str] = None, debug_mode: bool = False):
    """
    Performs all given analysis routines on a given scan. Optionally uploads results to google doc scanlog.

    :param tag: Tag representing the scan date, number, and experiment
    :param scan_analyzer_list: List of valid analyzers that can be run on the given scan
    :param upload_to_scanlog: If True, will upload index of files to Google scanlog
    :param documentID: If given, the Google doc ID of the scanlog. Otherwise, will default to today's.
    :param debug_mode: If True, will not attempt analysis.
    :return:
    """
    all_display_files = []

    for analyzer_info in scan_analyzer_list:
        device = analyzer_info.device_name if analyzer_info.device_name else ''
        print(tag, ":", analyzer_info.scan_analyzer_class.__name__, device)
        if not debug_mode:
            try:
                # Use the helper to instantiate the analyzer (with image analyzer and file pattern settings)
                logging.info(f'attempting to instantiate image the scan analyzer with an ImageAnalyzer config: {analyzer_info}')
                analyzer = instantiate_scan_analyzer(analyzer_info)
                index_of_files = analyzer.run_analysis(scan_tag=tag)
                print(f'index of files: {index_of_files}')

                # If analysis produces files, add them to the list.
                if index_of_files is not None:
                    all_display_files.append(index_of_files)
            except Exception as err:
                logging.error(f"Error in analyze_scan {tag.month}/{tag.day}/{tag.year}:Scan{tag.number:03d}): {err}")

    if loaded_docgen and upload_to_scanlog and len(all_display_files) > 0:
        flattened_file_paths = list(itertools.chain.from_iterable(all_display_files))
        print(f'flatten file list: {flattened_file_paths}')
        insert_display_content_to_doc(tag, flattened_file_paths, documentID=documentID)



def insert_display_content_to_doc(scan_tag: ScanTag, path_list: list[str], documentID: Optional[str] = None):
    """
    Inserts display content from a list of paths into a Google Doc for a given scan.

    Args:
        scan_tag: The scan tag containing metadata (e.g., year, month, day, number).
        path_list (list[str]): A list of file paths to insert into the Google Doc.
        documentID (str): If given, the Google doc ID of the scanlog.  Otherwise, will default to today's
    Returns:
        None
    """
    try:
        # Map content entries to table cells
        table_mapping = [
            (1, 0), (1, 1),  # Row 1: Col 1, Col 2
            (2, 0), (2, 1)   # Row 2: Col 1, Col 2
        ]

        # Iterate over the paths and insert them into the Google Doc
        for i, image_path in enumerate(path_list):
            if i >= len(table_mapping):
                print(f"Ignoring extra entry: {image_path} as the table is limited to 2x2.")
                break

            row, col = table_mapping[i]

            # Insert the image into the Google Doc
            docgen.insertImageToExperimentLog(
                scanNumber=scan_tag.number,
                row=row,
                column=col,
                image_path=image_path,  # Pass as string for use with Google scripts
                documentID=documentID,
                experiment=scan_tag.experiment
            )

        print(f"Successfully inserted display content for scan {scan_tag.number}.")

    except Exception as display_err:
        logging.error(f"Error processing display content for scan {scan_tag.number}: {display_err}")


if __name__ == '__main__':
    from geecs_data_utils import ScanData
    from scan_analysis.mapping.map_Undulator import undulator_analyzers
    test_tag = ScanData.get_scan_tag(2025, 4, 3, number=2, experiment='Undulator')
    test_analyzer = undulator_analyzers[0]

    analyze_scan(test_tag, scan_analyzer_list=[test_analyzer], upload_to_scanlog=False)

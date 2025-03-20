"""
Module containing the mapping of specific analyzers to their respective classes.  Gives an analysis command to the
specified analyzer with the scan folder location
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from scan_analysis.base import AnalyzerInfo
    from geecs_python_api.controls.api_defs import ScanTag

from pathlib import Path
import logging
import itertools

try:
    from logmaker_4_googledocs import docgen
    loaded_docgen = True
except:
    logging.warning(f'could not properly load docgen, results will not auto populate scan log')
    loaded_docgen = False

from scan_analysis.base import ScanAnalysis
from image_analysis.offline_analyzers.density_from_phase_analysis import PhaseAnalysisConfig  # import your config class

def instantiate_analyzer(tag: ScanTag, analyzer_info: AnalyzerInfo) -> ScanAnalysis:
    """
    Instantiate an analyzer from an AnalyzerInfo object.
    If an image analyzer class is specified, instantiate it and pass it to the analyzer's constructor.
    Also, if the analyzer supports a file pattern and one is provided, set it; otherwise, use the default pattern.
    Additionally, handle extra configuration that may be a path (to a config file) or a configuration object.
    """
    image_analyzer = None
    config = getattr(analyzer_info, "image_analysis_config", None)
    logging.info(f'the image analyzer config from the passed analyzer info is {config}')
    logging.info(f'the complete analyzer info is {analyzer_info}')
    logging.info(f'the image analyzer class is  {analyzer_info.image_analyzer_class}')


    if analyzer_info.image_analyzer_class:
        logging.info(f'image analyzer class recongnized, search for config')

        if config is not None:
            logging.info(f'image analyzer class recongnized, there is a config')
            logging.info(f'config type is {type(config)}')
            # If it's a string or Path, assume it's a path to a config file.
            if isinstance(config, (str, Path)):
                # You could load the configuration from file here.
                # For example: config_obj = load_config(Path(config))
                # For now, we'll simply pass the path.
                image_analyzer = analyzer_info.image_analyzer_class(config=config)
            # Otherwise, if it's an instance of PhaseAnalysisConfig (or similar), pass it directly.
            elif isinstance(config, PhaseAnalysisConfig):
                logging.info(f'config type matched the PhaseAnalysisConfig')

                image_analyzer = analyzer_info.image_analyzer_class(config=config)
                logging.info(f'config in the image analyzer set to {image_analyzer.config}')
            else:
                # Fallback: pass it as-is.
                image_analyzer = analyzer_info.image_analyzer_class(config)
        else:
            image_analyzer = analyzer_info.image_analyzer_class()
    logging.info(f'the image analyzer config is {image_analyzer.config}')
    analyzer_instance = analyzer_info.analyzer_class(
        scan_tag=tag,
        device_name=analyzer_info.device_name,
        skip_plt_show=True,
        image_analyzer=image_analyzer
    )

    if hasattr(analyzer_instance, "file_pattern"):
        default_pattern = "*_{shot_num:03d}.png"
        analyzer_instance.file_pattern = analyzer_info.file_pattern or default_pattern

    return analyzer_instance

def analyze_scan(tag: ScanTag, analyzer_list: list[AnalyzerInfo], upload_to_scanlog: bool = True,
                 documentID: Optional[str] = None, debug_mode: bool = False):
    """
    Performs all given analysis routines on a given scan. Optionally uploads results to google doc scanlog.

    :param tag: Tag representing the scan date, number, and experiment
    :param analyzer_list: List of valid analyzers that can be run on the given scan
    :param upload_to_scanlog: If True, will upload index of files to Google scanlog
    :param documentID: If given, the Google doc ID of the scanlog. Otherwise, will default to today's.
    :param debug_mode: If True, will not attempt analysis.
    :return:
    """
    all_display_files = []

    for analyzer_info in analyzer_list:
        device = analyzer_info.device_name if analyzer_info.device_name else ''
        print(tag, ":", analyzer_info.analyzer_class.__name__, device)
        if not debug_mode:
            try:
                # Use the helper to instantiate the analyzer (with image analyzer and file pattern settings)
                logging.info(f'attempting to instantiate image the scan analyzer with an ImageAnalyzer config: {analyzer_info}')
                analyzer = instantiate_analyzer(tag, analyzer_info)
                index_of_files = analyzer.run_analysis(config_options=analyzer_info.config_file)
                print(f'index of files: {index_of_files}')
                # If analysis produces files, add them to the list.
                if index_of_files:
                    all_display_files.append(index_of_files)
            except Exception as err:
                logging.error(f"Error in analyze_scan {tag.month}/{tag.day}/{tag.year}:Scan{tag.number:03d}): {err}")

    if loaded_docgen and upload_to_scanlog and len(all_display_files) > 0:
        flattened_file_paths = list(itertools.chain.from_iterable(all_display_files))
        print(f'flatten file list: {flattened_file_paths}')
        insert_display_content_to_doc(tag, flattened_file_paths, documentID=documentID)


# def analyze_scan(tag: ScanTag, analyzer_list: list[AnalyzerInfo], upload_to_scanlog: bool = True,
#                  documentID: Optional[str] = None, debug_mode: bool = False):
#     """
#     Performs all given analysis routines on a given scan.  Optionally uploads results to google doc scanlog
#
#     :param tag: Tag representing the scan date, number, and experiment
#     :param analyzer_list: List of valid analyzers that can be run on the given scan
#     :param upload_to_scanlog: If True, will upload index of files to Google scanlog
#     :param documentID: If given, the Google doc ID of the scanlog.  Otherwise, will default to today's
#     :param debug_mode: If True, will not attempt analysis.
#     :return:
#     """
#     all_display_files = []
#
#     for analyzer_info in analyzer_list:
#         device = analyzer_info.device_name if analyzer_info.device_name else ''
#         print(tag, ":", analyzer_info.analyzer_class.__name__, device)
#         if not debug_mode:
#             try:
#                 analyzer_class = analyzer_info.analyzer_class
#                 # Instantiate the image analyzer if specified
#                 if hasattr(analyzer_info, 'image_analyzer_class') and analyzer_info.image_analyzer_class:
#                     image_analyzer = analyzer_info.image_analyzer_class()
#                     analyzer = analyzer_class(
#                         scan_tag=tag,
#                         device_name=analyzer_info.device_name,
#                         skip_plt_show=True,
#                         image_analyzer=image_analyzer
#                     )
#                 else:
#                     analyzer = analyzer_class(
#                         scan_tag=tag,
#                         device_name=analyzer_info.device_name,
#                         skip_plt_show=True
#                     )
#                 index_of_files = analyzer.run_analysis(config_options=analyzer_info.config_file)
#                 print(f'index of files: {index_of_files}')
#                 # if index_of_files:  # TODO And if a Google doc procedure is defined for the given experiment
#                 #     # TODO Append the images to the appropriate location in the daily experiment log on Google
#                 #     pass
#                 if index_of_files:
#                     all_display_files.append(index_of_files)
#             except Exception as err:
#                 logging.error(f"Error in analyze_scan {tag.month}/{tag.day}/{tag.year}:Scan{tag.number:03d}): {err}")
#
#     if loaded_docgen and upload_to_scanlog and len(all_display_files) > 0:
#         flattened_file_paths = list(itertools.chain.from_iterable(all_display_files))
#         print(f'flatten file list: {flattened_file_paths}')
#         insert_display_content_to_doc(tag, flattened_file_paths, documentID=documentID)


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

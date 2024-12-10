"""
Module containing the mapping of specific analyzers to their respective classes.  Gives an analysis command to the
specified analyzer with the scan folder location
"""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scan_analysis.base import AnalyzerInfo
    from geecs_python_api.controls.api_defs import ScanTag

import logging
from pathlib import Path
import yaml
import itertools
from logmaker_4_googledocs import docgen

def analyze_scan(tag: 'ScanTag', analyzer_list: list['AnalyzerInfo'], debug_mode: bool = False):
    all_dispay_files = []
    for analyzer_info in analyzer_list:
        device = analyzer_info.device_name if analyzer_info.device_name else ''
        print(tag, ":", analyzer_info.analyzer_class.__name__, device)
        if not debug_mode:
            try:
                analyzer_class = analyzer_info.analyzer_class
                analyzer = analyzer_class(scan_tag=tag, device_name=analyzer_info.device_name, skip_plt_show=True)
                index_of_files = analyzer.run_analysis(config_options=analyzer_info.config_file)
                print(f'index of files: {index_of_files}')
                # if index_of_files:  # TODO And if a Google doc procedure is defined for the given experiment
                #     # TODO Append the images to the appropriate location in the daily experiment log on Google
                #     pass
                if index_of_files:
                    all_dispay_files.append(index_of_files)
            except Exception as err:
                logging.error(f"Error in analyze_scan {tag.month}/{tag.day}/{tag.year}:Scan{tag.number:03d}): {err}")
                
    if len(all_dispay_files)>0:
        flattened_file_paths = list(itertools.chain.from_iterable(all_dispay_files))
        print(f'flatten file list: {flattened_file_paths}')
        insert_display_content_to_doc(tag, flattened_file_paths)

def insert_display_content_to_doc(scan_tag, path_list, experiment='Undulator'):
    """
    Inserts display content from a list of paths into a Google Doc for a given scan.

    Args:
        scan_tag: The scan tag containing metadata (e.g., year, month, day, number).
        path_list (list[str]): A list of file paths to insert into the Google Doc.
        experiment (str): The experiment name used to map the correct configuration file.

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
                image_path=image_path,  # Pass as string
                experiment=experiment
            )

        print(f"Successfully inserted display content for scan {scan_tag.number}.")

    except Exception as display_err:
        logging.error(f"Error processing display content for scan {scan_tag.number}: {display_err}")

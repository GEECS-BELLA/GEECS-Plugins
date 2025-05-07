# -*- coding: utf-8 -*-
"""
Created on Thu May  1 14:40:25 2025

@author: loasis
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import numbers
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalysis

class Array2DHTTC14(Array2DScanAnalysis):
    
    def _process_all_shots_parallel(self) -> None:
        """
        Run the image analyzer on every shot in parallel and update self.data.
        This method uses both a ProcessPoolExecutor and a ThreadPoolExecutor
        (depending on whether the analyzer should run asynchronously) and mirrors
        the error handling and logging from your sample code.
        """
        # Clear existing data
        self.data = {'shot_num': [], 'images': []}

        # Define success and error handlers:
        def process_success(shot_number: int, analysis_result: dict) -> None:
            """
            On successful analysis of a shot, update self.data and auxiliary_data.
            If no processed image is returned but analysis results are available,
            update the auxiliary data and log the outcome, but do not add to self.data.
            """
            image = analysis_result.get("processed_image_uint16")
            analysis = analysis_result.get("analyzer_return_dictionary", {})

            # Always update auxiliary data if analysis exists.
            if analysis:
                for key, value in analysis.items():
                    if not isinstance(value, numbers.Number):
                        logging.warning(
                            f"[{self.__class__.__name__} using {self.image_analyzer.__class__.__name__}] "
                            f"Analysis result for shot {shot_number} key '{key}' is not numeric (got {type(value).__name__}). Skipping."
                        )
                    else:
                        self.auxiliary_data.loc[self.auxiliary_data['Shotnumber'] == shot_number, key] = value
                logging.info(f"Finished processing analysis for shot {shot_number}.")
            else:
                logging.warning(f"No analysis results returned for shot {shot_number}.")

            # If a processed image is available, update self.data.
            if image is not None:
                # try:
                #     image = image.astype(np.uint16)
                # except Exception as e:
                #     logging.error(f"Error converting image for shot {shot_number} to uint16: {e}")
                #     image = None
                if image is not None:
                    self.data['shot_num'].append(shot_number)
                    self.data['images'].append(image)
            else:
                logging.info(f"Shot {shot_number} returned no processed image; only auxiliary data was updated.")

        def process_error(shot_number: int, exception: Exception) -> None:
            """
            Log an error if processing a shot fails.
            """
            logging.error(f"Error while analyzing shot {shot_number}: {exception}")

        # Gather tasks: each shot number paired with its file path.
        tasks = []
        for shot_num in self.auxiliary_data['Shotnumber'].values:
            pattern = self.file_pattern.format(shot_num=int(shot_num))
            file_path = next(self.path_dict['data_img'].glob(pattern), None)
            logging.info(f'file path found is {file_path}')
            if file_path is not None:
                tasks.append((shot_num, file_path))

        # Dictionary to map futures to shot numbers.
        image_analysis_futures = {}

        # Submit image analysis jobs.
        # Use ProcessPoolExecutor if the analyzer is CPU-bound,
        # or ThreadPoolExecutor if run_analyze_image_asynchronously is True.

        with ProcessPoolExecutor(max_workers=4) as process_pool, ThreadPoolExecutor(max_workers=4) as thread_pool:
            for shot_num, file_path in tasks:
                if self.image_analyzer.run_analyze_image_asynchronously:
                    # For asynchronous (likely I/O-bound) processing, use the thread pool.
                    future = thread_pool.submit(self.image_analyzer.analyze_image_file, image_filepath=file_path)
                else:
                    # For CPU-bound tasks, use the process pool.
                    # Pass the analyzer's class so each process can create its own instance.
                    future = process_pool.submit(self.process_shot_parallel, shot_num, file_path,
                                                 self.image_analyzer)
                image_analysis_futures[future] = shot_num

            # Process completed futures.
            for future in as_completed(image_analysis_futures):
                shot_num = image_analysis_futures[future]
                if (exception := future.exception()) is not None:
                    process_error(shot_num, exception)
                else:
                    result = future.result()
                    # If using the process pool, result is a tuple: (shot_num, image, analysis).
                    if isinstance(result, tuple):
                        shot_num, image, analysis = result
                        analysis_result = {"processed_image_uint16": image, "analyzer_return_dictionary": analysis}
                        
                        
                        """ MAKE PLOT HERE"""
                        
                        """  """
                        
                        save_filename = self.path_dict['save'] / f"{shot_num:03d}.png"
                        plt.savefig(save_filename)
                        
                    else:
                        analysis_result = result
                    process_success(shot_num, analysis_result)

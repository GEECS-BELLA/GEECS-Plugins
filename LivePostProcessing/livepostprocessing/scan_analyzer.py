from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed as futures_as_completed
from argparse import ArgumentParser
from pathlib import Path

from configparser import ConfigParser
from inspect import signature

from typing import TYPE_CHECKING, Any, Optional, Union, get_type_hints, Callable
from .types import Array2D, ImageSubject, DeviceName, MetricName, RunID, ScanNumber, ShotNumber
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from image_analysis.base import ImageAnalyzer
    ShotKey = tuple[RunID, ScanNumber, ShotNumber]
    DeviceMetricKey = tuple[DeviceName, MetricName]
    from pathlib import Path
    from concurrent.futures import Future
    from collections.abc import Generator

import numpy as np
from imageio.v3 import imwrite

from .orm import Scan, SFile
from image_analysis.analyzers.U_PhasicsFileCopy import U_PhasicsFileCopyImageAnalyzer
from image_analysis.analyzers.U_FROG_Grenouille import U_FROG_GrenouilleImageAnalyzer, U_FROG_GrenouilleAWSLambdaImageAnalyzer
from image_analysis.utils import ROI
from .utils import find_undulator_folder

import logging
import logging.config
import json
logging.config.dictConfig(
    json.load(
        (Path(__file__).parents[1] / "logging_config.json")
            .open()
    )
)
logger = logging.getLogger("scan_analyzer")

class ScanAnalyzer:
    image_analyzer_classes: dict[DeviceName, type[ImageAnalyzer]] = {
        DeviceName('U_PhasicsFileCopy'): U_PhasicsFileCopyImageAnalyzer,
        DeviceName('U_FROG_Grenouille'): U_FROG_GrenouilleAWSLambdaImageAnalyzer,
    }

    def __init__(self, 
                 image_analyzer_kwargs: Optional[dict[DeviceName, dict[str, Any]]] = None,
                 num_processes: int = 5,
                 num_threads: int = 100,
                 experiment_data_folder: Optional[Union[Path, str]] = None,
                ):
        """ 
        Parameters
        ----------
        image_analyzer_kwargs : Optional[dict[DeviceName, dict[str, Any]]]
            keyword arguments for the various image analysers, in the form
                {'device1': {'kwarg1': 1.234, 'kwarg2': True}}
        num_processes : int
            number of processes running ImageAnalyzer.analyze_image() calls with 
            run_async = False, for local evaluation.
        num_async_workers : int
            number of threads running ImageAnalyzer.analyze_image() calls with 
            run_async = True, for external process evaluation.
        experiment_data_folder : Path
            A folder containing experiment data, such as Z:/data/Undulator        

        """
        if image_analyzer_kwargs is None:
            image_analyzer_kwargs = {}

        # instantiate imageanalyzers for each device.
        self.image_analyzers: dict[DeviceName, ImageAnalyzer] = {
            device_name: image_analyzer_class(**image_analyzer_kwargs.get(device_name, {})) 
            for device_name, image_analyzer_class in self.image_analyzer_classes.items()
        }

        # allow GUI to enable/disable certain image analyzers
        self.enable_image_analyzer: dict[DeviceName, bool] = {
            device_name: True
            for device_name in self.image_analyzers
        }

        self.num_processes = num_processes
        self.num_threads = num_threads

        # experiment folder
        # TODO: make more general than Undulator experiment
        if experiment_data_folder is None:
            self.experiment_data_folder = find_undulator_folder()
        else:
            self.experiment_data_folder = Path(experiment_data_folder)


    def analyze_scan(self, run_id: str, scan_number: int) -> None:
        self.scan = Scan(run_id, scan_number, experiment_data_folder=self.experiment_data_folder)

        # save config to this Scan's analysis folder
        self.scan.analysis_path.mkdir(parents=True, exist_ok=True)
        self.save_image_analyzer_config(self.scan.analysis_path / "image_analyzer_config.ini")

        # functions to process analysis results
        s_file = SFile(self.scan)

        def process_success(shot_number: ShotNumber, device_name: DeviceName, analysis_result: dict[MetricName, Union[float, np.ndarray]]):
            """ Save analysis_results to s-file and disk
            """
            try:
                for metric_name, metric_value in analysis_result.items():
                    def make_filename(ext: str):
                        filefolder = self.scan.analysis_path / f"{device_name}-{metric_name}"
                        filefolder.mkdir(parents=True, exist_ok=True)
                        return filefolder / f"Scan{self.scan.number:03d}_{device_name}-{metric_name}_{shot_number:03d}.{ext}"

                    # save scalars to s_file
                    if np.ndim(metric_value) == 0:
                        s_file.scalar_data.loc[(run_id, scan_number, shot_number), (device_name, metric_name)] = metric_value

                    # save 1d arrays to text files
                    elif np.ndim(metric_value) == 1:
                        np.savetxt(make_filename('dat'), metric_value, header=metric_name)

                    # save 2d arrays as tif images
                    elif np.ndim(metric_value) == 2:
                        imwrite(make_filename('tif'), metric_value)

                logger.info(f"Finished processing analysis result {run_id}/Scan{scan_number:03d}/Shot{shot_number:03d}/{device_name}")

            except Exception as exception:
                logger.error(f"Error while processing analysis result {run_id}/Scan{scan_number:03d}/Shot{shot_number:03d}/{device_name}: {exception}")

        def process_error(shot_number: ShotNumber, device_name: DeviceName, exception: Exception):
            logger.error(f"Error while analyzing {run_id}/Scan{scan_number:03d}/Shot{shot_number:03d}/{device_name}: {exception}")

        def iterate_image_objects() -> Generator[tuple[ShotNumber, DeviceName], None, None]:
            """ Iterate over Shot Image objects that need to be processed
            """
            for shot_number, shot in self.scan.shots.items():
                # analyze devices that have an image for this shot and have an image_analyzer.
                for device_name in (shot.images.keys() & self.image_analyzers.keys()):
                    if not self.enable_image_analyzer[device_name]:
                        continue

                    yield shot_number, device_name

        num_images_to_process = sum(1 for _ in iterate_image_objects())

        try:

            num_images_processed = 0

            with ProcessPoolExecutor(self.num_processes) as process_pool, ThreadPoolExecutor(self.num_threads) as thread_pool:

                image_analysis_result_futures: dict[Future[dict[str, Union[float, NDArray]]], tuple[ShotNumber, DeviceName]] = {}

                # submit image analysis jobs
                for shot_number, device_name in iterate_image_objects():
                    image = Array2D(self.scan.shots[shot_number].images[device_name][ImageSubject('raw')].load())
                    if self.image_analyzers[device_name].run_analyze_image_asynchronously:
                        analysis_result_future = thread_pool.submit(self.image_analyzers[device_name].analyze_image, image)
                    else:
                        analysis_result_future = process_pool.submit(self.image_analyzers[device_name].analyze_image, image)
                    image_analysis_result_futures[analysis_result_future] = (shot_number, device_name)

                # process finished analysis
                for analysis_result_future in futures_as_completed(image_analysis_result_futures):
                    shot_number, device_name = image_analysis_result_futures[analysis_result_future]

                    if exception := analysis_result_future.exception() is not None:
                        process_error(shot_number, device_name, exception)
                    else:
                        process_success(shot_number, device_name, analysis_result_future.result())

                    num_images_processed += 1

        finally:
            s_file.save_s_file()


    def save_scan_metrics(self):
        """ Saves floats to s_file and arrays to analysis folder.
        """
        
        s_file = SFile(self.scan)

        for shot_key, analysis in self.scan_metrics.items():
            run_id, scan_number, shot_number = shot_key
            assert ((run_id == self.scan.run_id) and (scan_number == self.scan.number)), "self.scan_metrics run_id/scan_number don't match self.scan"

            for (device_name, metric_name), metric_value in analysis.items():

                def make_filename(ext: str):
                    filefolder = (self.scan.path.parent.parent/'analysis'/f"Scan{self.scan.number:03d}"/
                                  f"{device_name}-{metric_name}"
                                 )
                    filefolder.mkdir(parents=True, exist_ok=True)
                    return filefolder/f"Scan{self.scan.number:03d}_{device_name}-{metric_name}_{shot_number:03d}.{ext}"

                # save scalars to s_file
                if np.ndim(metric_value) == 0:
                    s_file.scalar_data.loc[shot_key, (device_name, metric_name)] = metric_value

                # save 1d arrays to text files
                elif np.ndim(metric_value) == 1:
                    np.savetxt(make_filename('dat'), metric_value, header=metric_name)
                
                # save 2d arrays as tif images
                elif np.ndim(metric_value) == 2:
                    imwrite(make_filename('tif'), metric_value)

        s_file.save_s_file()


    def upload_scan_metrics(self):
        """ Upload contents of s-file to AWS DynamoDB
        """
        s_file = SFile(self.scan)
        s_file.upload_data_to_cloud()

    def save_image_analyzer_config(self, config_filename: str|Path):
        """ Save image analyzer parameters to a .ini style file
        
        """
        config_parser = ConfigParser()
        # set option-rename to identity function instead of (the default) str.lower in order to preserve case
        config_parser.optionxform = str

        # how to convert parameters of different types to a string in the config file. 
        # default is str
        object_to_config_string_functions: dict[type, Callable] = {
            ROI: lambda roi: f"{roi.top}, {roi.bottom}, {roi.left}, {roi.right}"
        }

        for device, image_analyzer in self.image_analyzers.items():
            # each device gets its own section
            config_parser.add_section(device)
            # start the section with an enable option, which corresponds to having checkbox marked in the GUI
            config_parser[device]['enable'] = str(self.enable_image_analyzer[device])
            # add values of the image_analyzer's parameters
            for parameter_name in signature(image_analyzer.__init__).parameters:
                if parameter_name == 'self':
                    continue
                parameter_value = getattr(image_analyzer, parameter_name)
                parameter_value_str = object_to_config_string_functions.get(type(parameter_value), str)(parameter_value)
                config_parser[device][parameter_name] = parameter_value_str

        with open(config_filename, 'w') as f:
            config_parser.write(f)

    def load_image_analyzer_config(self, config_filename: Union[str, Path]):
        config_parser = ConfigParser()
        # set option-rename to identity function instead of (the default) str.lower in order to preserve case
        config_parser.optionxform = str

        # how to convert parameters of different types to a string in the config file. 
        # default is to call the parameter's constructor
        config_string_to_object_functions: dict[type, Callable] = {
            ROI: lambda config_string: ROI(*[(int(config_string_part.strip()) if config_string_part.strip().lower() != 'none' else None) 
                                             for config_string_part in config_string.split(',')
                                            ]
                                          ),
        }

        with open(config_filename, 'r') as f:
            config_parser.read_file(f)

        for device_name in config_parser.sections():
            image_analyzer_parameter_types = get_type_hints(self.image_analyzers[device_name].__init__)
            for parameter_name, parameter_value_str in config_parser[device_name].items():
                # special option in each device section which relates to having checkbox marked in GUI
                if parameter_name == 'enable': 
                    self.enable_image_analyzer[device_name] = {'true': True,
                                                               'false': False,
                                                              }[parameter_value_str.lower()]
                # otherwise set image_analyzer's parameter to value in config file
                else:
                    parameter_value = config_string_to_object_functions.get(image_analyzer_parameter_types[parameter_name], image_analyzer_parameter_types[parameter_name])(parameter_value_str)
                    setattr(self.image_analyzers[device_name], parameter_name, parameter_value)

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("run_id",
                    type=str,
                    help="date folder in yy_mmdd format", 
                   )
    ap.add_argument("scan",
                    type=int,
                    help="scan number"
                   ) 

    # ap.add_argument("--no-save", 
    #                 dest='no_save',
    #                 action='store_true',
    #                 help="Do not save to s-file after analysis",
    #                )
    
    ap.add_argument("--no-upload", 
                    dest='no_upload',
                    action='store_true',
                    help="Do not upload to AWS after analysis",
                   )
    
    args = ap.parse_args()

    sa = ScanAnalyzer()
    sa.analyze_scan(args.run_id, args.scan)
    # if not args.no_save:
    #     sa.save_scan_metrics()
    if not args.no_upload:
        sa.upload_scan_metrics()

from __future__ import annotations

from multiprocessing import Pool
from argparse import ArgumentParser

from configparser import ConfigParser
from inspect import signature

from typing import TYPE_CHECKING, Any, Optional, Union, get_type_hints, Callable
from .types import Array2D, ImageSubject, DeviceName, MetricName, RunID, ScanNumber, ShotNumber
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from image_analysis.base import ImageAnalyzer
    ShotKey = tuple[RunID, ScanNumber, ShotNumber]
    DeviceMetricKey = tuple[DeviceName, MetricName]
    AnalysisDict = dict[DeviceMetricKey, Union[float, NDArray]]
    from pathlib import Path

import numpy as np
from imageio.v3 import imwrite

from .orm import Scan, SFile
from image_analysis.analyzers.U_PhasicsFileCopy import U_PhasicsFileCopyImageAnalyzer
from image_analysis.utils import ROI

class ScanAnalyzer:
    image_analyzer_classes: dict[DeviceName, type[ImageAnalyzer]] = {
        DeviceName('U_PhasicsFileCopy'): U_PhasicsFileCopyImageAnalyzer,
    }

    def __init__(self, image_analyzer_kwargs: Optional[dict[DeviceName, dict[str, Any]]] = None):
        """ 
        Parameters
        ----------
        image_analyzer_kwargs : Optional[dict[DeviceName, dict[str, Any]]]
            keyword arguments for the various image analysers, in the form
                {'device1': {'kwarg1': 1.234, 'kwarg2': True}}
        
        """
        if image_analyzer_kwargs is None:
            image_analyzer_kwargs = {}

        # instantiate imageanalyzers for each device.
        self.image_analyzers: dict[DeviceName, ImageAnalyzer] = {
            device_name: image_analyzer_class(**image_analyzer_kwargs.get(device_name, {})) 
            for device_name, image_analyzer_class in self.image_analyzer_classes.items()
        }

    def analyze_scan(self, run_id: str, scan_number: int):
        self.scan = Scan(run_id, scan_number)
        self.scan_metrics: dict[ShotKey, AnalysisDict] = {}

        pool = Pool(6)

        for shot_number, shot in self.scan.shots.items():
            # analyze devices that have an image for this shot and have an image_analyzer.
            shot_key = (RunID(run_id), ScanNumber(scan_number), ShotNumber(shot_number))
            self.scan_metrics[shot_key] = {}
            for device_name in (shot.images.keys() & self.image_analyzers.keys()):
                if not self.image_analyzers[device_name].enable:
                    continue

                try:
                    analysis: dict[str, float|NDArray] = self.image_analyzers[device_name].analyze_image(
                        Array2D(shot.images[device_name][ImageSubject('raw')].load())
                    )
                except Exception as err:
                    print(f"Error while analyzing {run_id}/Scan{scan_number:03d}/Shot{shot_number:03d}/{device_name}: {err}")
                    analysis = {}

                self.scan_metrics[shot_key].update({(device_name, metric_name): metric_value 
                                                    for metric_name, metric_value in analysis.items()
                                                  })

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
            config_parser.add_section(device)
            for parameter_name in signature(image_analyzer.__init__).parameters:
                if parameter_name == 'self':
                    continue
                parameter_value = getattr(image_analyzer, parameter_name)
                parameter_value_str = object_to_config_string_functions.get(type(parameter_value), str)(parameter_value)
                config_parser[device][parameter_name] = parameter_value_str

        with open(config_filename, 'w') as f:
            config_parser.write(f)

    def load_image_analyzer_config(self, config_filename: str|Path):
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

    ap.add_argument("--no-save", 
                    dest='no_save',
                    action='store_true',
                    help="Do not save to s-file after analysis",
                   )
    
    ap.add_argument("--no-upload", 
                    dest='no_upload',
                    action='store_true',
                    help="Do not upload to AWS after analysis",
                   )
    
    args = ap.parse_args()

    sa = ScanAnalyzer()
    sa.analyze_scan(args.run_id, args.scan)
    if not args.no_save:
        sa.save_scan_metrics()
    if not args.no_upload:
        sa.upload_scan_metrics()


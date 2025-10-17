"""
ALine3 electron beam size evaluator for optimization.

This module provides an evaluator for optimizing electron beam size on the ALine3
diagnostic camera (UC_ALineEBeam3). The evaluator computes beam size metrics using
full-width half-maximum (FWHM) measurements in both x and y directions, combining
them into a single objective function suitable for automated optimization.

The evaluator integrates with the GEECS scan analysis framework to process beam
profile images and extract quantitative beam size measurements with proper spatial
calibration.

Classes
-------
ALine3SizeEval
    Evaluator for ALine3 electron beam size optimization.

Notes
-----
This evaluator requires the UC_ALineEBeam3 camera device and uses a spatial
calibration of 24.4 μm/pixel for converting pixel measurements to physical units.
The objective function combines x and y FWHM measurements as a sum of squares.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from geecs_scanner.data_acquisition.scan_data_manager import ScanDataManager
    from geecs_scanner.data_acquisition.data_logger import DataLogger


from geecs_scanner.optimization.base_evaluator import BaseEvaluator

from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer
from image_analysis.utils import read_imaq_image
from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer


class ALine3SizeEval(BaseEvaluator):
    """
    Evaluator for ALine3 electron beam size optimization.

    This evaluator computes electron beam size metrics from the UC_ALineEBeam3
    diagnostic camera using FWHM measurements. It integrates with the GEECS scan
    analysis framework to process beam profile images and extract quantitative
    measurements suitable for automated optimization.

    The evaluator uses an Array2DScanAnalyzer with an EBeamProfileAnalyzer to
    process images and compute beam size statistics. The objective function
    combines x and y FWHM measurements with spatial calibration to produce
    a single metric for optimization.

    Parameters
    ----------
    device_requirements : dict, optional
        Dictionary specifying required devices and variables. If None,
        defaults will be used for UC_ALineEBeam3 camera.
    scan_data_manager : ScanDataManager, optional
        Manager for accessing scan data and file paths.
    data_logger : DataLogger, optional
        Logger instance for accessing shot data and bin information.

    Attributes
    ----------
    dev_name : str
        Name of the camera device ('UC_ALineEBeam3').
    scan_analyzer : Array2DScanAnalyzer
        Analyzer instance for processing beam profile images.
    output_key : str
        Key name for the optimization objective ('f').
    objective_tag : str
        Tag for logging objective values.

    Methods
    -------
    evaluate_all_shots(shot_entries)
        Evaluate beam size for all shots in a data bin.
    objective_fn(x, y)
        Static method computing objective function from FWHM values.

    Notes
    -----
    - Requires UC_ALineEBeam3 camera device to be available
    - Uses spatial calibration of 24.4 μm/pixel
    - Objective function is sum of squares: (x_fwhm * cal)² + (y_fwhm * cal)²
    - Configured for live analysis during optimization scans
    """

    def __init__(
        self,
        device_requirements=None,
        scan_data_manager: Optional[ScanDataManager] = None,
        data_logger: Optional[DataLogger] = None,
    ):
        required_keys = {}

        super().__init__(
            device_requirements=device_requirements,
            required_keys=required_keys,
            scan_data_manager=scan_data_manager,
            data_logger=data_logger,
        )

        self.dev_name = "UC_ALineEBeam3"
        self.scan_analyzer = Array2DScanAnalyzer(
            device_name=self.dev_name,
            image_analyzer=BeamAnalyzer(camera_config_name=self.dev_name),
        )

        # use live_analysis option for the scan_analyzer so that it knows not to try to load
        # data from an sFile already written to disk (which doesn't happen until the end of scan)
        self.scan_analyzer.live_analysis = True
        self.scan_analyzer.use_colon_scan_param = False  # default is file-style

        self.output_key = (
            "f"  # string name of optimization function defined in config, don't change
        )
        self.objective_tag: str = (
            "PlaceHolder"  # string to append to logged objective value
        )

    def evaluate_all_shots(self, shot_entries: list[dict]) -> float:
        """
        Evaluate beam size objective function for all shots in current bin.

        Processes all shots in the current data bin to compute an aggregate
        beam size metric. Uses the scan analyzer to create an averaged image
        from all shots, then analyzes the averaged image to extract FWHM
        measurements and compute the objective function.

        Parameters
        ----------
        shot_entries : list of dict
            List of shot entry dictionaries as produced by _gather_shot_entries.
            Each dictionary contains shot number, scalars, and image paths.

        Returns
        -------
        float
            Computed objective value combining x and y FWHM measurements
            with spatial calibration.
        """
        # set the 'aux' data manually to isolate the current bin to get analyzed by the ScanAnalyzer
        self.scan_analyzer.auxiliary_data = self.current_data_bin
        self.scan_analyzer.run_analysis(scan_tag=self.scan_tag)

        # grab the path to the saved average image from the scan analyzer and load
        avg_image_path = self.scan_analyzer.saved_avg_image_paths[self.bin_number]
        avg_image = read_imaq_image(avg_image_path)

        # run standalone analyis using the image_analyzer, passing the argument that preprocessing
        # has already been done, e.g. ROI, background etc.
        result = self.scan_analyzer.image_analyzer.analyze_image(
            avg_image, auxiliary_data={"preprocessed": True}
        )

        # extract the scalar results returned by the image analyzer
        scalar_results = result["analyzer_return_dictionary"]

        # define keys to extract values to use for the objective function
        x_key = f"{self.dev_name}_x_fwhm"
        y_key = f"{self.dev_name}_y_fwhm"

        objective_value = self.objective_fn(
            x=scalar_results[x_key], y=scalar_results[y_key]
        )

        for shot_entry in shot_entries:
            self.log_objective_result(
                shot_num=shot_entry["shot_number"], scalar_value=objective_value
            )

        return objective_value

    @staticmethod
    def objective_fn(x, y):
        """Compute beam size objective function from FWHM measurements."""
        calibration = 24.4e-3  # spatial calibration in um/pixel
        return (x * calibration) ** 2 + (y * calibration) ** 2

    def _get_value(self, input_data: Dict) -> Dict:
        """
        Evaluate beam size objective for current optimization step.

        This is the main evaluation method called by the optimization framework.
        It gathers shot data from the current bin, processes the beam profile
        images, and returns the computed objective function value.

        Parameters
        ----------
        input_data : dict
            Dictionary of input parameter values for the current optimization
            step. Not directly used by this evaluator as it operates on
            acquired image data.

        Returns
        -------
        dict
            Dictionary containing the objective function result with key 'f'.

        {'f': 0.00234}

        Notes
        -----
        This method:
        1. Gathers shot entries for the current data bin
        2. Processes all shots to compute beam size metric
        3. Returns result in format expected by optimization framework

        The 'f' key in the returned dictionary corresponds to the objective
        function name defined in the optimization configuration.
        """
        shot_entries = self._gather_shot_entries(
            shot_numbers=self.current_shot_numbers,
            scalar_variables=self.required_keys,
            non_scalar_variables=["UC_ALineEBeam3"],
        )

        result = self.evaluate_all_shots(shot_entries)

        return {self.output_key: result}

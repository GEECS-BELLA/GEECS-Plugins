"""
High-resolution magnetic camera evaluator for beam optimization.

This module provides an evaluator for optimizing electron beam properties using
the high-resolution magnetic camera (UC_HiResMagCam). The evaluator computes
beam quality metrics based on total photon counts and emittance proxy measurements,
combining them into an objective function suitable for automated optimization.

The evaluator integrates with the GEECS scan analysis framework to process
magnetic camera images and extract quantitative beam measurements for optimization
of beam transport and focusing systems.

Classes
-------
HiResMagCam
    Evaluator for high-resolution magnetic camera beam optimization.

Examples
--------
Basic usage in optimization:

>>> evaluator = HiResMagCam(
...     device_requirements=device_reqs,
...     scan_data_manager=sdm,
...     data_logger=logger
... )
>>> result = evaluator.get_value(input_parameters)
>>> beam_quality_metric = result['f']

Notes
-----
This evaluator requires the UC_HiResMagCam device and uses total counts and
emittance proxy measurements to compute a beam quality objective function.
The objective maximizes the ratio of total counts to emittance proxy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from geecs_scanner.data_acquisition.scan_data_manager import ScanDataManager
    from geecs_scanner.data_acquisition.data_logger import DataLogger


from geecs_scanner.optimization.base_evaluator import BaseEvaluator

from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer
from image_analysis.utils import read_imaq_image
from image_analysis.offline_analyzers.Undulator.hi_res_mag_cam_analyzer import (
    HiResMagCamAnalyzer,
)


class HiResMagCam(BaseEvaluator):
    """
    Evaluator for high-resolution magnetic camera beam optimization.

    This evaluator computes beam quality metrics from the UC_HiResMagCam device
    using total photon counts and emittance proxy measurements. It integrates
    with the GEECS scan analysis framework to process magnetic camera images
    and extract quantitative beam measurements for optimization.

    The evaluator uses an Array2DScanAnalyzer with a HiResMagCamAnalyzer to
    process images and compute beam statistics. The objective function maximizes
    the ratio of total counts to emittance proxy, effectively optimizing for
    high beam intensity with low emittance.

    Parameters
    ----------
    device_requirements : dict, optional
        Dictionary specifying required devices and variables. If None,
        defaults will be used for UC_HiResMagCam camera.
    scan_data_manager : ScanDataManager, optional
        Manager for accessing scan data and file paths.
    data_logger : DataLogger, optional
        Logger instance for accessing shot data and bin information.

    Attributes
    ----------
    dev_name : str
        Name of the camera device ('UC_HiResMagCam').
    scan_analyzer : Array2DScanAnalyzer
        Analyzer instance for processing magnetic camera images.
    output_key : str
        Key name for the optimization objective ('f').
    objective_tag : str
        Tag for logging objective values.

    Methods
    -------
    evaluate_all_shots(shot_entries)
        Evaluate beam quality for all shots in a data bin.
    objective_fn(x, y)
        Static method computing objective function from counts and emittance.

    Examples
    --------
    >>> evaluator = HiResMagCam(
    ...     scan_data_manager=sdm,
    ...     data_logger=logger
    ... )
    >>> result = evaluator.get_value({"quad_strength": 85.2})
    >>> beam_quality_metric = result['f']

    Notes
    -----
    - Requires UC_HiResMagCam camera device to be available
    - Objective function is -total_counts/emittance_proxy (negative for minimization)
    - Higher total counts and lower emittance proxy lead to better objective values
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

        self.dev_name = "UC_HiResMagCam"
        config_dict = {"camera_name": self.dev_name}
        self.scan_analyzer = Array2DScanAnalyzer(
            device_name=self.dev_name, image_analyzer=HiResMagCamAnalyzer(**config_dict)
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
        Evaluate beam quality objective function for all shots in current bin.

        Processes all shots in the current data bin to compute an aggregate
        beam quality metric. Uses the scan analyzer to create an averaged image
        from all shots, then analyzes the averaged image to extract total counts
        and emittance proxy measurements for the objective function.

        Parameters
        ----------
        shot_entries : list of dict
            List of shot entry dictionaries as produced by _gather_shot_entries.
            Each dictionary contains shot number, scalars, and image paths.

        Returns
        -------
        float
            Computed objective value combining total counts and emittance proxy
            measurements: -total_counts/emittance_proxy.

        Notes
        -----
        This method:
        1. Sets auxiliary data for the scan analyzer
        2. Runs analysis to create averaged image
        3. Loads the averaged image from disk
        4. Analyzes the averaged image for beam properties
        5. Extracts total counts and emittance proxy measurements
        6. Computes objective function
        7. Logs results for each shot

        The objective function is: -total_counts/emittance_proxy
        The negative sign converts maximization to minimization problem.

        Examples
        --------
        >>> shot_entries = evaluator._gather_shot_entries(...)
        >>> objective_value = evaluator.evaluate_all_shots(shot_entries)
        >>> print(f"Beam quality metric: {objective_value:.3f}")
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
        x_key = f"{self.dev_name}:total_counts"
        y_key = f"{self.dev_name}:emittance_proxy"

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
        """
        Compute beam quality objective function from counts and emittance proxy.

        Combines total photon counts and emittance proxy measurements to produce
        a single objective metric for optimization. The function computes the
        negative ratio of total counts to emittance proxy, effectively maximizing
        beam intensity while minimizing emittance.

        Parameters
        ----------
        x : float
            Total photon counts from the magnetic camera image.
        y : float
            Emittance proxy measurement (beam size metric).

        Returns
        -------
        float
            Objective function value: -total_counts/emittance_proxy.
            Negative sign converts maximization to minimization problem.

        Examples
        --------
        >>> total_counts = 1.5e6
        >>> emittance_proxy = 2.3
        >>> objective = HiResMagCam.objective_fn(total_counts, emittance_proxy)
        >>> print(f"Beam quality metric: {objective:.3f}")

        Notes
        -----
        The negative sign is used because optimization algorithms typically
        minimize objective functions. Higher total counts (more beam intensity)
        and lower emittance proxy (better beam quality) both contribute to
        a more negative (better) objective value.
        """

        return -x/y/20000000


    def _get_value(self, input_data: Dict) -> Dict:
        """
        Evaluate beam quality objective for current optimization step.

        This is the main evaluation method called by the optimization framework.
        It gathers shot data from the current bin, processes the magnetic camera
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

        Examples
        --------
        >>> input_params = {"quad_strength": 85.2, "focus_position": 1.5}
        >>> result = evaluator._get_value(input_params)
        >>> print(result)
        {'f': -652173.9}

        Notes
        -----
        This method:
        1. Gathers shot entries for the current data bin
        2. Processes all shots to compute beam quality metric
        3. Returns result in format expected by optimization framework

        The 'f' key in the returned dictionary corresponds to the objective
        function name defined in the optimization configuration.

        Note: There appears to be an inconsistency in the non_scalar_variables
        specification - it should be 'UC_HiResMagCam' instead of 'UC_ALineEBeam3'.
        """
        shot_entries = self._gather_shot_entries(
            shot_numbers=self.current_shot_numbers,
            scalar_variables=self.required_keys,
            non_scalar_variables=["UC_ALineEBeam3"],
        )

        result = self.evaluate_all_shots(shot_entries)

        return {self.output_key: result}

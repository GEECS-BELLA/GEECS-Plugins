"""
Generic evaluator using SingleDeviceScanAnalyzer(s) for optimization.

This module provides the MultiDeviceScanEvaluator base class, which handles
orchestration of one or more SingleDeviceScanAnalyzers (Array1D or Array2D)
for optimization tasks. It eliminates boilerplate code by managing analyzer
instantiation, data collection, and result extraction automatically.

The evaluator leverages the 'per_bin' analysis mode of SingleDeviceScanAnalyzer
to automatically handle image averaging and scalar result computation, allowing
subclasses to focus solely on defining their objective function logic.

Classes
-------
MultiDeviceScanEvaluator
    Base evaluator using SingleDeviceScanAnalyzer(s) for optimization.
"""

from __future__ import annotations

import importlib
import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from geecs_scanner.data_acquisition.data_logger import DataLogger
    from geecs_scanner.data_acquisition.scan_data_manager import ScanDataManager

from geecs_scanner.optimization.base_evaluator import BaseEvaluator
from geecs_scanner.optimization.config_models import (
    MultiDeviceScanEvaluatorConfig,
    SingleDeviceScanAnalyzerConfig,
)
from geecs_data_utils import ScanPaths
from image_analysis.config_loader import set_config_base_dir

set_config_base_dir(ScanPaths.paths_config.image_analysis_configs_path)

logger = logging.getLogger(__name__)


class MultiDeviceScanEvaluator(BaseEvaluator):
    """
    Base evaluator using one or more SingleDeviceScanAnalyzers.

    This evaluator handles orchestration of SingleDeviceScanAnalyzers (Array1D
    or Array2D), automatically managing analyzer instantiation, data collection,
    and result extraction. It uses the 'per_bin' analysis mode to leverage
    built-in averaging functionality, eliminating the need for manual image
    averaging code.

    Subclasses need only implement the `compute_objective` method to define
    their specific objective function logic based on the scalar results
    extracted from the analyzers.

    Parameters
    ----------
    analyzers : list of dict
        List of SingleDeviceScanAnalyzerConfig dictionaries specifying the
        analyzers to instantiate and use for data collection.
    scan_data_manager : ScanDataManager, optional
        Manager for accessing scan data and file paths.
    data_logger : DataLogger, optional
        Logger instance for accessing shot data and bin information.
    **kwargs
        Additional keyword arguments passed to BaseEvaluator.

    Attributes
    ----------
    analyzer_configs : list of SingleDeviceScanAnalyzerConfig
        Validated analyzer configuration objects.
    scan_analyzers : dict
        Dictionary mapping device names to instantiated scan analyzer objects.
    output_key : str
        Key name for the optimization objective ('f').
    objective_tag : str
        Tag for logging objective values.

    Methods
    -------
    get_scalar(device_name, metric_name, scalar_results)
        Helper method to extract scalar values with flexible key naming.
    compute_objective(scalar_results, bin_number)
        Abstract method for subclasses to implement objective function logic.

    Notes
    -----
    - Uses 'per_bin' analysis mode by default for automatic averaging
    - Automatically generates device_requirements from analyzer configs
    - Provides flexible scalar key lookup via get_scalar() helper
    - All analyzer results are combined into a single scalar_results dict
    """

    def __init__(
        self,
        analyzers: List[dict],
        scan_data_manager: Optional[ScanDataManager] = None,
        data_logger: Optional[DataLogger] = None,
        **kwargs,
    ):
        """Initialize the evaluator with analyzer configurations."""
        # Parse and validate analyzer configs
        self.analyzer_configs = [
            SingleDeviceScanAnalyzerConfig.model_validate(cfg) for cfg in analyzers
        ]

        # Auto-generate device_requirements
        evaluator_config = MultiDeviceScanEvaluatorConfig(
            analyzers=self.analyzer_configs
        )
        device_requirements = evaluator_config.generate_device_requirements()
        logger.info("Device requirements parsed from config: %s", device_requirements)
        # Initialize base evaluator
        super().__init__(
            device_requirements=device_requirements,
            required_keys={},
            scan_data_manager=scan_data_manager,
            data_logger=data_logger,
        )

        # Instantiate scan analyzers
        self.scan_analyzers = {}
        for config in self.analyzer_configs:
            analyzer = self._create_scan_analyzer(config)
            self.scan_analyzers[config.device_name] = analyzer

        # Set output configuration
        self.output_key = "f"  # Standard optimization objective key
        self.objective_tag: str = "MultiDeviceScan"

        logger.info(
            f"Initialized {self.__class__.__name__} with {len(self.scan_analyzers)} analyzer(s)"
        )

    def _create_scan_analyzer(self, config: SingleDeviceScanAnalyzerConfig):
        """
        Dynamically instantiate a scan analyzer from configuration.

        Parameters
        ----------
        config : SingleDeviceScanAnalyzerConfig
            Configuration specifying the analyzer to create.

        Returns
        -------
        SingleDeviceScanAnalyzer
            Instantiated and configured scan analyzer (Array1D or Array2D).

        Raises
        ------
        ImportError
            If the specified analyzer or image analyzer module cannot be imported.
        AttributeError
            If the specified class cannot be found in the module.
        """
        # Import the appropriate scan analyzer class
        from scan_analysis.analyzers.common.array1d_scan_analysis import (
            Array1DScanAnalyzer,
        )
        from scan_analysis.analyzers.common.array2D_scan_analysis import (
            Array2DScanAnalyzer,
        )

        analyzer_class_map = {
            "Array1DScanAnalyzer": Array1DScanAnalyzer,
            "Array2DScanAnalyzer": Array2DScanAnalyzer,
        }
        analyzer_class = analyzer_class_map[config.analyzer_type]

        # Import and instantiate the image analyzer
        image_analyzer_module = importlib.import_module(config.image_analyzer.module)
        image_analyzer_class = getattr(
            image_analyzer_module, config.image_analyzer.class_
        )
        image_analyzer = image_analyzer_class(**config.image_analyzer.kwargs)

        # Create scan analyzer with appropriate renderer
        # Note: We skip rendering during optimization by setting flag_save_data=False
        scan_analyzer = analyzer_class(
            device_name=config.device_name,
            image_analyzer=image_analyzer,
            file_tail=config.file_tail,
            analysis_mode=config.analysis_mode,
            # flag_save_data=False,  # Don't save outputs during optimization
        )

        # Configure for live analysis during optimization
        scan_analyzer.live_analysis = True
        scan_analyzer.use_colon_scan_param = False

        logger.info(
            f"Created {config.analyzer_type} for device '{config.device_name}' "
            f"with {image_analyzer.__class__.__name__}"
        )

        return scan_analyzer

    def get_scalar(
        self, device_name: str, metric_name: str, scalar_results: dict
    ) -> float:
        """
        Get scalar value with flexible key naming convention support.

        This helper method abstracts away the different key naming conventions
        used by various analyzers. It tries multiple formats to find the
        requested metric:
        1. device_name_metric (e.g., "UC_ALineEBeam3_x_fwhm")
        2. device_name:metric (e.g., "UC_HiResMagCam:total_counts")
        3. Just metric (if unique)

        Parameters
        ----------
        device_name : str
            Name of the device whose metric to retrieve.
        metric_name : str
            Name of the metric to retrieve (e.g., "x_fwhm", "total_counts").
        scalar_results : dict
            Dictionary of scalar results from analyzers.

        Returns
        -------
        float
            The requested scalar value.

        Raises
        ------
        KeyError
            If the metric cannot be found using any naming convention.
        """
        # Try underscore convention (e.g., UC_ALineEBeam3_x_fwhm)
        key1 = f"{device_name}_{metric_name}"
        if key1 in scalar_results:
            return scalar_results[key1]

        # Try colon convention (e.g., UC_HiResMagCam:total_counts)
        key2 = f"{device_name}:{metric_name}"
        if key2 in scalar_results:
            return scalar_results[key2]

        # Try just the metric name (if unique)
        if metric_name in scalar_results:
            return scalar_results[metric_name]

        # If none found, raise informative error
        raise KeyError(
            f"Could not find metric '{metric_name}' for device '{device_name}'. "
            f"Tried keys: {key1}, {key2}, {metric_name}. "
            f"Available keys: {list(scalar_results.keys())}"
        )

    def _get_value(self, input_data: Dict) -> Dict:
        """
        Main evaluation method called by the optimization framework.

        This method orchestrates the evaluation process:
        1. Updates current data from the data logger
        2. Runs all configured scan analyzers on the current bin
        3. Collects scalar results from all analyzers
        4. Calls compute_objective() to get the final objective value
        5. Logs the result for all shots in the bin

        Parameters
        ----------
        input_data : dict
            Dictionary of input parameter values for the current optimization
            step. Not directly used by this evaluator as it operates on
            acquired data.

        Returns
        -------
        dict
            Dictionary containing the objective function result with key 'f'
            plus any additional observables returned by ``compute_observables``.
        """
        # Update current data from logger
        self.get_current_data()

        # Run all analyzers and collect results
        all_scalar_results = {}

        for device_name, analyzer in self.scan_analyzers.items():
            logger.info(
                f"Running analyzer for device '{device_name}' on bin {self.bin_number}"
            )

            # Set auxiliary data for current bin
            analyzer.auxiliary_data = self.current_data_bin
            analyzer.run_analysis(scan_tag=self.scan_tag)

            # Extract scalar results from the bin
            # With per_bin mode, results are keyed by bin_number
            if self.bin_number in analyzer.results:
                result = analyzer.results[self.bin_number]
                scalars = result.get("analyzer_return_dictionary", {})
                all_scalar_results.update(scalars)
                logger.info(
                    f"Extracted {len(scalars)} scalar results from '{device_name}'"
                )
            else:
                logger.warning(
                    f"No results found for bin {self.bin_number} from '{device_name}'"
                )

        # Compute objective from all scalar results
        objective_value = self.compute_objective(
            scalar_results=all_scalar_results, bin_number=self.bin_number
        )

        logger.info(
            f"Computed objective value: {objective_value} for bin {self.bin_number}"
        )

        # Log for all shots in bin
        for shot_num in self.current_shot_numbers:
            self.log_objective_result(shot_num=shot_num, scalar_value=objective_value)

        outputs: Dict[str, float] = {self.output_key: objective_value}
        extra_observables = self.compute_observables(
            scalar_results=all_scalar_results, bin_number=self.bin_number
        )
        if extra_observables:
            # Avoid silent overwrites of the primary objective key
            if self.output_key in extra_observables:
                logger.warning(
                    "compute_observables returned key '%s'; overriding is not allowed",
                    self.output_key,
                )
                extra_observables = {
                    k: v for k, v in extra_observables.items() if k != self.output_key
                }
            outputs.update(extra_observables)

        return outputs

    @abstractmethod
    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """
        Compute objective function from analyzer scalar results.

        This abstract method must be implemented by subclasses to define
        their specific objective function logic. The method receives all
        scalar results from all configured analyzers and should combine
        them into a single objective value.

        Parameters
        ----------
        scalar_results : dict
            Combined analyzer_return_dictionary from all analyzers.
            Keys follow various naming conventions:
            - "device_name_metric" (e.g., "UC_ALineEBeam3_x_fwhm")
            - "device_name:metric" (e.g., "UC_HiResMagCam:total_counts")
            Use the get_scalar() helper method for flexible key lookup.
        bin_number : int
            Current bin number being evaluated. Can be used to access
            additional data from self.current_data_bin if needed.

        Returns
        -------
        float
            Objective function value to be minimized or maximized.

        Notes
        -----
        - Use self.get_scalar() for robust key lookup
        - Access self.current_data_bin for auxiliary scalar data if needed
        - Return a single float value
        - The sign convention depends on your VOCS (MINIMIZE vs MAXIMIZE)
        """
        pass

    def compute_observables(
        self, scalar_results: dict, bin_number: int
    ) -> Dict[str, float]:
        """
        Compute additional observables to include in the evaluator output.

        Subclasses can override this hook to return a dictionary of extra
        scalar quantities (e.g., beam centroid positions) that should be
        passed through to Xopt alongside the primary objective.

        Parameters
        ----------
        scalar_results : dict
            Combined analyzer_return_dictionary from all analyzers.
        bin_number : int
            Current bin number being evaluated.

        Returns
        -------
        dict
            Mapping of observable names to scalar values. The default
            implementation returns an empty dict.
        """
        return {}

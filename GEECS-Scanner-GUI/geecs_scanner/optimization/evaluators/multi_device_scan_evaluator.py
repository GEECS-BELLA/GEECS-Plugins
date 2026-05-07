"""
Generic evaluator using SingleDeviceScanAnalyzer(s) for optimization.

This module provides the MultiDeviceScanEvaluator base class, which handles
orchestration of one or more SingleDeviceScanAnalyzers (Array1D or Array2D)
for optimization tasks. It eliminates boilerplate code by managing analyzer
instantiation, data collection, and result extraction automatically.

The evaluator supports two analysis modes, configured per analyzer via
``analysis_mode`` in ``SingleDeviceScanAnalyzerConfig``:

- **per_bin** (default): one averaged result per bin →
  ``compute_objective(scalar_results, bin_number)``
- **per_shot**: one result per individual shot →
  ``compute_objective_from_shots(scalar_results_list, bin_number)``

``per_shot`` mode enables richer statistical treatment (median, std dev) and
allows returning ``f_noise`` to Xopt for improved GP surrogate modelling.

Classes
-------
MultiDeviceScanEvaluator
    Base evaluator using SingleDeviceScanAnalyzer(s) for optimization.
"""

from __future__ import annotations

import importlib
import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from geecs_scanner.data_acquisition.data_logger import DataLogger
    from geecs_scanner.data_acquisition.scan_data_manager import ScanDataManager
    from image_analysis.types import ImageAnalyzerResult

from geecs_scanner.optimization.base_evaluator import BaseEvaluator
from geecs_scanner.optimization.config_models import (
    MultiDeviceScanEvaluatorConfig,
    SingleDeviceScanAnalyzerConfig,
)
from geecs_data_utils import ScanPaths
from geecs_data_utils.config_roots import image_analysis_config

logger = logging.getLogger(__name__)


class MultiDeviceScanEvaluator(BaseEvaluator):
    """
    Base evaluator using one or more SingleDeviceScanAnalyzers.

    This evaluator handles orchestration of SingleDeviceScanAnalyzers (Array1D
    or Array2D), automatically managing analyzer instantiation, data collection,
    and result extraction.

    Two analysis modes are supported, selected per analyzer via ``analysis_mode``
    in ``SingleDeviceScanAnalyzerConfig``:

    - **per_bin** (default): images in a bin are averaged before analysis;
      the evaluator receives a single scalar dict and calls
      ``compute_objective(scalar_results, bin_number)``.
    - **per_shot**: each image is analyzed individually; the evaluator
      collects one scalar dict per shot and calls
      ``compute_objective_from_shots(scalar_results_list, bin_number)``.

    When analyzers have mixed modes, per_bin results are merged into every
    shot dict before ``compute_objective_from_shots`` is called.

    Subclasses must implement ``compute_objective``. Override
    ``compute_objective_from_shots`` to apply custom per-shot statistics
    (e.g. median, noise estimate for Xopt GP).

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
    compute_objective_from_shots(scalar_results_list, bin_number)
        Hook for per-shot statistical treatment; default delegates to
        ``compute_objective`` via mean aggregation.
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
        image_analysis_config.set_base_dir(
            ScanPaths.paths_config.image_analysis_configs_path
        )

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

    def _get_value(self, input_data: Dict) -> Dict[str, float]:
        """
        Execute ScanAnalyzers for relevant diagnostics, use results to compute objective.

        Assumes BaseEvaluator.get_value() already refreshed current data and will log results.
        Returns a dict of outputs. If `self.output_key` is not None, the dict will include it.

        Routes to ``compute_objective`` (per_bin) or ``compute_objective_from_shots``
        (per_shot) depending on each analyzer's ``analysis_mode``. When modes are
        mixed, per_bin scalars are merged into every shot dict before the per_shot
        path is taken.
        """
        per_bin_scalars: Dict[str, float] = {}
        # shot_scalars[i] holds the merged scalar dict for the i-th shot in the bin
        shot_scalars: List[Dict[str, float]] = []
        shot_numbers: List[int] = list(self.current_shot_numbers or [])
        has_per_shot = any(
            cfg.analysis_mode == "per_shot" for cfg in self.analyzer_configs
        )

        for config in self.analyzer_configs:
            device_name = config.device_name
            analyzer = self.scan_analyzers[device_name]

            logger.info(
                "Running %s analyzer for device '%s' on bin %s",
                config.analysis_mode,
                device_name,
                self.bin_number,
            )

            analyzer.auxiliary_data = self.current_data_bin
            analyzer.run_analysis(scan_tag=self.scan_tag)

            if config.analysis_mode == "per_bin":
                result: ImageAnalyzerResult = analyzer.results.get(self.bin_number)
                if result:
                    per_bin_scalars.update(result.scalars)
                    logger.info(
                        "Extracted %d per_bin scalars from '%s'",
                        len(result.scalars),
                        device_name,
                    )
                else:
                    logger.warning(
                        "No per_bin result for bin %s from '%s'",
                        self.bin_number,
                        device_name,
                    )
            else:  # per_shot
                if not shot_scalars:
                    shot_scalars = [{} for _ in shot_numbers]
                for i, shot in enumerate(shot_numbers):
                    result = analyzer.results.get(shot)
                    if result:
                        shot_scalars[i].update(result.scalars)
                    else:
                        logger.warning(
                            "No per_shot result for shot %s from '%s'",
                            shot,
                            device_name,
                        )

        outputs: Dict[str, float] = {}
        output_key = getattr(self, "output_key", None)

        if has_per_shot:
            # Merge per_bin scalars (e.g. from a mixed-mode setup) into each shot dict
            if per_bin_scalars:
                for shot_dict in shot_scalars:
                    for k, v in per_bin_scalars.items():
                        shot_dict.setdefault(k, v)

            if output_key is not None:
                objective_result = self.compute_objective_from_shots(
                    scalar_results_list=shot_scalars, bin_number=self.bin_number
                )
                if isinstance(objective_result, dict):
                    if output_key not in objective_result:
                        logger.warning(
                            "compute_objective_from_shots dict missing key '%s'",
                            output_key,
                        )
                    outputs.update(
                        {str(k): float(v) for k, v in objective_result.items()}
                    )
                else:
                    outputs[output_key] = float(objective_result)

            # Pass mean-aggregated scalars to compute_observables for consistency
            all_keys = (
                set().union(*(d.keys() for d in shot_scalars))
                if shot_scalars
                else set()
            )
            aggregated: Dict[str, float] = {}
            for key in all_keys:
                vals = [d[key] for d in shot_scalars if key in d]
                if vals:
                    aggregated[key] = float(np.mean(vals))
            aggregated.update(per_bin_scalars)
            scalar_results_for_observables = aggregated
        else:
            if output_key is not None:
                outputs[output_key] = float(
                    self.compute_objective(
                        scalar_results=per_bin_scalars, bin_number=self.bin_number
                    )
                )
            scalar_results_for_observables = per_bin_scalars

        # Optional extras
        extra = (
            self.compute_observables(
                scalar_results=scalar_results_for_observables,
                bin_number=self.bin_number,
            )
            or {}
        )

        # Prevent overriding the objective key
        if output_key is not None and output_key in extra:
            logger.warning(
                "compute_observables returned key '%s'; removing to avoid overriding the objective",
                output_key,
            )
            extra = {k: v for k, v in extra.items() if k != output_key}

        # Ensure plain floats (skip keys already written by per_shot dict path)
        for k, v in extra.items():
            if str(k) not in outputs:
                outputs[str(k)] = float(v)

        return outputs

    @abstractmethod
    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """
        Compute objective function from analyzer scalar results.

        This abstract method must be implemented by subclasses to define
        their specific objective function logic. The method receives all
        scalar results from all configured analyzers and should combine
        them into a single objective value.

        Called when all analyzers use ``analysis_mode="per_bin"``, or by the
        default ``compute_objective_from_shots`` implementation after
        mean-aggregating per-shot scalars.

        Parameters
        ----------
        scalar_results : dict
            Combined scalar results from all analyzers.
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

    def compute_objective_from_shots(
        self,
        scalar_results_list: List[Dict[str, float]],
        bin_number: int,
    ) -> Union[float, Dict[str, float]]:
        """
        Compute objective from per-shot scalar results.

        Called when any analyzer uses ``analysis_mode="per_shot"``.
        The default implementation mean-aggregates the per-shot scalars and
        delegates to ``compute_objective``, so existing subclasses require no
        changes when switching from ``per_bin`` to ``per_shot``.

        Override this method to apply custom statistical treatment, for example
        returning a noise estimate alongside the objective so Xopt's GP
        surrogate can account for shot-to-shot jitter::

            def compute_objective_from_shots(self, scalar_results_list, bin_number):
                values = [
                    self.get_scalar(self.device_name, "image_total", r)
                    for r in scalar_results_list
                ]
                return {"f": -float(np.median(values)), "f_noise": float(np.std(values))}

        Parameters
        ----------
        scalar_results_list : list of dict
            One scalar dict per shot in the current bin. Each dict contains
            the same keys as the ``scalar_results`` dict passed to
            ``compute_objective`` in per_bin mode.
        bin_number : int
            Current bin number being evaluated.

        Returns
        -------
        float or dict
            Either a single float objective value, or a dict mapping output
            keys to floats. When returning a dict, it must contain
            ``self.output_key`` (typically ``"f"``). Extra keys (e.g.
            ``"f_noise"``) are passed through to Xopt.
        """
        if not scalar_results_list:
            logger.warning(
                "compute_objective_from_shots received empty list for bin %s",
                bin_number,
            )
            return 0.0

        all_keys = set().union(*scalar_results_list)
        aggregated: Dict[str, float] = {}
        for key in all_keys:
            vals = [d[key] for d in scalar_results_list if key in d]
            if vals:
                aggregated[key] = float(np.mean(vals))

        return self.compute_objective(scalar_results=aggregated, bin_number=bin_number)

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
            Combined scalar results from all analyzers. In per_shot mode,
            this is the mean-aggregated dict across shots in the bin.
        bin_number : int
            Current bin number being evaluated.

        Returns
        -------
        dict
            Mapping of observable names to scalar values. The default
            implementation returns an empty dict.
        """
        return {}

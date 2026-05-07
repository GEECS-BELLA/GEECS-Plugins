"""
Generic evaluator using SingleDeviceScanAnalyzer(s) for optimization.

Subclasses implement :meth:`compute_objective` for simple per-bin averaging,
or override :meth:`compute_objective_from_shots` for richer per-shot treatment
(median, noise estimates, etc.).

Analysis mode is configured per-analyzer via ``analysis_mode`` in
:class:`~geecs_scanner.optimization.config_models.SingleDeviceScanAnalyzerConfig`:

- ``per_bin`` (default) — images averaged before analysis; one scalar dict
  per bin is broadcast to every slot before aggregation.
- ``per_shot`` — each image analyzed individually; one scalar dict per shot.

When analyzers have mixed modes, ``per_bin`` results are broadcast across all
shot slots so every slot has a full merged scalar dict before the objective hook
is called.

Classes
-------
MultiDeviceScanEvaluator
    Base evaluator using SingleDeviceScanAnalyzer(s) for optimization.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from geecs_scanner.data_acquisition.data_logger import DataLogger
    from geecs_scanner.data_acquisition.scan_data_manager import ScanDataManager

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

    Subclasses implement :meth:`compute_objective` (and optionally
    :meth:`compute_objective_from_shots` and :meth:`compute_observables`).

    Parameters
    ----------
    analyzers : list of dict
        Each entry is validated as a
        :class:`~geecs_scanner.optimization.config_models.SingleDeviceScanAnalyzerConfig`.
    scan_data_manager : ScanDataManager, optional
    data_logger : DataLogger, optional
    **kwargs
        Forwarded to :class:`~geecs_scanner.optimization.base_evaluator.BaseEvaluator`.
    """

    def __init__(
        self,
        analyzers: List[dict],
        scan_data_manager: Optional["ScanDataManager"] = None,
        data_logger: Optional["DataLogger"] = None,
        **kwargs,
    ):
        self.analyzer_configs: List[SingleDeviceScanAnalyzerConfig] = [
            SingleDeviceScanAnalyzerConfig.model_validate(cfg) for cfg in analyzers
        ]

        evaluator_config = MultiDeviceScanEvaluatorConfig(
            analyzers=self.analyzer_configs
        )
        device_requirements = evaluator_config.generate_device_requirements()

        super().__init__(
            device_requirements=device_requirements,
            scan_data_manager=scan_data_manager,
            data_logger=data_logger,
        )

        self.scan_analyzers: Dict = {}
        for config in self.analyzer_configs:
            self.scan_analyzers[config.device_name] = self._create_scan_analyzer(config)

        self.output_key = "f"
        self.objective_tag: str = "MultiDeviceScan"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def primary_device(self) -> str:
        """Device name from the first analyzer config."""
        return self.analyzer_configs[0].device_name

    # ------------------------------------------------------------------
    # Analyzer construction
    # ------------------------------------------------------------------

    def _create_scan_analyzer(self, config: SingleDeviceScanAnalyzerConfig):
        """Dynamically instantiate a scan analyzer from *config*."""
        image_analysis_config.set_base_dir(
            ScanPaths.paths_config.image_analysis_configs_path
        )

        from scan_analysis.analyzers.common.array1d_scan_analysis import (
            Array1DScanAnalyzer,
        )
        from scan_analysis.analyzers.common.array2D_scan_analysis import (
            Array2DScanAnalyzer,
        )

        analyzer_class = {
            "Array1DScanAnalyzer": Array1DScanAnalyzer,
            "Array2DScanAnalyzer": Array2DScanAnalyzer,
        }[config.analyzer_type]

        image_analyzer_module = importlib.import_module(config.image_analyzer.module)
        image_analyzer_class = getattr(
            image_analyzer_module, config.image_analyzer.class_
        )
        image_analyzer = image_analyzer_class(**config.image_analyzer.kwargs)

        scan_analyzer = analyzer_class(
            device_name=config.device_name,
            image_analyzer=image_analyzer,
            file_tail=config.file_tail,
            analysis_mode=config.analysis_mode,
        )
        scan_analyzer.live_analysis = True
        scan_analyzer.use_colon_scan_param = False
        return scan_analyzer

    # ------------------------------------------------------------------
    # Scalar key resolution
    # ------------------------------------------------------------------

    def get_scalar(
        self, device_name: str, metric_name: str, scalar_results: dict
    ) -> float:
        """
        Extract a scalar with flexible key-naming fallback.

        Tries ``device_name_metric``, ``device_name:metric``, then
        ``metric`` alone.

        Raises
        ------
        KeyError
            When none of the three forms are present.
        """
        for key in (
            f"{device_name}_{metric_name}",
            f"{device_name}:{metric_name}",
            metric_name,
        ):
            if key in scalar_results:
                return scalar_results[key]

        raise KeyError(
            f"Could not find metric '{metric_name}' for device '{device_name}'. "
            f"Tried: {device_name}_{metric_name}, {device_name}:{metric_name}, "
            f"{metric_name}. Available: {list(scalar_results.keys())}"
        )

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _get_value(self, input_data: Dict) -> Dict[str, float]:
        """
        Run all analyzers, merge results, call objective/observable hooks.

        Both ``per_bin`` and ``per_shot`` analyzers produce a unified list of
        scalar dicts (``merged``).  ``per_bin`` results are broadcast to every
        slot; ``per_shot`` results fill by position.  The objective hook always
        receives the full list, so subclasses can apply arbitrary statistics.
        """
        shot_numbers: List[int] = list(self.current_shot_numbers or [])
        has_per_shot = any(
            cfg.analysis_mode == "per_shot" for cfg in self.analyzer_configs
        )
        # per_bin-only → one slot (the bin average); per_shot → one slot per shot
        n_slots = len(shot_numbers) if has_per_shot else 1
        merged: List[Dict[str, float]] = [{} for _ in range(n_slots)]

        for config in self.analyzer_configs:
            device_name = config.device_name
            analyzer = self.scan_analyzers[device_name]

            logger.info(
                "Running %s analyzer for '%s' on bin %s",
                config.analysis_mode,
                device_name,
                self.bin_number,
            )
            analyzer.auxiliary_data = self.current_data_bin
            analyzer.run_analysis(scan_tag=self.scan_tag)

            if config.analysis_mode == "per_bin":
                result = analyzer.results.get(self.bin_number)
                if result is None:
                    logger.warning(
                        "No per_bin result for bin %s from '%s'",
                        self.bin_number,
                        device_name,
                    )
                    continue
                # broadcast to every slot; first writer wins on key conflicts
                for slot in merged:
                    for k, v in result.scalars.items():
                        slot.setdefault(k, v)
            else:  # per_shot
                for i, shot in enumerate(shot_numbers):
                    result = analyzer.results.get(shot)
                    if result:
                        merged[i].update(result.scalars)
                    else:
                        logger.warning(
                            "No per_shot result for shot %s from '%s'",
                            shot,
                            device_name,
                        )

        outputs: Dict[str, float] = {}
        output_key = self.output_key

        if output_key is not None:
            objective_result = self.compute_objective_from_shots(
                scalar_results_list=merged, bin_number=self.bin_number
            )
            if isinstance(objective_result, dict):
                if output_key not in objective_result:
                    logger.warning(
                        "compute_objective_from_shots dict missing key '%s'", output_key
                    )
                outputs.update({str(k): float(v) for k, v in objective_result.items()})
            else:
                outputs[output_key] = float(objective_result)

        # mean-aggregate merged slots for compute_observables
        all_keys = set().union(*(d.keys() for d in merged)) if merged else set()
        aggregated: Dict[str, float] = {
            k: float(np.mean([d[k] for d in merged if k in d])) for k in all_keys
        }

        extra = (
            self.compute_observables(
                scalar_results=aggregated, bin_number=self.bin_number
            )
            or {}
        )
        if output_key is not None and output_key in extra:
            logger.warning(
                "compute_observables returned objective key '%s'; removing", output_key
            )
            extra = {k: v for k, v in extra.items() if k != output_key}

        for k, v in extra.items():
            if str(k) not in outputs:
                outputs[str(k)] = float(v)

        return outputs

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """
        Compute the objective from aggregated scalar results.

        Override this for simple evaluators where mean aggregation across shots
        is sufficient.  For full per-shot control (median, noise estimate, etc.)
        override :meth:`compute_objective_from_shots` instead.

        Raises
        ------
        NotImplementedError
            When neither this method nor :meth:`compute_objective_from_shots`
            is overridden.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement compute_objective "
            "or override compute_objective_from_shots."
        )

    def compute_objective_from_shots(
        self,
        scalar_results_list: List[Dict[str, float]],
        bin_number: int,
    ) -> Union[float, Dict[str, float]]:
        """
        Compute the objective from per-slot scalar results.

        The default mean-aggregates all slots and delegates to
        :meth:`compute_objective`, so subclasses that implement only
        ``compute_objective`` work for all analysis modes without changes.

        Override this to apply custom statistics, e.g.::

            def compute_objective_from_shots(self, scalar_results_list, bin_number):
                vals = [self.get_scalar(self.primary_device, "image_total", r)
                        for r in scalar_results_list]
                return {"f": -float(np.median(vals)), "f_noise": float(np.std(vals))}

        Parameters
        ----------
        scalar_results_list : list of dict
            One dict per slot (shot or bin, depending on ``analysis_mode``).
        bin_number : int

        Returns
        -------
        float or dict
            Scalar objective, or a dict containing at least ``self.output_key``
            plus any extra keys (e.g. ``f_noise``) to pass through to Xopt.
        """
        if not scalar_results_list:
            logger.warning(
                "compute_objective_from_shots received empty list for bin %s",
                bin_number,
            )
            return 0.0

        all_keys = set().union(*scalar_results_list)
        aggregated: Dict[str, float] = {
            k: float(np.mean([d[k] for d in scalar_results_list if k in d]))
            for k in all_keys
        }
        return self.compute_objective(scalar_results=aggregated, bin_number=bin_number)

    def compute_observables(
        self, scalar_results: dict, bin_number: int
    ) -> Dict[str, float]:
        """
        Return extra scalar observables to include alongside the objective.

        *scalar_results* is always the mean-aggregated dict across all slots.
        The default returns an empty dict.
        """
        return {}

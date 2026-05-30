"""
Generic evaluator using ScanAnalyzers driven by unified diagnostics.

Subclasses implement :meth:`compute_objective` for simple per-bin averaging,
or override :meth:`compute_objective_from_shots` for richer per-shot treatment
(median, noise estimates, etc.).

Analysis mode is configured per-analyzer via ``analysis_mode`` in
:class:`~geecs_scanner.optimization.config_models.OptimizerAnalyzerRef`:

- ``per_bin`` (default) — images averaged before analysis; one scalar dict
  per bin is broadcast to every slot before aggregation.
- ``per_shot`` — each image analyzed individually; one scalar dict per shot.

When analyzers have mixed modes, ``per_bin`` results are broadcast across all
shot slots so every slot has a full merged scalar dict before the objective hook
is called.

Classes
-------
MultiDeviceScanEvaluator
    Base evaluator using diagnostic-driven ScanAnalyzers for optimization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from geecs_scanner.engine.data_logger import DataLogger
    from geecs_scanner.engine.scan_data_manager import ScanDataManager

from scan_analysis.config import create_scan_analyzer

from geecs_scanner.optimization.base_evaluator import BaseEvaluator
from geecs_scanner.optimization.config_models import OptimizerAnalyzerRef

logger = logging.getLogger(__name__)


class MultiDeviceScanEvaluator(BaseEvaluator):
    """
    Base evaluator using one or more diagnostic-driven ScanAnalyzers.

    The optimizer YAML lists each analyzer as a reference to a unified
    diagnostic (see
    :class:`~geecs_scanner.optimization.config_models.OptimizerAnalyzerRef`).
    Wrapper class, image config, file tail, and data folder are all
    inherited from the diagnostic on disk; the optimizer only needs the
    diagnostic name plus an optional per-run ``analysis_mode`` override.

    Each diagnostic is handed straight to
    :func:`scan_analysis.config.create_scan_analyzer` — the canonical
    factory used by the task queue and LiveWatch — with
    ``use_injected_data=True`` so the wrapper consumes the in-memory
    DataLogger frame rather than re-reading the s-file from disk after
    each scan. The caller is responsible for setting ``auxiliary_data``
    on each analyzer before invoking ``run_analysis`` (handled in
    :meth:`_get_value` below).

    Subclasses implement :meth:`compute_objective` (and optionally
    :meth:`compute_objective_from_shots` and :meth:`compute_observables`).

    Parameters
    ----------
    analyzers : list of dict
        Each entry is validated as an
        :class:`~geecs_scanner.optimization.config_models.OptimizerAnalyzerRef`.
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
        self.analyzer_refs: List[OptimizerAnalyzerRef] = [
            OptimizerAnalyzerRef.model_validate(cfg) for cfg in analyzers
        ]

        # Aggregate per-analyzer device blocks into the evaluator-level
        # device_requirements. Each ref already loaded its diagnostic
        # during validation, so this loop does no extra disk work.
        devices: Dict[str, dict] = {}
        for ref in self.analyzer_refs:
            devices.update(ref.to_device_requirement())
        device_requirements = {"Devices": devices}

        super().__init__(
            device_requirements=device_requirements,
            scan_data_manager=scan_data_manager,
            data_logger=data_logger,
        )

        # Build one ScanAnalyzer per diagnostic. ``use_injected_data=True``
        # tells the wrapper to skip disk-based s-file loading; the
        # in-memory DataLogger frame is supplied per evaluation via
        # ``auxiliary_data`` in ``_get_value``. The effective analysis
        # mode (``per_shot`` vs ``per_bin``) is resolved inside the
        # factory: ``ref.analysis_mode`` if set, else the diagnostic's
        # ``scan.mode``. Read it off ``analyzer.analysis_mode`` whenever
        # the dispatch loop needs it.
        self.scan_analyzers: Dict = {
            ref.device_name: create_scan_analyzer(
                ref.diag,
                analysis_mode=ref.analysis_mode,
                use_injected_data=True,
            )
            for ref in self.analyzer_refs
        }

        self.output_key = "f"
        self.objective_tag: str = "MultiDeviceScan"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def primary_device(self) -> str:
        """Device name from the first analyzer ref."""
        return self.analyzer_refs[0].device_name

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
            a.analysis_mode == "per_shot" for a in self.scan_analyzers.values()
        )
        # per_bin-only → one slot (the bin average); per_shot → one slot per shot
        n_slots = len(shot_numbers) if has_per_shot else 1
        merged: List[Dict[str, float]] = [{} for _ in range(n_slots)]

        for ref in self.analyzer_refs:
            device_name = ref.device_name
            analyzer = self.scan_analyzers[device_name]
            mode = analyzer.analysis_mode

            logger.info(
                "Running %s analyzer for '%s' on bin %s",
                mode,
                device_name,
                self.bin_number,
            )
            analyzer.auxiliary_data = self.current_data_bin
            analyzer.run_analysis(scan_tag=self.scan_tag)

            if mode == "per_bin":
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

        return self._compute_outputs(merged, self.bin_number)

"""
Evaluator that computes an objective directly from scalar log_entries columns.

No image analysis is involved. Scalars are read from ``current_data_bin``
columns (populated from ``DataLogger.log_entries``), making this the simplest
possible evaluator for optimizing a device-level scalar diagnostic.

Subclasses implement :meth:`compute_objective` for simple mean-aggregation, or
override :meth:`compute_objective_from_shots` for custom per-shot statistics
(median, noise estimates, etc.).

Classes
-------
ScalarLogEvaluator
    Evaluator that reads scalars from log_entries and computes an objective.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Union

import numpy as np

from geecs_scanner.optimization.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class ScalarLogEvaluator(BaseEvaluator):
    """
    Evaluator that reads scalars from ``log_entries`` columns and computes an objective.

    Subclasses implement :meth:`compute_objective` for simple evaluators where
    mean aggregation across shots is sufficient.  For full per-shot control
    (median, noise estimate, etc.) override :meth:`compute_objective_from_shots`
    instead.

    Parameters
    ----------
    scalar_keys : list of str
        Column names to extract from ``current_data_bin``.  These must match
        keys logged by ``DataLogger`` (e.g., device variable names or injected
        computed values).
    **kwargs
        Forwarded to :class:`~geecs_scanner.optimization.base_evaluator.BaseEvaluator`.

    Examples
    --------
    Minimize a scalar directly from log entries::

        class EnergyEvaluator(ScalarLogEvaluator):
            def compute_objective(self, scalar_results, bin_number):
                return -scalar_results["U_Laser:Energy"]

        evaluator = EnergyEvaluator(scalar_keys=["U_Laser:Energy"], ...)
    """

    def __init__(
        self,
        scalar_keys: List[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scalar_keys = scalar_keys
        self.output_key = "f"
        self.objective_tag: str = "ScalarLog"

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _get_value(self, input_data: Dict) -> Dict[str, float]:
        """
        Extract per-shot scalars from current_data_bin, call objective/observable hooks.

        Returns
        -------
        dict
            Scalar results including the objective (if ``output_key`` is set)
            and any extra observables from :meth:`compute_observables`.
        """
        shot_scalars: List[Dict[str, float]] = []
        for _, row in self.current_data_bin.iterrows():
            slot: Dict[str, float] = {}
            for key in self.scalar_keys:
                if key in row:
                    try:
                        slot[key] = float(row[key])
                    except (TypeError, ValueError):
                        logger.warning(
                            "Could not convert '%s' = %r to float for shot",
                            key,
                            row[key],
                        )
            shot_scalars.append(slot)

        outputs: Dict[str, float] = {}
        output_key = self.output_key

        if output_key is not None:
            objective_result = self.compute_objective_from_shots(
                scalar_results_list=shot_scalars, bin_number=self.bin_number
            )
            if isinstance(objective_result, dict):
                if output_key not in objective_result:
                    logger.warning(
                        "compute_objective_from_shots dict missing key '%s'", output_key
                    )
                outputs.update({str(k): float(v) for k, v in objective_result.items()})
            else:
                outputs[output_key] = float(objective_result)

        # Mean-aggregate shot scalars for compute_observables
        all_keys = (
            set().union(*(d.keys() for d in shot_scalars)) if shot_scalars else set()
        )
        aggregated: Dict[str, float] = {
            k: float(np.mean([d[k] for d in shot_scalars if k in d])) for k in all_keys
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
    # Hooks for subclasses — mirror the MultiDeviceScanEvaluator API
    # ------------------------------------------------------------------

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """
        Compute the objective from mean-aggregated scalars.

        Override this for simple evaluators.  For full per-shot control
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
        Compute the objective from per-shot scalar results.

        The default mean-aggregates all shots and delegates to
        :meth:`compute_objective`.

        Parameters
        ----------
        scalar_results_list : list of dict
            One dict per shot with extracted scalar values.
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
        Return extra scalar observables alongside the objective.

        *scalar_results* is the mean-aggregated dict across all shots.
        The default returns an empty dict.
        """
        return {}

    # ------------------------------------------------------------------
    # Observable-only mode (output_key = None)
    # ------------------------------------------------------------------

    @classmethod
    def observables_only(
        cls,
        scalar_keys: List[str],
        **kwargs,
    ) -> "ScalarLogEvaluator":
        """
        Create an observables-only instance with ``output_key = None``.

        Useful for diagnostics that should be logged alongside an optimization
        without contributing to the objective.

        Parameters
        ----------
        scalar_keys : list of str
            Column names to extract and log as observables.
        **kwargs
            Forwarded to the constructor.

        Returns
        -------
        ScalarLogEvaluator
        """

        class _ObservablesOnly(cls):  # type: ignore[valid-type]
            def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
                return 0.0

            def compute_observables(
                self, scalar_results: dict, bin_number: int
            ) -> Dict[str, float]:
                return {k: v for k, v in scalar_results.items() if k in scalar_keys}

        instance = _ObservablesOnly(scalar_keys=scalar_keys, **kwargs)
        instance.output_key = None
        return instance

    # ------------------------------------------------------------------
    # Optional: multi-scalar objective helper
    # ------------------------------------------------------------------

    def get_scalar(self, key: str, scalar_results: dict) -> float:
        """
        Extract *key* from *scalar_results* with a clear error.

        Parameters
        ----------
        key : str
            Column name to extract.
        scalar_results : dict
            The aggregated scalar dict passed to :meth:`compute_objective`.

        Raises
        ------
        KeyError
            When *key* is not present.
        """
        if key not in scalar_results:
            raise KeyError(
                f"Scalar key '{key}' not found. "
                f"Available: {list(scalar_results.keys())}. "
                f"Configured scalar_keys: {self.scalar_keys}"
            )
        return float(scalar_results[key])

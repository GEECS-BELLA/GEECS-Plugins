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
from typing import Dict, List

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
        """Extract per-shot scalars from current_data_bin and compute outputs."""
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
        return self._compute_outputs(shot_scalars, self.bin_number)

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

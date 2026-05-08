"""Abstract base class for GEECS optimization evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from geecs_scanner.engine.data_logger import DataLogger
    from geecs_scanner.engine.scan_data_manager import ScanDataManager

import logging

import numpy as np

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Template-method base for all optimization evaluators.

    Subclasses implement :meth:`_get_value`. The public entry point
    :meth:`get_value` handles data refresh, type normalization, and logging.

    Parameters
    ----------
    device_requirements : dict, optional
        Devices the optimizer needs saved; consumed externally by the scan
        executor, not by this class.
    scan_data_manager : ScanDataManager, optional
    data_logger : DataLogger, optional
    """

    def __init__(
        self,
        device_requirements: Optional[Dict[str, Any]] = None,
        scan_data_manager: Optional["ScanDataManager"] = None,
        data_logger: Optional["DataLogger"] = None,
    ):
        self.device_requirements = device_requirements or {}
        self.scan_data_manager = scan_data_manager
        self.data_logger = data_logger

        self.bin_number: int = 0
        self.log_df = None  # pd.DataFrame, populated by get_current_data
        self.current_data_bin = None  # pd.DataFrame filtered to current bin
        self.current_shot_numbers: Optional[List[int]] = None
        self.objective_tag: str = "default"
        self.output_key: Optional[str] = None

        self.scan_tag = (
            self.scan_data_manager.scan_paths.get_tag()
            if self.scan_data_manager is not None
            else None
        )

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def get_current_data(self) -> None:
        """
        Refresh current_data_bin and current_shot_numbers from data_logger.

        Converts log_entries to a DataFrame (sorted by elapsed time so that
        Shotnumber reflects acquisition order), then filters to the current bin.
        """
        import pandas as pd

        log_entries = self.data_logger.log_entries
        self.bin_number = self.data_logger.bin_num

        df = pd.DataFrame.from_dict(log_entries, orient="index")
        df = df.sort_values(by="Elapsed Time").reset_index(drop=True)
        df["Shotnumber"] = df.index + 1
        self.log_df = df

        bin_mask = df["Bin #"] == self.bin_number
        self.current_data_bin = df[bin_mask].copy()
        self.current_shot_numbers = df.loc[bin_mask, "Shotnumber"].tolist()

    @staticmethod
    def filter_log_entries_by_bin(
        log_entries: Dict[float, Dict[str, Any]], bin_num: int
    ) -> List[Dict[str, Any]]:
        """Return all log entries belonging to *bin_num*."""
        return [
            entry for entry in log_entries.values() if entry.get("Bin #") == bin_num
        ]

    # ------------------------------------------------------------------
    # Public entry point (called by Xopt via evaluate_function)
    # ------------------------------------------------------------------

    def get_value(self, input_data: Dict) -> Dict[str, float]:
        """
        Refresh data, evaluate, normalise types, log, and return results.

        Parameters
        ----------
        input_data : dict
            Control-variable values from the optimizer.

        Returns
        -------
        dict
            Scalar results, including ``self.output_key`` when set.
        """
        self.get_current_data()

        results = self._get_value(input_data=input_data)
        if not isinstance(results, dict):
            raise TypeError(
                f"{self.__class__.__name__}._get_value must return Dict[str, float]; "
                f"got {type(results)}"
            )

        results = {str(k): float(v) for k, v in results.items()}

        if self.output_key is not None and self.output_key not in results:
            raise KeyError(
                f"{self.__class__.__name__} requires objective key '{self.output_key}' "
                "in results, or set self.output_key = None for observables-only evaluators."
            )

        self.log_results_for_current_bin(results)
        return results

    def __call__(self, input_data: Dict) -> Dict:
        """Alias for :meth:`get_value`."""
        return self.get_value(input_data)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_results_for_current_bin(self, results: Dict[str, float]) -> None:
        """Write *results* into every shot entry for the current bin."""
        if not self.current_shot_numbers:
            logger.warning("No shots found for current bin %s", self.bin_number)
            return
        for shot_num in self.current_shot_numbers:
            self._log_results_for_shot(shot_num, results)

    def _log_results_for_shot(self, shot_num: int, results: Dict[str, float]) -> None:
        try:
            elapsed_time = self.current_data_bin.loc[
                self.current_data_bin["Shotnumber"] == shot_num, "Elapsed Time"
            ].values[0]
        except Exception as e:
            logger.warning(
                "Could not extract Elapsed Time for shot %s: %s", shot_num, e
            )
            return

        if not elapsed_time:
            logger.warning("No valid elapsed_time for shot %s", shot_num)
            return

        for k, v in results.items():
            key = (
                f"Objective:{self.objective_tag}"
                if self.output_key is not None and k == self.output_key
                else f"Observable:{k}"
            )
            self.data_logger.log_entries[elapsed_time][key] = v
            logger.info("Logged %s = %s for shot %s", key, v, shot_num)

    # ------------------------------------------------------------------
    # Hooks — override in subclasses
    # ------------------------------------------------------------------

    def compute_objective(
        self, scalar_results: Dict[str, float], bin_number: int
    ) -> float:
        """
        Compute the scalar objective from aggregated per-shot scalars.

        Override for simple evaluators where mean aggregation is sufficient.
        For full per-shot control (median, noise estimates) override
        :meth:`compute_objective_from_shots` instead.

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
        Compute the objective from a list of per-shot scalar dicts.

        The default mean-aggregates all shots and delegates to
        :meth:`compute_objective`. Override for custom statistics::

            def compute_objective_from_shots(self, scalar_results_list, bin_number):
                vals = [d["energy"] for d in scalar_results_list if "energy" in d]
                return {"f": -float(np.median(vals)), "f_noise": float(np.std(vals))}

        Returns
        -------
        float or dict
            Scalar objective, or a dict with at least ``self.output_key`` plus
            any extra keys (e.g. ``f_noise``) to pass through to Xopt.
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
        self, scalar_results: Dict[str, float], bin_number: int
    ) -> Dict[str, float]:
        """
        Return extra scalar observables to log alongside the objective.

        *scalar_results* is the mean-aggregated dict across all shots.
        The default returns an empty dict.
        """
        return {}

    def _compute_outputs(
        self,
        scalar_results_list: List[Dict[str, float]],
        bin_number: int,
    ) -> Dict[str, float]:
        """
        Build the final outputs dict from a list of per-shot scalar dicts.

        Called at the end of :meth:`_get_value` by subclasses after they have
        assembled their shot list.  Handles objective computation, observable
        merging, and the output-key shadowing check.
        """
        outputs: Dict[str, float] = {}
        output_key = self.output_key

        if output_key is not None:
            objective_result = self.compute_objective_from_shots(
                scalar_results_list=scalar_results_list, bin_number=bin_number
            )
            if isinstance(objective_result, dict):
                if output_key not in objective_result:
                    logger.warning(
                        "compute_objective_from_shots dict missing key '%s'", output_key
                    )
                outputs.update({str(k): float(v) for k, v in objective_result.items()})
            else:
                outputs[output_key] = float(objective_result)

        all_keys = (
            set().union(*(d.keys() for d in scalar_results_list))
            if scalar_results_list
            else set()
        )
        aggregated: Dict[str, float] = {
            k: float(np.mean([d[k] for d in scalar_results_list if k in d]))
            for k in all_keys
        }

        extra = (
            self.compute_observables(scalar_results=aggregated, bin_number=bin_number)
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
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_value(self, input_data: Dict) -> Dict[str, float]:
        """
        Compute and return results.

        Must include ``self.output_key`` when ``self.output_key`` is not None.
        """

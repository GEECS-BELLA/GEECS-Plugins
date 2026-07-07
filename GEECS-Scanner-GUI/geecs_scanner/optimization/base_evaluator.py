"""Unified base class for GEECS optimization evaluators.

A single :class:`BaseEvaluator` handles both diagnostic-driven scan analyzers
and direct s-file scalar columns. Subclasses implement
:meth:`compute_objective` and/or :meth:`compute_observables` (both peer
hooks; either or both can be implemented).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

from image_analysis.config import load_diagnostic
from scan_analysis.config import create_scan_analyzer

if TYPE_CHECKING:
    import pandas as pd

    from geecs_scanner.engine.data_logger import DataLogger
    from geecs_scanner.engine.scan_data_manager import ScanDataManager

logger = logging.getLogger(__name__)


class EvaluatorDataSource(Protocol):
    """Where the evaluator reads per-shot rows and writes results back.

    This is the single seam between the evaluator and whichever engine is
    running the scan. The legacy ScanManager path uses
    :class:`DataLoggerSource` (in-memory ``DataLogger.log_entries``); the
    Bluesky/CA path provides a source over the session's per-bin event rows.
    Everything downstream of :meth:`BaseEvaluator.get_current_data` —
    analyzers, s-file scalars, objective/observable hooks — sees only the
    DataFrame this produces and is engine-agnostic.
    """

    def fetch(self) -> Tuple["pd.DataFrame", int]:
        """Return (full log frame, current bin number).

        The frame must carry ``"Bin #"``, ``"Elapsed Time"`` and a
        ``"Shotnumber"`` column reflecting acquisition order.
        """
        ...

    def record(self, elapsed_time: float, key: str, value: float) -> None:
        """Write one result column back for the shot row keyed by *elapsed_time*."""
        ...


class DataLoggerSource:
    """Legacy source: the ScanManager path's in-memory ``DataLogger``."""

    def __init__(self, data_logger: "DataLogger") -> None:
        self.data_logger = data_logger

    def fetch(self) -> Tuple["pd.DataFrame", int]:
        """Build the sorted log frame from ``log_entries`` (Shotnumber = order)."""
        import pandas as pd

        df = pd.DataFrame.from_dict(self.data_logger.log_entries, orient="index")
        df = df.sort_values(by="Elapsed Time").reset_index(drop=True)
        df["Shotnumber"] = df.index + 1
        return df, self.data_logger.bin_num

    def record(self, elapsed_time: float, key: str, value: float) -> None:
        """Write into the live log entry so the value lands in the s-file."""
        self.data_logger.log_entries[elapsed_time][key] = value


# Per-analyzer device-requirements template. Lifted from the old
# multi_device_scan_evaluator + config_models constant so the unified base
# class can build device_requirements from the analyzer list directly.
_ANALYZER_DEVICE_REQUIREMENT_TEMPLATE = {
    "add_all_variables": False,
    "save_nonscalar_data": True,
    "synchronous": True,
    "variable_list": ["acq_timestamp"],
}


class BaseEvaluator:
    """
    Unified evaluator: runs diagnostic-driven analyzers and reads s-file scalars.

    The evaluator's job is to assemble a per-shot scalar dict for every shot
    in the current bin from two sources:

    - ``analyzers``: diagnostic-driven scan analyzers (run in-memory against
      the DataLogger frame). Each analyzer's ``ImageAnalyzerResult.scalars``
      dict comes back with keys already namespaced via
      ``{metric_prefix}_{key}{metric_suffix}`` (defaults to the
      diagnostic's ``name`` for the prefix and empty for the suffix
      when unset). The namespacing is applied by ScanAnalysis's
      ``SingleDeviceScanAnalyzer._consume_result`` per #412; the
      analyzer side emits bare keys and this evaluator just forwards
      them through. So ``scalars["UC_TopView_x_fwhm"]`` is the
      convention regardless of which side of the contract was running.
    - ``scalars``: column names pulled directly from the current-bin DataFrame
      (i.e. raw s-file values like ``"U_Laser:Energy"``). Used verbatim as keys.

    Subclasses then implement either or both of:

    - :meth:`compute_objective` — returns a float (the value Xopt optimizes)
    - :meth:`compute_observables` — returns a dict of named auxiliary metrics
      (also returned to Xopt; required for BAX-style algorithms that don't have
      an objective)

    Both hooks receive the same ``scalars`` dict and both can optionally
    override the per-shot list versions (:meth:`compute_objective_from_shots`,
    :meth:`compute_observables_from_shots`) for non-mean statistics or
    shot-level filtering.

    Parameters
    ----------
    analyzers : list of (str or dict), optional
        Diagnostic stems or dict-form entries ``{diagnostic: X, ...overrides}``.
        Each entry passes through :func:`image_analysis.config.load_diagnostic`
        and yields a scan analyzer built via
        :func:`scan_analysis.config.create_scan_analyzer` with
        ``use_injected_data=True``.
    scalars : list of str, optional
        Column names to pull from ``current_data_bin`` per shot. No
        analyzer is invoked for these; they're already in the DataLogger
        frame.
    device_requirements : dict, optional
        Override the auto-generated requirements. The default is the union
        of per-analyzer blocks (each keyed on the GEECS device name).
    scan_data_manager, data_logger : injected at construction time
    data_source : EvaluatorDataSource, optional
        The engine seam. Defaults to ``DataLoggerSource(data_logger)`` when
        only ``data_logger`` is given (legacy path); the Bluesky/CA path
        injects a source over the session's per-bin rows instead.
    scan_tag : ScanTag, optional
        Explicit scan tag for analyzer file loading. Defaults to
        ``scan_data_manager.scan_paths.get_tag()`` when a manager is given;
        engines without a ScanDataManager pass the tag directly.

    Attributes
    ----------
    diagnostics : list of DiagnosticAnalysisConfig
        Resolved diagnostics from ``analyzers``.
    scan_analyzers : dict[str, ScanAnalyzer]
        One ScanAnalyzer per diagnostic, keyed by GEECS device name.
    scalar_keys : list of str
        Column names that will be read from ``current_data_bin``.
    output_key : str or None
        The key in the returned outputs dict that Xopt should treat as the
        objective. Defaults to ``"f"``. Set to ``None`` for BAX evaluators
        that emit only observables.
    objective_tag : str, optional
        Human-readable label written to ``log_entries`` (under
        ``Objective:<tag>``). Xopt's actual objective key is hardcoded as
        ``"f"``; this tag exists purely so the s-file row that gets written
        to disk carries a recognisable name. Defaults to the subclass name.
        Override via this kwarg from YAML, or via a class attribute on the
        subclass.
    """

    # Default output_key — subclasses override to None for BAX-style
    # observables-only mode.
    output_key: Optional[str] = "f"

    # Defaults to class name in __init__ if not overridden by subclass.
    objective_tag: str = ""

    # Class-level default so the data_source property works on instances
    # built without running BaseEvaluator.__init__ (test doubles do this).
    _data_source: Optional["EvaluatorDataSource"] = None

    def __init__(
        self,
        analyzers: Optional[List[Union[str, Dict[str, Any]]]] = None,
        scalars: Optional[List[str]] = None,
        objective_tag: Optional[str] = None,
        device_requirements: Optional[Dict[str, Any]] = None,
        scan_data_manager: Optional["ScanDataManager"] = None,
        data_logger: Optional["DataLogger"] = None,
        data_source: Optional[EvaluatorDataSource] = None,
        scan_tag: Optional[Any] = None,  # ScanTag; Any avoids a hard dep here
    ):
        # Deferred: config_models' module-level model rebuild walks into
        # geecs_scanner.engine, which imports back through base_optimizer
        # to this module — a cycle when base_evaluator is imported first.
        from geecs_scanner.optimization.config_models import _split_analyzer_entry

        # --- Data sources ---------------------------------------------
        # Load each diagnostic into a typed config; build one scan
        # analyzer per diagnostic, keyed on the GEECS device name. Both
        # are stored so subclasses can introspect (e.g. for primary_device).
        self.diagnostics = []
        for entry in analyzers or []:
            name, overrides = _split_analyzer_entry(entry)
            self.diagnostics.append(load_diagnostic(name, overrides=overrides))

        self.scan_analyzers: Dict = {
            diag.name: create_scan_analyzer(diag, use_injected_data=True)
            for diag in self.diagnostics
        }

        # s-file scalar keys are plain column names from current_data_bin.
        # No analyzer needs to run for these.
        self.scalar_keys: List[str] = list(scalars or [])

        # --- Device requirements --------------------------------------
        # Auto-generate from analyzers if not supplied. s-file scalars
        # are assumed to already be in the DataLogger frame (configured
        # via the scanner's save-element layer), so they don't extend
        # device_requirements here.
        if device_requirements is None:
            devices = {
                diag.name: dict(_ANALYZER_DEVICE_REQUIREMENT_TEMPLATE)
                for diag in self.diagnostics
            }
            device_requirements = {"Devices": devices}
        self.device_requirements = device_requirements

        # --- Injected runtime context ---------------------------------
        # data_source is the engine seam (EvaluatorDataSource). When only
        # data_logger is present — including assigned after construction,
        # a long-standing pattern — the data_source property lazily wraps
        # it in a DataLoggerSource (legacy behavior).
        self.scan_data_manager = scan_data_manager
        self.data_logger = data_logger
        self._data_source = data_source
        if scan_tag is not None:
            self.scan_tag = scan_tag
        else:
            self.scan_tag = (
                self.scan_data_manager.scan_paths.get_tag()
                if self.scan_data_manager is not None
                else None
            )

        # --- Per-evaluation state -------------------------------------
        self.bin_number: int = 0
        self.log_df = None  # pd.DataFrame
        self.current_data_bin = None  # filtered to current bin
        self.current_shot_numbers: Optional[List[int]] = None

        # objective_tag resolution order:
        # 1. ``objective_tag`` kwarg (typically set from YAML kwargs)
        # 2. class attribute (``objective_tag = "BeamSize"`` on the subclass)
        # 3. fall back to the subclass name
        if objective_tag is not None:
            self.objective_tag = objective_tag
        elif not self.objective_tag:
            self.objective_tag = self.__class__.__name__

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def data_source(self) -> Optional[EvaluatorDataSource]:
        """The engine seam, lazily wrapping ``data_logger`` when not injected.

        Tracks ``data_logger`` reassignment (a long-standing pattern) so the
        wrapper never points at a stale logger.
        """
        if self.data_logger is not None and (
            self._data_source is None
            or (
                isinstance(self._data_source, DataLoggerSource)
                and self._data_source.data_logger is not self.data_logger
            )
        ):
            self._data_source = DataLoggerSource(self.data_logger)
        return self._data_source

    @data_source.setter
    def data_source(self, value: Optional[EvaluatorDataSource]) -> None:
        self._data_source = value

    @property
    def primary_device(self) -> Optional[str]:
        """GEECS device name of the first listed diagnostic, or None.

        Convenience for subclasses that want to reference "the" device
        when there's just one analyzer (the common case).
        """
        return self.diagnostics[0].name if self.diagnostics else None

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def get_current_data(self) -> None:
        """Refresh ``current_data_bin`` and ``current_shot_numbers`` from the data source.

        Fetches the full log frame and current bin number from
        :attr:`data_source`, then filters to the current bin.
        """
        if self.data_source is None:
            raise RuntimeError(
                "BaseEvaluator has no data source: construct with data_logger= "
                "(legacy ScanManager path) or data_source= (engine seam)."
            )
        df, self.bin_number = self.data_source.fetch()
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
        """Refresh data, evaluate, normalise types, log, and return results."""
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
                "in results, or set output_key = None for observables-only evaluators."
            )

        self.log_results_for_current_bin(results)
        return results

    def __call__(self, input_data: Dict) -> Dict:
        """Alias for :meth:`get_value`."""
        return self.get_value(input_data)

    # ------------------------------------------------------------------
    # Core evaluation: build per-shot scalars dict, call hooks
    # ------------------------------------------------------------------

    def _get_value(self, input_data: Dict) -> Dict[str, float]:
        """Build the per-shot scalars list from both sources, call hooks.

        - Each analyzer's per-shot or per-bin result merges into the slot
          list with keys prefixed ``<device>_<metric>``.
        - Each s-file scalar key is pulled from ``current_data_bin`` row
          by row.
        - The resulting list is handed to :meth:`_compute_outputs` which
          calls both ``compute_*_from_shots`` hooks.
        """
        shot_numbers: List[int] = list(self.current_shot_numbers or [])

        # Slot sizing.
        # - Scalars are always per-shot. per_bin aggregation is purely an
        #   image-analyzer concern; for raw s-file values the evaluator
        #   subclass decides what to do with the per-shot list (mean,
        #   median, whatever) via ``compute_*_from_shots``.
        # - If any analyzer runs per_shot, slots match shot count.
        # - Only when all analyzers are per_bin AND there are no scalars
        #   do we collapse to one slot (a single bin-averaged result).
        has_per_shot = any(
            a.analysis_mode == "per_shot" for a in self.scan_analyzers.values()
        )
        if has_per_shot or self.scalar_keys:
            n_slots = max(len(shot_numbers), 1)
        else:
            n_slots = 1
        merged: List[Dict[str, float]] = [{} for _ in range(n_slots)]

        # --- Run analyzers, prefix scalars by device name -----------------
        for device_name, analyzer in self.scan_analyzers.items():
            mode = analyzer.analysis_mode
            logger.info(
                "Running %s analyzer for '%s' on bin %s",
                mode,
                device_name,
                self.bin_number,
            )
            analyzer.auxiliary_data = self.current_data_bin
            analyzer.run_analysis(scan_tag=self.scan_tag)

            # Post-#412: ImageAnalysis emits bare scalar keys; ScanAnalysis's
            # SingleDeviceScanAnalyzer._consume_result applies the
            # ``{prefix}_{key}{suffix}`` namespacing (from the diagnostic
            # config's ``metric_prefix`` / ``metric_suffix``) before
            # stashing the result in ``analyzer.results[N]``. So
            # ``result.scalars`` here already has fully-namespaced keys
            # — pass them through unchanged. One source of truth lives
            # in ScanAnalysis; nothing for the optimizer evaluator to
            # do beyond forward.
            if mode == "per_bin":
                result = analyzer.results.get(self.bin_number)
                if result is None:
                    logger.warning(
                        "No per_bin result for bin %s from '%s'",
                        self.bin_number,
                        device_name,
                    )
                    continue
                # Broadcast already-prefixed scalars to every slot.
                for slot in merged:
                    for k, v in result.scalars.items():
                        slot[k] = v
            else:  # per_shot
                for i, shot in enumerate(shot_numbers):
                    if i >= n_slots:
                        break
                    result = analyzer.results.get(shot)
                    if result:
                        for k, v in result.scalars.items():
                            merged[i][k] = v
                    else:
                        logger.warning(
                            "No per_shot result for shot %s from '%s'",
                            shot,
                            device_name,
                        )

        # --- Pull s-file scalars: always per-shot, one row per slot ------
        # The slot count was sized to match shot count whenever scalars are
        # present, so this loop fills row-by-row. The subclass decides how
        # to aggregate (mean, median, filtered, ...) via its
        # ``compute_*_from_shots`` hook — no aggregation policy enforced
        # at the framework layer.
        if self.scalar_keys and self.current_data_bin is not None:
            for i, (_, row) in enumerate(self.current_data_bin.iterrows()):
                if i >= n_slots:
                    break
                for key in self.scalar_keys:
                    if key in row:
                        try:
                            merged[i][key] = float(row[key])
                        except (TypeError, ValueError):
                            logger.warning(
                                "Could not convert '%s' = %r to float for shot",
                                key,
                                row[key],
                            )

        return self._compute_outputs(merged, self.bin_number)

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
            self.data_source.record(elapsed_time, key, v)
            logger.info("Logged %s = %s for shot %s", key, v, shot_num)

    # ------------------------------------------------------------------
    # Hooks — subclasses override these
    # ------------------------------------------------------------------

    def compute_objective(
        self, scalars: Dict[str, float], bin_number: int
    ) -> Optional[float]:
        """Compute the scalar objective from mean-aggregated per-shot scalars.

        Override this for simple evaluators where mean aggregation is enough.
        For full per-shot control (median, percentile, shot-level filtering),
        override :meth:`compute_objective_from_shots` instead.

        ``scalars`` is a flat dict with analyzer outputs prefixed by device
        name (``"UC_TopView_x_fwhm"``) and s-file columns as their natural
        names (``"U_Laser:Energy"``).

        Returns ``None`` by default — signals "this evaluator has no
        objective" (BAX mode). Subclasses with an objective must override
        this OR :meth:`compute_objective_from_shots`.
        """
        return None

    def compute_objective_from_shots(
        self,
        scalars_list: List[Dict[str, float]],
        bin_number: int,
    ) -> Union[float, Dict[str, float], None]:
        """Compute the objective from a list of per-shot scalar dicts.

        The default mean-aggregates and delegates to :meth:`compute_objective`.
        Override for custom statistics::

            def compute_objective_from_shots(self, scalars_list, bin_number):
                vals = [d["UC_TopView_x_fwhm"] for d in scalars_list]
                return float(np.median(vals))

        Returns ``None`` to signal no objective (BAX mode), a ``float``, or
        a dict that includes at least ``self.output_key`` plus any extras
        (e.g. ``f_noise``) to pass through to Xopt.
        """
        if not scalars_list:
            logger.warning(
                "compute_objective_from_shots received empty list for bin %s",
                bin_number,
            )
            return None
        aggregated = _mean_aggregate(scalars_list)
        return self.compute_objective(scalars=aggregated, bin_number=bin_number)

    def compute_observables(
        self, scalars: Dict[str, float], bin_number: int
    ) -> Dict[str, float]:
        """Return auxiliary scalar observables.

        Override this for simple observables built from mean-aggregated
        per-shot scalars. For full per-shot control, override
        :meth:`compute_observables_from_shots` instead. Same ``scalars``
        namespace as :meth:`compute_objective`. Default returns ``{}``.

        Required for BAX evaluators (they have no objective; observables are
        what Xopt models).
        """
        return {}

    def compute_observables_from_shots(
        self,
        scalars_list: List[Dict[str, float]],
        bin_number: int,
    ) -> Dict[str, float]:
        """Return auxiliary observables from the per-shot scalar list.

        The default mean-aggregates and delegates to
        :meth:`compute_observables`. Override for per-shot statistics or
        shot-level filtering on observables — same shape as
        :meth:`compute_objective_from_shots`, the observables peer.
        """
        if not scalars_list:
            return {}
        aggregated = _mean_aggregate(scalars_list)
        return self.compute_observables(scalars=aggregated, bin_number=bin_number)

    # ------------------------------------------------------------------
    # Output assembly — calls both hooks, merges into one dict
    # ------------------------------------------------------------------

    def _compute_outputs(
        self,
        scalars_list: List[Dict[str, float]],
        bin_number: int,
    ) -> Dict[str, float]:
        """Build the final outputs dict by calling both hooks.

        Objective and observables are peer hooks. Either or both may
        contribute keys; the framework merges them with the objective key
        winning collisions.
        """
        outputs: Dict[str, float] = {}
        output_key = self.output_key

        # --- Objective hook ---
        if output_key is not None:
            objective_result = self.compute_objective_from_shots(
                scalars_list=scalars_list, bin_number=bin_number
            )
            if objective_result is None:
                # Subclass didn't implement objective despite output_key being
                # set — surface this loudly so the user adjusts either the
                # subclass or sets output_key=None for BAX mode.
                raise NotImplementedError(
                    f"{self.__class__.__name__}: output_key={output_key!r} but "
                    f"compute_objective / compute_objective_from_shots returned None. "
                    f"Either implement the objective hook or set output_key = None."
                )
            if isinstance(objective_result, dict):
                if output_key not in objective_result:
                    logger.warning(
                        "compute_objective_from_shots dict missing key '%s'",
                        output_key,
                    )
                outputs.update({str(k): float(v) for k, v in objective_result.items()})
            else:
                outputs[output_key] = float(objective_result)

        # --- Observables hook ---
        extra = (
            self.compute_observables_from_shots(
                scalars_list=scalars_list, bin_number=bin_number
            )
            or {}
        )
        if output_key is not None and output_key in extra:
            logger.warning(
                "compute_observables returned objective key '%s'; removing",
                output_key,
            )
            extra = {k: v for k, v in extra.items() if k != output_key}

        for k, v in extra.items():
            if str(k) not in outputs:
                outputs[str(k)] = float(v)

        return outputs


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _mean_aggregate(scalars_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean-aggregate a list of per-shot scalar dicts to a single dict.

    Keys present in any shot end up in the result; the mean is taken over
    only the shots that have that key (missing values don't contribute).
    """
    all_keys = set().union(*scalars_list)
    return {k: float(np.mean([d[k] for d in scalars_list if k in d])) for k in all_keys}

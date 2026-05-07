"""Shared fixtures for optimization unit tests.

All helpers here are network-free and require no scan files on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Lightweight log_entries factory
# ---------------------------------------------------------------------------


def make_log_entries(
    n_shots: int,
    bin_num: int,
    extra_columns: Optional[Dict[str, List[Any]]] = None,
) -> Dict[float, Dict[str, Any]]:
    """
    Build a ``DataLogger.log_entries``-compatible dict with synthetic data.

    Parameters
    ----------
    n_shots : int
        Number of shots to generate (all assigned to *bin_num*).
    bin_num : int
        Bin number to assign to every shot.
    extra_columns : dict of str -> list, optional
        Additional column values indexed by shot (0-based).  Lists must have
        length *n_shots*.

    Returns
    -------
    dict
        Keys are elapsed-time floats (0.1, 0.2, …); values are per-shot dicts
        that match the format expected by ``BaseEvaluator.get_current_data``.
    """
    extra_columns = extra_columns or {}
    entries: Dict[float, Dict[str, Any]] = {}
    for i in range(n_shots):
        elapsed = float(i + 1) * 0.1
        shot: Dict[str, Any] = {
            "Elapsed Time": elapsed,
            "Bin #": bin_num,
        }
        for col, values in extra_columns.items():
            shot[col] = values[i]
        entries[elapsed] = shot
    return entries


# ---------------------------------------------------------------------------
# FakeDataLogger
# ---------------------------------------------------------------------------


class FakeDataLogger:
    """
    Duck-typed replacement for ``DataLogger``.

    Only the attributes that ``BaseEvaluator.get_current_data`` accesses are
    implemented:
    - ``log_entries``
    - ``bin_num``
    """

    def __init__(
        self,
        log_entries: Dict[float, Dict[str, Any]],
        bin_num: int = 1,
    ):
        self.log_entries = log_entries
        self.bin_num = bin_num


# ---------------------------------------------------------------------------
# FakeResult
# ---------------------------------------------------------------------------


@dataclass
class FakeResult:
    """
    Minimal replacement for ``ImageAnalyzerResult``.

    Scan-analyzer results in tests only need ``.scalars``.
    """

    scalars: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Concrete BaseEvaluator subclass for tests
# ---------------------------------------------------------------------------


def make_base_evaluator(
    log_entries: Optional[Dict[float, Dict[str, Any]]] = None,
    bin_num: int = 1,
    n_shots: int = 3,
    output_key: Optional[str] = "f",
):
    """
    Build a minimal concrete :class:`BaseEvaluator` wired to a ``FakeDataLogger``.

    The returned evaluator's ``_get_value`` just returns ``{"f": 1.0}``.

    Parameters
    ----------
    log_entries : dict, optional
        Custom log_entries.  Defaults to 3 shots in bin 1 with no extra columns.
    bin_num : int
        Active bin number.
    n_shots : int
        Number of shots if *log_entries* is not provided.
    output_key : str or None
        ``output_key`` to set on the evaluator.

    Returns
    -------
    BaseEvaluator concrete instance
    """
    from geecs_scanner.optimization.base_evaluator import BaseEvaluator

    if log_entries is None:
        log_entries = make_log_entries(n_shots=n_shots, bin_num=bin_num)

    class _Concrete(BaseEvaluator):
        def _get_value(self, input_data: Dict) -> Dict[str, float]:
            return {output_key or "f": 1.0}

    obj = _Concrete.__new__(_Concrete)
    # Manually set attributes instead of calling __init__ to avoid needing
    # ScanDataManager / ScanPaths.
    obj.device_requirements = {}
    obj.scan_data_manager = None
    obj.data_logger = FakeDataLogger(log_entries=log_entries, bin_num=bin_num)
    obj.bin_number = 0
    obj.log_df = None
    obj.current_data_bin = None
    obj.current_shot_numbers = None
    obj.objective_tag = "test"
    obj.output_key = output_key
    obj.scan_tag = None
    return obj

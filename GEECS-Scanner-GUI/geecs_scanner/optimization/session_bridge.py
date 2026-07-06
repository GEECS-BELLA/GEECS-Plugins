"""Run the config-driven optimization stack on a Bluesky/CA session scan.

This is the adapter between the two worlds, and it deliberately adds no
optimization machinery of its own:

- **Everything Xopt/evaluator stays here (GUI side).**
  :class:`~geecs_scanner.optimization.base_optimizer.BaseOptimizer` is built
  from the same ``optimizer_config_path`` YAML as a legacy optimization scan —
  same VOCS, same generator factory (Xopt 3.1 / GEST), same evaluator with its
  ScanAnalysis analyzers (per-shot or per-bin-averaged image analysis).
- **Everything acquisition stays in geecs_bluesky.**
  :meth:`GeecsSession.optimize` runs the optimization *as a scan* (one scan
  number, iteration = bin, schema-v1 rows, native file saving); this module
  only supplies its ``objective`` and ``suggester``.

The seam on the evaluator side is
:class:`~geecs_scanner.optimization.base_evaluator.EvaluatorDataSource`:
:class:`SessionBinSource` presents the session's per-bin event rows as the
legacy-shaped DataFrame (``Device:Variable`` columns, ``Bin #``,
``Shotnumber``, ``Elapsed Time``) so everything downstream of
``get_current_data`` — analyzers, s-file scalars, objective hooks — runs
unchanged.  Analyzers load natively saved files by the real ``ScanTag``,
exactly as they do for any scan.

``BlueskyScanner`` receives :func:`load_session_optimization` via its
``optimization_loader`` constructor argument (wired in ``RunControl``); the
dependency direction stays GUI → geecs_bluesky.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from geecs_scanner.optimization.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


def _column_map(devices: List[Any]) -> Dict[str, str]:
    """Merge device ``_column_headers`` into an event-key → ``Device:Variable`` map.

    Each CA device carries ``_column_headers`` mapping its mangled event keys
    (``<ophyd>-<safe_var>``) to legacy ``Device Variable`` headers (the s-file
    convention).  The in-memory DataLogger frame the evaluator consumes uses
    colon form (``Device:Variable``), so convert the first space.
    """
    mapping: Dict[str, str] = {}
    for dev in devices:
        headers = getattr(dev, "_column_headers", None) or {}
        for event_key, header in headers.items():
            mapping[event_key] = header.replace(" ", ":", 1)
    return mapping


class SessionBinSource:
    """``EvaluatorDataSource`` over the session's accumulated per-bin rows.

    ``push_bin`` translates one iteration's schema-v1 event rows into
    legacy-shaped row dicts and appends them; ``fetch`` returns the full
    accumulated frame plus the current bin number, matching what
    ``DataLoggerSource`` produces from ``DataLogger.log_entries``:

    - ``Bin #`` ← ``bin_number`` (= optimization iteration)
    - ``Shotnumber`` ← ``scan_event_index`` (global, 1-based, acquisition order)
    - ``Elapsed Time`` ← ``scan_event_index`` as float (unique, monotonic row
      key; legacy wall-clock timing has no equivalent here and nothing
      downstream depends on its units)
    - ``Device:Variable`` data columns via the device column map; unmapped
      schema columns (``<det>-shot_id`` / ``-valid`` / …) are kept under their
      raw names for shot-level filtering in evaluator hooks.

    ``record`` writes evaluator results (``Objective:<tag>`` /
    ``Observable:<k>``) back into the kept rows, mirroring the legacy
    write-back into ``log_entries``.
    """

    def __init__(self, column_map: Optional[Dict[str, str]] = None) -> None:
        self.column_map: Dict[str, str] = dict(column_map or {})
        self._rows_by_time: Dict[float, Dict[str, Any]] = {}
        self._current_bin: int = 0
        self._fallback_index: int = 0

    def push_bin(self, bin_data: Any) -> None:
        """Translate and append one iteration's rows (``BinData``-shaped)."""
        self._current_bin = int(bin_data.iteration)
        for row in bin_data.rows:
            index = row.get("scan_event_index")
            if index is None:
                self._fallback_index += 1
                index = self._fallback_index
            else:
                index = int(index)
                self._fallback_index = max(self._fallback_index, index)
            out: Dict[str, Any] = {
                "Bin #": int(row.get("bin_number", bin_data.iteration)),
                "Shotnumber": index,
                "Elapsed Time": float(index),
            }
            for key, value in row.items():
                mapped = self.column_map.get(key)
                if mapped is not None:
                    out[mapped] = value
                elif key not in out:
                    out[key] = value
            self._rows_by_time[out["Elapsed Time"]] = out

    # --- EvaluatorDataSource protocol ---------------------------------

    def fetch(self) -> Tuple[pd.DataFrame, int]:
        """Return (accumulated legacy-shaped frame, current bin number)."""
        rows = [self._rows_by_time[t] for t in sorted(self._rows_by_time)]
        return pd.DataFrame(rows), self._current_bin

    def record(self, elapsed_time: float, key: str, value: float) -> None:
        """Attach an evaluator result to the row keyed by *elapsed_time*."""
        row = self._rows_by_time.get(float(elapsed_time))
        if row is None:
            logger.warning("record(): no row at Elapsed Time %s", elapsed_time)
            return
        row[key] = value


class SessionOptimizationBridge:
    """Drive a :class:`BaseOptimizer` through ``GeecsSession.optimize``.

    Implements the session's suggester protocol (``suggest`` / ``observe``)
    over the optimizer's ask/tell surface and supplies the ``objective``
    callable that runs the config-driven evaluator on each completed bin.
    Mirrors the legacy executor's behavior: the first
    ``max(0, 2 - n_seeded)`` proposals are random within the VOCS bounds
    before the generator takes over, and evaluated points (inputs + all
    evaluator outputs, observables included) are fed to Xopt after every bin.
    """

    def __init__(self, optimizer: BaseOptimizer) -> None:
        self.optimizer = optimizer
        self.source = SessionBinSource()
        optimizer.evaluator.data_source = self.source
        self._pending_inputs: Optional[Dict[str, float]] = None
        self._pending_outputs: Optional[Dict[str, float]] = None
        self._scan_folder: Optional[Path] = None
        self._n_random_init = max(0, 2 - optimizer.n_seeded)

    @property
    def variable_names(self) -> List[str]:
        """VOCS variable names (``"Device:Variable"`` keys)."""
        return list(self.optimizer.vocs.variable_names)

    @property
    def on_finish(self) -> str:
        """Session end-of-run policy from the optimizer config.

        Maps the legacy ``move_to_best_on_finish`` flag onto
        ``GeecsSession.optimize``'s ``on_finish``: ``"best"`` (move to the
        highest-objective iteration) when set, else ``"hold"`` (stay at the
        last visited point — the legacy default behavior).
        """
        return "best" if self.optimizer.move_to_best_on_finish else "hold"

    def bind(
        self, *, devices: List[Any], scan_tag: Any, scan_folder: Any = None
    ) -> Tuple[Any, Any]:
        """Attach the run context; return ``(objective, suggester)``.

        Parameters
        ----------
        devices : list
            The session devices in the run (movables + detectors); their
            ``_column_headers`` build the event-key → legacy-column map.
        scan_tag : ScanTag or None
            The claimed scan's tag, handed to the evaluator so its
            ScanAnalysis analyzers load natively saved files from the actual
            scan folder.
        scan_folder : str or Path, optional
            The claimed scan folder. When given, each bin's expected native
            files are awaited (bounded) before the evaluator runs — devices
            write over the network and directory listings lag behind (SMB
            caching), so the objective would otherwise race the filesystem.
        """
        self.source.column_map = _column_map(devices)
        self.optimizer.evaluator.scan_tag = scan_tag
        self._scan_folder = Path(scan_folder) if scan_folder else None
        return self._objective, self

    # --- objective (called by session.optimize after each bin) --------

    def _await_bin_assets(self, bin_data: Any, timeout_s: float = 10.0) -> None:
        """Wait (bounded) for this bin's expected native files to be visible.

        The bin rows carry each analyzer device's per-shot ``acq_timestamp``
        — the exact value in the file's name — so the expected paths are
        fully determined and each is checked by direct ``stat`` (which,
        unlike a directory listing, is not served from the SMB directory
        cache). Rows invalid for a device expect no file. On timeout, log
        and proceed: the analyzer maps whatever is visible and a failed
        objective becomes NaN, same as any other missing-data case.
        """
        if self._scan_folder is None:
            return
        expected: List[Path] = []
        for device_name, analyzer in self.optimizer.evaluator.scan_analyzers.items():
            token = re.sub(r"[^a-z0-9]+", "_", device_name.lower()).strip("_")
            tail = getattr(analyzer, "file_tail", None) or ".png"
            ts_key = valid_key = None
            for row in bin_data.rows[:1]:
                for key in row:
                    norm = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
                    if norm == f"{token}_acq_timestamp":
                        ts_key = key
                    elif norm == f"{token}_valid":
                        valid_key = key
            if ts_key is None:
                continue
            folder = self._scan_folder / device_name
            for row in bin_data.rows:
                if valid_key is not None and not bool(row.get(valid_key)):
                    continue
                ts = row.get(ts_key)
                if ts is None or not np.isfinite(ts) or ts <= 0:
                    continue
                expected.append(folder / f"{device_name}_{float(ts):.3f}{tail}")

        deadline = time.monotonic() + timeout_s
        missing = [p for p in expected if not p.exists()]
        while missing and time.monotonic() < deadline:
            time.sleep(0.5)
            missing = [p for p in missing if not p.exists()]
        if missing:
            logger.warning(
                "Bin %s: %d expected native file(s) not visible after %.0fs: %s",
                bin_data.iteration,
                len(missing),
                timeout_s,
                [p.name for p in missing[:4]],
            )

    def _objective(self, bin_data: Any) -> float:
        self.source.push_bin(bin_data)
        self._await_bin_assets(bin_data)
        inputs = dict(self._pending_inputs or {})
        outputs = self.optimizer.evaluator.get_value(inputs)
        self._pending_outputs = {str(k): float(v) for k, v in outputs.items()}
        output_key = self.optimizer.evaluator.output_key
        if output_key is None:
            # BAX-style observables-only evaluator: no scalar objective.
            return float("nan")
        return float(outputs[output_key])

    # --- suggester protocol --------------------------------------------

    def suggest(self) -> Optional[Dict[str, float]]:
        """Propose the next VOCS point (random init first, then the generator)."""
        if self._n_random_init > 0:
            from xopt.vocs import random_inputs

            self._n_random_init -= 1
            candidate = random_inputs(self.optimizer.vocs, 1)[0]
        else:
            candidate = self.optimizer.generate(1)[0]
        inputs = {name: float(candidate[name]) for name in self.variable_names}
        self._pending_inputs = inputs
        return inputs

    def observe(
        self, inputs: Dict[str, float], objective: float, bin_data: Any
    ) -> None:
        """Feed the evaluated point (inputs + evaluator outputs) to Xopt."""
        outputs = self._pending_outputs
        self._pending_outputs = None
        if outputs is None:
            # Objective raised before producing outputs (session recorded
            # NaN); tell the generator the point failed rather than nothing.
            vocs = self.optimizer.vocs
            outputs = {
                name: float("nan")
                for name in list(vocs.objective_names) + list(vocs.observable_names)
            }
        frame = pd.DataFrame([{**inputs, **outputs}])
        xopt = self.optimizer.xopt
        xopt.add_data(frame)
        generator = xopt.generator
        # Same top-up as BaseOptimizer.seed_from_dumps: make sure the
        # generator's own frame is populated.
        if generator.data is None or len(generator.data) == 0:
            generator.add_data(xopt.data)


def load_session_optimization(optimizer_config_path: str) -> SessionOptimizationBridge:
    """``optimization_loader`` for ``BlueskyScanner`` (injected by RunControl).

    Builds the :class:`BaseOptimizer` from the same YAML as a legacy
    optimization scan — no ScanDataManager or DataLogger; the session bridge
    supplies the data source and scan tag at :meth:`SessionOptimizationBridge.bind`.
    """
    optimizer = BaseOptimizer.from_config_file(config_path=optimizer_config_path)
    return SessionOptimizationBridge(optimizer)

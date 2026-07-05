"""Optimization support: suggesters and per-iteration bin data.

Optimization runs **as a scan** (see :func:`geecs_bluesky.plans.optimize.geecs_adaptive_scan`
and :meth:`geecs_bluesky.session.GeecsSession.optimize`): one scan number, one
Tiled run, iteration = ``bin_number``, every row shot-matched and
``valid``-labeled like any other scan.  This module holds the pieces around the
plan:

* the **suggester protocol** — ``suggest() -> dict | None`` proposes the next
  variable values (``None`` ends the optimization early);
  ``observe(inputs, objective, bin_data)`` feeds back the evaluated result.
* :class:`RandomSuggester` — dependency-free random search (useful for
  exploration and as the machinery test double).
* :class:`XoptSuggester` — thin adapter over `Xopt <https://xopt.xopt.org>`_
  generators (requires the ``optimize`` extra).
* :class:`BinData` — one iteration's shot-matched rows (scalar access only).
  Image/diagnostic analysis belongs to the configured evaluator path
  (GEECS-Scanner-GUI ``optimization/`` — ScanAnalysis analyzers with per-shot
  or per-bin-averaged modes), not here: analyzers load native files from the
  scan folder by scan tag, exactly as they do for any scan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Suggester(Protocol):
    """Ask/tell protocol driving :meth:`GeecsSession.optimize`."""

    def suggest(self) -> dict[str, float] | None:
        """Return the next variable values, or ``None`` to stop early."""
        ...

    def observe(
        self, inputs: dict[str, float], objective: float, bin_data: "BinData"
    ) -> None:
        """Record the evaluated objective for *inputs*."""
        ...


class RandomSuggester:
    """Uniform random search within bounds — dependency-free exploration.

    Parameters
    ----------
    bounds : dict
        ``{variable_name: (lo, hi)}``.
    seed : int, optional
        RNG seed for reproducibility.
    """

    def __init__(self, bounds: dict[str, tuple[float, float]], seed: int | None = None):
        self.bounds = bounds
        self._rng = np.random.default_rng(seed)
        self.history: list[dict[str, Any]] = []

    def suggest(self) -> dict[str, float] | None:
        """Sample each variable uniformly within its bounds."""
        return {
            name: float(self._rng.uniform(lo, hi))
            for name, (lo, hi) in self.bounds.items()
        }

    def observe(
        self, inputs: dict[str, float], objective: float, bin_data: "BinData"
    ) -> None:
        """Append to :attr:`history` (used for :attr:`best`)."""
        self.history.append({**inputs, "objective": objective})

    @property
    def best(self) -> dict[str, Any] | None:
        """The highest-objective observation so far (NaNs excluded)."""
        finite = [h for h in self.history if np.isfinite(h["objective"])]
        return max(finite, key=lambda h: h["objective"]) if finite else None


class XoptSuggester:
    """Adapter exposing an Xopt generator through the suggester protocol.

    Exploratory glue (requires ``poetry install --extras optimize``): builds a
    single-objective ``VOCS`` and drives any Xopt generator through
    ``generate(1)`` / ``add_data``.

    Parameters
    ----------
    bounds : dict
        ``{variable_name: (lo, hi)}``.
    generator : str or Any
        Xopt generator name (e.g. ``"expected_improvement"``, ``"random"``)
        or an already-constructed generator instance.
    maximize : bool
        Objective sense (default maximize).
    **generator_kwargs
        Forwarded to the generator constructor when *generator* is a name.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        generator: str | Any = "random",
        *,
        maximize: bool = True,
        **generator_kwargs: Any,
    ) -> None:
        try:
            import pandas as pd
            from xopt import VOCS
            from xopt.generators import get_generator
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise ImportError(
                "XoptSuggester requires xopt: poetry install --extras optimize"
            ) from exc
        self._pd = pd
        self.vocs = VOCS(
            variables={k: list(v) for k, v in bounds.items()},
            objectives={"objective": "MAXIMIZE" if maximize else "MINIMIZE"},
        )
        if isinstance(generator, str):
            generator = get_generator(generator)(vocs=self.vocs, **generator_kwargs)
        self.generator = generator

    def suggest(self) -> dict[str, float] | None:
        """Ask the generator for one candidate."""
        candidate = self.generator.generate(1)[0]
        return {k: float(candidate[k]) for k in self.vocs.variables}

    def observe(
        self, inputs: dict[str, float], objective: float, bin_data: "BinData"
    ) -> None:
        """Tell the generator the evaluated candidate."""
        self.generator.add_data(
            self._pd.DataFrame([{**inputs, "objective": objective}])
        )


@dataclass
class BinData:
    """One optimization iteration's shot-matched rows and native assets.

    ``rows`` are the primary-stream event data dicts for this bin (schema v1:
    per-device columns plus ``<det>-shot_id`` / ``<det>-valid`` / …).
    """

    iteration: int
    rows: list[dict[str, Any]]

    def valid_rows(self, detector: str | None = None) -> list[dict[str, Any]]:
        """Rows where the detector(s) captured the row's physical shot.

        With *detector* given, filters on that device's ``-valid`` column;
        otherwise requires every present ``-valid`` column to be true.
        """
        if detector is not None:
            key = f"{detector}-valid"
            return [r for r in self.rows if r.get(key)]
        return [
            r for r in self.rows if all(v for k, v in r.items() if k.endswith("-valid"))
        ]

    def column(self, key: str, *, valid_for: str | None = None) -> np.ndarray:
        """One column across the bin as an array (optionally valid-filtered)."""
        rows = self.valid_rows(valid_for) if valid_for is not None else self.rows
        return np.asarray([r[key] for r in rows if key in r], dtype=float)

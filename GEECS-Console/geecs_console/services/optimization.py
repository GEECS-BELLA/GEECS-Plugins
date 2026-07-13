"""The console's ``optimization_loader``: OptimizationSpec → session bridge.

The engine (GeecsBluesky ≥ 0.31.0) runs optimize-mode ScanRequests through a
GUI-injected ``optimization_loader`` — on the delegated path the loader is
called with the request's resolved
:class:`~geecs_schemas.OptimizationSpec` — because the config-driven
Xopt/evaluator stack lives in ``geecs_scanner.optimization``, which
geecs_bluesky must never import (dependency direction).  This module is the
console's implementation of that seam:

- :func:`optimizer_config_from_spec` — pure mapping from the schema
  ``OptimizationSpec`` onto the ``BaseOptimizerConfig`` dict shape; the
  exact inverse of ``geecs_schemas.convert.convert_optimizer_config``
  (``generator.options`` ↔ ``xopt_config_overrides[generator.name]``).
  ``max_iterations`` is deliberately not mapped — the engine consumes it
  from the spec directly.
- :func:`load_console_optimization` — the loader callable: builds a
  ``BaseOptimizer`` from the mapped config and wraps it in the
  ``SessionOptimizationBridge`` (the same bridge the legacy GUI used).
- :func:`make_optimization_loader` — availability-gated factory used by
  :func:`geecs_console.submission.make_bluesky_submitter`.  Returns the
  loader when ``geecs-scanner-gui`` is importable (installed via the
  console's optional ``optimization`` extra), else ``None`` — the engine
  then refuses optimize submissions at ``reinitialize`` with its explicit
  needs-a-loader message, which the window already surfaces in the status
  bar.  The check is a light ``find_spec`` so submitter construction never
  pays the Xopt import; the heavy imports happen inside the loader call,
  on the scan thread, once per process.

The stack behind the loader (evaluators → ScanAnalysis/ImageAnalysis
analyzers) is the legacy machinery kept for old-GUI parity; a redesigned
hook (bluesky-adaptive direction) is a planned follow-up, so this module is
written to be deletable — nothing else in the console imports
``geecs_scanner``.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def optimizer_config_from_spec(spec: Any) -> dict:
    """Map an ``OptimizationSpec`` onto the ``BaseOptimizerConfig`` dict shape.

    Pure and import-light (duck-typed off the spec's attributes), so the
    mapping is testable without ``geecs-scanner-gui`` installed.

    Parameters
    ----------
    spec : geecs_schemas.OptimizationSpec
        The request's resolved optimization block.

    Returns
    -------
    dict
        The parsed-YAML shape ``BaseOptimizer.from_config`` validates:
        ``vocs`` (variables/objectives/observables/constraints),
        ``evaluator`` (module/class/kwargs), ``generator`` (name),
        ``xopt_config_overrides`` keyed by the generator name (only when
        the spec carries generator options), ``seed_dump_files`` and
        ``move_to_best_on_finish``.
    """
    config: dict[str, Any] = {
        "vocs": {
            "variables": {
                name: list(bounds) for name, bounds in spec.variables.items()
            },
            "objectives": dict(spec.objectives),
            "observables": list(spec.observables),
            "constraints": {
                name: list(bound) for name, bound in spec.constraints.items()
            },
        },
        "evaluator": {
            "module": spec.evaluator.module,
            "class": spec.evaluator.class_name,
            "kwargs": dict(spec.evaluator.kwargs),
        },
        "generator": {"name": spec.generator.name},
        "seed_dump_files": [str(path) for path in spec.seed_dump_files],
        "move_to_best_on_finish": spec.move_to_best_on_finish,
    }
    if spec.generator.options:
        # Inverse of convert_optimizer_config: the legacy overrides dict is
        # keyed by the generator name.
        config["xopt_config_overrides"] = {
            spec.generator.name: dict(spec.generator.options)
        }
    return config


def load_console_optimization(spec: Any) -> Any:
    """Build the optimization bridge for one resolved ``OptimizationSpec``.

    The console's ``optimization_loader`` implementation: instantiates the
    config-driven optimizer stack (``BaseOptimizer`` — Xopt generator +
    dynamically imported evaluator with its analyzers) and wraps it in the
    ``SessionOptimizationBridge`` whose ``bind`` the engine calls with the
    connected devices and the claimed scan tag/folder.

    Relative ``seed_dump_files`` entries are not resolved on this path
    (there is no config-file directory once the spec is inline in the
    request) — use absolute paths in optimizer configs that warm-start.

    Parameters
    ----------
    spec : geecs_schemas.OptimizationSpec
        The request's resolved optimization block.

    Returns
    -------
    geecs_scanner.optimization.session_bridge.SessionOptimizationBridge
        The bridge exposing ``bind(devices=..., scan_tag=...,
        scan_folder=...) -> (objective, suggester)`` and ``finish()``.
    """
    from geecs_scanner.optimization.base_optimizer import BaseOptimizer
    from geecs_scanner.optimization.session_bridge import SessionOptimizationBridge

    optimizer = BaseOptimizer.from_config(optimizer_config_from_spec(spec))
    return SessionOptimizationBridge(optimizer)


def optimization_available() -> bool:
    """Whether the optimization stack (``geecs-scanner-gui``) is importable.

    A light ``find_spec`` probe — nothing heavy is imported.  Installed via
    the console's optional ``optimization`` extra.
    """
    try:
        return importlib.util.find_spec("geecs_scanner.optimization") is not None
    except (ImportError, ModuleNotFoundError):
        return False


def make_optimization_loader() -> Optional[Callable[[Any], Any]]:
    """Return the console's ``optimization_loader``, or ``None`` if unavailable.

    ``None`` (the ``optimization`` extra not installed) makes the engine
    refuse optimize-mode requests at ``reinitialize`` with its explicit
    needs-a-loader message — surfaced in the status bar like any other
    submit-time error; every other scan mode is unaffected.
    """
    if not optimization_available():
        logger.info(
            "geecs-scanner-gui is not installed (console 'optimization' "
            "extra) — optimize-mode submissions will be refused by the engine"
        )
        return None
    return load_console_optimization

"""The queueserver worker's ``optimization_loader``: OptimizationSpec → bridge.

Decision 5 (``Planning/cutover_strategy/02_queueserver_migration.md``):
optimization stays in-worker — the stack's code
(``geecs_bluesky.optimization``) does not move, only the *call site* that
invokes it moves from the GUI process into the queueserver worker's plan
preamble (``geecs_bluesky.plans.scan_request_plan``,
:func:`~geecs_bluesky.plans.scan_request_plan.set_optimization_loader`).

This module is the worker-side twin of GEECS-Console's
``geecs_console.services.optimization`` — same four functions (including
the startup warm-up), same shapes, independently implemented so
GeecsBluesky imports nothing from GEECS-Console (the dependency graph runs
the other way: Console depends on GeecsBluesky, never the reverse). Keep
the two in sync by hand if the ``OptimizationSpec``/``BaseOptimizerConfig``
mapping changes.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: The heavy modules :func:`load_worker_optimization` imports at call time —
#: pulling in the whole Xopt/botorch/torch stack transitively.  The warm-up
#: thread imports exactly these so a later loader call is a pure
#: ``sys.modules`` cache hit.
_HEAVY_MODULES = (
    "geecs_bluesky.optimization.base_optimizer",
    "geecs_bluesky.optimization.session_bridge",
)


def optimizer_config_from_spec(spec: Any) -> dict:
    """Map an ``OptimizationSpec`` onto the ``BaseOptimizerConfig`` dict shape.

    Pure and import-light (duck-typed off the spec's attributes), so the
    mapping is testable without the ``optimize`` extra installed. The exact
    inverse of ``geecs_schemas.convert.convert_optimizer_config``
    (``generator.options`` <-> ``xopt_config_overrides[generator.name]``).
    ``max_iterations`` is deliberately not mapped — the plan consumes it
    from the spec directly.

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
        config["xopt_config_overrides"] = {
            spec.generator.name: dict(spec.generator.options)
        }
    return config


def load_worker_optimization(spec: Any) -> Any:
    """Build the optimization bridge for one resolved ``OptimizationSpec``.

    The worker's ``optimization_loader`` implementation: instantiates the
    config-driven optimizer stack (``BaseOptimizer`` — Xopt generator +
    dynamically imported evaluator with its analyzers) and wraps it in the
    ``SessionOptimizationBridge`` whose ``bind`` the plan calls with the
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
    geecs_bluesky.optimization.session_bridge.SessionOptimizationBridge
        The bridge exposing ``bind(devices=..., scan_tag=...,
        scan_folder=...) -> (objective, suggester)``, ``finish()``, and
        ``device_requirements`` (auto-generated from the evaluator's
        analyzers).
    """
    from geecs_bluesky.optimization.base_optimizer import BaseOptimizer
    from geecs_bluesky.optimization.session_bridge import SessionOptimizationBridge

    optimizer = BaseOptimizer.from_config(optimizer_config_from_spec(spec))
    return SessionOptimizationBridge(optimizer)


def optimization_available() -> bool:
    """Whether the optimization stack's heavy dependencies are installed.

    The stack's *code* (``geecs_bluesky.optimization``) always ships with
    geecs-bluesky; the ``optimize`` extra adds the dependency tree (xopt ->
    torch/botorch, ScanAnalysis). A light ``find_spec`` probe on the
    stack's two import roots (``xopt`` AND ``scan_analysis`` — an
    environment with only one would die mid-scan instead of getting the
    clean needs-a-loader refusal); nothing heavy is imported.
    """
    try:
        return (
            importlib.util.find_spec("xopt") is not None
            and importlib.util.find_spec("scan_analysis") is not None
        )
    except (ImportError, ModuleNotFoundError):
        return False


def make_worker_optimization_loader() -> Optional[Callable[[Any], Any]]:
    """Return the worker's ``optimization_loader``, or ``None`` if unavailable.

    ``None`` (the ``optimize`` extra not installed) is what the worker
    startup profile passes to
    :func:`~geecs_bluesky.plans.scan_request_plan.set_optimization_loader`,
    which makes the plan refuse optimize-mode requests loudly rather than
    failing mid-scan on a missing import.
    """
    if not optimization_available():
        logger.info(
            "the optimization stack's dependencies are not installed "
            "(the `optimize` extra) — optimize-mode ScanRequests will be "
            "refused by geecs_scan_request_plan"
        )
        return None
    return load_worker_optimization


def warm_up_optimization_stack() -> Optional[threading.Thread]:
    """Pre-import the optimization stack on a daemon thread.

    The loader's first call pays the torch/botorch/xopt cold import — tens
    of seconds — which would otherwise freeze the worker's first optimize
    request. The worker startup profile (``qserver/startup/startup.py``)
    calls this once, right after registering a non-``None``
    :func:`make_worker_optimization_loader` result via ``set_optimization_
    loader``: when the ``optimize`` extra is present (the same light
    ``find_spec`` probe as :func:`optimization_available`), a daemon thread
    imports ``_HEAVY_MODULES`` in the background so the stack is warm
    before the first optimize-mode ``ScanRequest`` ever arrives on the
    queue. Without the extra this is a no-op.

    Warm-up failures are logged and swallowed — they never affect startup;
    the loader's own lazy import simply pays the cost (and raises its own
    error) later.

    No double-import guard is needed for a request arriving mid-warm-up:
    Python's per-module import locks already serialize concurrent imports,
    so :func:`load_worker_optimization`'s import statements block until
    the warm-up thread finishes that module, then reuse it from
    ``sys.modules``. The work happens exactly once per process either way.

    Returns
    -------
    threading.Thread or None
        The started daemon thread, or ``None`` when the ``optimize``
        extra is absent (nothing to warm). Callers need not join it.
    """
    if not optimization_available():
        logger.debug(
            "optimization stack warm-up skipped — its dependencies are not "
            "installed (the `optimize` extra)"
        )
        return None

    def _warm_up() -> None:
        start = time.perf_counter()
        try:
            for name in _HEAVY_MODULES:
                importlib.import_module(name)
        except Exception:
            logger.warning(
                "optimization stack warm-up failed — the first optimize "
                "request will pay the import cost instead",
                exc_info=True,
            )
            return
        logger.info(
            "optimization stack preloaded in %.1f s", time.perf_counter() - start
        )

    thread = threading.Thread(target=_warm_up, name="optimization-warmup", daemon=True)
    thread.start()
    return thread


__all__ = [
    "optimizer_config_from_spec",
    "load_worker_optimization",
    "optimization_available",
    "make_worker_optimization_loader",
    "warm_up_optimization_stack",
]

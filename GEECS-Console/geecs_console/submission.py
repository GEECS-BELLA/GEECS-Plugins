"""The submission seam: what the console needs from a scan engine.

:class:`Submitter` is the protocol the main window depends on — the four
methods of ``BlueskyScanner``'s ScanManager-compatible surface that the
console actually calls.  :func:`make_bluesky_submitter` builds the real
engine; the import is inside the function so the window opens (and the whole
package imports) without the ``ca`` extra or a reachable gateway.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from geecs_schemas import ScanRequest


@runtime_checkable
class Submitter(Protocol):
    """The scan-engine surface the console submits through."""

    def reinitialize(self, request: ScanRequest) -> bool:
        """Validate and store *request* for the next run."""
        ...

    def start_scan_thread(self) -> None:
        """Launch the stored scan in a background thread."""
        ...

    def stop_scanning_thread(self) -> None:
        """Abort the running scan and join its thread."""
        ...

    def is_scanning_active(self) -> bool:
        """Whether a scan is currently running."""
        ...


def make_bluesky_submitter(
    experiment: str,
    on_event: Callable[[Any], None] | None = None,
) -> Submitter:
    """Build the real :class:`~geecs_bluesky.scanner_bridge.BlueskyScanner`.

    Parameters
    ----------
    experiment : str
        Experiment name (configs-repo folder / PV prefix).
    on_event : callable, optional
        The engine's ``on_event`` callback — typically
        :meth:`geecs_console.events_adapter.ScanEventsAdapter.handle`.

    Returns
    -------
    Submitter
        A ready ``BlueskyScanner`` (RunEngine created, Tiled subscription
        attempted best-effort).

    Raises
    ------
    ImportError
        When ``geecs-bluesky`` is installed without the ``ca`` extra.

    Notes
    -----
    The engine's ``optimization_loader`` seam is wired here from
    :func:`geecs_console.services.optimization.make_optimization_loader`:
    with the console's optional ``optimization`` extra installed the loader
    turns an optimize request's ``OptimizationSpec`` into the Xopt/evaluator
    bridge; without it the loader is ``None`` and the engine refuses
    optimize-mode requests at ``reinitialize`` (surfaced in the status bar).
    """
    from geecs_bluesky.scanner_bridge.bluesky_scanner import BlueskyScanner

    from geecs_console.services.optimization import make_optimization_loader

    return BlueskyScanner(
        experiment_dir=experiment,
        on_event=on_event,
        optimization_loader=make_optimization_loader(),
    )

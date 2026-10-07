"""Subscribe the GEECS output callbacks to a RunEngine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from geecs_bluesky.callbacks.scan_info import ScanInfoCallback
from geecs_bluesky.callbacks.scan_log import ScanLogCallback
from geecs_bluesky.callbacks.sfile import SFileCallback
from geecs_bluesky.callbacks.stack_check import StackCheckCallback


@dataclass(frozen=True)
class ScanOutputs:
    """The subscribed output callbacks of a RunEngine, and their tokens.

    Returned so a caller can reach the two that finish their work on a
    thread: :meth:`join` blocks until every pending stack read and s-file
    write is done, which an orderly shutdown (or a test) needs and a
    bare subscription token cannot give.
    """

    stack_check: StackCheckCallback
    scan_log: ScanLogCallback
    scan_info: ScanInfoCallback
    sfile: SFileCallback
    tokens: tuple[int, int, int, int]

    def join(self, timeout: float | None = None) -> None:
        """Wait for the pending stack reads and s-file writes."""
        self.stack_check.join(timeout)
        self.sfile.join(timeout)


def subscribe_scan_outputs(run_engine: Any) -> ScanOutputs:
    """Subscribe the output callbacks; return them with their tokens.

    Order is not load-bearing: the stack check appends its verdict to
    ``scan.log`` itself, after the log callback has closed the file.

    Returns
    -------
    ScanOutputs
        The four callbacks and their subscription tokens.  Iterating it is
        not the same as the pre-0.85 four-tuple of tokens — read
        ``.tokens`` for those.
    """
    stack_check = StackCheckCallback()
    scan_log = ScanLogCallback()
    scan_info = ScanInfoCallback()
    sfile = SFileCallback()
    return ScanOutputs(
        stack_check=stack_check,
        scan_log=scan_log,
        scan_info=scan_info,
        sfile=sfile,
        tokens=(
            run_engine.subscribe(stack_check),
            run_engine.subscribe(scan_log),
            run_engine.subscribe(scan_info),
            run_engine.subscribe(sfile),
        ),
    )

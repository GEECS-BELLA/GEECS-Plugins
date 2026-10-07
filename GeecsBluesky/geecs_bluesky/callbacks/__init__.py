"""The GEECS outputs of a run, as RunEngine callbacks.

Four document callbacks, each best-effort: a failure is logged and never
raised back into the RunEngine.

- :class:`ScanInfoCallback` — ``ScanInfoScanNNN.ini`` at the start
  document (the ``[Scan Info]`` keys downstream readers parse), rewritten
  at the stop document with ``ScanEndInfo`` filled in.
- :class:`SFileCallback` — ``ScanDataScanNNN.txt`` + ``analysis/sNNN.txt``
  at the stop document, from the run's own per-shot rows
  (:func:`geecs_data_utils.write_scalar_files`), for any exit status that
  produced rows: a strict run's ``primary`` or a gated run's ``shots``
  events, with datum-only streams joined by offset-corrected stamp.
- :class:`ScanLogCallback` — ``scan.log`` from start to stop
  (:class:`geecs_bluesky.callbacks.scan_log.ScanLogFile`).
- :class:`StackCheckCallback` — at the stop document, checks each image
  stack's frame count and per-row stamps against the documents.  A
  mismatch is a WARNING in ``scan.log``, never a failure.

All four read the claim preprocessor's start-document keys
(``scan_number``, ``scan_folder``, ``geecs_scalar_headers``) and write
**into** the claimed folder, never creating it.  The two stream readers
share :class:`_StreamCallback` and read files on a thread that waits for
the plugin to finalize (the stop document precedes ``unstage``), so the
RunEngine is never blocked.
"""

from geecs_bluesky.callbacks._base import (
    DEFAULT_FINALIZE_TIMEOUT_S,
    ROW_STREAMS,
    await_finalized,
)
from geecs_bluesky.callbacks.outputs import ScanOutputs, subscribe_scan_outputs
from geecs_bluesky.callbacks.scan_info import (
    ScanInfoCallback,
    first_axis,
    scan_info_lines,
    scan_parameter,
    shots_per_step,
)
from geecs_bluesky.callbacks.scan_log import ScanLogCallback
from geecs_bluesky.callbacks.sfile import SFileCallback
from geecs_bluesky.callbacks.stack_check import StackCheckCallback

__all__ = [
    "DEFAULT_FINALIZE_TIMEOUT_S",
    "ROW_STREAMS",
    "SFileCallback",
    "ScanOutputs",
    "ScanInfoCallback",
    "ScanLogCallback",
    "StackCheckCallback",
    "await_finalized",
    "first_axis",
    "scan_info_lines",
    "scan_parameter",
    "shots_per_step",
    "subscribe_scan_outputs",
]

"""The GEECS outputs of a run, as RunEngine callbacks.

Document callbacks, each best-effort: a failure is logged and never
raised back into the RunEngine.

- :class:`ScanInfoCallback` — ``ScanInfoScanNNN.ini`` at the start
  document (the ``[Scan Info]`` keys downstream readers parse), rewritten
  at the stop document with ``ScanEndInfo`` filled in.
- :class:`SFileCallback` — ``ScanDataScanNNN.txt`` + ``analysis/sNNN.txt``
  at the stop document, from the run's own per-shot rows
  (:func:`geecs_data_utils.write_scalar_files`), for any exit status that
  produced rows: the run's ``primary`` events, with datum-only streams
  joined by offset-corrected stamp.
- :class:`ScanLogCallback` — ``scan.log`` from start to stop
  (:class:`geecs_bluesky.callbacks.scan_log.ScanLogFile`).
- :class:`StackCheckCallback` — at the stop document, checks each image
  stack's frame count and per-row stamps against the documents.  A
  mismatch is a WARNING in ``scan.log``, never a failure.
- :class:`EcsDumpCallback` — at the start document, Master Control's ECS
  live dump (``<day>/ECS Live dumps/Scan<N>.txt``): every served settable
  read once, one bounded batch, before the plan's first move.  The one
  output beside ``scans/`` rather than in the scan folder.

All read the claim preprocessor's start-document keys
(``scan_number``, ``scan_folder``, ``geecs_scalar_headers``); the first
four write **into** the claimed folder, never creating it, and the dump
creates only its own ``ECS Live dumps/`` folder in the existing day.  The two stream readers
share :class:`_StreamCallback` and read files on a thread that waits for
the plugin to finalize (the stop document precedes ``unstage``), so the
RunEngine is never blocked.
"""

from geecs_bluesky.callbacks._base import (
    DEFAULT_FINALIZE_TIMEOUT_S,
    ROW_STREAMS,
    await_finalized,
)
from geecs_bluesky.callbacks.ecs_dump import (
    EcsDumpCallback,
    ecs_dump_path,
    ecs_dump_targets,
    render_ecs_dump,
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
    "EcsDumpCallback",
    "ROW_STREAMS",
    "SFileCallback",
    "ScanOutputs",
    "ScanInfoCallback",
    "ScanLogCallback",
    "StackCheckCallback",
    "await_finalized",
    "ecs_dump_path",
    "ecs_dump_targets",
    "first_axis",
    "render_ecs_dump",
    "scan_info_lines",
    "scan_parameter",
    "shots_per_step",
    "subscribe_scan_outputs",
]

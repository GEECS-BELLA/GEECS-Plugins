"""The s-file callback.

``ScanDataScanNNN.txt`` + ``analysis/sNNN.txt`` at the stop document, from
the run's own per-shot rows.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from geecs_bluesky.callbacks._base import (
    Document,
    _await_all_finalized,
    _RunStreams,
    _Stack,
    _StreamCallback,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from geecs_data_utils.shot_table import DeviceRecords

logger = logging.getLogger(__name__)


class SFileCallback(_StreamCallback):
    """Write the legacy scalar files at the stop document from the run's per-shot rows.

    The rows are the ``primary`` events when the run has them (strict: the
    shot table, or an earlier run's per-shot reads) and the per-shot sampler's
    ``shots`` events otherwise (gated: ``primary`` carries only the
    cameras' frames).  The columns are whatever the rows carried — the run's
    devices and its background telemetry alike — renamed and ordered by
    ``geecs_scalar_headers`` inside
    :func:`geecs_data_utils.build_legacy_scalar_dataframe`.

    Every **datum-only** stream of the run — a gated run's cameras, a
    gated run's streamed camera, a shot-table run's device streams — has its per-frame columns joined onto those rows by
    offset-corrected stamp through the shot table
    (:mod:`geecs_data_utils.shot_table`): one
    s-file row per row of the run, a camera's per-frame scalars spelled as
    a strict row spells them, and a frame with no shot inside the window
    left where it is (in the stack and in Tiled, out of the s-file).

    A run with no such stream is written **synchronously**, as before.  A
    run with one is written on a thread, because a stack may only be read
    after the plugin finalizes it and that happens at ``unstage``, after
    the stop document; a stack that never finalizes costs its own columns
    and a warning, never the s-file.

    Parameters
    ----------
    finalize_timeout :
        Seconds to wait for the plugin to finalize each stack.
    """

    def on_streams(
        self, start: dict[str, Any], stop: Document, run: _RunStreams
    ) -> None:
        """Write the files from the run's rows, joining the stacks if it has any."""
        stream = run.row_stream()
        rows = run.stream_rows(stream) if stream else []
        if not rows:
            logger.info(
                "scan %s: no per-shot rows in any stream, no scalar files "
                "(exit_status=%s)",
                start.get("scan_number"),
                stop.get("exit_status"),
            )
            return
        events = run.event_stream_records(start)
        stacks = run.datum_only_stacks()
        if not stacks:
            self._write(start, rows, events, run.drain_offsets)
            return
        logger.info(
            "scan %s: s-file from the %s rows joined to %d stack(s): %s",
            start.get("scan_number"),
            stream,
            len(stacks),
            ", ".join(s.data_key for s in stacks),
        )
        self.spawn(
            f"s-file[{start.get('scan_number')}]",
            self._join_and_write,
            dict(start),
            rows,
            stacks,
            dict(run.drain_offsets),
            self.finalize_timeout,
            events,
        )

    def _join_and_write(
        self,
        start: Mapping[str, Any],
        rows: list[dict[str, Any]],
        stacks: list[_Stack],
        drain_offsets: Mapping[str, float],
        finalize_timeout: float,
        events: "Sequence[DeviceRecords]" = (),
    ) -> None:
        from geecs_core.db.variable_types import LABVIEW_EPOCH_OFFSET
        from geecs_data_utils.io.scan_stack import (
            read_stack_attributes,
            stack_scalar_variables,
        )
        from geecs_data_utils.shot_join import frame_columns_from_attributes
        from geecs_data_utils.shot_table import device_records

        # One window for every stack, not one each: a gated run with three
        # cameras whose plugin never finalizes must not hold its s-file for
        # three timeouts.
        finalized = {
            stack.data_key: ready
            for stack, ready in zip(
                stacks,
                _await_all_finalized([s.path for s in stacks], finalize_timeout),
                strict=True,
            )
        }
        frames = []
        for stack in stacks:
            columns = None
            attributes: dict[str, Any] = {}
            if not finalized[stack.data_key]:
                logger.warning(
                    "scan %s: %s not finalized within %.0f s (%s) — its per-frame "
                    "columns are absent from the s-file",
                    start.get("scan_number"),
                    stack.data_key,
                    finalize_timeout,
                    stack.path,
                )
            else:
                attributes = read_stack_attributes(stack.path)
                columns = frame_columns_from_attributes(
                    stack.data_key,
                    attributes,
                    variables=stack_scalar_variables(stack.path),
                    labview_epoch_offset=LABVIEW_EPOCH_OFFSET,
                )
            if columns is None:
                continue
            if stack.width and stack.width < len(columns):
                # Frames the documents do not reference: a streamed camera
                # keeps writing between its last ``collect`` and its ``unstage``, and
                # a value in the s-file for a frame Tiled has no datum for would
                # make the two disagree about the same shot.
                logger.info(
                    "scan %s: %s has %d frame(s) on disk and %d referenced by its "
                    "datums; the join uses the referenced ones",
                    start.get("scan_number"),
                    stack.data_key,
                    len(columns),
                    stack.width,
                )
                columns = columns.truncated(stack.width)
            # The stack's counter and witness attribute too, as the stack check
            # reads them: the two must assign a shared-stamp group alike.
            frames.append(device_records(stack.stream, columns, start, attributes))
        self._write(start, rows, [*frames, *events], drain_offsets)

    @staticmethod
    def _write(
        start: Mapping[str, Any],
        rows: list[dict[str, Any]],
        frames: "Sequence[DeviceRecords]",
        drain_offsets: Mapping[str, float],
    ) -> None:
        import pandas as pd
        from geecs_data_utils import write_scalar_files

        result = write_scalar_files(
            dict(start),
            pd.DataFrame(rows),
            frames,
            drain_offsets=drain_offsets,
        )
        if result is None:
            logger.warning(
                "scan %s: scalar files not written", start.get("scan_number")
            )

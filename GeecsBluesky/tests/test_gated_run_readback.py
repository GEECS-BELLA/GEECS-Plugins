"""Runs recorded in the retired gated mode stay readable, byte for byte.

``fixtures/gated_runs/<run>/`` holds what two gated runs emitted before the
mode was deleted (2026-11-14), recorded from the live plan and callbacks:
every document in order (``documents.json``, temporary paths replaced by
``/data/TestExp``), the cameras' finalized stacks (``<key>.h5``, by stream in
``stacks.json``) and the s-file the live callback wrote from them
(``s1.txt``).  The offline re-export (``geecs_data_utils.tiled_export``)
reads the run back as Tiled would serve it and must write that s-file
again, byte for byte: the readers of old gated runs (``shots`` rows, a
datum-only ``primary``, the ``non_essential`` key's ``<name>_stream``) are
kept, unchanged, for exactly these runs.

- ``sweep_two_cameras``: a two-position sweep, two shots per step, two
  plugin cameras with different drain offsets, a gauge and a magnet on the
  ``shots`` rows.
- ``count_streamed_camera``: a three-shot count, one required plugin
  camera (``primary``) and one not required (``uc_b_stream``, named by
  ``non_essential``) that saw a trailing edge no row owns.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

pytest.importorskip("aioca")

from tests.test_callbacks import _Run

FIXTURES = Path(__file__).parent / "fixtures" / "gated_runs"


class _Recorded:
    """A recorded run's documents in the shape ``_Run`` reads (``docs`` by name)."""

    def __init__(self, folder: Path) -> None:
        self.docs: dict[str, list[dict]] = defaultdict(list)
        for name, doc in json.loads((folder / "documents.json").read_text()):
            self.docs[name].append(doc)


@pytest.mark.parametrize("run_name", ["sweep_two_cameras", "count_streamed_camera"])
def test_a_recorded_gated_run_re_exports_its_live_s_file_byte_for_byte(
    run_name: str, tmp_path: Path
) -> None:
    from geecs_data_utils.tiled_export import (
        read_drain_offsets,
        read_frame_columns,
        read_run_rows,
        write_scalar_files,
    )

    folder = FIXTURES / run_name
    recorded = _Recorded(folder)
    start = recorded.docs["start"][0]
    assert start["acquisition"] == "gated"
    stacks = {
        stream: {key: folder / name for key, name in by_key.items()}
        for stream, by_key in json.loads((folder / "stacks.json").read_text()).items()
    }
    run = _Run(recorded, stacks)
    rows, row_stream = read_run_rows(run)
    assert row_stream == "shots"
    scan_folder = tmp_path / "scans" / "Scan001"
    scan_folder.mkdir(parents=True)
    result = write_scalar_files(
        {**start, "scan_folder": str(scan_folder)},
        rows,
        read_frame_columns(run, row_stream),
        drain_offsets=read_drain_offsets(run),
    )
    assert result is not None
    assert Path(result[0]).read_bytes() == (folder / "s1.txt").read_bytes()

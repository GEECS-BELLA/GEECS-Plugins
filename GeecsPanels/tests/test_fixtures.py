"""The recorded device-stream fixtures stay valid frames: the web repository replays them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geecs_panels.contracts import DeviceFrame

FIXTURES = sorted((Path(__file__).parent.parent / "fixtures").glob("*.ndjson"))


def test_there_are_fixtures() -> None:
    assert FIXTURES


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_a_fixture_is_a_full_frame_then_deltas_one_seq_apart(path: Path) -> None:
    frames = [
        DeviceFrame.model_validate(json.loads(line))
        for line in path.read_text().splitlines()
    ]
    assert frames[0].full and frames[0].connected
    assert not any(f.full for f in frames[1:])
    assert [f.seq for f in frames] == list(
        range(frames[0].seq, frames[0].seq + len(frames))
    )

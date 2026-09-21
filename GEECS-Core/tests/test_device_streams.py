"""device_streams — the per-devicetype capture declaration, pinned against recorded DB rows."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from geecs_core.db import device_streams
from geecs_core.db.device_streams import (
    DEVICE_TYPE_STREAMS,
    DeviceTypeStreams,
    capture_variables,
    streams_for,
)
from geecs_core.db.variable_types import SKIP_VARTYPES, effective_vartype

#: ``devicetype_variable`` rows per devicetype, recorded from the GEECS DB by
#: ``scripts/record_devicetype_variables.py`` — NOT derived from the table
#: under test, so a name the table gets wrong is a name the fixture lacks.
FIXTURE: dict[str, list[dict]] = json.loads(
    (Path(__file__).parent / "fixtures" / "devicetype_variables.json").read_text()
)["devicetypes"]


def test_every_table_entry_has_a_recorded_fixture_and_vice_versa() -> None:
    """A new entry must come with recorded rows, or the parity test below is vacuous."""
    assert {k.lower() for k in FIXTURE} == set(DEVICE_TYPE_STREAMS)


@pytest.mark.parametrize("devicetype", sorted(FIXTURE))
def test_every_declared_name_is_a_db_variable_of_that_devicetype(
    devicetype: str, caplog: pytest.LogCaptureFixture
) -> None:
    entry = streams_for(devicetype)
    assert entry is not None
    with caplog.at_level(logging.WARNING, logger="geecs_core.db.device_streams"):
        captured = capture_variables(devicetype, FIXTURE[devicetype])
    assert captured is not None and len(captured) == len(entry.capture)
    assert not caplog.records, [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize("devicetype", sorted(FIXTURE))
def test_declared_names_are_non_scalar_variables(devicetype: str) -> None:
    """The declaration is about streams: every name it carries is image/1darray-typed."""
    by_name = {str(r["name"]).lower(): r for r in FIXTURE[devicetype]}
    entry = streams_for(devicetype)
    assert entry is not None
    for name in entry.capture:
        row = by_name[name.lower()]
        assert (
            effective_vartype(row["variabletype"], row["choices"]) in SKIP_VARTYPES
        ), name


def test_point_grey_declaration_equals_the_historic_one_image_default() -> None:
    """Strictly additive for the 40 Point Greys: the table names exactly the ``image`` variable."""
    assert capture_variables("Point Grey Camera", FIXTURE["Point Grey Camera"]) == [
        "image"
    ]


def test_the_frog_captures_its_trace_and_nothing_alphabetical() -> None:
    assert capture_variables("FROG", FIXTURE["FROG"]) == ["frogTrace"]


def test_capture_order_is_the_declared_order_not_the_db_order() -> None:
    """The first declared stream is the primary one (the ``hdf`` child, the bare stream key)."""
    rows = FIXTURE["MagSpecCamera"]
    assert capture_variables("MagSpecCamera", rows) == [
        "Image",
        "ImageInterp",
        "interpSpec",
        "interpDiv",
    ]
    assert capture_variables("MagSpecCamera", list(reversed(rows))) == [
        "Image",
        "ImageInterp",
        "interpSpec",
        "interpDiv",
    ]


def test_a_misspelled_declaration_is_dropped_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A typo'd allowlist entry must not arm anything, and must not fail silently."""
    monkeypatch.setitem(
        device_streams.DEVICE_TYPE_STREAMS,
        "frog",
        DeviceTypeStreams(capture=("frogTrase", "frogTrace")),
    )
    with caplog.at_level(logging.WARNING, logger="geecs_core.db.device_streams"):
        assert capture_variables("FROG", FIXTURE["FROG"]) == ["frogTrace"]
    assert len(caplog.records) == 1
    assert "frogTrase" in caplog.records[0].getMessage()
    assert "FROG" in caplog.records[0].getMessage()


def test_resolution_is_case_insensitive_and_returns_the_db_spelling() -> None:
    assert capture_variables("Point Grey Camera", [{"name": "IMAGE"}]) == ["IMAGE"]
    assert capture_variables("frog", [{"name": "FROGTRACE"}]) == ["FROGTRACE"]


def test_devicetype_lookup_normalises_case_and_whitespace() -> None:
    assert streams_for("  MagSpecCamera ") is streams_for("magspeccamera")
    assert streams_for("Point Grey Camera") is DEVICE_TYPE_STREAMS["point grey camera"]


def test_an_undeclared_devicetype_yields_none_not_empty() -> None:
    """None = "apply your default"; [] = "declared: capture nothing" (the Picoscope, for now)."""
    rows = [{"name": "Image"}, {"name": "SpotfieldImage"}]
    assert streams_for("ThorlabsWFS") is None
    assert capture_variables("ThorlabsWFS", rows) is None
    assert capture_variables("PicoscopeV2", FIXTURE["PicoscopeV2"]) == []

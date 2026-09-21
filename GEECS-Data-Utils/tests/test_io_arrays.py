"""geecs_data_utils.io.arrays — the three array wire shapes, pinned on captured payloads."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from geecs_data_utils.io import (
    decode_array_payload,
    decode_csv_values,
    decode_labview_waveform,
    decode_nested_pairs,
)

WIRE = Path(__file__).parent / "data" / "wire"


def _wire(name: str) -> bytes:
    return (WIRE / name).read_bytes()


# ------------------------------------------------------------ nested pairs
def test_interp_spec_is_285_rows_of_axis_and_value() -> None:
    spec = decode_nested_pairs(_wire("magspec_interpSpec.bin"))
    assert spec.shape == (285, 2) and spec.dtype == np.float64
    assert spec[0].tolist() == pytest.approx([51.55832, 6487.3])
    assert spec[-1, 0] == pytest.approx(122.5583, abs=1e-3)
    steps = np.diff(spec[:, 0])
    assert np.all(steps > 0) and steps.mean() == pytest.approx(0.25, abs=1e-4)


def test_interp_div_is_189_rows_on_the_angle_axis() -> None:
    div = decode_nested_pairs(_wire("magspec_interpDiv.bin"))
    assert div.shape == (189, 2)
    assert div[0, 0] == pytest.approx(-23.5) and div[-1, 0] == pytest.approx(23.5)


def test_a_single_row_is_a_valid_value_not_an_error() -> None:
    """The magnet-off default of interpSpec is 1 x 2; it must decode, never raise."""
    assert decode_nested_pairs("[[5.155832E+1,0.000000E+0]]").shape == (1, 2)


@pytest.mark.parametrize(
    "bad",
    ["[[1,2],[3]]", "[[1,2,3]]", "[]", "[[a,b]]", "1,2,3", "[[1,2]", ""],
)
def test_malformed_pairs_raise(bad: str) -> None:
    with pytest.raises(ValueError):
        decode_nested_pairs(bad)


# ------------------------------------------------------------------- csv
def test_energy_axis_csv_matches_interp_spec_column_0() -> None:
    axis = decode_csv_values(_wire("magspec_EnergyAxis.bin"))
    spec = decode_nested_pairs(_wire("magspec_interpSpec.bin"))
    assert axis.shape == (285,)
    np.testing.assert_allclose(axis, spec[:, 0])


def test_angle_axis_csv_is_the_camera_geometry() -> None:
    axis = decode_csv_values(_wire("magspec_AngleAxis.bin"))
    assert axis.shape == (189,)
    np.testing.assert_allclose(np.diff(axis), 0.25, atol=1e-6)


@pytest.mark.parametrize("bad", ["", "   \r\n", "1,two,3"])
def test_malformed_csv_raises(bad: str) -> None:
    with pytest.raises(ValueError):
        decode_csv_values(bad)


# -------------------------------------------------------------- waveform
def test_picoscope_trace_decodes_to_volts_with_its_axis() -> None:
    wf = decode_labview_waveform(_wire("picoscope_scopeTrace_Channel0.bin"))
    assert wf.kind == "waveform"
    assert wf.values.shape == (3000,) and wf.values.dtype == np.float64
    assert wf.attributes["samples"] == 3000
    assert wf.attributes["dx"] == pytest.approx(4e-9)
    assert wf.attributes["x0"] == 0.0
    assert wf.attributes["name"] == "Channel0"
    # A 100 mV range with no beam: every sample well inside the range.
    assert np.all(np.abs(wf.values) < 0.1)
    assert wf.values.std() > 0  # real samples, not a flat line


def _waveform(
    raw: list[int], *, x0=0.0, dx=4e-9, offset=-0.5, gain=1e-3, name="ChanX"
) -> bytes:
    header = f"{len(raw)}.000000,{x0:.12f},{dx:.12f},{offset:.12f},{gain:.12f},{name}|".encode()
    return header + struct.pack(">I", len(raw)) + struct.pack(f">{len(raw)}h", *raw)


def test_waveform_scaling_and_the_str_transport_form() -> None:
    blob = _waveform([0, 1000, -1000])
    wf = decode_labview_waveform(blob)
    np.testing.assert_allclose(wf.values, [-0.5, 0.5, -1.5])
    # The transport hands values over as latin-1 str; bytes round-trip exactly.
    assert (
        decode_labview_waveform(blob.decode("latin-1")).values.tolist()
        == wf.values.tolist()
    )
    assert wf.attributes["name"] == "ChanX"


def test_waveform_two_samples_is_not_truncation() -> None:
    """The DaqPad publishes n=2 by configuration; a short record is a record."""
    assert decode_labview_waveform(_waveform([7, 8])).values.shape == (2,)


def test_waveform_refuses_a_payload_it_cannot_account_for() -> None:
    good = _waveform([1, 2, 3])
    with pytest.raises(ValueError, match="expected"):
        decode_labview_waveform(good[:-1])  # one byte short
    with pytest.raises(ValueError, match="expected"):
        decode_labview_waveform(good + b"\x00")  # one byte over
    header_end = good.index(b"|") + 1
    with pytest.raises(ValueError, match="count field"):
        decode_labview_waveform(
            good[:header_end] + struct.pack(">I", 2) + good[header_end + 4 :]
        )
    with pytest.raises(ValueError, match="header"):
        decode_labview_waveform(b"1,2,3|" + b"\x00" * 6)


# -------------------------------------------------------------- dispatch
def test_dispatch_sniffs_the_payload_not_the_devicetype() -> None:
    assert decode_array_payload(_wire("magspec_interpSpec.bin")).kind == "pairs"
    assert decode_array_payload(_wire("magspec_EnergyAxis.bin")).kind == "csv"
    assert (
        decode_array_payload(_wire("picoscope_scopeTrace_Channel0.bin")).kind
        == "waveform"
    )
    assert decode_array_payload("  [[1,2]]").values.shape == (1, 2)
    assert decode_array_payload("1,2,3\r\n").values.tolist() == [1.0, 2.0, 3.0]


def test_dispatch_never_returns_a_partial_decode() -> None:
    with pytest.raises(ValueError):
        decode_array_payload("[[1,2],[3]]")
    with pytest.raises(ValueError):
        decode_array_payload(_waveform([1, 2])[:-1])

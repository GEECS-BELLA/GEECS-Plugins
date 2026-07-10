"""Tests for safe_name and device legacy header maps."""

from __future__ import annotations

import pytest

from geecs_bluesky.utils import safe_name

pytest.importorskip("aioca")  # devices are CA-backed

from geecs_bluesky.devices.ca import (  # noqa: E402
    CaGenericDetector,
    CaMotor,
    CaSnapshotReadable,
)


def test_safe_name_mangles_and_lowercases() -> None:
    # Each non-alphanumeric char becomes "_"; adjacent specials (space + "(")
    # collapse only via strip, so "Wavelength (nm)" keeps a double underscore.
    assert safe_name("Wavelength (nm)") == "wavelength__nm"
    assert safe_name("Position.Axis 1") == "position_axis_1"
    assert safe_name("") == "var"


def test_generic_detector_column_headers() -> None:
    det = CaGenericDetector(
        "UC_Wavemeter",
        ["Wavelength (nm)", "Power (mW)"],
        name="wavemeter",
    )
    assert det._column_headers == {
        "wavemeter-wavelength__nm": "UC_Wavemeter Wavelength (nm)",
        "wavemeter-power__mw": "UC_Wavemeter Power (mW)",
    }


def test_snapshot_column_headers() -> None:
    snap = CaSnapshotReadable("U_Stage", ["Position"], name="stage")
    assert snap._column_headers == {"stage-position": "U_Stage Position"}


def test_motor_column_headers_uses_position_attr() -> None:
    motor = CaMotor("U_ESP_JetXYZ", "Position.Axis 1", name="jet_x")
    assert motor._column_headers == {"jet_x-position": "U_ESP_JetXYZ Position.Axis 1"}

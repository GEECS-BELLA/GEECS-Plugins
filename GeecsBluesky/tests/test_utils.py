"""Tests for safe_name / build_signal_attrs and device legacy header maps."""

from __future__ import annotations

from geecs_bluesky.devices.generic_detector import GeecsGenericDetector
from geecs_bluesky.devices.motor import GeecsMotor
from geecs_bluesky.devices.snapshot import GeecsSnapshotReadable
from geecs_bluesky.utils import build_signal_attrs, safe_name


def test_safe_name_mangles_and_lowercases() -> None:
    # Each non-alphanumeric char becomes "_"; adjacent specials (space + "(")
    # collapse only via strip, so "Wavelength (nm)" keeps a double underscore.
    assert safe_name("Wavelength (nm)") == "wavelength__nm"
    assert safe_name("Position.Axis 1") == "position_axis_1"
    assert safe_name("") == "var"


def test_build_signal_attrs_disambiguates_collisions() -> None:
    # Two variables that mangle to the same attr get _2 appended in order.
    pairs = build_signal_attrs(["A (x)", "A [x]", "B"])
    attrs = [attr for attr, _ in pairs]
    assert attrs == ["a__x", "a__x_2", "b"]
    # The original variable name is preserved alongside each attr.
    assert pairs == [("a__x", "A (x)"), ("a__x_2", "A [x]"), ("b", "B")]


def test_generic_detector_column_headers() -> None:
    det = GeecsGenericDetector(
        "UC_Wavemeter",
        ["Wavelength (nm)", "Power (mW)"],
        "127.0.0.1",
        0,
        name="wavemeter",
    )
    assert det._column_headers == {
        "wavemeter-wavelength__nm": "UC_Wavemeter Wavelength (nm)",
        "wavemeter-power__mw": "UC_Wavemeter Power (mW)",
    }


def test_snapshot_column_headers() -> None:
    snap = GeecsSnapshotReadable("U_Stage", ["Position"], "127.0.0.1", 0, name="stage")
    assert snap._column_headers == {"stage-position": "U_Stage Position"}


def test_motor_column_headers_uses_position_attr() -> None:
    motor = GeecsMotor("U_ESP_JetXYZ", "Position.Axis 1", "127.0.0.1", 0, name="jet_x")
    assert motor._column_headers == {"jet_x-position": "U_ESP_JetXYZ Position.Axis 1"}

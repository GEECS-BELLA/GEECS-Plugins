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
    # Runs of non-alphanumeric chars collapse to one "_" (the shared
    # pv_naming.normalize_component policy), lowercase.
    assert safe_name("Wavelength (nm)") == "wavelength_nm"
    assert safe_name("Position.Axis 1") == "position_axis_1"
    assert safe_name("") == "var"


def test_safe_name_agrees_with_the_pv_naming_contract() -> None:
    """One lossy encoding, not two: column components == PV components.

    ``safe_name`` must stay a delegation to the gateway's
    ``normalize_component`` (plus the non-empty fallback) so a GEECS name
    mangles identically into an event column and a PV.
    """
    from geecs_ca_gateway.pv_naming import normalize_component

    for raw in ("Position.Axis 1", "Amplitude.Ch AB", "Beam Current (A)", "ypos"):
        assert safe_name(raw) == normalize_component(raw)


def test_generic_detector_column_headers() -> None:
    det = CaGenericDetector(
        "UC_Wavemeter",
        ["Wavelength (nm)", "Power (mW)"],
        name="wavemeter",
    )
    assert det._column_headers == {
        "wavemeter-wavelength_nm": "UC_Wavemeter Wavelength (nm)",
        "wavemeter-power_mw": "UC_Wavemeter Power (mW)",
    }


def test_snapshot_column_headers() -> None:
    snap = CaSnapshotReadable("U_Stage", ["Position"], name="stage")
    assert snap._column_headers == {"stage-position": "U_Stage Position"}


def test_motor_column_headers_uses_position_attr() -> None:
    motor = CaMotor("U_ESP_JetXYZ", "Position.Axis 1", name="jet_x")
    assert motor._column_headers == {"jet_x-position": "U_ESP_JetXYZ Position.Axis 1"}

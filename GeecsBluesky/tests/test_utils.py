"""Tests for safe_name and device legacy header maps."""

from __future__ import annotations

import pytest

from geecs_bluesky.utils import safe_name

pytest.importorskip("aioca")  # devices are CA-backed

from geecs_bluesky.devices.ca import CaMotor, CaSnapshotReadable  # noqa: E402
from geecs_bluesky.devices.detector import GeecsDetector  # noqa: E402


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
    from geecs_core.pv_naming import normalize_component

    for raw in ("Position.Axis 1", "Amplitude.Ch AB", "Beam Current (A)", "ypos"):
        assert safe_name(raw) == normalize_component(raw)


def test_detector_column_headers() -> None:
    det = GeecsDetector(
        "UC_Wavemeter",
        ["Wavelength (nm)", "Power (mW)"],
        name="wavemeter",
    )
    assert det._column_headers == {
        "wavemeter-wavelength_nm": "UC_Wavemeter Wavelength (nm)",
        "wavemeter-power_mw": "UC_Wavemeter Power (mW)",
        "wavemeter-acq_timestamp": "UC_Wavemeter acq_timestamp",
    }


def test_snapshot_column_headers() -> None:
    snap = CaSnapshotReadable("U_Stage", ["Position"], name="stage")
    assert snap._column_headers == {"stage-position": "U_Stage Position"}


def test_motor_column_headers_uses_position_attr() -> None:
    motor = CaMotor("U_ESP_JetXYZ", "Position.Axis 1", name="jet_x")
    assert motor._column_headers == {"jet_x-position": "U_ESP_JetXYZ Position.Axis 1"}


def test_resolve_annotations_refuses_an_unmapped_parameter() -> None:
    """Silence would cost the manager's validation for that argument.

    ``resolve_annotations`` rewrites a registered plan's ``__signature__``
    from a name-keyed mapping. Substituting ``Parameter.empty`` for anything
    the mapping forgets is invisible — the plan still registers, still
    submits, and just loses the manager's type validation and device-name
    conversion for that argument. A parameter added later would inherit that
    silently, so an omission has to be an error (review of #861, finding 7).
    """
    import pytest

    from geecs_bluesky.utils import resolve_annotations

    def plan(detectors, *, shots: int = 1):
        yield None

    with pytest.raises(ValueError, match="no resolved annotation given for shots"):
        resolve_annotations(plan, {"detectors": list})

    # Mapping to None is how you drop one deliberately.
    resolved = resolve_annotations(plan, {"detectors": list, "shots": None})
    import inspect

    assert (
        inspect.signature(resolved).parameters["shots"].annotation
        is inspect.Parameter.empty
    )

"""Unit tests for PV name mapping (no network)."""

from __future__ import annotations

from geecs_ca_gateway.config import DeviceSpec, VariableSpec
from geecs_ca_gateway.naming import normalize_pv_component, pv_name


def test_devicevar_maps_one_to_one() -> None:
    """``deviceName:variable`` is already valid CA — passes through unchanged."""
    assert pv_name("U_ESP_JetXYZ", "Position") == "U_ESP_JetXYZ:Position"


def test_spaces_collapse_to_underscores() -> None:
    """GEECS variable names with spaces become CA-safe underscores."""
    assert normalize_pv_component("Jet X pos") == "Jet_X_pos"
    assert normalize_pv_component("  padded  name  ") == "padded_name"


def test_variable_spec_default_suffix_normalizes() -> None:
    """A VariableSpec with no explicit ``pv`` normalizes the GEECS var name."""
    spec = VariableSpec(geecs_var="Beam Current")
    assert spec.pv_suffix == "Beam_Current"


def test_variable_spec_explicit_pv_wins() -> None:
    """An explicit ``pv`` overrides the derived suffix."""
    spec = VariableSpec(geecs_var="acq_timestamp", pv="AcqTime")
    assert spec.pv_suffix == "AcqTime"


def test_device_prefix_defaults_to_name() -> None:
    """``pv_prefix`` falls back to the device name when unset."""
    dev = DeviceSpec(name="U_HexapodXYZ", host="127.0.0.1", port=1)
    assert dev.pv_prefix == "U_HexapodXYZ"

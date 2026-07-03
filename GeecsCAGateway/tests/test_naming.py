"""Unit tests for PV name mapping (no network)."""

from __future__ import annotations

from geecs_ca_gateway.config import DeviceSpec, VariableSpec
from geecs_ca_gateway.naming import normalize_pv_component


def test_spaces_collapse_to_underscores() -> None:
    """GEECS variable names with spaces become CA-safe underscores."""
    assert normalize_pv_component("Jet X pos") == "Jet_X_pos"
    assert normalize_pv_component("  padded  name  ") == "padded_name"


def test_dot_becomes_underscore() -> None:
    """The dot is critical: EPICS reads it as the record/field separator."""
    assert normalize_pv_component("Trigger.Source") == "Trigger_Source"


def test_mixed_bad_chars_collapse() -> None:
    """Dashes, parens, and other non-[A-Za-z0-9_] chars map to single ``_``."""
    assert normalize_pv_component("Beam-Current (A)") == "Beam_Current_A"


def test_pv_name_for_with_experiment_prefix() -> None:
    """Experiment prefix yields ``Experiment:Device:Variable``."""
    dev = DeviceSpec(name="U_S1H", host="h", port=1, experiment="Undulator")
    assert (
        dev.pv_name_for(VariableSpec(geecs_var="Current")) == "Undulator:U_S1H:Current"
    )


def test_pv_name_for_without_experiment() -> None:
    """No experiment prefix yields ``Device:Variable``."""
    dev = DeviceSpec(name="U_S1H", host="h", port=1)
    assert dev.pv_name_for(VariableSpec(geecs_var="Current")) == "U_S1H:Current"


def test_pv_name_for_normalizes_variable_dot() -> None:
    """A dotted GEECS variable is CA-safe in the full PV name."""
    dev = DeviceSpec(name="U_DG645", host="h", port=1)
    assert (
        dev.pv_name_for(VariableSpec(geecs_var="Trigger.Source"))
        == "U_DG645:Trigger_Source"
    )


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

"""Unit tests for full PV name assembly via DeviceSpec/VariableSpec (no network).

The naming *policy* primitives (normalize/join/setpoint) live in
``geecs_core.pv_naming`` and are pinned by ``GEECS-Core/tests/test_pv_naming.py``;
this file covers the gateway-side assembly of complete PV names.
"""

from __future__ import annotations

from geecs_ca_gateway.config import DeviceSpec, VariableSpec


def test_pv_name_for_with_experiment_prefix() -> None:
    """Experiment prefix yields ``Experiment:Device:Variable``."""
    dev = DeviceSpec(name="U_S1H", host="h", port=1, experiment="Undulator")
    assert (
        dev.pv_name_for(VariableSpec(geecs_var="Current")) == "undulator:u_s1h:current"
    )


def test_pv_name_for_without_experiment() -> None:
    """No experiment prefix yields ``Device:Variable``."""
    dev = DeviceSpec(name="U_S1H", host="h", port=1)
    assert dev.pv_name_for(VariableSpec(geecs_var="Current")) == "u_s1h:current"


def test_pv_name_for_normalizes_variable_dot() -> None:
    """A dotted GEECS variable is CA-safe in the full PV name."""
    dev = DeviceSpec(name="U_DG645", host="h", port=1)
    assert (
        dev.pv_name_for(VariableSpec(geecs_var="Trigger.Source"))
        == "u_dg645:trigger_source"
    )


def test_variable_spec_default_suffix_normalizes() -> None:
    """A VariableSpec with no explicit ``pv`` normalizes the GEECS var name."""
    spec = VariableSpec(geecs_var="Beam Current")
    assert spec.pv_suffix == "beam_current"


def test_variable_spec_explicit_pv_wins() -> None:
    """An explicit ``pv`` overrides the derived suffix."""
    spec = VariableSpec(geecs_var="acq_timestamp", pv="AcqTime")
    assert spec.pv_suffix == "acqtime"


def test_device_prefix_defaults_to_name() -> None:
    """``pv_prefix`` falls back to the device name when unset."""
    dev = DeviceSpec(name="U_HexapodXYZ", host="127.0.0.1", port=1)
    assert dev.pv_prefix == "U_HexapodXYZ"


def test_full_pv_names_are_lowercased() -> None:
    """Case carries no meaning in assembled PV names."""
    dev = DeviceSpec(name="U_S1H", host="h", port=1, experiment="Undulator")
    pv = dev.pv_name_for(VariableSpec(geecs_var="Trigger.Source"))
    assert pv == pv.lower()

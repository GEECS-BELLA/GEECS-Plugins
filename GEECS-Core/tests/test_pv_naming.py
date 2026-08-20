"""Unit tests for the shared PV naming policy (no network).

Full PV assembly (experiment:device:variable) is the gateway's concern and is
tested with ``DeviceSpec`` in ``GeecsCAGateway/tests/test_naming.py``; this
file pins the policy primitives every producer and consumer shares.
"""

from __future__ import annotations

from geecs_core.pv_naming import normalize_component, pv_name, setpoint_pv


def test_spaces_collapse_to_underscores() -> None:
    """GEECS variable names with spaces become CA-safe underscores."""
    assert normalize_component("Jet X pos") == "jet_x_pos"
    assert normalize_component("  padded  name  ") == "padded_name"


def test_dot_becomes_underscore() -> None:
    """The dot is critical: EPICS reads it as the record/field separator."""
    assert normalize_component("Trigger.Source") == "trigger_source"


def test_mixed_bad_chars_collapse() -> None:
    """Dashes, parens, and other non-[A-Za-z0-9_] chars map to single ``_``."""
    assert normalize_component("Beam-Current (A)") == "beam_current_a"


def test_components_are_lowercased() -> None:
    """Case carries no meaning: all derived PV components are lowercase."""
    assert normalize_component("MiXeD Case") == "mixed_case"


def test_pv_name_joins_and_drops_empty_parts() -> None:
    """``pv_name`` colon-joins normalized parts, skipping empty ones."""
    assert pv_name("Undulator", "U_S1H", "Current") == "undulator:u_s1h:current"
    assert pv_name("", "U_S1H", "Current") == "u_s1h:current"


def test_setpoint_pv_appends_suffix() -> None:
    """Setpoint PVs are the readback name plus ``:SP``."""
    assert setpoint_pv("u_s1h:current") == "u_s1h:current:SP"

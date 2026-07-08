"""ScanVariables — the catalog of things a scan is allowed to sweep.

This file gives every scannable knob a friendly name — "jet_z", "EMQ1
Current" — and says which device variable it actually moves.  You would edit
it to expose a new stage or power supply as a scan variable, or to define a
*pseudo* variable that moves several devices together from one number (for
example a jet position that also tracks a probe camera stage).

Developer notes
---------------
Successor of the legacy pair ``scan_devices.yaml`` (``single_scan_devices:``
friendly-name → ``Device:Variable`` strings) + ``composite_variables.yaml``
(``composite_variables:`` with per-component numexpr ``relation`` strings and
an ``absolute``/``relative`` mode).

- ``kind: setpoint`` — plain blocking set through the device's setpoint;
  matches legacy set-and-wait semantics and is the converter default.
- ``kind: motor`` — blocking move *plus* readback-tolerance polling; an
  explicit opt-in for real positioners (renders to ``CaMotor``).
- ``kind: pseudo`` — a pseudo-positioner: one scanned number fanned out to
  several targets through ``forward`` expressions.  v1 keeps the legacy
  numexpr semantics **verbatim**: each expression is written in terms of
  ``composite_var`` (the scanned value), and ``mode`` keeps the legacy
  meaning — ``absolute`` sets each target to its expression's value,
  ``relative`` offsets each target from its position at scan start.

Limits, units, and tolerances deliberately do **not** live here — device
facts belong below the configs (gateway PV metadata; vision doc §4.3).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Union

from pydantic import Field, field_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel


def _validate_target(value: str) -> str:
    """Check that a target string looks like ``Device:Variable``.

    Parameters
    ----------
    value : str
        Candidate target string.

    Returns
    -------
    str
        The validated target string, unchanged.

    Raises
    ------
    ValueError
        If the string has no ``:`` separator or an empty device/variable part.
    """
    device, sep, variable = value.partition(":")
    if not sep or not device.strip() or not variable.strip():
        raise ValueError(
            f"Target {value!r} must look like 'Device:Variable', e.g. "
            "'U_ESP_JetXYZ:Position.Axis 3'."
        )
    return value


class CompositeMode(str, Enum):
    """How a pseudo variable's targets interpret the scanned value.

    Attributes
    ----------
    ABSOLUTE : str
        Each target is set to exactly what its expression computes.
    RELATIVE : str
        Each target is offset from where it was when the scan started, by
        the amount its expression computes.
    """

    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class ScanVariable(SchemaModel):
    """A friendly name for one device variable you can scan.

    Points a human-readable name at the ``Device:Variable`` it moves, and
    says whether it is a plain setpoint or a motor with position readback.
    """

    target: str = Field(
        description=(
            "The device variable this name moves, written as "
            "'Device:Variable', e.g. 'U_ESP_JetXYZ:Position.Axis 3'."
        )
    )
    kind: Literal["motor", "setpoint"] = Field(
        "setpoint",
        description=(
            "'setpoint' = write the value and wait for the device to accept "
            "it (the default). 'motor' = additionally poll the readback until "
            "the device reports it arrived — use for real positioners."
        ),
    )

    _check_target = field_validator("target")(_validate_target)


class PseudoTarget(SchemaModel):
    """One device a pseudo variable moves, and the formula for its value.

    The ``forward`` expression computes this target's setting from the single
    scanned number, which appears in the formula as ``composite_var``.
    """

    target: str = Field(
        description=(
            "The device variable to move, written as 'Device:Variable', "
            "e.g. 'U_S1H:Current'."
        )
    )
    forward: str = Field(
        min_length=1,
        description=(
            "Formula for this device's value in terms of the scanned number, "
            "written with 'composite_var' as the scanned value — e.g. "
            "'composite_var * -2' or '8.5 + (composite_var-10)*2.5'."
        ),
    )

    _check_target = field_validator("target")(_validate_target)


class PseudoScanVariable(SchemaModel):
    """A friendly name that moves several devices together from one number.

    Scanning this variable sweeps one number; each listed target follows it
    through its own formula.  Use it for coordinated moves — e.g. moving a
    gas jet while a probe camera stage tracks it.

    Notes
    -----
    Successor of a legacy ``composite_variables.yaml`` entry.  ``mode`` and
    the numexpr expressions keep their legacy semantics verbatim in v1.
    ``inverse`` (a readback formula recovering the scanned number from the
    first target's position) has no legacy counterpart and is optional.
    """

    kind: Literal["pseudo"] = Field(
        description="Variable type. 'pseudo' moves several devices from one number."
    )
    targets: list[PseudoTarget] = Field(
        min_length=1,
        description="The devices this variable moves, each with its own formula.",
    )
    mode: CompositeMode = Field(
        description=(
            "'absolute' = each device goes exactly where its formula says. "
            "'relative' = each device is offset from where it was when the "
            "scan started."
        )
    )
    inverse: Optional[str] = Field(
        None,
        description=(
            "Optional formula recovering the scanned number from the first "
            "target's readback. Leave unset if you don't need a readback for "
            "this variable."
        ),
    )


# Plain (smart) union rather than a discriminated one: 'kind' has a default
# on ScanVariable ('setpoint' may be omitted in YAML), which tagged unions do
# not allow. The two shapes are disjoint under extra="forbid" ('target' vs
# 'targets'), so smart-union resolution is unambiguous.
ScanVariableSpec = Union[ScanVariable, PseudoScanVariable]


class ScanVariables(VersionedSchemaModel):
    """The experiment's catalog of scannable variables, keyed by friendly name.

    Everything a scan's "Variable" dropdown offers comes from this file.
    Edit it to expose a new device knob under a readable name, or to define
    a coordinated multi-device (pseudo) variable.

    Notes
    -----
    Replaces the legacy ``scan_devices.yaml`` + ``composite_variables.yaml``
    pair with one document.  Simple entries default to ``kind: setpoint``;
    entries with ``kind: pseudo`` carry targets and formulas.
    """

    variables: dict[str, ScanVariableSpec] = Field(
        description=(
            "All scannable variables, keyed by the friendly name shown when "
            "setting up a scan."
        )
    )

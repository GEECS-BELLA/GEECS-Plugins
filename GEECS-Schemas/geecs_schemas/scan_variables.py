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
  several components through ``forward`` expressions.  v1 keeps the legacy
  numexpr semantics **verbatim**: each expression is written in terms of
  ``composite_var`` (the scanned value), and ``mode`` keeps the legacy
  meaning — ``absolute`` sets each component to its expression's value,
  ``relative`` offsets each component from its position at scan start.

Limits, units, and tolerances deliberately do **not** live here — device
facts belong below the configs (gateway PV metadata; vision doc §4.3).

The optional ``confirm`` field on a simple :class:`ScanVariable` is the one
place a scan variable names a *second* device variable: it covers the case
where the variable you *set* is not the variable that *measures* the result
(the "topology C" gap — see :class:`ScanVariable`).  It is declared but not
yet enforced by the engine in v1.
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


def _validate_optional_target(value: Optional[str]) -> Optional[str]:
    """Validate an optional ``Device:Variable`` string, passing ``None`` through.

    Parameters
    ----------
    value : str or None
        Candidate target string, or ``None`` when the field is unset.

    Returns
    -------
    str or None
        The validated string, or ``None`` unchanged.
    """
    return value if value is None else _validate_target(value)


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

    Notes
    -----
    Most scan variables are the simple case: the variable you *set* is also
    the variable that *measures* the result (a stage's ``Position.Axis N``, a
    steering-magnet ``Current``), so completion can be judged on the same
    name.  For those, leave ``confirm`` unset.

    Some devices split the two — you set one variable but a *different*
    variable reports the physical truth.  The production example is the EMQ
    triplet: the catalog's "EMQ1 Current" writes
    ``U_EMQTripletBipolar:Current_Limit.Ch1`` (a software limit), while the
    measured current is a separate variable.  GEECS binds its set-completion
    tolerance to the *written* variable, so such a set "confirms" against a
    value that is trivially correct and the real current is never checked.
    ``confirm`` names the measured variable so this split is at least
    **visible** in the config; enforcing it (a device whose ``set()``
    completes on the confirming variable) is a later engine milestone.  The
    match *tolerance* is a device fact and stays below the configs (the DB /
    gateway PV metadata), not here.
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
    confirm: Optional[str] = Field(
        None,
        description=(
            "Optional 'Device:Variable' that *measures* the result when it "
            "differs from the variable being set — e.g. set a supply's "
            "current limit but confirm on its measured current. Leave unset "
            "when the set variable is also the readback (the common case). "
            "Declared but not yet enforced by the engine in v1."
        ),
    )

    _check_target = field_validator("target")(_validate_target)
    _check_confirm = field_validator("confirm")(_validate_optional_target)


class PseudoComponent(SchemaModel):
    """One device a pseudo variable moves, and the formula for its value.

    The ``forward`` expression computes this component's setting from the
    single scanned number, which appears in the formula as ``composite_var``.
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
    targets: list[PseudoComponent] = Field(
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

"""Snapshot — a beamline-level set of setpoints across devices and IOCs (``Planning/epics_estate.md`` §2.6).

The LabVIEW Preset Manager's job, which neither autosave (one IOC's last
state, its configMenu named sets) nor the archiver (history) covers: a
named set of every writable setpoint of a scope's devices, with the
readbacks beside them at the time, who took it and why. The PV bridge
captures, diffs and restores one; the restore semantics (a tick per row,
unreachable devices skipped, motion rows confirmed and waited on) are the
bridge's (``geecs_panels.bridge.snapshots``), not this document's.

**v0 keeps one authoritative copy**, a JSON file per snapshot in the
bridge's ``StateDirectory``. The ``{experiment}/snapshots/`` mirror on the
data tree, and golden snapshots promoted into the configs repository by
PR, come later; ``golden`` is the mark that promotion will read.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from geecs_schemas._base import SchemaModel, VersionedSchemaModel
from geecs_schemas.epics.family import RoleName, refuse_duplicates
from geecs_schemas.epics.panel_devices import DeviceName

#: A value a row keeps: a number, a flag, text, or an enum's state name.
SnapshotValue = Union[StrictBool, StrictInt, StrictFloat, StrictStr]


class SnapshotScope(SchemaModel):
    """What a snapshot covers: a list of devices, or a composed panel's devices.

    Exactly one of the two. A panel's devices are those its device and
    component widgets name, resolved when the snapshot is taken (the rows
    record them).
    """

    devices: Optional[list[DeviceName]] = Field(
        None,
        min_length=1,
        description="The devices whose writable roles are kept, by GEECS name.",
    )
    panel: Optional[str] = Field(
        None,
        min_length=1,
        pattern=r"^[^/\s]+$",
        description="A composed panel's name (its file stem under <experiment>/panels/).",
    )

    @field_validator("devices")
    @classmethod
    def _each_once(cls, devices: Optional[list[str]]) -> Optional[list[str]]:
        if devices is not None:
            refuse_duplicates("devices", devices)
        return devices

    @model_validator(mode="after")
    def _one_of(self) -> "SnapshotScope":
        if (self.devices is None) == (self.panel is None):
            raise ValueError("a scope names devices or a panel, exactly one")
        return self


class SnapshotRow(SchemaModel):
    """One writable role of one device (and axis) as it was: its setpoint, and its readback beside it."""

    device: DeviceName = Field(description="The device's GEECS name.")
    axis: Optional[str] = Field(
        None,
        min_length=1,
        description="The axis, for a per-axis family (the motor record); else absent.",
    )
    role: RoleName = Field(description="The role (a writable setpoint, enum or bit).")
    setpoint: SnapshotValue = Field(
        description="The role's value at capture: what a restore writes back."
    )
    readback: Optional[SnapshotValue] = Field(
        None,
        description=(
            "The value of the role's readback (its family's 'readback') at capture; "
            "null when the family names none or it was not read."
        ),
    )


class Snapshot(VersionedSchemaModel):
    """A named set of setpoints across devices, with the readbacks at the time: what a restore returns the beamline to.

    Taken by the PV bridge from its live device streams (every writable
    setpoint, enum and bit role of the scope's devices), never written by
    hand; kept as one JSON file per snapshot in the bridge's state
    directory.
    """

    schema_version: Literal[1] = Field(
        1,
        description=(
            "Format version of this config file. Leave at 1 — tools update "
            "this automatically when the file format changes."
        ),
    )
    name: str = Field(
        min_length=1,
        max_length=120,
        description="What the operator called it: 'chicane tuned, 25 pC'.",
    )
    created: float = Field(description="When it was taken, Unix seconds.")
    author: str = Field(
        min_length=1,
        max_length=60,
        description="Who took it (the operator's identity, as a put journals it).",
    )
    note: str = Field("", max_length=2000, description="Why, in a line or a few.")
    golden: bool = Field(
        False,
        description=(
            "Marked as a reference setup (a later release promotes golden "
            "snapshots into the configs repository)."
        ),
    )
    scope: SnapshotScope = Field(description="The devices it covers, or the panel.")
    rows: list[SnapshotRow] = Field(
        min_length=1,
        description="Every writable role of the scope's devices, per axis, as it was.",
    )

    @model_validator(mode="after")
    def _each_row_once(self) -> "Snapshot":
        refuse_duplicates(
            "rows", [f"{r.device}/{r.axis or ''}/{r.role}" for r in self.rows]
        )
        return self

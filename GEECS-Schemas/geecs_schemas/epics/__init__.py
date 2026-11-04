"""NativeDevices — the experiment's devices served by native EPICS IOCs.

A device can be served by its own EPICS IOC (geecs-epics) instead of a
LabVIEW device app.  This registry names those devices, per experiment, in
the configs repository::

    scanner_configs/experiments/<Experiment>/native_devices.yaml

One owner per device: a name listed here must not also be an enabled device
of the experiment in the GEECS DB (the scan worker refuses to open
otherwise).  The registry carries names, kind and the kind's few options
only — the PV prefix is derived from the experiment and device name
(``<experiment>:<device>:``, GEECS normalization), the record names under it
are the kind's own, and hosts and paths are facility values with a home of
their own.  An absent file means the experiment has no native devices.

Four kinds so far: ``aravis_camera`` (an areaDetector camera, a detector
of the scan), ``epics_motor`` (one or more EPICS motor records, scan axes),
``setpoint_device`` (a supply's plain records: settables confirmed on
their readbacks, extra readbacks, status bits) and ``witness`` (a hardware
count of one trigger domain's edges, which confirms every strict fire).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Annotated, Literal, Optional, Union

from pydantic import (
    Field,
    PositiveFloat,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    field_validator,
    model_serializer,
    model_validator,
)

from geecs_schemas._base import SchemaModel, VersionedSchemaModel
from geecs_schemas.epics.family import (
    REGISTRY_KIND_FAMILIES,
    FamilyFile,
    TemplateName,
    families_from,
)

#: The registry's file name in the experiment's configs folder.
NATIVE_DEVICES_FILENAME = "native_devices.yaml"


class _NativeEntry(SchemaModel):
    """What every registry entry carries beside its kind: the family it instantiates."""

    kind: str
    template: Optional[TemplateName] = Field(
        None,
        description=(
            "The family file the device instantiates, one of its kind's; "
            "absent: the kind's default (REGISTRY_KIND_FAMILIES). A device also "
            "in panel_devices.yaml names the same family there. Older readers "
            "refuse the key: write it only once every reader has this schema."
        ),
    )

    @model_validator(mode="after")
    def _a_family_of_this_kind(self, info: ValidationInfo) -> "_NativeEntry":
        if self.template is None:
            return self
        families = families_from(info)
        family = families.get(self.template)
        if family is None:
            raise ValueError(
                f"template {self.template!r} is not an installed family "
                f"({', '.join(sorted(families))})"
            )
        if family.kind != self.kind:
            raise ValueError(
                f"template {self.template!r} is a {family.kind!r} family, "
                f"not {self.kind!r}"
            )
        return self

    @model_serializer(mode="wrap")
    def _absent_stays_absent(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        # The deployed worker refuses unknown keys: never write a 'template' nobody set.
        data = handler(self)
        if self.template is None:
            data.pop("template", None)
        return data

    @property
    def family_template(self) -> str:
        """The family this device instantiates: ``template``, else its kind's default."""
        return self.template or REGISTRY_KIND_FAMILIES[self.kind]


class AravisCamera(_NativeEntry):
    """A GigE Vision camera behind ADAravis, writing HDF5 through its IOC's own file plugin."""

    kind: Literal["aravis_camera"] = Field(
        description="Device kind: 'aravis_camera' (an areaDetector ADAravis camera)."
    )
    trigger_source: Literal["line0", "line1", "software"] = Field(
        description=(
            "Where each frame comes from. 'line0' / 'line1': the camera's "
            "hardware input on that line (a Basler's opto-isolated input is "
            "Line1, a FLIR Blackfly's Line0), wired to a timing-system output "
            "gated by the trigger box; the line must be one of the camera's "
            "own TriggerSource choices. 'software': the scan worker presses "
            "the camera's software trigger at each shot (exposed when the "
            "worker says go, not when the laser fires)."
        )
    )


def _normalised(name: str) -> str:
    """The GEECS normalization a name binds by: runs outside ``[A-Za-z0-9_]`` → ``'_'``, stripped, lowercase.

    A copy of ``geecs_core.pv_naming.normalize_component`` (this package
    sits below GEECS-Core); ``tests/test_native_devices.py`` pins the two
    to agree where GEECS-Core is importable.
    """
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()


#: The setpoint record beside a readback by the gateway contract (``current`` → ``current:SP``).
SETPOINT_SUFFIX = ":SP"


def _record_names(
    what: str, names: Iterable[str], seen: dict[str, str], *, same: str = "record"
) -> None:
    """Refuse a name that is no record name, or one that normalises like another.

    *seen* maps each normalised name already taken to its spelling: the
    worker binds every name to an attribute by the GEECS normalization
    (:func:`_normalised`), so two that normalise alike would be one attribute.
    """
    for name in names:
        if not name or name != name.strip() or any(c.isspace() for c in name):
            raise ValueError(
                f"{what} {name!r} is not a record name (no blanks, no whitespace)"
            )
        key = _normalised(name)
        if key in seen:
            raise ValueError(
                f"{seen[key]!r} and {name!r} are the same {same} after normalization"
            )
        seen[key] = name


class EpicsMotor(_NativeEntry):
    """A positioner: one device, one EPICS motor record per axis.

    The axes are the record names under the device's prefix
    (``<experiment>:<device>:<axis>``), so the entry still carries names
    only; tolerance, units and limits live in the record (``.RDBD``,
    ``.EGU``, ``.HLM``/``.LLM``), the one home for device facts.  Each axis
    binds as a scan axis ``Device:axis`` (the same spelling as a DB
    device's settable), and the device as a whole is a per-row scalar
    device (every axis's readback).
    """

    kind: Literal["epics_motor"] = Field(
        description="Device kind: 'epics_motor' (EPICS motor records, one per axis)."
    )
    axes: list[str] = Field(
        min_length=1,
        description=(
            "The axes, as the motor record names under the device's prefix "
            "(e.g. ['x', 'y', 'z'] for '<experiment>:<device>:x' …). Each is "
            "a scan axis 'Device:axis'. The record owns the axis's limits, "
            "units and deadband."
        ),
    )

    @field_validator("axes")
    @classmethod
    def _axes_are_distinct_names(cls, axes: list[str]) -> list[str]:
        _record_names("axis", axes, {}, same="axis")
        return axes


class SetpointSettable(SchemaModel):
    """One settable of a setpoint device: how close its readback must come, and how soon."""

    # Finite: an infinite tolerance confirms any readback, an infinite
    # timeout removes the deadline (PositiveFloat alone admits +inf).
    tolerance: PositiveFloat = Field(
        allow_inf_nan=False,
        description=(
            "How close the readback must come to the requested value for the "
            "set to be done, in the record's units (e.g. 0.005 for 5 mA on a "
            "supply in A). The one device fact a plain 'ao' record cannot hold."
        ),
    )
    timeout: PositiveFloat = Field(
        10.0,
        allow_inf_nan=False,
        description=(
            "Seconds the set waits for the readback to arrive within the "
            "tolerance after the write, before it fails. Size it for the "
            "slowest ramp plus the readback's scan period."
        ),
    )


class SetpointDevice(_NativeEntry):
    """A setpoint device: a supply's plain records (``NATIVE_DEVICES.md`` §8).

    The entry names the records under the device's prefix
    (``<experiment>:<device>:<record>``), since the worker has no DB rows to
    enumerate them: each settable is its readback record, written at its
    setpoint record (``<record>:SP``, or the family's setpoint role for that
    readback) and confirmed on the readback within its tolerance; readbacks
    are extra row columns; status bits are two-state records that fail a
    set when they read 1. Limits and units live in the records
    (``DRVH``/``DRVL``, ``EGU``), the one home for device facts.
    """

    kind: Literal["setpoint_device"] = Field(
        description=(
            "Device kind: 'setpoint_device' (a supply's plain records: "
            "setpoint, readback, status bits)."
        )
    )
    settables: dict[str, SetpointSettable] = Field(
        min_length=1,
        description=(
            "The settables, by readback record name under the device's prefix "
            "(e.g. {'current': {tolerance: 0.005}} for "
            "'<experiment>:<device>:current', written at 'current:SP'). Each "
            "is a scan axis 'Device:record' and an action's set target."
        ),
    )
    readbacks: list[str] = Field(
        default_factory=list,
        description=(
            "Further numeric readback records, read into each row beside the "
            "settables' own (e.g. ['voltage'])."
        ),
    )
    status: list[str] = Field(
        default_factory=list,
        description=(
            "Status bits, two-state records that read 1 when tripped (e.g. "
            "['fault', 'interlock']): a set is refused while one reads 1 and "
            "fails if one rises while it waits."
        ),
    )

    @model_validator(mode="after")
    def _names_are_distinct_records(self, info: ValidationInfo) -> "SetpointDevice":
        seen: dict[str, str] = {}
        _record_names("settable", self.settables, seen)
        _record_names("readback", self.readbacks, seen)
        _record_names("status bit", self.status, seen)
        for record in self.settables:
            if record.endswith(SETPOINT_SUFFIX):
                raise ValueError(
                    f"settable {record!r} is a setpoint record: name its "
                    f"readback ({record[: -len(SETPOINT_SUFFIX)]!r})"
                )
        family = families_from(info).get(self.family_template)
        if family is not None:  # an unknown template is _NativeEntry's refusal
            self._check_against(family)
        written = set(self.setpoint_records(family).values())
        clash = sorted(written & {*self.readbacks, *self.status})
        if clash:
            raise ValueError(f"{clash} are setpoint records of this device's settables")
        return self

    def _check_against(self, family: FamilyFile) -> None:
        """Refuse a record the family knows as something else, and a status bit it lacks.

        A family lists every bit its template has but may omit a readback
        or a setpoint (the Easy-Driver's ``slewrate``, ``temperature``): a
        record the family does not name is the template's own business.
        """
        by_suffix = {r.suffix: r for r in family.roles}
        for record in self.settables:
            role = by_suffix.get(record)
            if role is not None and family.setpoint_for(record) is None:
                raise ValueError(
                    f"settable {record!r}: family {family.template!r} has it as a "
                    f"{role.type} with no setpoint role"
                )
        for record in self.readbacks:
            role = by_suffix.get(record)
            if role is not None and role.type != "readback":
                raise ValueError(
                    f"readback {record!r}: family {family.template!r} has it as a "
                    f"{role.type}, not a readback"
                )
        faults = sorted(r.suffix for r in family.roles if r.type == "bit" and r.fault)
        for record in self.status:
            if record not in faults:
                raise ValueError(
                    f"status bit {record!r} is not a fault bit of family "
                    f"{family.template!r} ({', '.join(faults) or 'none'})"
                )

    def setpoint_records(self, family: Optional[FamilyFile]) -> dict[str, str]:
        """Each settable's setpoint record: the family's setpoint role for it, else ``<record>:SP``.

        The one rule every reader writes or archives by (the worker, and
        later the scanner's list, the panel and the archiver); *family* is
        the device's (``load_families()[entry.family_template]``).
        """
        return {
            record: (family.setpoint_for(record) if family is not None else None)
            or record + SETPOINT_SUFFIX
            for record in self.settables
        }


class Witness(_NativeEntry):
    """A trigger domain's witness: a hardware count of its edges, the count's stamp, an epoch.

    The worker waits on it in every strict shot: the fire is confirmed when
    the count (``<prefix><counter>Counts``) advances, and each row records
    the count, the count record's own timestamp and the epoch
    (``<prefix>WitnessEpoch``, a string that changes whenever the count
    restarts). At most one witness per domain. Names only, as every entry:
    the records are the family's.
    """

    kind: Literal["witness"] = Field(
        description="Device kind: 'witness' (a counter of a trigger domain's edges)."
    )
    counter: str = Field(
        "Counter1",
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description=(
            "The counter record's name stem under the device's prefix: the "
            "count is '<counter>Counts'. It must be the family's count record."
        ),
    )
    domain: str = Field(
        "main",
        pattern=r"^[a-z][a-z0-9_]*$",
        description=(
            "The trigger domain it counts (a plain word): the edges every "
            "synchronous device of that domain sees. One witness per domain."
        ),
    )

    @model_validator(mode="after")
    def _the_familys_counter(self, info: ValidationInfo) -> "Witness":
        family = families_from(info).get(self.family_template)
        if family is None:  # an unknown template is _NativeEntry's refusal
            return self
        role = next((r for r in family.roles if r.name == "count"), None)
        if role is not None and role.suffix != self.count_record:
            raise ValueError(
                f"counter {self.counter!r}: family {family.template!r} counts at "
                f"{role.suffix!r}, not {self.count_record!r}"
            )
        return self

    @property
    def count_record(self) -> str:
        """The count record under the device's prefix (``Counter1Counts``)."""
        return f"{self.counter}Counts"


#: The epoch record under a witness's prefix: a string that changes when the count restarts.
WITNESS_EPOCH_RECORD = "WitnessEpoch"


#: Every native device kind, discriminated by ``kind`` (a new kind is one more member).
NativeDevice = Annotated[
    Union[AravisCamera, EpicsMotor, SetpointDevice, Witness],
    Field(discriminator="kind"),
]

#: The kinds that are detectors of the scan — they acquire per shot (a
#: stack, a stamp) and take the detector rules (``save_images``, the run
#: shapes a native camera is refused in).  A positioner and a setpoint
#: device are settables: never in this set.
DETECTOR_KINDS: frozenset[str] = frozenset({"aravis_camera"})


class NativeDevices(VersionedSchemaModel):
    """The experiment's devices served by native EPICS IOCs, by GEECS device name."""

    schema_version: Literal[1] = Field(
        1,
        description=(
            "Format version of this config file. Leave at 1 — tools update "
            "this automatically when the file format changes."
        ),
    )
    devices: dict[str, NativeDevice] = Field(
        default_factory=dict,
        description=(
            "GEECS device name (e.g. 'UC_ExampleCam') → its kind and "
            "settings. The IOC serves it under the derived prefix "
            "'<experiment>:<device>:'; the device must not also be enabled "
            "in the GEECS DB."
        ),
    )

    @field_validator("devices")
    @classmethod
    def _names_are_names(
        cls, devices: dict[str, NativeDevice]
    ) -> dict[str, NativeDevice]:
        blank = [name for name in devices if not name.strip()]
        if blank:
            raise ValueError("a native device name must not be blank")
        domains: dict[str, str] = {}
        for name, entry in devices.items():
            if isinstance(entry, Witness):
                if entry.domain in domains:
                    raise ValueError(
                        f"witnesses {domains[entry.domain]!r} and {name!r} both "
                        f"count domain {entry.domain!r}: one witness per domain"
                    )
                domains[entry.domain] = name
        return devices

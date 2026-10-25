"""FamilyFile — what a record template's records are, by role (layer 0 of the web estate).

One file per **record template** (the motor record, the areaDetector camera
chain, the scope contract, a supply's StreamDevice template …), shipped as
package data under ``geecs_schemas/epics/families/<template>.json`` and read
by :func:`load_families`.  It is device-type knowledge, the driver's
variable list: every role names its record suffix under the device's
prefix, what kind of value it is, how a page shows it and what a write to
it means (who may, whether it needs a hazard confirm, how it settles).
Per-device differences (a supply's own PV names, a bit's label, the
permitted actions) are instance data, :mod:`geecs_schemas.epics.panel_devices`.

JSON, not YAML: the files are read at run time wherever the bridge, the
scanner, the archiver and the MCP contract tests run, and this package
keeps PyYAML out of its runtime dependencies (YAML is serialization, not
the contract; see :meth:`VersionedSchemaModel.from_path`).

The contents were transcribed from GeecsPanels' per-kind tables
(``motor.py``, ``camera.py``, ``scope.py``, ``power.py``); a pin in
GeecsPanels' tests holds the two in step until those tables are deleted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
from typing import Annotated, Literal, Optional

from pydantic import Field, ValidationInfo, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel

#: What kind of value a role carries, which is also the widget a page draws for it.
RoleType = Literal[
    "readback", "setpoint", "enum", "bit", "command", "array", "image", "counter"
]  # fmt: skip

#: Who sees a role by default: every operator, or the expert view only.
Level = Literal["operator", "expert"]

#: How the bridge reads a role's record.
ReadMode = Literal["monitor", "on_request", "once", "never"]

#: The registry kind a family serves: ``geecs_schemas.epics``' built kinds,
#: and the two ``GeecsBluesky/NATIVE_DEVICES.md`` §10 proposes.
FamilyKind = Literal[
    "aravis_camera", "epics_motor", "setpoint_device", "array_detector"
]

#: A template's name: the family file's stem.
TemplateName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]

#: A role's (or an action's) name: a plain word, it keys the stream.
RoleName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]

#: A PV suffix under the device's prefix: no whitespace.
Suffix = Annotated[str, Field(pattern=r"^\S+$")]

#: Roles a write goes to; the others are read only.
_ALWAYS_WRITABLE = frozenset({"setpoint", "command"})
_NEVER_WRITABLE = frozenset({"readback", "counter", "array", "image"})


class Role(SchemaModel):
    """One record of the template, by what it does for a page.

    A role is read from ``suffix`` under the device's prefix (for a family
    whose records repeat per axis, under ``<prefix><axis>``).  A writable
    role is written at ``write`` when the template keeps its setpoint and
    readback in two records (areaDetector's ``X`` / ``X_RBV``), else at
    ``suffix`` itself.
    """

    name: RoleName = Field(
        description="The role's name: a plain word, unique in the family; it keys the device stream."
    )
    suffix: Suffix = Field(
        description=(
            "The record (or record field, e.g. '.RBV') under the device's prefix "
            "that the role reads; for a command, the record pressed."
        )
    )
    type: RoleType = Field(
        description=(
            "What the value is and the widget a page draws: readback, setpoint, "
            "enum (shown by its state name), bit (0/1), command (a press), array "
            "(read on request), image (the frame source), counter (a plugin's "
            "frame counter, which judges freshness)."
        )
    )
    label: str = Field(min_length=1, description="What a page calls it.")
    units: Optional[str] = Field(
        None,
        description=(
            "The units the value travels in, where the record carries none "
            "(a record's EGU wins when it has one)."
        ),
    )
    level: Level = Field(
        "operator",
        description="'operator': on the page by default; 'expert': in the expert view only.",
    )
    writable: bool = Field(
        False,
        description=(
            "Whether a page may write it. Setpoints and commands always are; "
            "readbacks, counters, arrays and images never are; an enum or a bit "
            "says."
        ),
    )
    write: Optional[Suffix] = Field(
        None,
        description=(
            "The record a write goes to, when it is not 'suffix' (the setpoint "
            "beside an _RBV readback). Implies writable."
        ),
    )
    confirm: Optional[Literal["hazard"]] = Field(
        None,
        description=(
            "'hazard': the page asks the operator to confirm the write, naming "
            "device, role and value (a supply's output, a motor's home)."
        ),
    )
    archive: bool = Field(
        False,
        description=(
            "The archiver takes it: the history an engineer reads later "
            "(readbacks, setpoints, limit switches, fault bits)."
        ),
    )
    locked: bool = Field(
        False,
        description=(
            "Refused while a run holds the device (the run lock), or while the "
            "lock cannot be read: what a scan saves."
        ),
    )
    settled_by: Optional[RoleName] = Field(
        None,
        description=(
            "A write whose put completes only when the work ends (an acquire, a "
            "move) is dispatched without waiting and settles when this role "
            "says so: the readback that confirms it, or the motor's done flag."
        ),
    )
    readback: Optional[RoleName] = Field(
        None,
        description="A setpoint's readback, shown beside it (a motor's .RBV beside .VAL).",
    )
    fresh_by: Optional[RoleName] = Field(
        None,
        description=(
            "The counter role whose advance makes this value fresh (the producing "
            "plugin's ArrayCounter_RBV): statistics post only on change, so a "
            "steady value is judged by its producer. Absent: fresh while connected."
        ),
    )
    computed_by: Optional[RoleName] = Field(
        None,
        description=(
            "The flag that must be on for the value to be computed at all "
            "(ComputeCentroid): with it off the value reads 'off', not fresh."
        ),
    )
    text: Optional[Literal["string", "char_array"]] = Field(
        None,
        description=(
            "Read as text: a string record, or a char-array record longer than a "
            "string's 40 characters. Enums are read by state name regardless."
        ),
    )
    fields: list[Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9]*$")]] = Field(
        default_factory=list,
        description=(
            "Record fields read beside the value and carried with it, e.g. "
            "['DRVH', 'DRVL'] for a setpoint's drive limits."
        ),
    )
    limits: Optional[tuple[float, float]] = Field(
        None,
        description=(
            "The bridge's own bounds on a write, low then high, where the record "
            "enforces none (a pixel, an exposure). A motor's soft limits and a "
            "supply's drive limits are the record's, read at put time."
        ),
    )
    integer: bool = Field(False, description="A write takes a whole number.")
    choices: Optional[list[str]] = Field(
        None,
        description=(
            "The values a write may take, where the page offers fewer than the "
            "record's own (an enum's choices are otherwise read from the PV)."
        ),
    )
    fault: bool = Field(
        False,
        description="A bit that reads 1 when tripped: any tripped fault makes the device 'failed'.",
    )
    read: Optional[ReadMode] = Field(
        None,
        description=(
            "How the bridge reads it, when not the type's own: monitored (scalars), "
            "on request (arrays), once (the image's URL), never (a command's record)."
        ),
    )

    @property
    def read_mode(self) -> ReadMode:
        """How the bridge reads the role: ``read`` if set, else by type."""
        if self.read is not None:
            return self.read
        if self.type == "array":
            return "on_request"
        if self.type == "image":
            return "once"
        if self.type == "command" and self.write is None:
            return "never"
        return "monitor"

    @property
    def write_suffix(self) -> Optional[str]:
        """The record a write goes to, ``None`` for a read-only role."""
        if not self.writable:
            return None
        return self.write or self.suffix

    @model_validator(mode="after")
    def _type_and_writes_agree(self) -> "Role":
        if self.type in _ALWAYS_WRITABLE and not self.writable:
            raise ValueError(f"role {self.name!r}: a {self.type} is writable")
        if self.type in _NEVER_WRITABLE and self.writable:
            raise ValueError(f"role {self.name!r}: a {self.type} is never writable")
        if self.write is not None and not self.writable:
            raise ValueError(f"role {self.name!r}: 'write' needs writable: true")
        if not self.writable:
            set_ = [
                name
                for name in ("confirm", "locked", "settled_by", "limits", "choices")
                if getattr(self, name)
            ] + (["integer"] if self.integer else [])
            if set_:
                raise ValueError(
                    f"role {self.name!r}: {set_} apply to writable roles only"
                )
        if self.fault and self.type != "bit":
            raise ValueError(f"role {self.name!r}: only a bit is a fault")
        if self.limits is not None and not self.limits[0] < self.limits[1]:
            raise ValueError(f"role {self.name!r}: limits are low then high")
        return self


class FamilyAction(SchemaModel):
    """One thing a page does to a device of the family, and the roles it writes.

    Instance data permits a subset of a family's actions per device
    (``allow``); a family without actions permits every writable role.
    """

    label: str = Field(min_length=1, description="What the page's control is called.")
    roles: list[RoleName] = Field(
        min_length=1, description="The writable roles the action writes."
    )
    requires: list[RoleName] = Field(
        default_factory=list,
        description=(
            "Actions a device permitting this one must permit too (a motion "
            "needs stop)."
        ),
    )


class FamilyFile(VersionedSchemaModel):
    """The records of one IOC record template, by role: what every device of the family serves.

    Shipped as package data (``geecs_schemas/epics/families/<template>.json``),
    one file per template; read with :func:`load_families`. The order of
    ``roles`` is the page's order.
    """

    schema_version: Literal[1] = Field(
        1,
        description=(
            "Format version of this config file. Leave at 1 — tools update "
            "this automatically when the file format changes."
        ),
    )
    template: TemplateName = Field(
        description="The record template's name (the file's stem), e.g. 'motor_record'."
    )
    kind: FamilyKind = Field(
        description="The native-device kind whose devices instantiate the template."
    )
    description: str = Field(
        min_length=1,
        description="One line: what the template is and where it comes from.",
    )
    per_axis: bool = Field(
        False,
        description=(
            "The records repeat per axis (the motor record): every suffix is "
            "under '<prefix><axis>' for each axis the device names."
        ),
    )
    roles: list[Role] = Field(
        min_length=1,
        description="The template's records, by role, in the page's order.",
    )
    actions: dict[RoleName, FamilyAction] = Field(
        default_factory=dict,
        description=(
            "The page's actions and the roles each writes; a device may permit a "
            "subset. Empty: every writable role is its own permission."
        ),
    )

    def role(self, name: str) -> Role:
        """The role called *name*; ``KeyError`` if the family has none."""
        for role in self.roles:
            if role.name == name:
                return role
        raise KeyError(f"family {self.template!r} has no role {name!r}")

    def role_names(self) -> list[str]:
        """Every role's name, in the page's order."""
        return [role.name for role in self.roles]

    @model_validator(mode="after")
    def _roles_are_one_table(self) -> "FamilyFile":
        _refuse_duplicates("role names", self.role_names())
        check_suffixes(self.template, self.roles)
        by_name = {role.name: role for role in self.roles}
        for role in self.roles:
            _check_links(role, by_name)
        for action_name, action in self.actions.items():
            for name in action.roles:
                if name not in by_name or not by_name[name].writable:
                    raise ValueError(
                        f"action {action_name!r}: {name!r} is not a writable role"
                    )
            missing = sorted(set(action.requires) - set(self.actions))
            if missing:
                raise ValueError(
                    f"action {action_name!r} requires unknown action(s) {missing}"
                )
        return self


def _check_links(role: Role, by_name: Mapping[str, Role]) -> None:
    """Refuse a link naming no role, an unmonitored one, or a freshness source that is no counter."""
    for link in ("settled_by", "readback", "fresh_by", "computed_by"):
        target = getattr(role, link)
        if target is None:
            continue
        other = by_name.get(target)
        if other is None:
            raise ValueError(f"role {role.name!r}: {link} names no role ({target!r})")
        if other.read_mode != "monitor":
            raise ValueError(f"role {role.name!r}: {link} {target!r} is not monitored")
        if link == "fresh_by" and other.type != "counter":
            raise ValueError(
                f"role {role.name!r}: fresh_by names a counter, not "
                f"{other.type} {target!r}"
            )


def _refuse_duplicates(what: str, items: list[str]) -> None:
    seen: set[str] = set()
    twice = sorted({item for item in items if item in seen or seen.add(item)})
    if twice:
        raise ValueError(f"{what} must be unique; repeated: {twice}")


def check_suffixes(template: str, roles: list[Role]) -> None:
    """Refuse two roles reading one record, or two writable roles writing one.

    Raises
    ------
    ValueError
        Naming the template and the repeated suffixes.
    """
    try:
        _refuse_duplicates("read suffixes", [r.suffix for r in roles])
        _refuse_duplicates(
            "write suffixes", [s for r in roles if (s := r.write_suffix) is not None]
        )
    except ValueError as exc:
        raise ValueError(f"family {template!r}: {exc}") from None


#: The package-data folder holding the family files.
FAMILIES_PACKAGE = "geecs_schemas.epics"
FAMILIES_FOLDER = "families"


@lru_cache(maxsize=1)
def load_families() -> Mapping[str, FamilyFile]:
    """Every installed family file, by template name, read once per process.

    Raises
    ------
    ValueError
        A file that does not validate, or whose ``template`` is not its stem.
    """
    folder = resources.files(FAMILIES_PACKAGE).joinpath(FAMILIES_FOLDER)
    families: dict[str, FamilyFile] = {}
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if not entry.name.endswith(".json"):
            continue
        stem = entry.name.removesuffix(".json")
        family = FamilyFile.model_validate(
            json.loads(entry.read_text(encoding="utf-8"))
        )
        if family.template != stem:
            raise ValueError(
                f"family file {entry.name} declares template {family.template!r}"
            )
        families[stem] = family
    return families


def families_from(info: ValidationInfo) -> Mapping[str, FamilyFile]:
    """The family set a document validates against: ``context['families']``, else the installed ones.

    Tests pass tiny families through the validation context
    (``Model.model_validate(doc, context={"families": {...}})``); every
    other caller validates against the families this package ships.
    """
    context = info.context if isinstance(info.context, Mapping) else {}
    families = context.get("families")
    return families if families is not None else load_families()

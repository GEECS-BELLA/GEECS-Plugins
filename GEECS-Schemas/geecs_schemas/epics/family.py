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

JSON, not YAML: the files are read at run time by every native-device
reader (GeecsCanopy and the archiver's registry pass), and this package
keeps PyYAML out of its runtime dependencies (YAML is serialization, not
the contract; see :meth:`VersionedSchemaModel.from_path`).

The contents were transcribed from GeecsPanels' per-kind tables
(``motor.py``, ``camera.py``, ``scope.py``, ``power.py``); the pin that
holds the two in step (web estate S2) is kept with the PV bridge at
the tag ``web-estate-final``, not beside these files.

The installed families are shared by everything in the process, so the
models are frozen and :func:`load_families` returns a read-only mapping.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import Annotated, Literal, Optional

from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationInfo,
    model_validator,
)

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
#: and the one ``GeecsBluesky/NATIVE_DEVICES.md`` §10 still proposes.
FamilyKind = Literal[
    "aravis_camera", "epics_motor", "setpoint_device", "array_detector"
]

#: Registry kind (``NativeDevices``) → the family a registry device
#: instantiates when its panel data names none. One table for every
#: reader (GeecsCanopy and the archiver's registry pass), so no two pick
#: different families.
REGISTRY_KIND_FAMILIES: Mapping[str, str] = MappingProxyType(
    {
        "aravis_camera": "ad_camera_chain",
        "epics_motor": "motor_record",
        "setpoint_device": "easydriver",
    }
)

#: A template's name: the family file's stem.
TemplateName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]

#: Words YAML 1.1 reads as a boolean or null: as a role name, an unquoted
#: one in an override or a panel would arrive as ``False``, not the role.
_YAML_WORDS = frozenset({"y", "n", "yes", "no", "on", "off", "true", "false", "null"})


def _not_a_yaml_word(name: str) -> str:
    if name in _YAML_WORDS:
        raise ValueError(
            f"{name!r} reads as a boolean or null in YAML; name it otherwise"
        )
    return name


#: A role's (or an action's) name: a plain word, it keys the stream.
RoleName = Annotated[
    str, Field(pattern=r"^[a-z][a-z0-9_]*$"), AfterValidator(_not_a_yaml_word)
]

#: A role as a binding, a grant or a hazard names it: a family role, or, on a
#: device the GEECS DB serves, its GEECS variable name ("Position.Axis 1").
AnyRoleName = Annotated[str, Field(min_length=1, pattern=r"^\S(.*\S)?$")]

#: A device's family word: its family template, or the GEECS DB device type
#: of a device the DB serves ("Point Grey Camera").
FamilyWord = Annotated[str, Field(min_length=1, pattern=r"^\S(.*\S)?$")]

#: A PV suffix under the device's prefix: no whitespace.
Suffix = Annotated[str, Field(pattern=r"^\S+$")]


def _one_or_many(value: object) -> object:
    return [value] if isinstance(value, str) else value


#: The arrays an axis is the x axis of: one name, or a list of them.
AxisTargets = Annotated[
    list[RoleName], Field(min_length=1), BeforeValidator(_one_or_many)
]

#: Roles a write goes to; the others are read only.
_ALWAYS_WRITABLE = frozenset({"setpoint", "command"})
_NEVER_WRITABLE = frozenset({"readback", "counter", "array", "image"})

#: Shared through the process-wide cache: never changed in place.
_FROZEN = ConfigDict(extra="forbid", frozen=True)


class Role(SchemaModel):
    """One record of the template, by what it does for a page.

    A role is read from ``suffix`` under the device's prefix (for a family
    whose records repeat per axis, under ``<prefix><axis>``).  A writable
    role is written at ``write`` when the template keeps its setpoint and
    readback in two records (areaDetector's ``X`` / ``X_RBV``), else at
    ``suffix`` itself.
    """

    model_config = _FROZEN

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
    autosaved: bool = Field(
        False,
        description=(
            "The IOC's autosave restores this setting's last value at boot (its "
            "request files in geecs-epics); descriptive only, not a power-on "
            "default an operator chooses. False also where no request file "
            "documents it."
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
    axis_of: Optional[AxisTargets] = Field(
        None,
        description=(
            "On an array: the array role (or the list of them) this one is the x "
            "axis of, one value per sample of each of its channels (a scope's time "
            "axis beside its capture, a spectrometer's wavelengths beside its raw "
            "and its processed spectrum). A single name is read as a list of one. "
            "All are arrays; an array has one axis at most, and an axis is no "
            "array's trace."
        ),
    )
    channels_by: Optional[RoleName] = Field(
        None,
        description=(
            "On an array of several channels: the readback that counts them. The "
            "array is then [channels, samples], channels fastest (a scope's ARR "
            "capture, a disabled channel NaN). Absent: one channel."
        ),
    )
    read: Optional[ReadMode] = Field(
        None,
        description=(
            "How the bridge reads it, when not the type's own: monitored (scalars), "
            "on request (arrays), once (the image's URL, or a constant record such as a camera's FullScale: read, never judged), never (a command's record, or an array the bridge never reads because a route of its own serves it: a camera's raw frame)."
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

    model_config = _FROZEN

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


#: A button's label on the page.
ButtonLabel = Annotated[str, Field(min_length=1)]


class PanelGroup(SchemaModel):
    """One titled group of a device page's control rail: the roles it draws, in order."""

    model_config = _FROZEN

    title: str = Field(min_length=1, description="The group's heading on the rail.")
    roles: list[RoleName] = Field(
        min_length=1,
        description=(
            "The family roles the group draws, in order; the page picks each "
            "control by the role's type."
        ),
    )
    layout: Literal["fields", "grid"] = Field(
        "fields",
        description=(
            "'fields': one labelled control per row; 'grid': a compact labelled "
            "grid of short controls (a PLC's channels), for a group of bits, enums "
            "and two-state whole numbers only."
        ),
    )


class TweakHint(SchemaModel):
    """A setpoint drawn with a step and a pair of step buttons beside it (the motor record's tweak)."""

    model_config = _FROZEN

    step: RoleName = Field(
        description="The setpoint holding the step size (the motor record's 'twv')."
    )
    forward: RoleName = Field(
        description="The command that steps forward by it ('twf'): the record moves, the page computes nothing."
    )
    reverse: RoleName = Field(description="The command that steps back by it ('twr').")


class Panel(SchemaModel):
    """How a device page lays out the family's roles: the control rail and the readings strip.

    Layout only. The page draws each rail role by its type (an enum a
    select of the record's own choices, a setpoint a number beside its
    readback, a bit a checkbox, a command a button, a readback its text), so
    a plain group constrains no role type; two hints do: a ``grid`` group
    holds short controls only (bits, enums, two-state whole numbers) and a
    ``tweak`` names a setpoint, its step setpoint and two commands. A reading
    is a monitored role, since it carries the module's word. The panel
    places roles whatever their ``level``: ``level`` decides the role
    table's operator and expert views, the panel what the page draws above
    it. Whether a write is permitted stays the module's.
    """

    model_config = _FROZEN

    rail: list[PanelGroup] = Field(
        default_factory=list,
        description="The rail's control groups, in order; a role sits in one group at most.",
    )
    readings: list[RoleName] = Field(
        default_factory=list,
        description="The roles of the readings strip, in order, each with its freshness word.",
    )
    buttons: dict[RoleName, tuple[ButtonLabel, ButtonLabel]] = Field(
        default_factory=dict,
        description=(
            "A two-state rail role drawn as a pair of buttons, its labels for 0 "
            "then 1 (an acquire as Stop / Start), in place of its type's control. "
            "The role is writable, in the rail, and a bit or a whole number "
            "limited to [0, 1]."
        ),
    )
    tweak: dict[RoleName, TweakHint] = Field(
        default_factory=dict,
        description=(
            "A rail setpoint drawn with its step and step buttons beside it, in "
            "place of three separate controls: the key is the setpoint, the hint "
            "its step setpoint and its forward and reverse commands, all writable "
            "and in the key's rail group."
        ),
    )
    show_while: dict[RoleName, RoleName] = Field(
        default_factory=dict,
        description=(
            "An array an axis draws, shown by default while a monitored role reads "
            "1 (a processed spectrum while background subtraction is on); otherwise "
            "the axis' first array. A viewer's own choice wins."
        ),
    )


class FamilyFile(VersionedSchemaModel):
    """The records of one IOC record template, by role: what every device of the family serves.

    Shipped as package data (``geecs_schemas/epics/families/<template>.json``),
    one file per template; read with :func:`load_families`. The order of
    ``roles`` is the page's order.
    """

    model_config = _FROZEN

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
            "subset. When there are actions, every writable role belongs to one. "
            "Empty: every writable role is its own permission."
        ),
    )
    panel: Optional[Panel] = Field(
        None,
        description=(
            "How the kind's device page lays the roles out (the rail's groups, the "
            "readings strip); absent: the page's own default."
        ),
    )

    def role(self, name: str) -> Role:
        """The role called *name*; ``KeyError`` if the family has none."""
        for role in self.roles:
            if role.name == name:
                return role
        raise KeyError(f"family {self.template!r} has no role {name!r}")

    def setpoint_for(self, readback: str) -> Optional[str]:
        """The record a write to the readback record *readback* goes to; ``None`` if no role says.

        The setpoint role whose ``readback`` is the role reading
        *readback* (the Easy-Driver's ``current:SP`` beside ``current``, the
        FAST-PS's ``current_ramp:SP``).
        """
        names = {r.name for r in self.roles if r.suffix == readback}
        for role in self.roles:
            if role.type == "setpoint" and role.readback in names:
                return role.write_suffix
        return None

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
        _check_axes(self.roles, by_name)
        if self.panel is not None:
            _check_panel(self.panel, by_name)
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
        if self.actions:
            acted = {name for action in self.actions.values() for name in action.roles}
            loose = [r.name for r in self.roles if r.writable and r.name not in acted]
            if loose:
                raise ValueError(
                    f"writable role(s) {loose} belong to no action: with actions, "
                    "a device's allow decides every write"
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


def _check_axes(roles: list[Role], by_name: Mapping[str, Role]) -> None:
    """Refuse an axis that is no array, names no array, is a second axis of one array, or is itself a trace; a channel count that is no readback."""
    drawn = [target for r in roles for target in r.axis_of or ()]
    for role in roles:
        if role.channels_by is not None:
            counter = by_name.get(role.channels_by)
            if role.type != "array" or counter is None or counter.type != "readback":
                raise ValueError(
                    f"role {role.name!r}: channels_by is an array's, naming a readback "
                    f"({role.channels_by!r})"
                )
        if role.axis_of is None:
            continue
        if role.type != "array":
            raise ValueError(
                f"role {role.name!r}: only an array is an axis ({role.type})"
            )
        for target in role.axis_of:
            other = by_name.get(target)
            if other is None:
                raise ValueError(
                    f"role {role.name!r}: axis_of names no role ({target!r})"
                )
            if other.type != "array" or target == role.name:
                raise ValueError(
                    f"role {role.name!r}: axis_of names an array other than itself, "
                    f"not {other.type} {target!r}"
                )
        if role.name in drawn:
            raise ValueError(f"role {role.name!r}: an axis is no array's trace")
    _refuse_duplicates("axes (one per array)", drawn)


def _check_panel(panel: Panel, by_name: Mapping[str, Role]) -> None:
    """Refuse a panel naming a role the family lacks, one rail role in two places, an unmonitored reading, a button pair on no two-state rail role, a tweak on no writable setpoint and commands of its group, a grid of more than bits and enums, or a default array no axis draws."""
    named = [name for group in panel.rail for name in group.roles]
    tweaked = [
        name
        for key, hint in panel.tweak.items()
        for name in (key, hint.step, hint.forward, hint.reverse)
    ]
    hinted = [*panel.buttons, *tweaked, *panel.show_while, *panel.show_while.values()]
    unknown = sorted(
        {name for name in [*named, *panel.readings, *hinted] if name not in by_name}
    )
    if unknown:
        raise ValueError(f"panel names no role {unknown}")
    _refuse_duplicates("panel rail roles", named)
    _refuse_duplicates("panel readings", list(panel.readings))
    unjudged = [n for n in panel.readings if by_name[n].read_mode != "monitor"]
    if unjudged:
        raise ValueError(
            f"panel readings {unjudged} are not monitored: a reading carries the "
            "module's word, and the module judges monitored roles only"
        )
    for name in panel.buttons:
        role = by_name[name]
        if name not in named or not role.writable or not _two_state(role):
            raise ValueError(
                f"panel buttons {name!r}: a button pair draws a writable two-state "
                "rail role (a bit, or a whole number limited to [0, 1])"
            )
    _check_tweaks(panel, by_name, tweaked)
    for group in panel.rail:
        long = [n for n in group.roles if not _short(by_name[n])]
        if group.layout == "grid" and long:
            raise ValueError(
                f"panel group {group.title!r}: a grid draws short controls only "
                f"(bits, enums, two-state whole numbers), not {long}"
            )
    drawn = {target for r in by_name.values() for target in r.axis_of or ()}
    for array, flag in panel.show_while.items():
        if array not in drawn:
            raise ValueError(f"panel show_while {array!r}: no axis draws it")
        if by_name[flag].read_mode != "monitor":
            raise ValueError(
                f"panel show_while {array!r}: {flag!r} is not monitored, so the "
                "page cannot read it"
            )


def _two_state(role: Role) -> bool:
    """A bit, or a whole number limited to [0, 1]: what a button pair draws."""
    return role.type == "bit" or (role.integer and role.limits == (0, 1))


def _short(role: Role) -> bool:
    """A two-state role or an enum: what a grid cell draws."""
    return role.type == "enum" or _two_state(role)


def _check_tweaks(
    panel: Panel, by_name: Mapping[str, Role], tweaked: list[str]
) -> None:
    """Refuse a tweak whose roles are not a writable setpoint, its step setpoint and two commands, all in the key's rail group and drawn by no other hint.

    A group is the group itself (its place in the rail), never its title:
    two groups may share a heading.
    """
    group_of = {name: i for i, group in enumerate(panel.rail) for name in group.roles}

    def where(name: str) -> str:
        i = group_of[name]
        return f"{panel.rail[i].title!r}, group {i + 1}"

    for key, hint in panel.tweak.items():
        wanted = ((key, "setpoint"), (hint.step, "setpoint"),
                  (hint.forward, "command"), (hint.reverse, "command"))  # fmt: skip
        for name, type_ in wanted:
            role = by_name[name]
            if role.type != type_ or name not in group_of:
                raise ValueError(
                    f"panel tweak {key!r}: {name!r} must be a rail {type_} "
                    f"({'an' if role.type[0] in 'aeiou' else 'a'} {role.type}"
                    f"{'' if name in group_of else ' off the rail'})"
                )
            if group_of[name] != group_of[key]:
                raise ValueError(
                    f"panel tweak {key!r}: {name!r} sits in the setpoint's rail "
                    f"group ({where(key)}), not {panel.rail[group_of[name]].title!r} "
                    f"(group {group_of[name] + 1})"
                )
    _refuse_duplicates("panel tweak roles", tweaked)
    both = sorted(set(tweaked) & set(panel.buttons))
    if both:
        raise ValueError(f"panel roles {both} are drawn by a tweak and a button pair")


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
    return MappingProxyType(families)


def families_from(info: ValidationInfo) -> Mapping[str, FamilyFile]:
    """The family set a document validates against: ``context['families']``, else the installed ones.

    Tests pass tiny families through the validation context
    (``Model.model_validate(doc, context={"families": {...}})``); every
    other caller validates against the families this package ships.
    """
    context = info.context if isinstance(info.context, Mapping) else {}
    families = context.get("families")
    return families if families is not None else load_families()

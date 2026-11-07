"""Screen — a control GUI as a file: titled groups of widgets, each a label and a binding.

``<experiment>/screens/<name>.yaml`` in the configs repository, drawn by
GeecsControlRoom's one renderer (``Planning/operator_surfaces.md`` §3). A
screen binds, groups and confirms; it never computes: there are no
expressions, scripts or conditions here, and no colour but a kit token. A
binding names a GEECS device and a role (and an axis), or a bare PV. On a
device the GEECS DB serves, the role is its GEECS variable name. A bare PV
is read-only, for anything servable (a timestamp, a derived channel): a
writing kind on one is drawn disabled; a write binds ``{device, role}``.
A button may instead name an action plan (``{action: <plan>}``) from the
experiment's action library: the device module runs it, step by step
through its gates; the screen only calls it.
Whether the role exists on the device and fits the widget depends on the
family files and the instance data, so it is checked by :func:`resolve`,
not by the model; a DB device's roles are the device module's, so
:func:`resolve` defers them to the page.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel
from geecs_schemas.epics import NativeDevices
from geecs_schemas.epics.family import AnyRoleName, FamilyFile, Role, Suffix
from geecs_schemas.epics.panel_devices import DeviceName, PanelDevices

#: The screens' folder in the experiment's configs folder.
SCREENS_DIRNAME = "screens"

#: The eight widgets: setpoint (with its readback), two-state toggle, flipper
#: (a toggle with in/out indicators), latching button, a choice of an enum's
#: options, read-only value, text, picture.
WidgetKind = Literal[
    "setpoint", "toggle", "flipper", "button", "select", "readout", "annotation", "image"
]  # fmt: skip

#: The kit's colour tokens a group may wear: hues with no status meaning.
GroupTone = Literal[
    "accent", "slate", "muted", "trace-1", "trace-2", "trace-3", "trace-4"
]  # fmt: skip

#: Role types each bound kind accepts, and whether the role must be writable.
_FITS: dict[str, tuple[frozenset[str], bool]] = {
    "setpoint": (frozenset({"setpoint"}), True),
    "toggle": (frozenset({"bit", "enum"}), True),
    "flipper": (frozenset({"bit", "enum"}), True),
    "button": (frozenset({"command", "bit"}), True),
    "select": (frozenset({"enum"}), True),
    "readout": (frozenset({"readback", "setpoint", "counter", "bit", "enum"}), False),
}
_WRITES = frozenset({"setpoint", "toggle", "flipper", "button", "select"})


class ToggleValues(SchemaModel):
    """The two numbers a toggle writes to a numeric role (a LabVIEW boolean kept as a number)."""

    on: float = Field(allow_inf_nan=False, description="Written to switch it on; a readback equal to it reads on.")  # fmt: skip
    off: float = Field(allow_inf_nan=False, description="Written to switch it off; a readback equal to it reads off. Any other readback reads unknown.")  # fmt: skip

    @model_validator(mode="before")
    @classmethod
    def _yaml_booleans(cls, data: object) -> object:
        """YAML 1.1 (PyYAML) reads a bare ``on:``/``off:`` key as true/false: take those as the names."""
        if isinstance(data, dict):
            return {{True: "on", False: "off"}.get(k, k) if isinstance(k, bool) else k: v for k, v in data.items()}  # fmt: skip
        return data

    @model_validator(mode="after")
    def _two_states(self) -> "ToggleValues":
        if self.on == self.off:
            raise ValueError("on and off are two different numbers")
        return self


class Binding(SchemaModel):
    """What a widget shows and writes: a device's role (on an axis), or a bare PV."""

    device: Optional[DeviceName] = Field(None, description="The GEECS device name.")
    role: Optional[AnyRoleName] = Field(None, description="The device family's role; a DB device's GEECS variable name.")  # fmt: skip
    axis: Optional[Suffix] = Field(
        None, description="The axis, for a per-axis family (a motor's m1)."
    )
    pv: Optional[Suffix] = Field(None, description="A bare PV, read only: a writing kind on one is drawn disabled.")  # fmt: skip
    action: Optional[str] = Field(None, min_length=1, description="An action plan's name in the experiment's library: a button runs it.")  # fmt: skip

    @model_validator(mode="after")
    def _one_kind_of_binding(self) -> "Binding":
        named = self.device is not None and self.role is not None
        loose = (self.device or self.role) and not named
        if loose or named + (self.pv is not None) + (self.action is not None) != 1:
            raise ValueError(
                "a binding names a device and a role, a pv, or an action: one of them"
            )
        if self.axis is not None and not named:
            raise ValueError("an axis belongs to a device's role")
        return self


class Widget(SchemaModel):
    """One control: its kind, the label a person reads, and what it is bound to."""

    kind: WidgetKind = Field(description="The widget, one of the eight.")
    label: str = Field("", description="Free text shown beside it (an annotation's text); empty: the role's label.")  # fmt: skip
    bind: Optional[Binding] = Field(None, description="What it shows and writes; none for an annotation or an image.")  # fmt: skip
    indicators: list[Binding] = Field([], max_length=2, description="A flipper's in and out indicators, in that order.")  # fmt: skip
    confirm: Optional[str] = Field(None, min_length=1, description="A hazard note the operator confirms before the write.")  # fmt: skip
    values: Optional[ToggleValues] = Field(None, description="A toggle's or flipper's on and off numbers: with them it binds a numeric setpoint role.")  # fmt: skip
    units: Optional[str] = Field(
        None, description="Units shown, where the record carries none."
    )
    precision: Optional[int] = Field(
        None, ge=0, le=12, description="Digits after the point."
    )
    src: Optional[str] = Field(None, pattern=r"^[\w-]+\.(png|jpe?g|gif|webp|PNG|JPE?G|GIF|WEBP)$", description="An image's file name in the screens folder.")  # fmt: skip

    @model_validator(mode="after")
    def _fields_fit_the_kind(self) -> "Widget":
        bound = self.kind not in ("annotation", "image")
        if bound != (self.bind is not None):
            raise ValueError(
                f"a {self.kind} {'needs' if bound else 'takes no'} binding"
            )
        if (self.kind == "image") != (self.src is not None):
            raise ValueError("an image, and only an image, has a src")
        if self.kind == "annotation" and not self.label:
            raise ValueError("an annotation's label is its text")
        if self.indicators and self.kind != "flipper":
            raise ValueError("only a flipper has indicators")
        if self.values and self.kind not in ("toggle", "flipper"):
            raise ValueError("only a toggle or a flipper has values")
        if self.confirm and self.kind not in _WRITES:
            raise ValueError(f"a {self.kind} writes nothing to confirm")
        runs = [b for b in (self.bind, *self.indicators) if b and b.action]
        if runs and (self.kind != "button" or runs[0] is not self.bind):
            raise ValueError("only a button runs an action")
        return self


class Group(SchemaModel):
    """A titled block of widgets on the screen's grid, worn in a kit colour token."""

    title: str = Field(description="The group's heading.")
    at: tuple[int, int] = Field(
        description="Its top-left grid cell: row, column (from 1)."
    )
    size: tuple[int, int] = Field((1, 1), description="Rows and columns it spans.")
    columns: int = Field(
        1, ge=1, le=8, description="Widgets side by side inside it, filled row by row."
    )
    tone: GroupTone = Field(
        "slate", description="Its colour, a kit token (never a literal)."
    )
    widgets: list[Widget] = Field(min_length=1, description="Its controls, in order.")

    @model_validator(mode="after")
    def _on_the_grid(self) -> "Group":
        if min(*self.at, *self.size) < 1:
            raise ValueError("grid cells and spans count from 1")
        return self


class Screen(VersionedSchemaModel):
    """A control GUI: a title and groups of widgets, drawn by one renderer."""

    schema_version: Literal[1] = Field(1, description="Format version. Leave at 1.")
    title: str = Field(min_length=1, description="The screen's title.")
    groups: list[Group] = Field(
        min_length=1, description="Its groups, placed on the grid."
    )


@dataclass(frozen=True)
class Bound:
    """A widget with what it is bound to: the role, its readback, the indicators' roles.

    ``deferred``: bound to a device the module lists but neither device
    file names (a DB device); its role is judged at page time by the
    module's device route, against ``fits``, the role types the kind takes.
    """

    widget: Widget
    role: Optional[Role] = None
    readback: Optional[Role] = None
    indicators: tuple[
        Optional[Role], ...
    ] = ()  # one per indicator; None for a bare PV or a deferred device
    deferred: bool = False
    fits: tuple[str, ...] = ()


class ScreenError(ValueError):
    """The screen does not fit the experiment's devices; ``problems`` names each."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def _device_roles(
    name: str,
    families: Mapping[str, FamilyFile],
    panel: PanelDevices,
    native: NativeDevices,
) -> tuple[dict[str, Role], tuple[str, ...]]:
    """The device's roles and axes, joined from both files as GeecsCanopy's ``native.py`` joins them."""
    entry, registered = panel.devices.get(name), native.devices.get(name)
    template = entry.family if entry else registered and registered.family_template
    family = families.get(template or "")
    if family is None:
        raise KeyError(f"no device {name!r} in the experiment")
    if entry and registered and registered.family_template != entry.family:
        raise KeyError(f"{name} is a {registered.family_template} in the registry, a {entry.family} in the panel file")  # fmt: skip
    roles = entry.resolve(family) if entry else list(family.roles)
    axes = getattr(registered, "axes", None) or (entry and entry.axes) or []
    if family.per_axis and not axes:
        raise KeyError(f"{name}: the {template} family needs axes")
    return {r.name: r for r in roles}, tuple(axes)


def _accepts(widget: Widget) -> frozenset[str]:
    """The role types ``widget`` takes: its kind's, and a numeric setpoint for a toggle with values."""
    types = _FITS[widget.kind][0]
    return types | {"setpoint"} if widget.values else types


def _misfit(widget: Widget, role: Optional[Role]) -> Optional[str]:
    """Why ``widget`` cannot be bound to ``role``, or ``None`` when it fits."""
    if role is None or widget.kind not in _FITS:
        return None
    types, writes = _accepts(widget), _FITS[widget.kind][1]
    if role.type not in types or (writes and not role.writable):
        return f"{role.name} is a {role.type}, not for a {widget.kind}"
    if widget.values and role.type == "setpoint" and role.text:
        return f"{role.name} is text, not for a {widget.kind}'s values"
    return None


def _unreadable(marks: tuple[Optional[Role], ...], where: str) -> list[str]:
    """The indicators that cannot be read as a readout is (a bare PV's or a deferred device's have no role)."""
    readable = _FITS["readout"][0]
    return [f"{where}: indicator {m.name} is a {m.type}, not readable" for m in marks if m and m.type not in readable]  # fmt: skip


def _later(
    b: Optional[Binding],
    deferred: Collection[str],
    panel: PanelDevices,
    known: NativeDevices,
) -> bool:
    """A device only the module lists (a DB device): its roles are the page's to judge."""
    name = b.device or "" if b is not None else ""
    return bool(name) and name in deferred and name not in panel.devices and name not in known.devices  # fmt: skip


def _fits(widget: Widget, deferred: bool) -> tuple[str, ...]:
    """The role types a deferred widget's kind takes, for the page to check the module's type against."""
    return tuple(sorted(_accepts(widget))) if deferred and widget.kind in _FITS else ()


def _roleless(
    b: Binding,
    deferred: Collection[str],
    panel: PanelDevices,
    known: NativeDevices,
) -> bool:
    """A bare PV, an action, or a deferred device's binding: no Role to look up."""
    return b.pv is not None or b.action is not None or _later(b, deferred, panel, known)


def _unplanned(widget: Widget, actions: Collection[str], where: str) -> list[str]:
    """A button naming a plan the library does not hold."""
    plan = widget.bind.action if widget.bind else None
    return [f"{where}: no action plan {plan!r} in the library"] if plan and plan not in actions else []  # fmt: skip


def resolve(
    screen: Screen,
    families: Mapping[str, FamilyFile],
    panel: PanelDevices,
    native: Optional[NativeDevices] = None,
    deferred: Collection[str] = (),
    actions: Collection[str] = (),
) -> list[list[Bound]]:
    """Bind every widget to its role, group by group; raise :class:`ScreenError` naming every misfit.

    ``deferred``: the devices the module lists (its roster); one that
    neither device file names is bound without a role, for the page to
    judge by the module's device route. A device in neither is refused.
    ``actions``: the plan names in the experiment's library; a button
    naming another is refused.
    """
    known = native or NativeDevices.model_validate({})
    problems: list[str] = []

    later = partial(_later, deferred=deferred, panel=panel, known=known)
    roleless = partial(_roleless, deferred=deferred, panel=panel, known=known)

    def role_of(b: Binding, where: str) -> Optional[Role]:
        if roleless(b):
            return None
        try:
            roles, axes = _device_roles(b.device or "", families, panel, known)
        except KeyError as exc:
            problems.append(f"{where}: {exc.args[0]}")
            return None
        if (b.axis or None) not in (axes or (None,)):
            problems.append(f"{where}: {b.device} takes an axis of {list(axes)}")
        if (role := roles.get(b.role or "")) is None:
            problems.append(f"{where}: {b.device} has no role {b.role!r}")
        return role

    def bind(widget: Widget, where: str) -> Bound:
        problems.extend(_unplanned(widget, actions, where))
        role = role_of(widget.bind, where) if widget.bind else None  # deferred: None
        if why := _misfit(widget, role):
            problems.append(f"{where}: {why}")
        readback = None
        if role is not None and role.readback and widget.bind:
            readback = role_of(widget.bind.model_copy(update={"role": role.readback}), where)  # fmt: skip
        marks = tuple(role_of(i, where) for i in widget.indicators)
        problems.extend(_unreadable(marks, where))
        later_ = later(widget.bind)  # no Role: the module's device route says
        return Bound(widget, role, readback, marks, later_, _fits(widget, later_))

    out = [
        [
            bind(w, f"{g.title!r} widget {n} ({w.kind})")
            for n, w in enumerate(g.widgets, 1)
        ]  # fmt: skip
        for g in screen.groups
    ]
    if problems:
        raise ScreenError(problems)
    return out

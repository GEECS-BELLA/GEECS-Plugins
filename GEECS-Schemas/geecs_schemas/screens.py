"""Screen — a control GUI as a file: titled groups of widgets, each a label and a binding.

``<experiment>/screens/<name>.yaml`` in the configs repository, drawn by
GeecsControlRoom's one renderer (``Planning/operator_surfaces.md`` §3). A
screen binds, groups and confirms; it never computes: there are no
expressions, scripts or conditions here, and no colour but a kit token. A
binding names a GEECS device and a family role (and an axis), or a bare PV
for a device the GEECS DB serves. Whether the role exists on the device and
fits the widget depends on the family files and the instance data, so it is
checked by :func:`resolve`, not by the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel
from geecs_schemas.epics import NativeDevices
from geecs_schemas.epics.family import FamilyFile, Role, RoleName, Suffix
from geecs_schemas.epics.panel_devices import DeviceName, PanelDevices

#: The screens' folder in the experiment's configs folder.
SCREENS_DIRNAME = "screens"

#: The seven widgets: setpoint (with its readback), two-state toggle, flipper
#: (a toggle with in/out indicators), latching button, read-only value, text, picture.
WidgetKind = Literal[
    "setpoint", "toggle", "flipper", "button", "readout", "annotation", "image"
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
    "readout": (frozenset({"readback", "setpoint", "counter", "bit", "enum"}), False),
}
_WRITES = frozenset({"setpoint", "toggle", "flipper", "button"})


class Binding(SchemaModel):
    """What a widget shows and writes: a device's role (on an axis), or a bare PV."""

    device: Optional[DeviceName] = Field(None, description="The GEECS device name.")
    role: Optional[RoleName] = Field(None, description="The device family's role.")
    axis: Optional[Suffix] = Field(
        None, description="The axis, for a per-axis family (a motor's m1)."
    )
    pv: Optional[Suffix] = Field(None, description="A bare PV, for a device the GEECS DB serves (read only).")  # fmt: skip

    @model_validator(mode="after")
    def _one_kind_of_binding(self) -> "Binding":
        named = self.device is not None and self.role is not None
        if named == (self.pv is not None) or (self.pv and (self.device or self.role)):
            raise ValueError("a binding names a device and a role, or a pv, not both")
        if self.axis is not None and not named:
            raise ValueError("an axis belongs to a device's role")
        return self


class Widget(SchemaModel):
    """One control: its kind, the label a person reads, and what it is bound to."""

    kind: WidgetKind = Field(description="The widget, one of the seven.")
    label: str = Field("", description="Free text shown beside it (an annotation's text); empty: the role's label.")  # fmt: skip
    bind: Optional[Binding] = Field(None, description="What it shows and writes; none for an annotation or an image.")  # fmt: skip
    indicators: list[Binding] = Field([], max_length=2, description="A flipper's in and out indicators.")  # fmt: skip
    confirm: Optional[str] = Field(None, min_length=1, description="A hazard note the operator confirms before the write.")  # fmt: skip
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
        if self.confirm and self.kind not in _WRITES:
            raise ValueError(f"a {self.kind} writes nothing to confirm")
        return self


class Group(SchemaModel):
    """A titled block of widgets on the screen's grid, worn in a kit colour token."""

    title: str = Field(description="The group's heading.")
    at: tuple[int, int] = Field(
        description="Its top-left grid cell: row, column (from 1)."
    )
    size: tuple[int, int] = Field((1, 1), description="Rows and columns it spans.")
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
    """A widget with what it is bound to: the role, its readback, the indicators' roles."""

    widget: Widget
    role: Optional[Role] = None
    readback: Optional[Role] = None
    indicators: tuple[Role, ...] = ()


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


def _misfit(widget: Widget, role: Optional[Role]) -> Optional[str]:
    """Why ``widget`` cannot be bound to ``role``, or ``None`` when it fits."""
    if widget.bind and widget.bind.pv and widget.kind in _WRITES:
        return "a bare PV is read only; bind the device's role"
    if any(i.pv for i in widget.indicators):
        return "an indicator is a device's role, not a bare PV"
    if role is None or widget.kind not in _FITS:
        return None
    types, writes = _FITS[widget.kind]
    if role.type not in types or (writes and not role.writable):
        return f"{role.name} is a {role.type}, not for a {widget.kind}"
    return None


def resolve(
    screen: Screen,
    families: Mapping[str, FamilyFile],
    panel: PanelDevices,
    native: Optional[NativeDevices] = None,
) -> list[list[Bound]]:
    """Bind every widget to its role, group by group; raise :class:`ScreenError` naming every misfit."""
    known = native or NativeDevices.model_validate({})
    problems: list[str] = []

    def role_of(b: Binding, where: str) -> Optional[Role]:
        if b.pv is not None:
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
        role = role_of(widget.bind, where) if widget.bind else None
        if why := _misfit(widget, role):
            problems.append(f"{where}: {why}")
        readback = None
        if role is not None and role.readback and widget.bind:
            readback = role_of(widget.bind.model_copy(update={"role": role.readback}), where)  # fmt: skip
        marks = tuple(r for i in widget.indicators if (r := role_of(i, where)))
        problems.extend(
            f"{where}: indicator {m.name} is a {m.type}, not readable"
            for m in marks
            if m.type not in _FITS["readout"][0]  # an indicator reads like a readout
        )
        return Bound(widget, role, readback, marks)

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

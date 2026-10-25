"""ComposedPanel and PanelOverride — the composed-panel spec (layer 2 of the web estate).

A composed panel is ``<experiment>/panels/<name>.yaml`` in the configs
repository, served by the PV bridge: named sections, each a grid of
widgets.  The spec is closed so it never becomes a language
(``Planning/epics_estate.md`` §2.4): a derived value is an IOC record (a
calc, an mbbi), never an expression here; a multi-step button is an
``action`` widget naming an action plan; conditional visibility is one
declarative ``show_when: {pv, equals}`` and nothing more.

A ``PanelOverride`` (``<experiment>/panels/overrides/<family>.yaml``)
adjusts a family's device page for one experiment: hide roles, relabel,
reorder.  It may not add a field or logic.

Both validate their role names against the family files: a panel through
the validation context's ``devices`` (device name → family template, the
bridge's join of the registry and ``panel_devices.yaml``), an override
through its own ``family``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Optional, Union

from pydantic import (
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from geecs_schemas._base import SchemaModel, VersionedSchemaModel
from geecs_schemas.epics.family import (
    FamilyFile,
    RoleName,
    TemplateName,
    families_from,
    refuse_duplicates,
)
from geecs_schemas.epics.panel_devices import DeviceName

#: The layer-3 components a panel may embed, and the family each draws.
COMPONENT_FAMILIES: dict[str, str] = {
    "camera_viewer": "ad_camera_chain",
    "scope_traces": "scope_contract",
}

#: A full PV name: no whitespace.
PvName = Annotated[str, Field(pattern=r"^\S+$")]

#: What a bare PV may be drawn as: read only. A bare PV comes from the raw
#: stream, which writes nothing: a write goes to a device's role, through
#: its allow-list and hazard confirm.
PvType = Literal["readback", "enum", "bit", "counter"]

#: Role types a trend cannot plot: no history is kept of a press, an array
#: or the image.
_UNTRENDABLE = frozenset({"command", "array", "image"})


class ShowWhen(SchemaModel):
    """Show the widget or section only while one PV reads one value."""

    pv: PvName = Field(description="The full PV name read.")
    equals: Union[StrictInt, StrictFloat, StrictStr] = Field(
        description=(
            "The value it must read: a number (a bit as 0 or 1), or an enum's "
            "state name, quoted ('On', not On: YAML reads a bare On as true)."
        )
    )


class _Widget(SchemaModel):
    """What every widget carries: its width in grid columns, an optional title and condition."""

    span: int = Field(1, ge=1, le=12, description="Grid columns the widget spans.")
    title: Optional[str] = Field(None, description="A title above the widget.")
    show_when: Optional[ShowWhen] = Field(
        None, description="Shown only while this PV reads this value."
    )


class DeviceWidget(_Widget):
    """A device's own page, embedded whole or reduced to some of its roles."""

    kind: Literal["device"] = Field(description="Widget kind: 'device'.")
    device: DeviceName = Field(description="The device's GEECS name.")
    roles: Optional[list[RoleName]] = Field(
        None,
        description="Show only these roles, in this order. Absent: the whole page.",
    )

    @field_validator("roles")
    @classmethod
    def _each_once(cls, roles: Optional[list[str]]) -> Optional[list[str]]:
        if roles is not None:
            refuse_duplicates("roles", roles)
        return roles


class ComponentWidget(_Widget):
    """A bespoke view of one device: the camera viewer, the scope traces."""

    kind: Literal["component"] = Field(description="Widget kind: 'component'.")
    component: Literal["camera_viewer", "scope_traces"] = Field(
        description="Which component: 'camera_viewer' or 'scope_traces'."
    )
    device: DeviceName = Field(description="The device it shows.")


class PvWidget(_Widget):
    """One bare PV, shown read-only as a role of the given type (for a PV no family names)."""

    kind: Literal["pv"] = Field(description="Widget kind: 'pv'.")
    pv: PvName = Field(description="The full PV name.")
    type: PvType = Field(description="Drawn as: readback, enum, bit or counter.")
    label: str = Field(min_length=1, description="What the panel calls it.")


class TrendSeries(SchemaModel):
    """One line of a trend: a device's role (from its stream's history), or a bare PV."""

    pv: Optional[PvName] = Field(
        None,
        description=(
            "A bare PV; or give device and role. The raw stream keeps no "
            "history, so a bare PV's line starts when the page opens."
        ),
    )
    device: Optional[DeviceName] = Field(
        None, description="The device whose role is plotted."
    )
    role: Optional[RoleName] = Field(None, description="The role plotted.")
    axis: Optional[str] = Field(
        None,
        min_length=1,
        description="The device's axis, for a per-axis family (a motor's 'm1'); else absent.",
    )
    y_axis: Literal["left", "right"] = Field(
        "left", description="Which of the trend's two y axes it is drawn against."
    )

    @model_validator(mode="after")
    def _a_pv_or_a_role(self) -> "TrendSeries":
        by_pv = self.pv is not None and self.device is None and self.role is None
        by_role = self.pv is None and self.device is not None and self.role is not None
        if not (by_pv or by_role) or (by_pv and self.axis is not None):
            raise ValueError(
                "a trend series names a pv, or a device and a role (and an axis)"
            )
        return self


class TrendWidget(_Widget):
    """A live trend: device roles over their streams' server-kept history, bare PVs from when the page opens."""

    kind: Literal["trend"] = Field(description="Widget kind: 'trend'.")
    series: list[TrendSeries] = Field(min_length=1, description="The lines plotted.")
    minutes: int = Field(10, ge=1, le=120, description="The last N minutes shown.")
    left: Optional[str] = Field(None, description="The left y axis's title.")
    right: Optional[str] = Field(None, description="The right y axis's title.")


class ActionWidget(_Widget):
    """A button that runs an action plan through the scanner (idle only)."""

    kind: Literal["action"] = Field(description="Widget kind: 'action'.")
    plan: str = Field(min_length=1, description="The action plan's name.")


class JournalWidget(_Widget):
    """The bridge's put journal: who wrote what, when, accepted or refused."""

    kind: Literal["journal"] = Field(description="Widget kind: 'journal'.")
    entries: int = Field(10, ge=1, le=200, description="How many recent entries.")


class RosterWidget(_Widget):
    """The experiment's devices with their IOC liveness."""

    kind: Literal["roster"] = Field(description="Widget kind: 'roster'.")


class AlarmsWidget(_Widget):
    """The alarm list, reduced from the device streams' severities."""

    kind: Literal["alarms"] = Field(description="Widget kind: 'alarms'.")


#: Every widget kind, discriminated by ``kind``.
Widget = Annotated[
    Union[
        DeviceWidget,
        ComponentWidget,
        PvWidget,
        TrendWidget,
        ActionWidget,
        JournalWidget,
        RosterWidget,
        AlarmsWidget,
    ],
    Field(discriminator="kind"),
]


class PanelSection(SchemaModel):
    """A titled grid of widgets."""

    title: str = Field(min_length=1, description="The section's title.")
    columns: int = Field(2, ge=1, le=12, description="The grid's columns.")
    widgets: list[Widget] = Field(min_length=1, description="The widgets, row by row.")
    show_when: Optional[ShowWhen] = Field(
        None, description="Shown only while this PV reads this value."
    )

    @model_validator(mode="after")
    def _widgets_fit_the_grid(self) -> "PanelSection":
        wide = [w.kind for w in self.widgets if w.span > self.columns]
        if wide:
            raise ValueError(
                f"section {self.title!r}: widget(s) {wide} span more than its "
                f"{self.columns} columns"
            )
        return self


class ComposedPanel(VersionedSchemaModel):
    """A composed panel: named sections, each a grid of device pages, components, PVs, trends and buttons.

    ``<experiment>/panels/<name>.yaml`` in the configs repository. Derived
    values are IOC records and multi-step buttons are action plans: the
    spec holds no expressions and no logic.
    """

    schema_version: Literal[1] = Field(
        1,
        description=(
            "Format version of this config file. Leave at 1 — tools update "
            "this automatically when the file format changes."
        ),
    )
    title: str = Field(min_length=1, description="The panel's title.")
    sections: list[PanelSection] = Field(
        min_length=1, description="The sections, in order."
    )

    def check_roles(
        self, devices: Mapping[str, str], families: Mapping[str, FamilyFile]
    ) -> None:
        """Refuse a widget naming an unknown device, a role its family lacks, an untrendable role, or a component of another family.

        Raises
        ------
        ValueError
            Naming the section, the device and what is wrong.
        """
        for section in self.sections:
            where = f"section {section.title!r}"
            for widget in section.widgets:
                if isinstance(widget, DeviceWidget):
                    family = _family_of(where, widget.device, devices, families)
                    _has_roles(where, widget.device, family, widget.roles or [])
                elif isinstance(widget, ComponentWidget):
                    family = _family_of(where, widget.device, devices, families)
                    wanted = COMPONENT_FAMILIES[widget.component]
                    if family.template != wanted:
                        raise ValueError(
                            f"{where}: {widget.component} draws a {wanted} device; "
                            f"{widget.device!r} is {family.template}"
                        )
                elif isinstance(widget, TrendWidget):
                    for series in widget.series:
                        if series.device is not None and series.role is not None:
                            _check_series(where, series, devices, families)

    @model_validator(mode="after")
    def _roles_exist(self, info: ValidationInfo) -> "ComposedPanel":
        context = info.context if isinstance(info.context, Mapping) else {}
        devices = context.get("devices")
        if devices is not None:
            self.check_roles(devices, families_from(info))
        return self


def _family_of(
    where: str,
    device: str,
    devices: Mapping[str, str],
    families: Mapping[str, FamilyFile],
) -> FamilyFile:
    template = devices.get(device)
    if template is None:
        raise ValueError(f"{where}: no device {device!r}")
    family = families.get(template)
    if family is None:
        raise ValueError(f"{where}: {device!r} has no family {template!r}")
    return family


def _has_roles(where: str, device: str, family: FamilyFile, roles: list[str]) -> None:
    missing = sorted(set(roles) - set(family.role_names()))
    if missing:
        raise ValueError(
            f"{where}: {device!r} ({family.template}) has no role(s) {missing}"
        )


def _check_series(
    where: str,
    series: TrendSeries,
    devices: Mapping[str, str],
    families: Mapping[str, FamilyFile],
) -> None:
    """Refuse a trend line on a role that is not there, not kept in history, or on no axis of a per-axis device."""
    device, role = str(series.device), str(series.role)
    family = _family_of(where, device, devices, families)
    _has_roles(where, device, family, [role])
    found = family.role(role)
    if found.read_mode != "monitor" or found.type in _UNTRENDABLE:
        raise ValueError(
            f"{where}: {device!r} {role!r} is a {found.type}, not trendable"
        )
    if family.per_axis != (series.axis is not None):
        need = "names its axis" if family.per_axis else "has no axis"
        raise ValueError(f"{where}: a trend of {device!r} ({family.template}) {need}")


class PanelOverride(VersionedSchemaModel):
    """One experiment's adjustments to a family's device page: hide roles, relabel, reorder — nothing else.

    ``<experiment>/panels/overrides/<family>.yaml`` in the configs
    repository. It may not add a field or logic.
    """

    schema_version: Literal[1] = Field(
        1,
        description=(
            "Format version of this config file. Leave at 1 — tools update "
            "this automatically when the file format changes."
        ),
    )
    family: TemplateName = Field(
        description="The family whose page it adjusts (the file's stem)."
    )
    hide: list[RoleName] = Field(
        default_factory=list, description="Roles the page leaves out."
    )
    labels: dict[RoleName, str] = Field(
        default_factory=dict, description="Role → what the page calls it here."
    )
    order: list[RoleName] = Field(
        default_factory=list,
        description="Roles drawn first, in this order; the rest follow in the family's order.",
    )

    @model_validator(mode="after")
    def _roles_exist(self, info: ValidationInfo) -> "PanelOverride":
        families = families_from(info)
        family = families.get(self.family)
        if family is None:
            raise ValueError(
                f"no family {self.family!r}; the families are {sorted(families)}"
            )
        known = set(family.role_names())
        for what, names in (
            ("hide", self.hide),
            ("labels", self.labels),
            ("order", self.order),
        ):
            unknown = sorted(set(names) - known)
            if unknown:
                raise ValueError(
                    f"{what} names role(s) {unknown} that family {self.family!r} does not have"
                )
        refuse_duplicates("hidden roles", self.hide)
        refuse_duplicates("ordered roles", self.order)
        shown = sorted(set(self.hide) & {*self.labels, *self.order})
        if shown:
            raise ValueError(
                f"role(s) {shown} are hidden and also relabelled or ordered"
            )
        return self

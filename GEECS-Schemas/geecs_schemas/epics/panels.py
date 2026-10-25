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
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationInfo,
)
from pydantic import model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel
from geecs_schemas.epics.family import (
    FamilyFile,
    RoleName,
    RoleType,
    TemplateName,
    families_from,
)

#: The layer-3 components a panel may embed, and the family each draws.
COMPONENT_FAMILIES: dict[str, str] = {
    "camera_viewer": "ad_camera_chain",
    "scope_traces": "scope_contract",
}

#: A full PV name: no whitespace.
PvName = Annotated[str, Field(pattern=r"^\S+$")]


class ShowWhen(SchemaModel):
    """Show the widget or section only while one PV reads one value."""

    pv: PvName = Field(description="The full PV name read.")
    equals: Union[StrictBool, StrictInt, StrictFloat, StrictStr] = Field(
        description="The value it must read (an enum by its state name)."
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
    device: str = Field(min_length=1, description="The device's GEECS name.")
    roles: Optional[list[RoleName]] = Field(
        None,
        description="Show only these roles, in this order. Absent: the whole page.",
    )


class ComponentWidget(_Widget):
    """A bespoke view of one device: the camera viewer, the scope traces."""

    kind: Literal["component"] = Field(description="Widget kind: 'component'.")
    component: Literal["camera_viewer", "scope_traces"] = Field(
        description="Which component: 'camera_viewer' or 'scope_traces'."
    )
    device: str = Field(min_length=1, description="The device it shows.")


class PvWidget(_Widget):
    """One bare PV, drawn as a role of the given type (for a PV no family names)."""

    kind: Literal["pv"] = Field(description="Widget kind: 'pv'.")
    pv: PvName = Field(description="The full PV name.")
    type: RoleType = Field(description="The widget to draw it as, by role type.")
    label: str = Field(min_length=1, description="What the panel calls it.")


class TrendSeries(SchemaModel):
    """One line of a trend: a device's role, or a bare PV."""

    pv: Optional[PvName] = Field(
        None, description="A bare PV; or give device and role."
    )
    device: Optional[str] = Field(None, description="The device whose role is plotted.")
    role: Optional[RoleName] = Field(None, description="The role plotted.")
    axis: Literal["left", "right"] = Field(
        "left", description="Which of the trend's two y axes it is drawn against."
    )

    @model_validator(mode="after")
    def _a_pv_or_a_role(self) -> "TrendSeries":
        by_pv = self.pv is not None and self.device is None and self.role is None
        by_role = self.pv is None and self.device is not None and self.role is not None
        if not (by_pv or by_role):
            raise ValueError("a trend series names a pv, or a device and a role")
        return self


class TrendWidget(_Widget):
    """A live trend over the device streams' server-kept history."""

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
        """Refuse a widget naming a device not in *devices*, a role its family lacks, or a component of another family.

        Raises
        ------
        ValueError
            Naming the section, the device and what is wrong.
        """
        for section in self.sections:
            for widget in section.widgets:
                named: list[tuple[str, list[str]]] = []
                if isinstance(widget, DeviceWidget):
                    named.append((widget.device, widget.roles or []))
                elif isinstance(widget, ComponentWidget):
                    named.append((widget.device, []))
                elif isinstance(widget, TrendWidget):
                    named += [
                        (s.device, [s.role])
                        for s in widget.series
                        if s.device is not None and s.role is not None
                    ]
                for device, roles in named:
                    family = _family_of(section.title, device, devices, families)
                    missing = sorted(set(roles) - set(family.role_names()))
                    if missing:
                        raise ValueError(
                            f"section {section.title!r}: {device!r} ({family.template}) "
                            f"has no role(s) {missing}"
                        )
                if isinstance(widget, ComponentWidget):
                    wanted = COMPONENT_FAMILIES[widget.component]
                    if devices[widget.device] != wanted:
                        raise ValueError(
                            f"section {section.title!r}: {widget.component} draws a "
                            f"{wanted} device; {widget.device!r} is "
                            f"{devices[widget.device]}"
                        )

    @model_validator(mode="after")
    def _roles_exist(self, info: ValidationInfo) -> "ComposedPanel":
        context = info.context if isinstance(info.context, Mapping) else {}
        devices = context.get("devices")
        if devices is not None:
            self.check_roles(devices, families_from(info))
        return self


def _family_of(
    section: str,
    device: str,
    devices: Mapping[str, str],
    families: Mapping[str, FamilyFile],
) -> FamilyFile:
    template = devices.get(device)
    if template is None:
        raise ValueError(f"section {section!r}: no device {device!r}")
    family = families.get(template)
    if family is None:
        raise ValueError(f"section {section!r}: {device!r} has no family {template!r}")
    return family


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
        if len(set(self.order)) != len(self.order):
            raise ValueError("order names a role twice")
        return self

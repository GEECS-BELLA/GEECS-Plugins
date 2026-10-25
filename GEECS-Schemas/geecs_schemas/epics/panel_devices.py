"""PanelDevices — per-device panel data, one document per experiment (layer 0b of the web estate).

The configs repository's ``<experiment>/panel_devices.yaml`` (the panels'
``--bench`` file, promoted): for each device a page is drawn for, the
family (record template) it instantiates and what differs from that
family on this device — its own PV names for a role (the two supplies have
no shared record set), a bit's label, the actions the page may take on it
(``allow``), the line the page shows (``note``), the roles it does not
have (``omit``), the roles that need a hazard confirm on it alone
(``hazard``), the roles the archiver takes beyond the family's.

Read by the PV bridge; the archiver's registry pass reads ``archive``
alone (``Planning/epics_estate.md`` §2.5).  The scan worker never reads it, so a device
listed here may still be owned by the GEECS DB (a bench device on a bench
prefix): ``native_devices.yaml`` stays what the worker binds, and a device
in both takes its axes from the registry and its panel data from here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Optional

from pydantic import AfterValidator, Field, ValidationInfo, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel
from geecs_schemas.epics.family import (
    FamilyFile,
    Role,
    RoleName,
    Suffix,
    TemplateName,
    check_suffixes,
    families_from,
)

#: The document's file name in the experiment's configs folder.
PANEL_DEVICES_FILENAME = "panel_devices.yaml"

#: What a URL path segment cannot carry as itself: the path and its ends
#: (``/``, ``?``, ``#``), the escape (``%``), and ``\`` (a browser's ``/``).
_NOT_IN_A_SEGMENT = frozenset("/?#%\\")


def _one_path_segment(name: str) -> str:
    """Refuse a name a page's URL cannot carry as one segment (``/devices/<name>``)."""
    if set(name) <= {"."} or any(
        c in _NOT_IN_A_SEGMENT or c.isspace() or not c.isprintable() for c in name
    ):
        raise ValueError(
            f"{name!r} is not one URL path segment: no / ? # % \\ or whitespace, "
            "and not . or .."
        )
    return name


#: A device's name: one URL path segment.
DeviceName = Annotated[str, AfterValidator(_one_path_segment)]


class PanelDevice(SchemaModel):
    """One device's panel data: its family and what differs from it on this device."""

    family: TemplateName = Field(
        description="The record template the device instantiates (a family file's name)."
    )
    prefix: Optional[str] = Field(
        None,
        pattern=r"^[^\s:]+(:[^\s:]+)*:$",
        description=(
            "The device's PV prefix, ending in ':', for a device on a bench IOC "
            "(not in native_devices.yaml). Absent: '<experiment>:<device>:'."
        ),
    )
    axes: Optional[list[Suffix]] = Field(
        None,
        min_length=1,
        description=(
            "A per-axis family's axes (the motor record names under the prefix), "
            "for a device the registry does not list."
        ),
    )
    roles: dict[RoleName, Suffix] = Field(
        default_factory=dict,
        description=(
            "Role → this device's own record suffix, where it differs from the "
            "family's (a supply's IOC names its own PVs)."
        ),
    )
    labels: dict[RoleName, str] = Field(
        default_factory=dict,
        description="Role → what this device's page calls it (a bit's label: 'chiller flow').",
    )
    allow: Optional[list[RoleName]] = Field(
        None,
        description=(
            "The family actions the page may take on this device (no homing for "
            "actuators on a mirror mount). Absent: every action."
        ),
    )
    note: str = Field("", description="A line the device's page shows.")
    omit: list[RoleName] = Field(
        default_factory=list,
        description=(
            "Family roles this device does not have or use (an unwired "
            "interlock): never read, never shown, and their actions are unavailable."
        ),
    )
    hazard: list[RoleName] = Field(
        default_factory=list,
        description=(
            "Writable roles that need the hazard confirm on this device, beyond "
            "the family's own (one stage's move)."
        ),
    )
    archive: list[RoleName] = Field(
        default_factory=list,
        description="Roles the archiver takes on this device beyond the family's own.",
    )

    def resolve(self, family: FamilyFile) -> list[Role]:
        """The family's roles as this device has them: omissions, suffixes, labels, hazards and archive applied."""
        return [
            role.model_copy(
                deep=True,
                update={
                    "suffix": self.roles.get(role.name, role.suffix),
                    "label": self.labels.get(role.name, role.label),
                    "confirm": "hazard" if role.name in self.hazard else role.confirm,
                    "archive": role.archive or role.name in self.archive,
                },
            )
            for role in family.roles
            if role.name not in self.omit
        ]

    def allowed_actions(self, family: FamilyFile) -> list[str]:
        """The family's actions available on this device (none of its roles omitted), within ``allow``, in order."""
        return [
            a
            for a, action in family.actions.items()
            if (self.allow is None or a in self.allow)
            and not set(action.roles) & set(self.omit)
        ]

    def check_against(self, name: str, family: FamilyFile) -> None:
        """Refuse panel data naming a role, an action or an axis the family does not have.

        Raises
        ------
        ValueError
            Naming the device and what it names that the family lacks.
        """
        known = set(family.role_names())
        for what, names in (
            ("roles", self.roles),
            ("labels", self.labels),
            ("omit", self.omit),
            ("hazard", self.hazard),
            ("archive", self.archive),
        ):
            unknown = sorted(set(names) - known)
            if unknown:
                raise ValueError(
                    f"device {name!r}: {what} names role(s) {unknown} that family "
                    f"{family.template!r} does not have"
                )
        if self.axes is not None and not family.per_axis:
            raise ValueError(f"device {name!r}: family {family.template!r} has no axes")
        self._check_roles(name, family)
        if self.allow is not None:
            self._check_allow(name, family)
        available = set(self.allowed_actions(family))
        for action in available:
            missing = sorted(set(family.actions[action].requires) - available)
            if missing:
                raise ValueError(
                    f"device {name!r}: {action!r} would be available without "
                    f"{missing} (omitted or not allowed)"
                )
        check_suffixes(f"{family.template} on {name}", self.resolve(family))

    def _check_roles(self, name: str, family: FamilyFile) -> None:
        """Refuse a hazard on a read-only role, a split record overridden, an omission a kept role links to."""
        read_only = sorted(r for r in self.hazard if not family.role(r).writable)
        if read_only:
            raise ValueError(
                f"device {name!r}: hazard on read-only role(s) {read_only}"
            )
        both = sorted(
            set(self.omit) & {*self.roles, *self.labels, *self.hazard, *self.archive}
        )
        if both:
            raise ValueError(
                f"device {name!r}: omits role(s) {both} it also configures"
            )
        split = sorted(r for r in self.roles if family.role(r).write is not None)
        if split:
            raise ValueError(
                f"device {name!r}: roles {split} read and write two records; an "
                "override of one would leave the other on the family's record"
            )
        for role in family.roles:
            if role.name in self.omit:
                continue
            for link in ("settled_by", "readback", "fresh_by", "computed_by"):
                if getattr(role, link) in self.omit:
                    raise ValueError(
                        f"device {name!r}: omits {getattr(role, link)!r}, which "
                        f"{role.name!r} needs ({link})"
                    )

    def _check_allow(self, name: str, family: FamilyFile) -> None:
        """Refuse an ``allow`` the family cannot honour on this device."""
        if not family.actions:
            raise ValueError(
                f"device {name!r}: family {family.template!r} has no actions "
                "to allow (each writable role is its own permission)"
            )
        allow = self.allow or []
        unknown = sorted(set(allow) - set(family.actions))
        if unknown:
            raise ValueError(
                f"device {name!r}: unknown action(s) {unknown}; family "
                f"{family.template!r} has {list(family.actions)}"
            )
        for action in allow:
            missing = sorted(set(family.actions[action].requires) - set(allow))
            if missing:
                raise ValueError(
                    f"device {name!r}: allowing {action!r} needs {missing} too"
                )
            gone = sorted(set(family.actions[action].roles) & set(self.omit))
            if gone:
                raise ValueError(
                    f"device {name!r}: allows {action!r} but omits its role(s) {gone}"
                )


class PanelDevices(VersionedSchemaModel):
    """The experiment's devices with a panel: each one's family and per-device panel data.

    ``<experiment>/panel_devices.yaml`` in the configs repository, read by
    the PV bridge (and, for ``archive`` alone, the archiver's registry
    pass). A device here may still be owned by the GEECS DB
    (a bench device): the scan worker reads ``native_devices.yaml``, never
    this file.
    """

    schema_version: Literal[1] = Field(
        1,
        description=(
            "Format version of this config file. Leave at 1 — tools update "
            "this automatically when the file format changes."
        ),
    )
    devices: dict[DeviceName, PanelDevice] = Field(
        default_factory=dict,
        description="GEECS device name → its family and panel data.",
    )

    def families(self) -> Mapping[str, str]:
        """Device name → its family's template name."""
        return {name: device.family for name, device in self.devices.items()}

    @model_validator(mode="after")
    def _devices_fit_their_families(self, info: ValidationInfo) -> "PanelDevices":
        families = families_from(info)
        for name, device in self.devices.items():
            family = families.get(device.family)
            if family is None:
                raise ValueError(
                    f"device {name!r}: no family {device.family!r}; the families "
                    f"are {sorted(families)}"
                )
            device.check_against(name, family)
        return self

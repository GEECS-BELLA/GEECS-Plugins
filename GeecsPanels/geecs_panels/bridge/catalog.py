"""Which devices the bridge serves: the registry joined with the experiment's panel data.

Two documents, both in the configs repository's experiment folder and both
found by the resolver the registry uses (:func:`geecs_core.configs_repo.scanner_configs_base`):

- ``native_devices.yaml`` (the worker's registry, :mod:`geecs_panels.registry`):
  a camera, a positioner, read with their kind's family (:data:`KIND_FAMILIES`);
- ``panel_devices.yaml`` (GEECS-Schemas' ``PanelDevices``, layer 0b): per
  device the family it instantiates and what differs on it (its own
  record names, labels, the permitted actions, a note, the roles it does
  not have). A device in both takes its prefix and axes from the registry
  and its panel data from here; a device here alone is a bench device
  (a prefix of its own, a device the GEECS DB still owns).

The ``--bench`` file is a ``PanelDevices`` document too
(:func:`load_bench_file`); the older bench form (``kind:`` per device) is
still read for the per-kind pages, and its positioners join the bridge as
motor records. Its power supplies cannot (the old form names no family):
:func:`build_catalog` says so in its errors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml
from geecs_core.configs_repo import scanner_configs_base
from geecs_core.pv_naming import native_device_prefix
from geecs_schemas.epics.family import FamilyFile, Role
from geecs_schemas.epics.panel_devices import (
    PANEL_DEVICES_FILENAME,
    PanelDevice,
    PanelDevices,
)
from pydantic import BaseModel, ConfigDict, Field

from geecs_panels.motor import MOTOR_KIND
from geecs_panels.power import POWER_KIND, REQUIRED_ROLES, ROLES, check_roles
from geecs_panels.registry import (
    Device,
    RegistryError,
    load_bench,
    load_devices,
    registry_path,
)
from geecs_panels.scope import SCOPE_KIND

#: Registry kind → the family a registry device instantiates when
#: ``panel_devices.yaml`` names none for it. ``scope`` is the panels' own
#: kind (``--scope``), not yet a registry kind.
KIND_FAMILIES: dict[str, str] = {
    "aravis_camera": "ad_camera_chain",
    MOTOR_KIND: "motor_record",
    SCOPE_KIND: "scope_contract",
}

#: Family kind → the per-kind page's registry kind, for a device that only
#: ``panel_devices.yaml`` (or a bench file in its form) lists.
_PAGE_KINDS: dict[str, str] = {
    "aravis_camera": "aravis_camera",
    "epics_motor": MOTOR_KIND,
    "array_detector": SCOPE_KIND,
    "setpoint_device": POWER_KIND,
}

Source = Literal["registry", "panel_devices"]


@dataclass(frozen=True)
class BridgeDevice:
    """One device the bridge serves: its family, its roles as it has them, where its records are."""

    name: str
    family: FamilyFile
    #: The family's roles with this device's overrides applied (``PanelDevice.resolve``).
    roles: tuple[Role, ...]
    #: The PV prefix, ending in ':'.
    prefix: str
    #: A per-axis family's axes (record names under the prefix); empty otherwise.
    axes: tuple[str, ...] = ()
    #: The family actions permitted on this device, in the family's order.
    actions: tuple[str, ...] = ()
    note: str = ""
    source: Source = "registry"

    def role(self, name: str) -> Role:
        """The role called *name* as this device has it; ``KeyError`` if it has none."""
        for role in self.roles:
            if role.name == name:
                return role
        raise KeyError(f"{self.name} has no role {name!r}")

    def has_role(self, name: str) -> bool:
        """Whether the device has the role (the family's, not omitted)."""
        return any(role.name == name for role in self.roles)

    @property
    def channel_axes(self) -> tuple[str, ...]:
        """The axes its records repeat under: ``("",)`` for a family without axes."""
        return self.axes if self.family.per_axis else ("",)


class DeviceInfo(BaseModel):
    """One row of ``GET /api/devices``: what a page needs to draw the device before its stream opens."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The device's GEECS name, one URL path segment.")
    family: str = Field(description="Its family (record template).")
    kind: str = Field(description="The family's native-device kind.")
    prefix: str = Field(description="Its PV prefix, ending in ':'.")
    axes: list[str] = Field(
        default_factory=list, description="A per-axis family's axes, else empty."
    )
    actions: list[str] = Field(
        default_factory=list,
        description="The family actions permitted on it, in the family's order.",
    )
    note: str = Field("", description="A line its page shows.")
    source: Source = Field(
        description="'registry' (native_devices.yaml) or 'panel_devices' (panel data alone: a bench device)."
    )
    roles: list[Role] = Field(
        description="The family's roles as this device has them: overrides, labels, hazards applied, omissions gone."
    )


class DeviceCatalog(BaseModel):
    """``GET /api/devices``: the experiment's devices with a stream, and why any are missing."""

    model_config = ConfigDict(extra="forbid")

    experiment: str = Field(description="The experiment whose documents were read.")
    devices: list[DeviceInfo] = Field(description="Every device served, in order.")
    errors: list[str] = Field(
        default_factory=list,
        description="Documents or devices that could not be read; the page shows them.",
    )


def device_info(device: BridgeDevice) -> DeviceInfo:
    """A device's row for ``GET /api/devices``."""
    return DeviceInfo(
        name=device.name,
        family=device.family.template,
        kind=device.family.kind,
        prefix=device.prefix,
        axes=list(device.axes),
        actions=list(device.actions),
        note=device.note,
        source=device.source,
        roles=list(device.roles),
    )


def panel_devices_path(experiment: str, base: Optional[Path] = None) -> Optional[Path]:
    """Where the experiment's ``panel_devices.yaml`` would be; ``None`` without a configs repository."""
    base = base if base is not None else scanner_configs_base()
    if base is None:
        return None
    return base / experiment / PANEL_DEVICES_FILENAME


def load_panel_devices(path: Optional[Path]) -> PanelDevices:
    """Read a ``PanelDevices`` document; an absent file (or no configs repository) is an empty one.

    Raises
    ------
    RegistryError
        A file that exists but does not validate (configured-but-broken is
        not "none").
    """
    if path is None or not path.exists():
        return PanelDevices()
    try:
        return PanelDevices.model_validate(yaml.safe_load(path.read_text()) or {})
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise RegistryError(f"panel devices {path} are invalid: {exc}") from exc


def _is_panel_devices_form(document: object) -> bool:
    """Whether a bench document is in the ``PanelDevices`` form (each device names a family)."""
    if not isinstance(document, Mapping):
        return False
    if "schema_version" in document:
        return True
    devices = document.get("devices")
    return isinstance(devices, Mapping) and any(
        isinstance(entry, Mapping) and "family" in entry for entry in devices.values()
    )


def load_bench_file(
    path: Path, experiment: str, families: Mapping[str, FamilyFile]
) -> tuple[list[Device], Optional[PanelDevices]]:
    """Read a ``--bench`` file: its devices for the per-kind pages, and its panel data if it has the new form.

    A ``PanelDevices`` document gives both (:func:`page_devices` derives
    the pages' devices); the older bench form gives the pages' devices and
    no panel data.

    Raises
    ------
    RegistryError
        A file that cannot be read or does not validate.
    """
    try:
        document = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RegistryError(f"bench file {path} is invalid: {exc}") from exc
    if not _is_panel_devices_form(document):
        return load_bench(path), None
    try:
        panels = PanelDevices.model_validate(document)
    except ValueError as exc:
        raise RegistryError(f"bench file {path} is invalid: {exc}") from exc
    return page_devices(panels, experiment, families), panels


def _prefix(experiment: str, name: str, entry: PanelDevice) -> str:
    return entry.prefix or native_device_prefix(experiment, name)


def page_devices(
    panels: PanelDevices, experiment: str, families: Mapping[str, FamilyFile]
) -> list[Device]:
    """The per-kind pages' devices for panel data: each device in the shape its kind's page reads.

    A supply's roles and bits come from its family's resolved roles (a
    fault bit keyed by its role name); one whose roles the supply page
    cannot draw (a required role omitted) gets no page, only a stream.
    """
    out: list[Device] = []
    for name, entry in panels.devices.items():
        family = families[entry.family]
        kind = _PAGE_KINDS[family.kind]
        allow = None if entry.allow is None else frozenset(entry.allow)
        prefix = _prefix(experiment, name, entry)
        if kind == POWER_KIND:
            resolved = entry.resolve(family)
            roles = {r.name: r.suffix for r in resolved if r.name in ROLES}
            if any(r not in roles for r in REQUIRED_ROLES):
                continue
            check_roles(roles, entry.allow)
            bits = tuple((r.name, r.suffix) for r in resolved if r.fault)
            out.append(
                Device(
                    name,
                    kind,
                    prefix,
                    allow=allow,
                    note=entry.note,
                    roles=tuple(roles.items()),
                    bits=bits,
                )
            )
            continue
        out.append(
            Device(
                name,
                kind,
                prefix,
                axes=tuple(entry.axes or ()),
                allow=allow,
                note=entry.note,
            )
        )
    return out


def _from_panel(
    name: str,
    entry: PanelDevice,
    family: FamilyFile,
    *,
    prefix: str,
    axes: Sequence[str],
    note: str,
    source: Source,
) -> BridgeDevice:
    return BridgeDevice(
        name=name,
        family=family,
        roles=tuple(entry.resolve(family)),
        prefix=prefix,
        axes=tuple(axes) if family.per_axis else (),
        actions=tuple(entry.allowed_actions(family)),
        note=entry.note or note,
        source=source,
    )


def build_catalog(
    experiment: str,
    registry: Sequence[Device],
    panels: PanelDevices,
    families: Mapping[str, FamilyFile],
) -> tuple[list[BridgeDevice], list[str]]:
    """The bridge's devices, in order (the registry's, then panel data's own), and why any are left out.

    A registry device takes the family ``panel_devices.yaml`` names for it
    (which must serve its kind), else its kind's (:data:`KIND_FAMILIES`);
    a device of a kind without a family has no stream. A per-axis device
    needs its axes from one of the two documents.
    """
    devices: list[BridgeDevice] = []
    errors: list[str] = []
    for device in registry:
        entry = panels.devices.get(device.name)
        template = entry.family if entry is not None else KIND_FAMILIES.get(device.kind)
        if template is None:
            if device.kind == POWER_KIND:
                errors.append(
                    f"{device.name}: a bench supply in the old form names no family; "
                    "write the bench file as panel devices to stream it"
                )
            continue
        family = families[template]
        if entry is not None and _PAGE_KINDS[family.kind] != device.kind:
            errors.append(
                f"{device.name}: panel data names family {template!r} ({family.kind}), "
                f"but the registry lists a {device.kind}"
            )
            continue
        entry = entry if entry is not None else _default_entry(template, device)
        axes = device.axes or tuple(entry.axes or ())
        if family.per_axis and not axes:
            errors.append(f"{device.name}: a {template} device names no axes")
            continue
        devices.append(
            _from_panel(
                device.name,
                entry,
                family,
                prefix=device.prefix,
                axes=axes,
                note=device.note,
                source="registry",
            )
        )
    listed = {d.name for d in registry}
    for name, entry in panels.devices.items():
        if name in listed:
            continue
        family = families[entry.family]
        if family.per_axis and not entry.axes:
            errors.append(f"{name}: a {entry.family} device names no axes")
            continue
        devices.append(
            _from_panel(
                name,
                entry,
                family,
                prefix=_prefix(experiment, name, entry),
                axes=entry.axes or (),
                note="",
                source="panel_devices",
            )
        )
    return devices, errors


def _default_entry(template: str, device: Device) -> PanelDevice:
    """A registry device's panel data when ``panel_devices.yaml`` has none: its family, its bench allow."""
    allow = sorted(device.allow) if device.allow is not None else None
    return PanelDevice(family=template, allow=allow, note=device.note)


@dataclass(frozen=True)
class Loaded:
    """Everything the process serves, from the experiment's documents and the command line."""

    #: The per-kind pages' devices, in the landing page's order.
    pages: list[Device]
    #: The bridge's devices.
    served: list[BridgeDevice]
    #: Documents that could not be read (the landing page and /api/devices say so).
    errors: list[str]
    #: Devices the bridge cannot stream, and why.
    unserved: list[str]
    registry_path: Optional[Path]


def load_experiment(
    experiment: str,
    families: Mapping[str, FamilyFile],
    *,
    bench: Optional[Path] = None,
    scopes: Sequence[str] = (),
    base: Optional[Path] = None,
) -> Loaded:
    """Read the registry, ``panel_devices.yaml``, the bench file and the ``--scope`` devices.

    An unreadable registry or ``panel_devices.yaml`` is an error the pages
    show, never "no devices"; an unreadable bench file, a device named twice
    or a malformed ``--scope`` raises :class:`RegistryError` (the command
    line is wrong).
    """
    errors: list[str] = []
    path = registry_path(experiment, base)
    try:
        pages = load_devices(experiment, path)
    except RegistryError as exc:
        pages = []
        errors.append(str(exc))
    try:
        panels = load_panel_devices(panel_devices_path(experiment, base))
    except RegistryError as exc:
        panels = PanelDevices()
        errors.append(str(exc))
    # The bridge's registry-shaped devices: the registry, an old-form bench
    # file, the scopes.
    listed = list(pages)
    if bench is not None:
        bench_pages, bench_panels = load_bench_file(bench, experiment, families)
        clash = sorted({d.name for d in pages} & {d.name for d in bench_pages})
        if clash:
            raise RegistryError(f"{clash} in both the registry and the bench file")
        pages += bench_pages
        if bench_panels is None:
            listed += bench_pages
        else:
            twice = sorted(set(panels.devices) & set(bench_panels.devices))
            if twice:
                raise RegistryError(
                    f"{twice} in both panel_devices.yaml and the bench file"
                )
            panels = PanelDevices.model_construct(
                devices={**panels.devices, **bench_panels.devices}
            )
    # Panel data's own devices get the per-kind pages too.
    named = {d.name for d in pages}
    pages += [
        d for d in page_devices(panels, experiment, families) if d.name not in named
    ]
    for spec in scopes:
        name, sep, prefix = spec.partition("=")
        if not sep or not name or not prefix:
            raise RegistryError(f"--scope takes NAME=PREFIX, not {spec!r}")
        scope = Device(name=name, kind=SCOPE_KIND, prefix=prefix)
        pages.append(scope)
        listed.append(scope)
    served, unserved = build_catalog(experiment, listed, panels, families)
    return Loaded(pages, served, errors, unserved, path)

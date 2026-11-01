"""The native source: the experiment's IOC-served devices, their roles expanded into PVs.

Two documents in the experiment's configs folder, joined by GEECS device
name: ``native_devices.yaml`` (what the scan worker binds: kind, axes, an
optional ``template``) and ``panel_devices.yaml`` (per-device panel data:
the family, a device's own suffixes, omitted roles; a bench device's
prefix). Both validate through GEECS-Schemas and the family files come
from its package data. Nothing here reads the GEECS DB.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml
from geecs_core.pv_naming import native_device_prefix
from geecs_schemas import (
    NATIVE_DEVICES_FILENAME,
    PANEL_DEVICES_FILENAME,
    EpicsMotor,
    FamilyFile,
    NativeDevices,
    PanelDevices,
    Role,
    load_families,
)


class SourceError(Exception):
    """A source is configured but cannot be read."""


@dataclass(frozen=True)
class Element:
    """One role or served variable of a device, and its PVs (readback first)."""

    name: str
    pvs: tuple[str, ...]


@dataclass(frozen=True)
class Device:
    """One GEECS device as the directory lists it."""

    name: str
    source: str  # "native" | "gateway": Raccoon's element_type
    tag: str  # the family template (native) or the DB device type (gateway)
    elements: tuple[Element, ...]


@dataclass(frozen=True)
class Resolved:
    """One device the module writes: its roles as this device has them (a native family's, or a DB device's variables)."""

    name: str
    family: str  # the family template (native) or the GEECS DB device type (gateway)
    prefix: str
    axes: tuple[str, ...]  # () unless the family repeats per axis
    roles: tuple[Role, ...]
    actions: tuple[str, ...]  # the family actions available on this device
    ioc: str | None
    writable: frozenset[str]  # the roles a write may name
    source: str = "native"  # "native" | "gateway"

    def bases(self) -> list[tuple[str | None, str]]:
        """``(axis, PV base)`` for each axis, or ``(None, prefix)``."""
        return [(a, self.prefix + a) for a in self.axes] or [(None, self.prefix)]

    def pv(self, role: Role, axis: str | None, *, write: bool = False) -> str:
        """The role's PV on ``axis`` (its write record when ``write`` and it has one)."""
        suffix = (role.write_suffix or role.suffix) if write else role.suffix
        return self.prefix + (axis or "") + suffix


def _load(path: Path, model: type, families: Mapping[str, FamilyFile]):
    if not path.exists():
        return model()
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return model.model_validate(doc, context={"families": families})
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SourceError(f"{path} does not validate: {exc}") from exc


class NativeSource:
    """Reads both documents on every call: they are small, and Raccoon caches the answers."""

    def __init__(
        self,
        experiment: str,
        experiments_root: Path,
        families: Mapping[str, FamilyFile] | None = None,
    ) -> None:
        self.experiment, self.root = experiment, experiments_root
        self.families = load_families() if families is None else families

    def resolve(self) -> list[Resolved]:
        """Every device in either document, joined by name and validated."""
        if not self.root.is_dir():  # an unmounted share is not "no devices"
            raise SourceError(f"experiments root {self.root} is not reachable")
        folder = self.root / self.experiment
        registry = _load(folder / NATIVE_DEVICES_FILENAME, NativeDevices, self.families)
        panels = _load(folder / PANEL_DEVICES_FILENAME, PanelDevices, self.families)
        names = dict.fromkeys([*registry.devices, *panels.devices])
        return [
            self._resolve(name, registry.devices.get(name), panels.devices.get(name))
            for name in names
        ]

    def devices(self) -> list[Device]:
        """Every device, with the family's roles as PVs (Raccoon's elements)."""
        return [_listed(device) for device in self.resolve()]

    def _resolve(self, name: str, entry, panel) -> Resolved:
        template = entry.family_template if entry is not None else panel.family
        if panel is not None and panel.family != template:
            raise SourceError(
                f"{name}: {NATIVE_DEVICES_FILENAME} names family {template!r}, "
                f"{PANEL_DEVICES_FILENAME} {panel.family!r} (one family per device)"
            )
        derived = native_device_prefix(self.experiment, name)
        if (
            entry is not None
            and panel is not None
            and panel.prefix not in (None, derived)
        ):
            raise SourceError(
                f"{name}: a registry device's prefix is derived ({derived!r}); "
                f"{PANEL_DEVICES_FILENAME} must not set {panel.prefix!r}"
            )
        prefix = panel.prefix if panel is not None and panel.prefix else derived
        family = self.families[template]
        roles = panel.resolve(family) if panel is not None else family.roles
        axes: list[str] = []
        if family.per_axis:
            axes = entry.axes if isinstance(entry, EpicsMotor) else (panel.axes or [])
            if not axes:
                raise SourceError(f"{name}: the {template} family needs axes")
        actions = panel.allowed_actions(family) if panel else list(family.actions)
        ioc = panel.ioc if panel is not None else None
        # those of the available actions, else every writable role
        writable = {r for a in actions for r in family.actions[a].roles} if family.actions else {r.name for r in roles if r.writable}  # fmt: skip
        return Resolved(
            name, template, prefix, tuple(axes), tuple(roles), tuple(actions), ioc,
            frozenset(writable),
        )  # fmt: skip


def _listed(device: Resolved) -> Device:
    bases = [
        (device.name if axis is None else f"{device.name}:{axis}", base)
        for axis, base in device.bases()
    ]
    elements = tuple(
        Element(
            f"{label}:{role.name}",
            tuple(
                base + s
                for s in dict.fromkeys([role.suffix, role.write or role.suffix])
            ),
        )
        for label, base in bases
        for role in device.roles
    )
    return Device(device.name, "native", device.family, elements)

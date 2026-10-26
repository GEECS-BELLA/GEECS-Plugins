"""GEECS's three sources of device knowledge, as one flat device list.

A device is a GEECS device name; an element is one of its family roles (native)
or served variables (gateway), named ``<Device>:<role>`` (``<Device>:<axis>:<role>``
per axis) or ``<Device>:<variable>``; it owns its PVs, readback first.  Native:
the registry, prefix ``pv_name(experiment, device) + ":"``, the family whose
``kind`` is the registry kind (a role's ``write`` record is a second PV).
Gateway: the CA gateway's served set from ``GeecsDb`` (PV_CONTRACT §1), ``:SP``
when settable, plus ``<expt>:<dev>:connected``.  ``families/`` is vendored from
``GEECS-Schemas/geecs_schemas/epics/families/`` at 7f38c12ac (``feat/web-estate``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml
from geecs_core.db.geecs_db import GeecsDb
from geecs_core.db.served_set import served_variables
from geecs_core.pv_naming import device_status_pv, normalize_component, pv_name
from geecs_core.pv_naming import setpoint_pv


class SourceError(Exception):
    """A source is configured but cannot be read (maps to 503)."""


@dataclass(frozen=True)
class Element:
    """One role or served variable of a device, and its PVs."""

    name: str
    pvs: tuple[str, ...]


@dataclass(frozen=True)
class Device:
    """One GEECS device as the directory lists it."""

    name: str
    source: str  # "native" | "gateway"
    tag: str  # the family template (native) or the DB device type (gateway)
    elements: tuple[Element, ...]


@dataclass(frozen=True)
class Family:
    """A family file reduced to what the directory needs."""

    template: str
    kind: str
    per_axis: bool
    roles: tuple[tuple[str, tuple[str, ...]], ...]  # (role, (suffix[, write]))


def load_families() -> dict[str, Family]:
    """Read the vendored family files, keyed by the registry kind they serve."""
    families: dict[str, Family] = {}
    for entry in sorted(resources.files(__package__).joinpath("families").iterdir()):
        if entry.name.endswith(".json"):
            doc = json.loads(entry.read_text())
            roles = tuple(
                (
                    r["name"],
                    (r["suffix"], r["write"]) if "write" in r else (r["suffix"],),
                )
                for r in doc["roles"]
            )
            per_axis = bool(doc.get("per_axis"))
            # easydriver and fastps both claim "setpoint_device": first wins;
            # no registry kind names either today (see the report).
            families.setdefault(
                doc["kind"], Family(doc["template"], doc["kind"], per_axis, roles)
            )
    return families


def _native(experiment: str, name: str, entry: dict, family: Family) -> Device:
    prefix = pv_name(experiment, name) + ":"
    bases = [(f"{name}:{a}", prefix + a) for a in entry.get("axes", [])]
    if not family.per_axis:
        bases = [(name, prefix)]
    elements = tuple(
        Element(f"{label}:{role}", tuple(base + s for s in suffixes))
        for label, base in bases
        for role, suffixes in family.roles
    )
    return Device(name, "native", family.template, elements)


def native_devices(
    experiment: str, experiments_root: Path, families: dict[str, Family]
) -> list[Device]:
    """The registry's devices with their family roles expanded into PVs."""
    if not experiments_root.is_dir():  # an unmounted share is not "no devices"
        raise SourceError(f"experiments root {experiments_root} is not reachable")
    path = experiments_root / experiment / "native_devices.yaml"
    if not path.exists():
        return []
    try:
        entries = (yaml.safe_load(path.read_text()) or {}).get("devices") or {}
        found = [(n, e, families.get(e.get("kind", ""))) for n, e in entries.items()]
    except (OSError, yaml.YAMLError, AttributeError) as exc:
        raise SourceError(f"native registry {path} is unreadable: {exc}") from exc
    unknown = [n for n, _, family in found if family is None]
    if unknown:
        raise SourceError(f"native registry: no family for the kind of {unknown}")
    return [_native(experiment, n, e, family) for n, e, family in found]


def gateway_devices(experiment: str) -> list[Device]:
    """The CA gateway's devices: the served-set rule over batched DB queries."""
    try:
        endpoints = GeecsDb.get_experiment_devices(experiment)
        var_map = GeecsDb.get_experiment_device_variables(experiment)
        sub_map = GeecsDb.get_subscribed_variables(experiment)
        types = GeecsDb.get_experiment_device_types(experiment)
    except Exception as exc:  # the DB is the one upstream: any failure is 503
        raise SourceError(f"GEECS DB unavailable: {exc}") from exc
    devices = []
    for name, served in served_variables(var_map, sub_map, devices=endpoints).items():
        elements = []
        for var in served:
            readback = pv_name(experiment, name, var.name)
            pvs = (readback, setpoint_pv(readback)) if var.settable else (readback,)
            elements.append(Element(f"{name}:{var.name}", pvs))
        status = Element(f"{name}:connected", (device_status_pv(experiment, name),))
        elements.append(status)
        devices.append(Device(name, "gateway", types.get(name, ""), tuple(elements)))
    return devices


def merged(native: list[Device], gateway: list[Device]) -> list[Device]:
    """Both lists by name; a native device shadows its DB twin.

    The scan worker refuses a name in both (one owner per device); the
    directory keeps answering and lets the native entry win.
    """
    owned = {normalize_component(d.name) for d in native}
    rest = [d for d in gateway if normalize_component(d.name) not in owned]
    return sorted(native + rest, key=lambda d: d.name.lower())

"""Family files and device instances: the data a GEECS device route is built from.

The four files under ``families/`` are vendored verbatim from
``GEECS-Schemas/geecs_schemas/epics/families/`` at GEECS-Plugins commit
``7f38c12ace88`` (branch ``feat/web-estate``); ``devices.json`` maps the
simulator's devices onto them. Nothing here imports Canopy.
"""

import json
from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class Role:
    """One named PV of a family: what it is and how its freshness is judged."""

    name: str
    suffix: str
    type: str
    writable: bool = False
    write: str | None = None
    fresh_by: str | None = None
    computed_by: str | None = None
    settled_by: str | None = None


@dataclass(frozen=True)
class Action:
    """A named group of writable roles, the unit a device's allow list names."""

    roles: tuple[str, ...]
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class Family:
    """A family file: roles (per axis or not) and the actions over them."""

    template: str
    per_axis: bool
    roles: dict[str, Role]
    actions: dict[str, Action]


@dataclass(frozen=True)
class Device:
    """A device instance: a family at a PV prefix, its axes and allowed actions."""

    name: str
    family: Family
    prefix: str
    axes: tuple[str, ...]
    allow: tuple[str, ...]

    def pv(self, role: Role, axis: str | None, *, write: bool = False) -> str:
        """Return the PV name of ``role`` on ``axis`` (the write PV if asked)."""
        suffix = role.write if write and role.write else role.suffix
        return f"{self.prefix}{axis or ''}{suffix}"

    def axis_ok(self, axis: str | None) -> bool:
        """Whether ``axis`` names one of the device's axes (``None`` iff it has none)."""
        return axis in self.axes if self.axes else axis is None

    def allowed_roles(self) -> set[str]:
        """Return every role reachable through one of the device's allowed actions."""
        return {
            r
            for a in self.allow
            if a in self.family.actions
            for r in self.family.actions[a].roles
        }


_ROLE_KEYS = ("writable", "write", "fresh_by", "computed_by", "settled_by")


def _read_json(path: str) -> dict:
    return json.loads(resources.files(__package__).joinpath(path).read_text())


def load_family(template: str) -> Family:
    """Read ``families/<template>.json`` from the package."""
    raw = _read_json(f"families/{template}.json")
    roles = {
        r["name"]: Role(
            name=r["name"],
            suffix=r["suffix"],
            type=r["type"],
            **{k: r[k] for k in _ROLE_KEYS if k in r},
        )
        for r in raw["roles"]
    }
    actions = {
        name: Action(roles=tuple(a["roles"]), requires=tuple(a.get("requires", ())))
        for name, a in raw.get("actions", {}).items()
    }
    return Family(
        template=raw["template"],
        per_axis=raw.get("per_axis", False),
        roles=roles,
        actions=actions,
    )


def load_devices() -> dict[str, Device]:
    """Read the packaged ``devices.json`` into device instances."""
    raw = _read_json("devices.json")
    families: dict[str, Family] = {}
    devices = {}
    for name, d in raw.items():
        family = families.setdefault(d["family"], load_family(d["family"]))
        devices[name] = Device(
            name=name,
            family=family,
            prefix=d["prefix"],
            axes=tuple(d.get("axes", ())),
            allow=tuple(d.get("allow", ())),
        )
    return devices

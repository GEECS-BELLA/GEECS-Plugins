"""Which devices get a panel: the experiment's native-device registry.

The registry is GEECS-Schemas' ``NativeDevices`` document,
``<configs>/experiments/<Experiment>/native_devices.yaml`` — the same file
the scan worker reads. A device's PV prefix is derived, never stored:
``<experiment>:<device>:`` under GEECS-Core's naming rule. No hosts, no
ports: the IOC publishes its own stream URL.

**The bench file** (``--bench``) is the panels' own, beside the registry:
bench positioners the registry does not list (a bench prefix, a device the GEECS
DB still owns, so the worker must not see it), each with its kind, its
prefix and the kind's options, and the panel's per-device options —
``allow``, the actions the panel may take on the device (a subset of the
kind's, e.g. no homing for actuators on mirror mounts), and ``note``, a
line the page shows. It is not the worker's document: a panel's
permissions are the panel's, and a device the worker cannot open must not
appear in its registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import yaml
from geecs_core.configs_repo import scanner_configs_base
from geecs_core.pv_naming import pv_name
from geecs_schemas import NATIVE_DEVICES_FILENAME, EpicsMotor, NativeDevices
from pydantic import BaseModel, ConfigDict, Field, model_validator

from geecs_panels.motor import MOTOR_KIND, check_actions


class RegistryError(RuntimeError):
    """The registry is configured but cannot be read."""


@dataclass(frozen=True)
class Device:
    """One native device: its GEECS name, its kind, its PV prefix and, for a positioner, its axes."""

    name: str
    kind: str
    prefix: str
    #: A positioner's axes — the motor record names under the prefix; empty for other kinds.
    axes: tuple[str, ...] = ()
    #: The panel actions the device permits (its kind's names); ``None``: every one.
    allow: Optional[frozenset[str]] = None
    #: A line the device's page shows (the bench file's ``note``).
    note: str = ""


def registry_path(experiment: str, base: Path | None = None) -> Path | None:
    """Where the experiment's ``native_devices.yaml`` would be; ``None`` without a configs repository."""
    base = base if base is not None else scanner_configs_base()
    if base is None:
        return None
    return base / experiment / NATIVE_DEVICES_FILENAME


def load_devices(experiment: str, path: Path | None) -> list[Device]:
    """Read the registry into devices, in the file's order.

    An absent file is an experiment with no native devices. A file that
    exists but does not validate raises: configured-but-broken is not
    "none".
    """
    if path is None:
        return []
    if not path.parent.parent.is_dir():
        # An unmounted share must not read as "no native devices".
        raise RegistryError(f"configs directory {path.parent.parent} is not reachable")
    if not path.exists():
        return []
    try:
        document = yaml.safe_load(path.read_text()) or {}
        registry = NativeDevices.model_validate(document)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise RegistryError(f"native device registry {path} is invalid: {exc}") from exc
    return [
        Device(
            name=name,
            kind=entry.kind,
            prefix=pv_name(experiment, name) + ":",
            axes=tuple(entry.axes) if isinstance(entry, EpicsMotor) else (),
        )
        for name, entry in registry.devices.items()
    ]


class BenchMotor(EpicsMotor):
    """A positioner on a bench IOC: the registry entry's rules, plus its prefix and the panel's options."""

    prefix: str = Field(pattern=r"^[^\s:]+(:[^\s:]+)*:$")
    allow: Optional[list[str]] = None
    note: str = ""

    @model_validator(mode="after")
    def _actions_are_the_kinds(self) -> "BenchMotor":
        if self.allow is not None:
            check_actions(self.allow)
        return self


#: A bench device's name: it is a URL path segment (no slash, no whitespace).
BenchName = Annotated[str, Field(pattern=r"^[^/\s]+$")]


class BenchFile(BaseModel):
    """The panels' bench file: bench devices by name."""

    model_config = ConfigDict(extra="forbid")

    devices: dict[BenchName, BenchMotor] = Field(default_factory=dict)


def load_bench(path: Path) -> list[Device]:
    """Read a bench file into devices, in its order; :class:`RegistryError` if it cannot be."""
    try:
        document = yaml.safe_load(path.read_text()) or {}
        bench = BenchFile.model_validate(document)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise RegistryError(f"bench file {path} is invalid: {exc}") from exc
    return [
        Device(
            name=name,
            kind=MOTOR_KIND,
            prefix=entry.prefix,
            axes=tuple(entry.axes),
            allow=None if entry.allow is None else frozenset(entry.allow),
            note=entry.note,
        )
        for name, entry in bench.devices.items()
    ]

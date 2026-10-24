"""Which devices get a panel: the experiment's native-device registry.

The registry is GEECS-Schemas' ``NativeDevices`` document,
``<configs>/experiments/<Experiment>/native_devices.yaml`` — the same file
the scan worker reads. A device's PV prefix is derived, never stored:
``<experiment>:<device>:`` under GEECS-Core's naming rule. No hosts, no
ports: the IOC publishes its own stream URL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from geecs_core.configs_repo import scanner_configs_base
from geecs_core.pv_naming import pv_name
from geecs_schemas import NATIVE_DEVICES_FILENAME, NativeDevices


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
            axes=tuple(getattr(entry, "axes", ())),
        )
        for name, entry in registry.devices.items()
    ]

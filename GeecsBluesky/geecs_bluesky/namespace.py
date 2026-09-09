"""The GEECS device namespace: every device of an experiment as a long-lived noun.

Built once at queue-server ``environment open`` (or by a headless session)
from the GEECS DB roster — the same batch queries GeecsPvaGateway uses to
decide what it serves — and exported into the worker namespace so stock
plans can be given devices **by name** (``count([UC_Amp4Input])``,
``scan([...], U_S1H.current, -1, 1, 5)``) exactly as the queue server
expects.  Design: ``Planning/native_bluesky/01_device_namespace.md``.

Constructing a :class:`~geecs_bluesky.devices.geecs_device.GeecsDevice`
touches no hardware; connection happens on first use through
:func:`geecs_bluesky.preprocessors.connect_on_demand`.

A DB failure **raises** (:class:`~geecs_bluesky.exceptions.GeecsConfigurationError`):
a worker with a silently empty roster would fail every plan with "unknown
device", which is worse than a loud failure at environment open.  Tests
and offline tooling build from an explicit :class:`DeviceRoster` instead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from geecs_bluesky.devices.geecs_device import (
    ACQ_TIMESTAMP_VARIABLE,
    GeecsDevice,
    GeecsTriggeredDevice,
    VariableMeta,
    identifier_name,
)
from geecs_bluesky.exceptions import GeecsConfigurationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceRoster:
    """What the GEECS DB says about one experiment's devices.

    ``variables`` is ``{device: [variable metadata dict, ...]}`` in the
    ``GeecsDb.get_experiment_device_variables`` shape; ``types`` maps device
    → devicetype; ``subscribed`` maps device → the ``get='yes'`` variable
    names (the default read selection); ``endpoints`` maps device → the
    GEECS ``(ip, port)`` (informational).
    """

    experiment: str
    variables: Mapping[str, list[Mapping[str, Any]]]
    types: Mapping[str, str] = field(default_factory=dict)
    subscribed: Mapping[str, list[str]] = field(default_factory=dict)
    endpoints: Mapping[str, tuple[str, int]] = field(default_factory=dict)

    @classmethod
    def from_geecs_db(
        cls, experiment: str, *, geecs_db: Any | None = None
    ) -> DeviceRoster:
        """Load the roster with the DB's batch queries (one connection each).

        Raises :class:`GeecsConfigurationError` on any DB failure — see the
        module docstring for why this is loud rather than empty.
        """
        if geecs_db is None:
            try:
                from geecs_core.db.geecs_db import GeecsDb as geecs_db
            except Exception as exc:  # pragma: no cover - environment-dependent
                raise GeecsConfigurationError(
                    "device namespace: geecs_core (the GEECS DB client) is not "
                    "importable — install GEECS-Core or build the namespace from "
                    "an explicit DeviceRoster"
                ) from exc
        try:
            variables = geecs_db.get_experiment_device_variables(experiment)
            types = geecs_db.get_experiment_device_types(experiment)
            subscribed = geecs_db.get_subscribed_variables(experiment)
            endpoints = geecs_db.get_experiment_devices(experiment)
        except Exception as exc:
            raise GeecsConfigurationError(
                f"device namespace: could not load the {experiment!r} device "
                f"roster from the GEECS DB ({type(exc).__name__}: {exc}). The "
                "worker cannot register devices without it — check the DB "
                "configuration ([Database] in config.ini) and connectivity."
            ) from exc
        return cls(
            experiment=experiment,
            variables=variables,
            types=types,
            subscribed=subscribed,
            endpoints=endpoints,
        )


class GeecsNamespace:
    """The experiment's devices as ophyd-async objects, addressable by name.

    Parameters
    ----------
    roster : DeviceRoster
        The DB description of the devices.
    motor_targets : mapping, optional
        ``{device: {variable, ...}}`` of settable variables the scan-variable
        catalog marks ``kind: motor`` (see :meth:`motor_targets_from_catalog`).
    """

    def __init__(
        self,
        roster: DeviceRoster,
        *,
        motor_targets: Mapping[str, set[str]] | None = None,
    ) -> None:
        self.experiment = roster.experiment
        self.roster = roster
        self._devices: dict[str, GeecsDevice] = {}
        self._by_geecs_name: dict[str, GeecsDevice] = {}
        motor_targets = motor_targets or {}
        skipped: list[str] = []
        for device, rows in roster.variables.items():
            metas = [VariableMeta.from_db(row) for row in rows]
            triggered = any(m.name.lower() == ACQ_TIMESTAMP_VARIABLE for m in metas)
            if not triggered and not any(m.is_scalar for m in metas):
                skipped.append(device)
                continue
            ns_name = identifier_name(device)
            clash = self._devices.get(ns_name)
            if clash is not None:
                raise GeecsConfigurationError(
                    f"device namespace: GEECS devices {clash.geecs_name!r} and "
                    f"{device!r} both normalise to the name {ns_name!r}"
                )
            cls = GeecsTriggeredDevice if triggered else GeecsDevice
            dev = cls(
                device,
                metas,
                experiment=roster.experiment,
                subscribed=roster.subscribed.get(device),
                motor_variables=motor_targets.get(device, ()),
                devicetype=roster.types.get(device, ""),
                name=ns_name,
            )
            self._devices[ns_name] = dev
            self._by_geecs_name[device.lower()] = dev
        if skipped:
            logger.info(
                "device namespace: %d device(s) without scalar variables not "
                "registered: %s",
                len(skipped),
                ", ".join(sorted(skipped)),
            )
        logger.info(
            "device namespace: %d device(s) registered for %s",
            len(self._devices),
            roster.experiment,
        )

    # ------------------------------------------------------------ builders
    @classmethod
    def from_experiment(
        cls,
        experiment: str,
        *,
        resolver: Any | None = None,
        geecs_db: Any | None = None,
    ) -> GeecsNamespace:
        """Build from the GEECS DB (loud on failure); catalog motor kinds if a resolver is given."""
        roster = DeviceRoster.from_geecs_db(experiment, geecs_db=geecs_db)
        motors = (
            cls.motor_targets_from_catalog(resolver) if resolver is not None else {}
        )
        return cls(roster, motor_targets=motors)

    @staticmethod
    def motor_targets_from_catalog(resolver: Any) -> dict[str, set[str]]:
        """Return ``{device: {variable}}`` for catalog entries with ``kind: motor``.

        Best effort: a missing or unreadable catalog yields ``{}`` (the DB
        tolerance still promotes a settable to a motor), logged at WARNING.
        """
        try:
            catalog = resolver.scan_variable_catalog()
        except Exception:
            logger.warning(
                "device namespace: scan-variable catalog unavailable; motor "
                "kinds come from DB tolerances only",
                exc_info=True,
            )
            return {}
        motors: dict[str, set[str]] = {}
        for spec in catalog.variables.values():
            if getattr(spec, "kind", None) != "motor":
                continue
            target = str(getattr(spec, "target", ""))
            device, sep, variable = target.partition(":")
            if sep:
                motors.setdefault(device, set()).add(variable)
        return motors

    # ------------------------------------------------------------- lookups
    @property
    def devices(self) -> Mapping[str, GeecsDevice]:
        """Namespace name → device, in roster order."""
        return self._devices

    def __len__(self) -> int:
        """The number of registered devices."""
        return len(self._devices)

    def __iter__(self) -> Iterator[GeecsDevice]:
        """Iterate the devices in roster order."""
        return iter(self._devices.values())

    def __contains__(self, name: object) -> bool:
        """Whether *name* (GEECS or attribute spelling) is a registered device."""
        return isinstance(name, str) and self._lookup(name) is not None

    def __getitem__(self, name: str) -> GeecsDevice:
        """Return the device for *name* (GEECS or attribute spelling); ``KeyError`` if absent."""
        dev = self._lookup(name)
        if dev is None:
            raise KeyError(
                f"device namespace: no device {name!r} in {self.experiment!r} "
                f"({len(self._devices)} registered)"
            )
        return dev

    def _lookup(self, name: str) -> GeecsDevice | None:
        return self._devices.get(name) or self._by_geecs_name.get(name.lower())

    def resolve(self, target: str) -> Any:
        """Return the object for ``"Device"`` or ``"Device:Variable"``.

        ``"U_S1H:current"`` → the Movable child; ``"UC_Amp4Input"`` → the
        device.  Either spelling (GEECS or attribute) of both parts works.
        """
        device_name, sep, variable = target.partition(":")
        device = self[device_name]
        return device.child(variable) if sep else device

    # -------------------------------------------------------------- export
    def export_into(self, namespace: dict[str, Any]) -> list[str]:
        """Bind every device into *namespace* (a module's ``globals()``).

        Returns the bound names, for the profile's ``__all__``.  Refuses to
        overwrite an existing binding — a device named like a plan or the
        RunEngine would silently shadow it otherwise.
        """
        clashes = [n for n in self._devices if n in namespace]
        if clashes:
            raise GeecsConfigurationError(
                "device namespace: these device names already exist in the "
                f"startup namespace and would shadow them: {', '.join(clashes)}"
            )
        namespace.update(self._devices)
        return list(self._devices)

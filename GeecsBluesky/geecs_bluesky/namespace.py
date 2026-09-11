"""The GEECS device namespace: every device of an experiment as a long-lived noun.

Built once at queue-server ``environment open`` (or by a headless session)
from the GEECS DB roster and exported into the worker namespace so stock
plans can be given devices **by name** — ``count([UC_Amp4_IR_input])``,
``scan([...], U_S1H.Current, -1, 1, 5)`` — exactly as the queue server
expects.  Design: ``Planning/native_bluesky/01_device_namespace.md`` and the
device-layer audit beside it.

The namespace owns **no device behaviour**.  It composes the existing
device layer (``Planning/native_bluesky/01a_device_layer_audit.md``):

* a device that acquires per shot (:func:`looks_triggerable`) is a
  :class:`~geecs_bluesky.devices.detector.GeecsDetector` — a stock
  ``StandardDetector`` whose ``trigger()`` waits for its ``acq_timestamp``
  to advance; ``native_save`` iff the DB lists both ``save`` and
  ``localsavingpath`` for it (so the gateway serves their ``:SP``), in
  which case the detector owns those two controls and a ``PathProvider``
  given at build points its files at the run (§4.A of the plan of record);
* any other device is a
  :class:`~geecs_bluesky.devices.ca.snapshot.CaSnapshotReadable`;
* each served **settable** variable is attached to that object as a child
  Movable — :class:`~geecs_bluesky.devices.ca.motor.CaMotor` when the DB
  gives it a tolerance (readback convergence), else
  :class:`~geecs_bluesky.devices.ca.settable.CaSettable` — so
  ``bps.mv(U_S1H.Current, 0.5)`` moves with the GEECS semantics those
  classes already implement.

What each object *reads* is the DB's subscribed (``get='yes'``) list — what
GEECS itself logs — resolved by the same
:class:`~geecs_bluesky.db_runtime.GeecsDbScalarPolicy` the scan path uses;
which variables *exist* as children is the gateway's served set, from the
same :class:`~geecs_bluesky.db_runtime.GeecsDbServedSetProvider` the
unserved-variables preflight uses; every variable's CA type is
:func:`geecs_core.db.variable_types.effective_vartype`, the rule the gateway
typed the PV with.  Nothing here restates a rule that has a home elsewhere.

Constructing a device touches no hardware; connection happens on first use
(:func:`geecs_bluesky.preprocessors.connect_on_demand`).  A DB failure at
build **raises**: a worker with a silently empty roster would fail every
plan with "unknown device", which is worse than a loud failure at
environment open.  Tests and offline tooling build from an explicit
:class:`DeviceRoster`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ophyd_async.core import Device, PathProvider

from geecs_core.db.variable_types import (
    VARTYPE_TO_DTYPE,
    effective_vartype,
    image_variables,
    is_scalar_vartype,
)

from geecs_bluesky.db_runtime import (
    GeecsDbDeviceTypes,
    GeecsDbScalarPolicy,
    GeecsDbServedSetProvider,
)
from geecs_bluesky.devices.ca.motor import CaMotor
from geecs_bluesky.devices.ca.settable import CaSettable
from geecs_bluesky.devices.ca.snapshot import CaSnapshotReadable
from geecs_bluesky.devices.detector import GeecsDetector
from geecs_bluesky.devices.hdf_plugin import PluginPathProvider
from geecs_bluesky.devices.hdf_plugin import file_plugin_hosts as _hosts_from_config
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.utils import identifier_name, safe_name, settable_attribute

logger = logging.getLogger(__name__)

#: GEECS variable that advances once per shot (the shot join key).
ACQ_TIMESTAMP_VARIABLE = "acq_timestamp"

#: Devicetypes that carry ``Trigger.*`` variables because they *generate* or
#: *route* triggers (delay generators, a bipolar supply with a trigger input)
#: — never acquirers.  :func:`looks_triggerable` excludes them.
TRIGGER_SOURCE_DEVICETYPES: frozenset[str] = frozenset(
    {"dg645", "dg535", "highland t564 ddg", "tdk-lambda z bipolar"}
)

#: Served scalar dtype (the gateway's vocabulary) → the Python type ophyd-async
#: declares.  ``str`` on an enum PV reads the label with the choices as
#: metadata; ``str`` on a char-array (path) PV is the long-string convention.
_DTYPE_TO_PYTHON: dict[str, type] = {
    "float": float,
    "string": str,
    "path": str,
    "enum": str,
}

#: Variables the gateway synthesises for every device that are never children.
_SYNTHESIZED: frozenset[str] = frozenset({"connected", ACQ_TIMESTAMP_VARIABLE})

#: The two LabVIEW-native saving controls.  A triggerable device whose DB
#: rows list both as **settable** (only settable variables get a gateway
#: ``:SP``, PV_CONTRACT.md §1) gets ``native_save`` and the detector owns
#: them (``localsavingpath`` / ``save`` children driven by its data logic) —
#: they are never bound as scan-settable children, so no plan can write
#: them behind the lifecycle.
NATIVE_SAVE_VARIABLES: frozenset[str] = frozenset({"save", "localsavingpath"})

_TRIGGER_VARIABLE = re.compile("trig", re.IGNORECASE)


# --------------------------------------------------------------------- rules
def looks_triggerable(rows: Sequence[Mapping[str, Any]], devicetype: str = "") -> bool:
    """Whether a device acquires per shot (so it gets a Bluesky ``trigger()``).

    ``acq_timestamp`` is generated inside LabVIEW and is not (yet) a DB
    variable, so this is the agreed shortcut (Sam, 2026-09-09): a device
    whose devicetype variables mention a trigger (``trigger``,
    ``TriggerDelay``, ``EnableTrigger``, …) is a triggered acquirer —
    cameras, spectrometers, ICT scopes, DAQ pads — unless its devicetype is a
    trigger *source* (:data:`TRIGGER_SOURCE_DEVICETYPES`).  A DB row named
    ``acq_timestamp`` is authoritative once the DB grows it.  Checked live
    against every Undulator device pushing ``acq_timestamp``: no misses; the
    extras were idle acquirers and the excluded sources.
    """
    names = [str(r["name"]) for r in rows]
    if any(n.lower() == ACQ_TIMESTAMP_VARIABLE for n in names):
        return True
    kind = devicetype.strip().lower()
    if any(source in kind for source in TRIGGER_SOURCE_DEVICETYPES):
        return False
    return any(_TRIGGER_VARIABLE.search(n) for n in names)


def python_type(row: Mapping[str, Any]) -> type | None:
    """The ophyd datatype for a DB variable row, or ``None`` for a non-scalar."""
    effective = effective_vartype(row.get("variabletype"), row.get("choices"))
    if not is_scalar_vartype(effective):
        return None
    return _DTYPE_TO_PYTHON[VARTYPE_TO_DTYPE.get(effective, "float")]


# -------------------------------------------------------------------- roster
@dataclass(frozen=True)
class DeviceRoster:
    """What the GEECS DB says about one experiment's devices.

    ``variables``: ``{device: [row, ...]}`` in the
    ``GeecsDb.get_experiment_device_variables`` shape (``name``, ``settable``,
    ``variabletype``, ``choices``, ``tolerance`` …).  ``types``: device →
    devicetype.  ``subscribed``: device → the ``get='yes'`` variables (what
    ``read()`` returns).  ``served``: device → the gateway's served set;
    ``None`` means "compute it from the rows" (subscribed ∪ settable, the
    provider's rule).  ``triggered``: explicit per-device overrides of
    :func:`looks_triggerable`.  ``endpoints``: device → the GEECS endpoint
    IP (the camera server that would serve its file plugin, #806).
    """

    experiment: str
    variables: Mapping[str, list[Mapping[str, Any]]]
    types: Mapping[str, str] = field(default_factory=dict)
    subscribed: Mapping[str, list[str]] = field(default_factory=dict)
    served: Mapping[str, set[str]] | None = None
    triggered: Mapping[str, bool] = field(default_factory=dict)
    endpoints: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_geecs_db(
        cls, experiment: str, *, geecs_db: Any | None = None
    ) -> DeviceRoster:
        """Load the roster; the rules come from the ``db_runtime`` providers.

        Raises :class:`GeecsConfigurationError` on any DB failure — see the
        module docstring for why this is loud rather than empty.
        """
        if geecs_db is None:
            try:
                from geecs_core.db.geecs_db import GeecsDb as geecs_db
            except Exception as exc:  # pragma: no cover - environment-dependent
                raise GeecsConfigurationError(
                    "device namespace: geecs_core (the GEECS DB client) is not "
                    "importable — install GEECS-Core or build from a DeviceRoster"
                ) from exc
        try:
            variables = geecs_db.get_experiment_device_variables(experiment)
        except Exception as exc:
            raise GeecsConfigurationError(
                f"device namespace: could not load the {experiment!r} device "
                f"roster from the GEECS DB ({type(exc).__name__}: {exc}). The "
                "worker cannot register devices without it — check the DB "
                "configuration ([Database] in config.ini) and connectivity."
            ) from exc
        served = GeecsDbServedSetProvider(experiment, db=geecs_db).served_by_device()
        if served is None:
            raise GeecsConfigurationError(
                f"device namespace: the gateway served set for {experiment!r} could "
                "not be read from the GEECS DB (see the warning above)"
            )
        try:
            endpoints = {
                device: ip
                for device, (ip, _port) in geecs_db.get_experiment_devices(
                    experiment
                ).items()
            }
        except Exception as exc:
            raise GeecsConfigurationError(
                f"device namespace: could not load the {experiment!r} device "
                f"endpoints from the GEECS DB ({type(exc).__name__}: {exc})"
            ) from exc
        return cls(
            experiment=experiment,
            variables=variables,
            types=GeecsDbDeviceTypes(experiment, db=geecs_db).by_device(),
            subscribed=GeecsDbScalarPolicy(
                experiment, db=geecs_db
            ).subscribed_by_device(),
            served=served,
            endpoints=endpoints,
        )

    def served_for(self, device: str) -> set[str]:
        """The served variable names for *device* (lower-cased).

        Without an explicit ``served`` map (rosters built in tests / offline)
        the gateway rule is still the provider's: it is run over this roster
        through :class:`_RosterDb`, so the rule is never restated here.
        """
        served = self.served
        if served is None:
            served = GeecsDbServedSetProvider(
                self.experiment, db=_RosterDb(self)
            ).served_by_device()
        return {v.lower() for v in (served or {}).get(device, ())}


class _RosterDb:
    """A :class:`DeviceRoster` behind the two ``GeecsDb`` calls the served-set provider makes."""

    def __init__(self, roster: DeviceRoster) -> None:
        self._roster = roster

    def get_subscribed_variables(self, experiment: str, *, enabled_only: bool = True):
        return {d: list(v) for d, v in self._roster.subscribed.items()}

    def get_experiment_device_variables(
        self, experiment: str, *, enabled_only: bool = True
    ):
        return {d: list(rows) for d, rows in self._roster.variables.items()}


# ----------------------------------------------------------------- namespace
_HOSTS_FROM_CONFIG = object()


class GeecsNamespace:
    """The experiment's devices as ophyd-async objects, addressable by name.

    Parameters
    ----------
    roster :
        What the DB says.
    path_provider :
        The run-scoped provider every file-writing detector shares (the
        claim preprocessor points it at each run's folder).
    file_plugin_hosts :
        Camera-server IPs whose gateway serves the file plugin (#806): a
        triggerable device with an image-typed variable on one of them is
        plugin-backed (stock ``ADHDFDataLogic`` over the plugin's PVs); the
        same device elsewhere keeps LabVIEW-native saving.  Defaults to
        ``config.ini [pva] file_plugin_addr_list``; absent or ``None``
        means no host (the rollout is opt-in per box).
    """

    def __init__(
        self,
        roster: DeviceRoster,
        *,
        path_provider: PathProvider | None = None,
        file_plugin_hosts: set[str] | None | object = _HOSTS_FROM_CONFIG,
    ) -> None:
        self.experiment = roster.experiment
        self.roster = roster
        self._path_provider = path_provider
        hosts = (
            _hosts_from_config()
            if file_plugin_hosts is _HOSTS_FROM_CONFIG
            else file_plugin_hosts
        )
        self._file_plugin_hosts: set[str] = set(hosts or ())
        self._devices: dict[str, Any] = {}
        self._by_geecs_name: dict[str, Any] = {}
        self._attrs: dict[str, dict[str, str]] = {}  # ns name → {lower var → attr}
        skipped: list[str] = []
        for device, rows in roster.variables.items():
            built = self._build(device, rows, roster)
            if built is None:
                skipped.append(device)
                continue
            ns_name, dev = built
            clash = self._devices.get(ns_name)
            if clash is not None:
                raise GeecsConfigurationError(
                    f"device namespace: GEECS devices {clash._geecs_device_name!r} "
                    f"and {device!r} both normalise to the name {ns_name!r}"
                )
            self._devices[ns_name] = dev
            self._by_geecs_name[device.lower()] = dev
        detectors = [d for d in self._devices.values() if isinstance(d, GeecsDetector)]
        plugin = sorted(d._geecs_device_name for d in detectors if d.plugin_backed)
        saving = sorted(
            d._geecs_device_name
            for d in detectors
            if d.native_save and not d.plugin_backed
        )
        logger.info(
            "device namespace: %d device(s) registered for %s (%d detectors, "
            "%d on the file plugin: %s; %d with native saving: %s)",
            len(self._devices),
            roster.experiment,
            len(detectors),
            len(plugin),
            ", ".join(plugin) or "none",
            len(saving),
            ", ".join(saving) or "none",
        )
        if skipped:
            logger.info(
                "device namespace: %d device(s) without served scalars not registered: %s",
                len(skipped),
                ", ".join(sorted(skipped)),
            )

    @classmethod
    def from_experiment(
        cls,
        experiment: str,
        *,
        geecs_db: Any | None = None,
        path_provider: PathProvider | None = None,
        file_plugin_hosts: set[str] | None | object = _HOSTS_FROM_CONFIG,
    ) -> GeecsNamespace:
        """Build from the GEECS DB (loud on failure)."""
        return cls(
            DeviceRoster.from_geecs_db(experiment, geecs_db=geecs_db),
            path_provider=path_provider,
            file_plugin_hosts=file_plugin_hosts,
        )

    # ------------------------------------------------------------------ build
    def _build(
        self, device: str, rows: Sequence[Mapping[str, Any]], roster: DeviceRoster
    ) -> tuple[str, Any] | None:
        served = roster.served_for(device)
        typed: dict[str, tuple[Mapping[str, Any], type]] = {}
        for row in rows:
            name = str(row["name"])
            if name.lower() not in served or name.lower() in _SYNTHESIZED:
                continue
            py = python_type(row)
            if py is not None:
                typed[name] = (row, py)
        devicetype = roster.types.get(device, "")
        triggered = (
            bool(roster.triggered[device])
            if device in roster.triggered
            else looks_triggerable(rows, devicetype)
        )
        if not typed and not triggered:
            return None
        ns_name = identifier_name(device)
        ophyd_name = safe_name(device)  # event keys per EVENT_SCHEMA.md
        settable_names = {
            n.lower() for n, (row, _) in typed.items() if row.get("settable")
        }
        native_save = triggered and NATIVE_SAVE_VARIABLES <= settable_names
        if native_save:
            # The detector owns the saving controls (§10.5 namespace rule).
            typed = {
                n: v for n, v in typed.items() if n.lower() not in NATIVE_SAVE_VARIABLES
            }
        settables = {
            n: (row, py) for n, (row, py) in typed.items() if row.get("settable")
        }
        # A GEECS name mangles into an attribute / event-column component
        # through safe_name — lowercased, runs of punctuation and whitespace
        # collapsed to one underscore — so 'Trigger'/'trigger' AND
        # 'Position.Axis 1'/'Position Axis 1' all land on one attribute.
        # Refuse rather than silently drop one, the way both gateways raise on
        # a PV-name collision after normalization (gateway.py, PVA server.py).
        by_lower: dict[str, str] = {}
        by_attr: dict[str, str] = {}
        for n in typed:
            attr = safe_name(n)
            if attr in by_attr:
                raise GeecsConfigurationError(
                    f"device namespace: {device}: variables {by_attr[attr]!r} and "
                    f"{n!r} both normalise to {attr!r}"
                )
            by_attr[attr] = n
            by_lower[n.lower()] = n
        # Readable columns: the subscribed list, minus settables (their
        # Movable child carries the readback) — the same subscribed list the
        # scan path's scalar policy resolves for a db_scalars save-set entry.
        readables = [
            by_lower[v.lower()]
            for v in roster.subscribed.get(device, ())
            if v.lower() in by_lower and by_lower[v.lower()] not in settables
        ]
        datatypes = {n: py for n, (_, py) in typed.items()}
        dev: Any
        if triggered:
            # Plugin-backed iff the DB lists an image variable and the
            # device's camera server serves the file plugin (#806).  The
            # LabVIEW-native path stays on beside it — PNG dual-write until
            # PNG retirement (#738), the parity evidence of the rollout.
            plugin_vars = (
                image_variables(rows)
                if self._path_provider is not None
                and roster.endpoints.get(device) in self._file_plugin_hosts
                else []
            )
            dev = GeecsDetector(
                device,
                readables,
                experiment=roster.experiment,
                name=ophyd_name,
                datatypes=datatypes,
                path_provider=self._path_provider if native_save else None,
                native_save=native_save,
                hdf_plugins=[
                    (var, PluginPathProvider(self._path_provider, device))
                    for var in plugin_vars
                ],
            )
        else:
            dev = CaSnapshotReadable(
                device,
                readables,
                experiment=roster.experiment,
                name=ophyd_name,
                datatypes=datatypes,
            )
        dev._geecs_namespace_member = True  # connect_on_demand's marker
        attrs: dict[str, str] = {safe_name(v).lower(): safe_name(v) for v in readables}
        for var, (row, py) in settables.items():
            attr = self._attribute_for(dev, var)
            if attr.lower() in attrs.values() or attr.lower() in attrs:
                raise GeecsConfigurationError(
                    f"device namespace: {device}: variable {var!r} collides on "
                    f"attribute {attr!r}"
                )
            child = self._movable(device, var, row, py, roster.experiment)
            setattr(dev, attr, child)  # ophyd-async registers + names the child
            # The parent carries every child's "Device Variable" header too
            # (the scalar_headers preprocessor walks descendants anyway).
            dev._column_headers.update(child._column_headers)
            if var.lower() in {v.lower() for v in roster.subscribed.get(device, ())}:
                dev.add_readables([child])  # subscribed settable: log its readback
            attrs[var.lower()] = attr
            attrs[attr.lower()] = attr
        for var in readables:
            attrs[var.lower()] = safe_name(var)
        self._attrs[dev.name] = attrs  # keyed by the ophyd name variable() sees
        return ns_name, dev

    @staticmethod
    def _attribute_for(dev: Any, variable: str) -> str:
        """The attribute a settable binds to; a Bluesky/ophyd name gets a trailing ``_``.

        The Amp4 camera has a settable enum called ``trigger`` (external
        trigger on/off) — bound verbatim it would overwrite ``trigger()``.
        :func:`~geecs_bluesky.utils.settable_attribute` is the rule (a frozen
        set of the detector class's names, pinned by a test, so the client
        seam spells the same attribute).  A name already bound to a *child* of this
        device (a readable signal, ``acq_timestamp``, ``connected_status``) is
        a real collision and raises rather than being renamed (review N1).
        """
        attr = settable_attribute(variable)  # the shared rule (utils)
        existing = getattr(dev, attr, None)
        if isinstance(existing, Device):
            raise GeecsConfigurationError(
                f"device namespace: {dev._geecs_device_name}: settable {variable!r} "
                f"collides with the child already bound as {attr!r}"
            )
        if existing is not None:
            return attr + "_"
        return attr

    @staticmethod
    def _movable(
        device: str, var: str, row: Mapping[str, Any], py: type, experiment: str
    ) -> Any:
        """One served settable → ``CaMotor`` (DB tolerance) or ``CaSettable``."""
        tolerance = row.get("tolerance")
        if py is float and tolerance is not None and float(tolerance) > 0:
            # A positive DB tolerance means "confirm the readback converged".
            # 0 / NULL → a plain setpoint: many numeric settables (exposure,
            # command-like values) never echo within a tolerance, and the
            # catalog's `kind: motor` opt-in arrives with the axes in phase 3.
            return CaMotor(
                device, var, experiment=experiment, tolerance=float(tolerance)
            )
        return CaSettable(device, var, experiment=experiment, datatype=py)

    # ---------------------------------------------------------------- lookups
    @property
    def devices(self) -> Mapping[str, Any]:
        """Namespace name → device, in roster order."""
        return self._devices

    def __len__(self) -> int:
        """The number of registered devices."""
        return len(self._devices)

    def __iter__(self) -> Iterator[Any]:
        """Iterate the devices in roster order."""
        return iter(self._devices.values())

    def __contains__(self, name: object) -> bool:
        """Whether *name* (GEECS or namespace spelling) is a registered device."""
        return isinstance(name, str) and self._lookup(name) is not None

    def __getitem__(self, name: str) -> Any:
        """The device for *name* (GEECS or namespace spelling); ``KeyError`` if absent."""
        dev = self._lookup(name)
        if dev is None:
            raise KeyError(
                f"device namespace: no device {name!r} in {self.experiment!r} "
                f"({len(self._devices)} registered)"
            )
        return dev

    def _lookup(self, name: str) -> Any | None:
        return self._devices.get(name) or self._by_geecs_name.get(name.lower())

    def variable(self, device: str, variable: str) -> Any:
        """The child for a variable: the Movable for a settable, the signal otherwise."""
        dev = self[device]
        attrs = self._attrs[dev.name]
        try:
            return getattr(dev, attrs[variable.lower()])
        except KeyError:
            raise KeyError(
                f"device namespace: {dev._geecs_device_name}: no served scalar "
                f"variable {variable!r}"
            ) from None

    def resolve(self, target: str) -> Any:
        """The object for ``"Device"`` or ``"Device:Variable"`` (either spelling)."""
        device, sep, variable = target.partition(":")
        return self.variable(device, variable) if sep else self[device]

    def telemetry(self) -> list[Any]:
        """Every subscribed scalar of the experiment, readable without a trigger.

        The ``SupplementalData`` baseline list (plan of record §4.B): each
        scalar-only device whole, and each detector's scalar **signals**
        individually — a detector itself is ``Triggerable`` and a baseline
        read would wait for a shot that ARMED never delivers.
        """
        objects: list[Any] = []
        for dev in self._devices.values():
            if isinstance(dev, GeecsDetector):
                objects.extend(dev._scalar_signals())
            else:
                objects.append(dev)
        return objects

    # ----------------------------------------------------------------- export
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

"""GeecsDevice — one ophyd-async device per GEECS device, built from the DB roster.

The noun the stock plans take.  Where the per-scan path builds a fresh
role-specific object (detector / contributor / snapshot / motor / settable)
from a request, a :class:`GeecsDevice` is built **once** from the GEECS DB's
description of the device (``Planning/native_bluesky/01_device_namespace.md``):

* one child per **scalar** variable, attribute-named after the variable —
  a read-only ``epics_signal_r`` for readbacks, a
  :class:`~geecs_bluesky.devices.ca.settable.CaSettable` /
  :class:`~geecs_bluesky.devices.ca.motor.CaMotor` (both ``Movable``) for
  settable variables, so ``bps.mv(U_S1H.current, 0.5)`` moves with the
  GEECS convergence semantics those classes already implement;
* non-scalar variables (``image``, ``1darray``, …) are **not** CA children —
  they are served over PVA and belong to the image-writer track (#806);
* a non-readable ``connected_status`` child (the gateway's ``CONNECTED`` PV);
* :class:`GeecsTriggeredDevice` — for devices with an ``acq_timestamp``
  variable — adds the persistent shot monitor and ``trigger()`` from
  :mod:`geecs_bluesky.devices.ca.shot_monitor`, so a stock
  ``trigger_and_read`` is one GEECS shot.

**What ``read()`` returns is a selection.**  A GEECS save set names the
variables to log; a stock plan just says ``count([UC_Amp4Input])``.  The
device therefore reads its *selected* variables: by default the DB's
subscribed (``get='yes'``) list — what GEECS itself logs — else every scalar
variable.  A plan changes the selection with the stock
``bps.configure(dev, variables=[...])`` (the Bluesky ``Configurable``
convention: ``configure`` returns ``(old, new)`` and the selection is part
of ``read_configuration``, so every descriptor records what was logged).
A triggered device always reads ``acq_timestamp`` too — it is the shot join
key downstream and is never deselectable.

Naming: the ophyd name of a child is the GEECS variable name when it is a
valid identifier, else :func:`~geecs_bluesky.utils.safe_name` of it; the
device keeps the original GEECS names for PV minting and lookups
(:meth:`GeecsDevice.child`).  Lookups accept either spelling.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bluesky.protocols import Reading
from event_model import DataKey
from ophyd_async.core import AsyncStatus, Device, SignalR, merge_gathered_dicts
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.devices.ca._pv import ca_pv
from geecs_bluesky.devices.ca.motor import CaMotor
from geecs_bluesky.devices.ca.settable import CaSettable
from geecs_bluesky.devices.ca.shot_monitor import ShotTriggerMixin
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)

#: GEECS variable that advances once per shot (the shot join key).
ACQ_TIMESTAMP_VARIABLE = "acq_timestamp"

#: GEECS ``variabletype`` values that are not scalars — never CA children here.
NON_SCALAR_TYPES: frozenset[str] = frozenset(
    {"image", "1darray", "2darray", "3darray", "array", "waveform"}
)

#: GEECS ``variabletype`` values that are definitely CA numerics.  Anything
#: else — including the many DB rows with **no** ``variabletype`` — is left to
#: ophyd-async to infer from the PV at connect (``datatype=None``): the
#: gateway serves enums, strings and char-array paths that a guessed
#: ``float`` cannot coerce, and a guess wrong on one child fails the whole
#: device's connect.  A DB ``tolerance``/``min``/``max`` is taken as a numeric
#: hint (``U_S1H:Current`` has no variabletype but a tolerance).
NUMERIC_TYPES: frozenset[str] = frozenset(
    {"numeric", "double", "float", "int", "integer"}
)

#: Variables the gateway serves for every device that are never data columns.
_RESERVED_VARIABLES: frozenset[str] = frozenset({"connected"})


def identifier_name(geecs_name: str) -> str:
    """Return the ophyd attribute / namespace name for a GEECS name.

    The GEECS spelling is kept when it is already a Python identifier
    (``U_S1H``, ``UC_Amp4Input``, ``current``) — that is what operators see
    in GEECS and what they will type into a plan argument.  Anything else
    (``Position.Axis 1``) goes through :func:`~geecs_bluesky.utils.safe_name`.
    """
    return geecs_name if geecs_name.isidentifier() else safe_name(geecs_name)


@dataclass(frozen=True)
class VariableMeta:
    """One GEECS variable as the DB describes it (``GeecsDb`` metadata shape)."""

    name: str
    settable: bool = False
    variabletype: str | None = None
    units: str = ""
    min: float | None = None
    max: float | None = None
    tolerance: float | None = None
    choices: str | None = None

    @classmethod
    def from_db(cls, row: Mapping[str, Any]) -> VariableMeta:
        """Build from a ``GeecsDb.get_experiment_device_variables`` entry."""
        vtype = row.get("variabletype")
        return cls(
            name=str(row["name"]),
            settable=bool(row.get("settable", False)),
            variabletype=str(vtype).strip().lower() if vtype else None,
            units=str(row.get("units") or ""),
            min=row.get("min"),
            max=row.get("max"),
            tolerance=row.get("tolerance"),
            choices=row.get("choices"),
        )

    @property
    def is_scalar(self) -> bool:
        """Whether this variable is a CA scalar (else PVA/non-scalar, skipped)."""
        return (self.variabletype or "") not in NON_SCALAR_TYPES

    @property
    def is_numeric(self) -> bool:
        """Whether the DB says (or hints) this is a numeric CA scalar."""
        if (self.variabletype or "") in NUMERIC_TYPES:
            return True
        if self.variabletype:
            return False  # an explicit non-numeric type (choice, string, path, …)
        return any(v is not None for v in (self.tolerance, self.min, self.max))

    @property
    def datatype(self) -> type | None:
        """``float`` for numerics; ``None`` = let ophyd-async infer from the PV."""
        return float if self.is_numeric else None


class GeecsDevice(Device):
    """A GEECS device as a long-lived ophyd-async noun (see module docstring).

    Parameters
    ----------
    device : str
        GEECS device name (e.g. ``"U_S1H"``).
    variables : iterable of VariableMeta or DB metadata dicts
        Every variable the DB lists for the device.  Non-scalars are skipped.
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    subscribed : sequence of str, optional
        The DB's ``get='yes'`` variables — the default read selection.
    motor_variables : iterable of str, optional
        Settable variables the scan-variable catalog marks ``kind: motor``;
        they become :class:`CaMotor` children even without a DB tolerance.
    devicetype : str
        GEECS devicetype, kept for callers (asset registry, capture policy).
    name : str
        ophyd-async device name (defaults to the identifier form of *device*).
    """

    #: Marker the connect-on-demand preprocessor keys on (walking ``.parent``).
    _geecs_namespace_member: bool = True

    def __init__(
        self,
        device: str,
        variables: Iterable[VariableMeta | Mapping[str, Any]],
        *,
        experiment: str | None = None,
        subscribed: Sequence[str] | None = None,
        motor_variables: Iterable[str] = (),
        devicetype: str = "",
        name: str = "",
    ) -> None:
        self._geecs_device_name = device
        self._experiment = experiment
        self.devicetype = devicetype
        self._variables: dict[str, VariableMeta] = {}
        self._attr_of: dict[str, str] = {}
        self._readback_of: dict[str, SignalR] = {}
        self._by_lookup: dict[str, str] = {}  # lowercase GEECS / attr → GEECS name
        motor_attrs = {identifier_name(v).lower() for v in motor_variables}

        for raw in variables:
            meta = raw if isinstance(raw, VariableMeta) else VariableMeta.from_db(raw)
            if not meta.is_scalar or meta.name.lower() in _RESERVED_VARIABLES:
                continue
            if meta.name.lower() == ACQ_TIMESTAMP_VARIABLE:
                continue  # GeecsTriggeredDevice owns the shot stamp child
            attr = identifier_name(meta.name)
            if attr.lower() in {a.lower() for a in self._attr_of.values()}:
                raise ValueError(
                    f"{device}: variables {meta.name!r} and "
                    f"{self._by_lookup[attr.lower()]!r} collide on attribute {attr!r}"
                )
            child, readback = self._build_child(meta, motor_attrs)
            setattr(self, attr, child)
            self._variables[meta.name] = meta
            self._attr_of[meta.name] = attr
            self._readback_of[meta.name] = readback
            self._by_lookup[meta.name.lower()] = meta.name
            self._by_lookup[attr.lower()] = meta.name

        # Liveness: the gateway's per-device CONNECTED PV — a child (connects
        # and mocks with the device) but never a data column.
        self.connected_status = epics_signal_r(
            str, ca_pv(experiment, device, "CONNECTED")
        )
        super().__init__(name=name or identifier_name(device))

        wanted = [self._by_lookup.get(v.lower()) for v in (subscribed or ())]
        default = [v for v in wanted if v is not None] or list(self._variables)
        self._default_selected: tuple[str, ...] = tuple(default)
        self._selected: tuple[str, ...] = self._default_selected

    # ------------------------------------------------------------------ build
    def _build_child(
        self, meta: VariableMeta, motor_attrs: set[str]
    ) -> tuple[Device, SignalR]:
        """Return ``(child, readback_signal)`` for one scalar variable."""
        device, experiment = self._geecs_device_name, self._experiment
        if not meta.settable:
            signal = epics_signal_r(meta.datatype, ca_pv(experiment, device, meta.name))
            return signal, signal
        if not meta.is_numeric:
            # enum / string / path setpoint: type inferred at connect
            child = CaSettable(device, meta.name, experiment=experiment, datatype=None)
            return child, child.readback
        is_motor = identifier_name(meta.name).lower() in motor_attrs or bool(
            meta.tolerance and meta.tolerance > 0
        )
        if is_motor:
            motor = CaMotor(
                device,
                meta.name,
                experiment=experiment,
                tolerance=float(meta.tolerance) if meta.tolerance else 0.005,
            )
            return motor, motor.position
        settable = CaSettable(device, meta.name, experiment=experiment)
        return settable, settable.readback

    # --------------------------------------------------------------- lookups
    @property
    def geecs_name(self) -> str:
        """The GEECS device name (PV component spelling)."""
        return self._geecs_device_name

    @property
    def variables(self) -> tuple[str, ...]:
        """Every scalar variable, GEECS spelling, DB order."""
        return tuple(self._variables)

    @property
    def settable_variables(self) -> tuple[str, ...]:
        """The scalar variables the gateway accepts ``:SP`` writes for."""
        return tuple(v for v, m in self._variables.items() if m.settable)

    @property
    def selected(self) -> tuple[str, ...]:
        """The variables ``read()`` currently returns (GEECS spelling)."""
        return self._selected

    def variable_name(self, lookup: str) -> str:
        """Return the GEECS spelling for a GEECS or attribute spelling; ``KeyError`` if unknown."""
        try:
            return self._by_lookup[lookup.lower()]
        except KeyError:
            raise KeyError(
                f"{self._geecs_device_name}: no scalar variable {lookup!r} "
                f"(known: {', '.join(self._variables) or 'none'})"
            ) from None

    def child(self, lookup: str) -> Device:
        """Return the child object for a variable (the Movable for a settable)."""
        return getattr(self, self._attr_of[self.variable_name(lookup)])

    def readback(self, lookup: str) -> SignalR:
        """Return the read-only signal that carries a variable's streamed value."""
        return self._readback_of[self.variable_name(lookup)]

    def meta(self, lookup: str) -> VariableMeta:
        """Return the DB metadata for a variable."""
        return self._variables[self.variable_name(lookup)]

    # -------------------------------------------------------------- reading
    def _read_signals(self) -> list[SignalR]:
        return [self._readback_of[v] for v in self._selected]

    async def read(self) -> dict[str, Reading]:
        """Read the selected variables (Bluesky ``Readable``)."""
        return await merge_gathered_dicts([sig.read() for sig in self._read_signals()])

    async def describe(self) -> dict[str, DataKey]:
        """Describe the selected variables (Bluesky ``Readable``)."""
        return await merge_gathered_dicts(
            [sig.describe() for sig in self._read_signals()]
        )

    @AsyncStatus.wrap
    async def stage(self) -> None:
        """Start caching the selected signals (monitor-backed reads)."""
        for sig in self._read_signals():
            await sig.stage()

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        """Stop caching the selected signals."""
        for sig in self._read_signals():
            await sig.unstage()

    # -------------------------------------------------------- configuration
    def _configuration_key(self) -> str:
        return f"{self.name}-variables"

    def _configuration_reading(self) -> dict[str, Reading]:
        return {
            self._configuration_key(): Reading(
                value=",".join(self._selected), timestamp=time.time()
            )
        }

    def configure(
        self, *, variables: Sequence[str] | None = None
    ) -> tuple[dict[str, Reading], dict[str, Reading]]:
        """Select which variables ``read()`` returns (Bluesky ``configure`` convention).

        ``variables=None`` restores the default selection.  Names may be in
        GEECS or attribute spelling; unknown names raise ``KeyError`` before
        anything changes.  Returns ``(old, new)`` configuration readings.
        """
        old = self._configuration_reading()
        if variables is None:
            self._selected = self._default_selected
        else:
            self._selected = tuple(
                dict.fromkeys(self.variable_name(v) for v in variables)
            )
        return old, self._configuration_reading()

    async def read_configuration(self) -> dict[str, Reading]:
        """Report the selection (Bluesky ``Configurable``)."""
        return self._configuration_reading()

    async def describe_configuration(self) -> dict[str, DataKey]:
        """Describe the selection column (Bluesky ``Configurable``)."""
        return {
            self._configuration_key(): DataKey(
                dtype="string",
                shape=[],
                source=f"geecs://{self._geecs_device_name}/variables",
            )
        }

    # --------------------------------------------------------------- teardown
    async def disconnect(self) -> None:
        """Release per-child resources (uniform teardown hook; idempotent)."""
        for attr in self._attr_of.values():
            child = getattr(self, attr)
            teardown = getattr(child, "disconnect", None)
            if teardown is not None and not isinstance(child, SignalR):
                await teardown()

    def __repr__(self) -> str:
        """A short device summary (kind, GEECS name, variable/selection counts)."""
        kind = type(self).__name__
        return (
            f"<{kind} {self.name!r} geecs={self._geecs_device_name!r} "
            f"variables={len(self._variables)} selected={len(self._selected)}>"
        )


class GeecsTriggeredDevice(ShotTriggerMixin, GeecsDevice):
    """A :class:`GeecsDevice` that is shot-triggered (it has ``acq_timestamp``).

    ``trigger()`` completes when the device's ``acq_timestamp`` advances (one
    real shot) — the machinery of :class:`~geecs_bluesky.devices.ca.triggerable.CaTriggerable`,
    reused through :class:`~geecs_bluesky.devices.ca.shot_monitor.ShotTriggerMixin`.
    ``acq_timestamp`` is always read alongside the selection: it is the shot
    join key every downstream consumer relies on.
    """

    def __init__(
        self,
        device: str,
        variables: Iterable[VariableMeta | Mapping[str, Any]],
        *,
        experiment: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.acq_timestamp = epics_signal_r(
            float, ca_pv(experiment, device, self._acq_timestamp_variable)
        )
        super().__init__(device, variables, experiment=experiment, **kwargs)
        self._init_shot_monitor()

    def _read_signals(self) -> list[SignalR]:
        return [self.acq_timestamp, *super()._read_signals()]

"""GeecsDetector — a GEECS acquirer as a stock ophyd-async ``StandardDetector``.

ophyd-async 0.19 composes a detector from three logics
(``Planning/native_bluesky/03_clean_room_rebuild.md`` §4.A, §7):

- :class:`GeecsTriggerLogic` — external edges only.  The DG645 fires,
  LabVIEW acquires; there is nothing to program.  Its one config signal is
  the calibrated **drain offset**, the per-device constant between the edge
  and the stamp (§11.4), which ``get_deadtime`` returns so a plan can budget
  the per-shot wait.
- :class:`GeecsAcquireLogic` — the GEECS shot contract: a shot **is**
  ``acq_timestamp`` advancing (§11.3).  LabVIEW is always acquiring, so
  ``start_acquiring``/``ensure_stopped`` are no-ops; what this logic owns is
  the wait, and the synchronous baseline that makes the wait exact.
- data logics — :class:`ScalarsDataLogic` reads the device's own scalar
  variables into the event row; :class:`LvNativeFileDataLogic` drives
  LabVIEW's native file saving (``localsavingpath`` / ``save``) from a
  ``PathProvider``.  #806 swaps the second for the stock
  ``ADHDFDataLogic`` over the PVA-gateway plugin; the other two survive it.

Every per-run fact about the device is set through its own lifecycle —
``stage → prepare → trigger → unstage`` — never from outside it (§3, the
second leg).  A plain ``bp.count([cam])`` is refused at prepare: a GEECS
camera cannot self-trigger, so the fire must come from the plan
(:mod:`geecs_bluesky.plans.strict`).  (With
``OPHYD_ASYNC_PRESERVE_DETECTOR_STATE=YES`` ophyd-async takes
:meth:`GeecsTriggerLogic.default_trigger_info` instead and the implicit
prepare succeeds — the shot then times out waiting for a fire nobody sends.)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bluesky.protocols import Reading
from event_model import DataKey
from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    AsyncStatus,
    DetectorAcquireLogic,
    DetectorDataLogic,
    DetectorTrigger,
    DetectorTriggerLogic,
    PathProvider,
    ReadableDataProvider,
    SignalDict,
    SignalR,
    SignalRW,
    StandardDetector,
    TriggerInfo,
    merge_gathered_dicts,
    soft_signal_rw,
)
from ophyd_async.epics.core import epics_signal_r, epics_signal_rw

from geecs_bluesky.data_paths import device_server_save_path
from geecs_bluesky.devices.ca._pv import ca_pv, setpoint_pv
from geecs_bluesky.exceptions import GeecsTriggerTimeoutError
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)

#: The GEECS shot stamp variable (generated inside LabVIEW, not a DB row yet).
ACQ_TIMESTAMP = "acq_timestamp"

#: How a strict shot prepares a GeecsDetector: one externally edge-triggered
#: event.  The plan fires the box; the detector waits for its stamp.
STRICT_TRIGGER_INFO = TriggerInfo(
    trigger=DetectorTrigger.EXTERNAL_EDGE, number_of_events=1
)


class GeecsTriggerLogic(DetectorTriggerLogic):
    """External edges only; the drain offset is the one configuration value.

    Parameters
    ----------
    drain_offset :
        Signal carrying the device's edge-to-stamp latency in seconds
        (calibrated once, §4.F; ``0.0`` until then).
    """

    def __init__(self, drain_offset: SignalR[float]) -> None:
        self.drain_offset = drain_offset

    def config_sigs(self) -> set[SignalR]:
        """The drain offset rides in ``read_configuration`` (every descriptor)."""
        return {self.drain_offset}

    def get_deadtime(self, config_values: SignalDict) -> float:
        """Edge-to-stamp latency: the per-shot budget a plan adds to the period."""
        return float(config_values[self.drain_offset])

    async def prepare_edge(self, num: int, livetime: float) -> None:
        """Nothing to program: LabVIEW acquires on every edge it receives.

        The count is the plan's business; exposure is a camera setting, not a
        per-scan livetime, so a non-zero *livetime* is refused.
        """
        if livetime:
            raise ValueError(
                "a GEECS camera's exposure is a device setting (set it as a "
                "scan action), not a per-scan TriggerInfo livetime"
            )

    async def default_trigger_info(self) -> TriggerInfo:
        """The truthful hardware state: externally edge-triggered, one event."""
        return STRICT_TRIGGER_INFO


class GeecsAcquireLogic(DetectorAcquireLogic):
    """A shot is ``acq_timestamp`` advancing past the baseline.

    The stamp PV is monitored from ``connect`` on (:meth:`attach`); updates
    land in a bounded drop-oldest queue.  :meth:`baseline` — called
    **synchronously** from :meth:`GeecsDetector.trigger` — records the
    latest stamp and drains the queue, so a shot fired immediately after the
    trigger message can never fall in a blind window (pinned by a mock race
    test).  :meth:`wait_for_idle`, which ``StandardDetector.trigger`` awaits
    after the plan's fire, returns on the first update that differs from the
    baseline or raises :exc:`GeecsTriggerTimeoutError`.

    Parameters
    ----------
    acq_timestamp :
        The device's stamp signal.
    device_name :
        GEECS device name, for the timeout error.
    shot_timeout :
        Seconds to wait for the stamp after a fire.  The hardware budget is
        one trigger period (the single shot fires on the *next* external
        edge) plus the device's exposure and drain (§7, M1).
    """

    _queue_maxsize: int = 128

    def __init__(
        self,
        acq_timestamp: SignalR[float],
        device_name: str,
        *,
        shot_timeout: float = 3.0,
        queue_maxsize: int | None = None,
    ) -> None:
        self._signal = acq_timestamp
        self._device = device_name
        self.shot_timeout = shot_timeout
        self._last: float | None = None
        self._t0: float | None = None
        self._queue: asyncio.Queue[float] = asyncio.Queue(
            maxsize=queue_maxsize or self._queue_maxsize
        )
        self._monitoring = False

    @property
    def last_acq_timestamp(self) -> float | None:
        """Latest stamp seen by the monitor (``None`` before the first shot)."""
        return self._last

    @property
    def queue(self) -> asyncio.Queue[float]:
        """The bounded drop-oldest queue of stamp updates (drained by ``baseline``)."""
        return self._queue

    @property
    def monitoring(self) -> bool:
        """Whether the persistent stamp monitor is attached."""
        return self._monitoring

    def attach(self) -> None:
        """Start the persistent stamp monitor (from the detector's ``connect``)."""
        if not self._monitoring:
            self._signal.subscribe_reading(self._on_update)
            self._monitoring = True

    def detach(self) -> None:
        """Stop the monitor and forget the shot state (``disconnect``)."""
        if self._monitoring:
            self._signal.clear_sub(self._on_update)
            self._monitoring = False
        self._drain()
        self._last = None

    def _on_update(self, reading: dict[str, Any]) -> None:
        value = reading[self._signal.name]["value"]
        if value is None or value <= 0:
            return  # 0.0 is the gateway's pre-acquisition placeholder
        self._last = value
        try:
            self._queue.put_nowait(value)
        except asyncio.QueueFull:
            self._queue.get_nowait()  # drop the oldest update
            self._queue.put_nowait(value)

    def _drain(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()

    def baseline(self) -> None:
        """Record the current stamp and drop stale updates — synchronously."""
        self._t0 = self._last
        self._drain()

    async def start_acquiring(self) -> None:
        """Nothing to start: LabVIEW is always acquiring; the plan fires the box."""

    async def wait_for_idle(self) -> None:
        """Wait for the stamp to advance past the baseline.

        Cold cache (baseline ``None``, no update since the monitor attached):
        deliberately **no** CA-get baseline — a get raced the shot itself, so
        the first positive arrival *is* the shot; ``baseline()`` already
        drained anything older.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.shot_timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise GeecsTriggerTimeoutError(self._device, self.shot_timeout)
            try:
                value = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                raise GeecsTriggerTimeoutError(
                    self._device, self.shot_timeout
                ) from None
            if value != self._t0:
                logger.debug("%s: shot (%s → %s)", self._device, self._t0, value)
                return

    async def ensure_stopped(self) -> None:
        """Nothing to stop: acquisition is LabVIEW's, saving is the data logic's."""


class _SignalsProvider(ReadableDataProvider):
    """Read a fixed set of signals into the event row."""

    def __init__(self, signals: Sequence[SignalR]) -> None:
        self._signals = tuple(signals)

    async def make_datakeys(self) -> dict[str, DataKey]:
        return await merge_gathered_dicts(sig.describe() for sig in self._signals)

    async def make_readings(self) -> dict[str, Reading]:
        return await merge_gathered_dicts(sig.read() for sig in self._signals)


class ScalarsDataLogic(DetectorDataLogic):
    """The device's own scalar variables (and its stamp) as event columns."""

    def __init__(self, signals: Sequence[SignalR]) -> None:
        self._signals: list[SignalR] = list(signals)

    def add(self, *signals: SignalR) -> None:
        """Add columns (a subscribed settable child's readback, at namespace build)."""
        self._signals.extend(signals)

    async def prepare_single(self, datakey_name: str) -> ReadableDataProvider:
        """One reading per event: the signals' current values."""
        return _SignalsProvider(self._signals)


class _ConstantProvider(ReadableDataProvider):
    def __init__(self, datakey_name: str, value: str) -> None:
        self._key = datakey_name
        self._value = value

    async def make_datakeys(self) -> dict[str, DataKey]:
        return {
            self._key: DataKey(
                source=f"derived://{self._key}", dtype="string", shape=[]
            )
        }

    async def make_readings(self) -> dict[str, Reading]:
        return {
            self._key: Reading(
                value=self._value, timestamp=time.time(), alarm_severity=0
            )
        }


class _NoProvider(ReadableDataProvider):
    async def make_datakeys(self) -> dict[str, DataKey]:
        return {}

    async def make_readings(self) -> dict[str, Reading]:
        return {}


class LvNativeFileDataLogic(DetectorDataLogic):
    """LabVIEW's native file saving as the detector's data logic.

    The device writes its own files (one per shot, named with the stamp —
    ``geecs_data_utils.native_files``) once ``localsavingpath`` points at a
    directory and ``save`` is on.  There is no write-complete readback
    (§10.1), so the provider is a per-event reading — the save directory —
    and files join to rows by stamp.  Opened in ``prepare`` from the path
    provider; closed (``save=off``) by :meth:`stop`, which ``stage`` and
    ``unstage`` both call, so a stale ``save=on`` left by a crash is switched
    off before the next run writes anywhere.

    A camera whose frames are **not** wanted this run still carries the
    logic (``native_save=True`` on the detector, no path provider): a
    ``save=on`` left by a crash would otherwise write today's shots into
    yesterday's folder (found live 26_0828) — so ``stage`` always clears it.

    The device directory is created with ``mkdir(exist_ok=True)`` **inside an
    existing scan folder** only — the scan folder itself is claimed by the
    scanner (root ``CLAUDE.md``, "Analysis code is a consumer of scan
    folders"); a missing parent is an error here, never a ``mkdir``.

    Parameters
    ----------
    localsavingpath, save :
        The device's two save controls (gateway readback + ``:SP``).
    path_provider :
        Where this run's files go; called with the datakey name.
    device_path :
        Worker path → the path the device server understands (the Windows
        share path); defaults to the config.ini mapping.
    """

    #: The column name is the event-schema contract (``EVENT_SCHEMA.md``,
    #: ``geecs_data_utils.tiled_schema.COMPANION_SUFFIXES``): renaming it is
    #: a contract change that travels with those files, not a detector edit.
    datakey_suffix = "-nonscalar_save_path"

    def __init__(
        self,
        localsavingpath: SignalRW[str],
        save: SignalRW[str],
        path_provider: PathProvider | None,
        *,
        device_path: Callable[[str], str] = device_server_save_path,
    ) -> None:
        self._localsavingpath = localsavingpath
        self._save = save
        self._path_provider = path_provider
        self._device_path = device_path
        self.directory: Path | None = None

    async def prepare_single(self, datakey_name: str) -> ReadableDataProvider:
        """Point the device at this run's directory and switch saving on.

        Without a path provider the device records scalars only: saving
        stays off (``stop`` already cleared a stale flag at ``stage``) and no
        column is produced.
        """
        if self._path_provider is None:
            return _NoProvider()
        info = self._path_provider(datakey_name)
        directory = Path(info.directory_path)
        if not directory.parent.is_dir():
            raise FileNotFoundError(
                f"{directory.parent} does not exist: the scan folder is claimed "
                "by the scanner, never created by a detector"
            )
        directory.mkdir(exist_ok=True)
        await self._localsavingpath.set(self._device_path(str(directory)))
        await self._save.set("on")
        self.directory = directory
        logger.info("%s: native saving on → %s", datakey_name, directory)
        return _ConstantProvider(datakey_name, str(directory))

    async def stop(self) -> None:
        """Switch saving off (``stage`` and ``unstage`` both call this)."""
        await self._save.set("off")
        self.directory = None


class GeecsDetector(StandardDetector):
    """One GEECS acquirer (camera, spectrometer, scope) as a StandardDetector.

    Parameters
    ----------
    device :
        GEECS device name (``"UC_Amp4_IR_input"``).
    variables :
        Scalar variables to read into the event row (the DB's subscribed
        set).  ``acq_timestamp`` is always present as its own child.
    experiment :
        Experiment PV-namespace prefix.
    name :
        ophyd-async device name (event keys are ``<name>-<variable>``).
    datatypes :
        Per-variable CA datatypes (DB-derived), keyed by variable name;
        ``float`` otherwise.
    path_provider :
        When given, the device saves its native files there
        (:class:`LvNativeFileDataLogic`).
    native_save :
        The device has ``localsavingpath``/``save`` controls.  Implied by
        *path_provider*; set it without one for a camera whose frames are
        not wanted this run, so a stale ``save=on`` is still cleared at
        ``stage`` (see :class:`LvNativeFileDataLogic`).  Without either the
        detector records scalars only.
    shot_timeout :
        Seconds to wait for the stamp after a fire.
    """

    def __init__(
        self,
        device: str,
        variables: Sequence[str],
        *,
        experiment: str | None = None,
        name: str = "",
        datatypes: Mapping[str, type | None] | None = None,
        path_provider: PathProvider | None = None,
        native_save: bool = False,
        shot_timeout: float = 3.0,
    ) -> None:
        self._geecs_device_name = device
        per_variable = {k.lower(): v for k, v in (datatypes or {}).items()}
        scalars: list[SignalR] = []
        for var in variables:
            if var.lower() == ACQ_TIMESTAMP:
                continue
            signal = epics_signal_r(
                per_variable.get(var.lower(), float), ca_pv(experiment, device, var)
            )
            setattr(self, safe_name(var), signal)
            scalars.append(signal)
        self.acq_timestamp = epics_signal_r(
            float, ca_pv(experiment, device, ACQ_TIMESTAMP)
        )
        # The gateway's per-device liveness PV: never an event column, read by
        # the refire gate to tell a dropped frame from a dead device.
        self.connected_status = epics_signal_r(
            str, ca_pv(experiment, device, "CONNECTED")
        )
        self.drain_offset = soft_signal_rw(float, 0.0, units="s")
        self._scalars: list[SignalR] = list(scalars)
        self._acquire = GeecsAcquireLogic(
            self.acq_timestamp, device, shot_timeout=shot_timeout
        )
        self._scalars_logic = ScalarsDataLogic((*scalars, self.acq_timestamp))
        logics: list[Any] = [
            GeecsTriggerLogic(self.drain_offset),
            self._acquire,
            self._scalars_logic,
        ]
        self.native_save = bool(native_save or path_provider is not None)
        if self.native_save:
            path_pv = ca_pv(experiment, device, "localsavingpath")
            save_pv = ca_pv(experiment, device, "save")
            self.localsavingpath = epics_signal_rw(str, path_pv, setpoint_pv(path_pv))
            self.save = epics_signal_rw(str, save_pv, setpoint_pv(save_pv))
            logics.append(
                LvNativeFileDataLogic(self.localsavingpath, self.save, path_provider)
            )
        self.add_detector_logics(*logics)
        super().__init__(name=name)
        # Legacy "Device Variable" headers for the Tiled → s-file exporter.
        self._column_headers = {
            f"{self.name}-{safe_name(var)}": f"{device} {var}"
            for var in variables
            if var.lower() != ACQ_TIMESTAMP
        }
        self._column_headers[f"{self.name}-{ACQ_TIMESTAMP}"] = (
            f"{device} {ACQ_TIMESTAMP}"
        )

    @property
    def last_acq_timestamp(self) -> float | None:
        """Latest stamp seen by the persistent monitor."""
        return self._acquire.last_acq_timestamp

    def add_readables(self, signals: Sequence[Any]) -> None:
        """Add event columns beyond the constructor's *variables*.

        The namespace binds each served settable as a Movable child and,
        when the DB also subscribes that variable, logs its readback here —
        the same rule ``StandardReadable.add_readables`` gives the
        scalar-only devices.  Anything with ``read`` / ``describe`` /
        ``stage`` / ``unstage`` qualifies (a signal or a child device);
        staged with the other scalars.
        """
        self._scalars.extend(signals)
        self._scalars_logic.add(*signals)

    async def connect(
        self,
        mock: Any = False,
        timeout: float = DEFAULT_TIMEOUT,
        force_reconnect: bool = False,
    ) -> None:
        """Connect every signal, then start the stamp monitor."""
        await super().connect(
            mock=mock, timeout=timeout, force_reconnect=force_reconnect
        )
        self._acquire.attach()

    async def disconnect(self) -> None:
        """Stop the stamp monitor; the signals drop their caches with it."""
        self._acquire.detach()

    @AsyncStatus.wrap
    async def stage(self) -> None:
        """Stage the scalar signals (monitor-backed reads), then the detector.

        Staged signals read from their monitor cache: one CA get per column
        per shot was the 0.7 s/row regression (``GeecsBluesky/CLAUDE.md``,
        "Read path: staging & shot coherence").
        """
        await asyncio.gather(
            *(sig.stage() for sig in (*self._scalars, self.acq_timestamp))
        )
        await super().stage()

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        """Unstage the detector (saving off), then release the signal caches."""
        await super().unstage()
        await asyncio.gather(
            *(sig.unstage() for sig in (*self._scalars, self.acq_timestamp))
        )

    def trigger(self) -> AsyncStatus:
        """Baseline the stamp **now**, then wait for it to advance.

        The baseline must happen before this returns: the plan's very next
        message is the fire, and a stamp that lands before an asynchronous
        baseline would be counted as the pre-shot value and the real shot
        missed.
        """
        self._acquire.baseline()
        return super().trigger()

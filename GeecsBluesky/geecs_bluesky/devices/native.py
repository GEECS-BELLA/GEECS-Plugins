"""NativeCamera — a native EPICS areaDetector camera as a GEECS strict or fly detector.

A camera served by its own IOC (geecs-epics: ADAravis + the stock
``NDFileHDF5`` plugin), not ophyd-async's ``AravisDetector`` — that one
starts an acquisition per shot, clears the writer's layout and lets frames
be skipped.  The GEECS camera contract over stock records
(``NATIVE_DEVICES.md``):

- **armed once per run** — the first ``prepare`` stops the camera and sets
  trigger mode *before* the HDF capture starts (no live-view frame reaches
  the stack); ``unstage`` restores the camera (live view between scans);
- **a shot is the writer's frame count advancing**: ``Software`` presses
  ``TriggerSoftware`` inside ``trigger()`` (exposed when the worker says go
  — the accepted caveat), or through :meth:`NativeCamera.press` at the fire
  of a stream run; ``Line0`` / ``Line1`` wait for the timing system's edge
  on that input;
- **never rewound**: the writer appends, and Bluesky's HDF5 consolidator
  reads frames by datum count from 0 — a strict run streams the camera, so
  a skipped frame is a gap in its stream and a pause replays nothing;
- **the row stamp** is the frame's EPICS timestamp through
  ``scan_stack.epics_timestamp_to_unix``, in the LabVIEW epoch.

A strict shot per row, or a datum-only **stream** after an unbounded
prepare (its datums the frames written, from 0; a skip is absent, never a
placeholder), both armed alike; a bounded fly prepare is refused.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import math
import os
import socket
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated as A
from typing import Any

from bluesky.protocols import Reading
from event_model import DataKey
from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    AsyncStatus,
    DetectorAcquireLogic,
    DetectorDataLogic,
    DetectorTriggerLogic,
    Device,
    OnOff,
    PathProvider,
    ReadableDataProvider,
    SignalDict,
    SignalR,
    SignalRW,
    StandardDetector,
    StreamableDataProvider,
    StreamResourceDataProvider,
    StreamResourceInfo,
    StrictEnum,
    TriggerInfo,
    TriggerableCommand,
    derived_signal_r,
    set_and_wait_for_value,
    soft_signal_rw,
    wait_for_value,
)
from ophyd_async.epics.adaravis import AravisDriverIO
from ophyd_async.epics.adcore import (
    ADHDFDataLogic,
    ADImageMode,
    NDArrayDescription,
    NDFileHDF5IO,
)
from ophyd_async.epics.core import PvSuffix, epics_signal_rw

from geecs_data_utils.io.scan_stack import (
    ATTRIBUTES_GROUP,
    EPICS_TS_NSEC,
    EPICS_TS_SEC,
    LABVIEW_EPOCH_OFFSET,
    epics_timestamp_to_unix,
)

from geecs_bluesky.devices.ca._view import owner_of
from geecs_bluesky.devices.ca.liveness import down_natives
from geecs_bluesky.devices.detector import (
    ACQ_TIMESTAMP,
    STRICT_TRIGGER_INFO,
    FlyTriggerInfo,
    plugin_reason,
)
from geecs_bluesky.exceptions import GeecsConfigurationError, GeecsTriggerTimeoutError
from geecs_bluesky.native_scope import native_refusal
from geecs_bluesky.utils import is_connected

logger = logging.getLogger(__name__)

#: The driver and writer record prefixes under a native camera's PV prefix
#: (the areaDetector / ophyd-async defaults; geecs-epics IOCs use them).
DRIVER_SUFFIX = "cam1:"
HDF_SUFFIX = "HDF1:"
#: The run lock's records under the prefix (geecs-epics run_lock.db).
RUN_LOCK_SUFFIX = "RUN:"
#: Seconds the trigger setup's readbacks get to settle before a refusal is declared.
READBACK_TIMEOUT_S = 2.0
#: Seconds a native camera's IOC gets to connect, then to answer, at the run's liveness gate.
LIVENESS_CONNECT_TIMEOUT_S = 5.0


class NativeTriggerSource(enum.StrEnum):
    """Where a native camera's frame comes from (the registry's ``trigger_source``).

    Not the driver signal's type: an enum-typed signal demands *every* member
    among the PV's choices at connect, and the choices are the camera's own
    (a Basler has no ``Line0``, a FLIR Blackfly's opto input is ``Line0``).
    The signal is a plain string; the first prepare checks the configured
    source against the camera's choices (:class:`NativeTriggerLogic`).
    """

    LINE0 = "Line0"
    LINE1 = "Line1"
    SOFTWARE = "Software"


class AsynConnect(StrictEnum):
    """The asyn record's ``CNCT`` field: the driver's connection to the camera."""

    DISCONNECT = "Disconnect"
    CONNECT = "Connect"


class NativeDriverIO(AravisDriverIO):
    """ophyd-async's ADAravis driver, its trigger source typed as the camera's own choices."""

    # A string, not an enum: connect must not demand a line this camera lacks.
    trigger_source: A[SignalRW[str], PvSuffix.rbv("TriggerSource")]  # type: ignore[assignment]
    # Processing the command record fires it (verified on the bench camera).
    trigger_software: A[TriggerableCommand, PvSuffix("TriggerSoftware.PROC")]
    asyn_connect: A[SignalR[AsynConnect], PvSuffix("AsynIO.CNCT")]


class NativeHdfIO(NDFileHDF5IO):
    """The stock HDF writer plus what GEECS reads off it."""

    write_message: A[SignalR[str], PvSuffix("WriteMessage")]
    epics_ts_sec: A[SignalR[int], PvSuffix("EpicsTSSec_RBV")]
    epics_ts_nsec: A[SignalR[int], PvSuffix("EpicsTSNsec_RBV")]


class RunLockIO(Device):
    """The IOC's run lock: ``RUN:Lock`` and ``RUN:Owner``.

    While ``lock`` is set, the IOC's access security refuses acquisition
    writes from anyone but the scan worker (geecs-epics ``access.acf``).
    """

    def __init__(self, prefix: str, name: str = "") -> None:
        self.lock = epics_signal_rw(bool, prefix + "Lock")
        self.owner = epics_signal_rw(str, prefix + "Owner")
        super().__init__(name=name)


def run_owner() -> str:
    """The lock's owner text, ``host:pid @HH:MM`` — who holds the camera, and since when.

    Within a ``stringout``'s 39 characters whatever the host's name (cut to
    22): the override dialog shows it for a stuck-lock judgement.
    """
    return f"{socket.gethostname()[:22]}:{os.getpid()} @{time.strftime('%H:%M')}"


def liveness_word(cnct: AsynConnect) -> str:
    """The liveness verdict's words: ``Disconnected`` only when the driver lost the camera."""
    return "Disconnected" if cnct == AsynConnect.DISCONNECT else "Connected"


@dataclass
class _Session:
    """Run state shared by the trigger and acquire logics (reset at ``stage``)."""

    programmed: bool = False
    acquiring: bool = False
    restore: tuple[Any, ...] | None = None
    #: This session set the run lock and has not released it.
    locked: bool = False
    #: Serialises the camera's whole ``prepare`` with its whole ``unstage``:
    #: an unstage sent mid-prepare (a sibling's prepare failed fast, a halt's
    #: sweep) waits, so the lock is never released while the camera is armed
    #: or its capture opening. Held by ``NativeCamera`` only (not reentrant).
    guard: asyncio.Lock = field(default_factory=asyncio.Lock)
    fire_pending: bool = False
    missed: bool = False


class NativeTriggerLogic(DetectorTriggerLogic):
    """External edges (Line0/Line1) or the software command; the camera is programmed once per run.

    ``prepare_edge`` runs before the data logic starts the HDF capture, so
    this is where the camera (free-running, or resting in trigger mode) is
    stopped and put in trigger mode: no frame of its live view can reach
    the stack.
    """

    def __init__(
        self,
        driver: NativeDriverIO,
        source: NativeTriggerSource,
        session: _Session,
        drain_offset: SignalR[float],
        run_lock: RunLockIO,
    ) -> None:
        self.driver = driver
        self.source = source
        self.session = session
        self.drain_offset = drain_offset
        self.run_lock = run_lock

    def config_sigs(self) -> set[SignalR]:
        """The drain offset rides in every descriptor, as on a GEECS camera."""
        return {self.drain_offset}

    def get_deadtime(self, config_values: SignalDict) -> float:
        """Edge-to-frame latency, the GEECS calibration's quantity."""
        return float(config_values[self.drain_offset])

    async def prepare_edge(self, num: int, livetime: float) -> None:
        """First prepare of the run: remember the camera's state, stop it, set triggering."""
        if livetime:
            raise ValueError(
                "a native camera's exposure is a camera setting, not a per-scan "
                "TriggerInfo livetime"
            )
        if self.session.programmed:
            return
        d = self.driver
        # The run owns the camera from here until it is restored: the IOC
        # refuses acquisition writes from anyone else (panels, Phoebus) while
        # the lock is set. First, before the snapshot, so nothing can change
        # the state being saved or set below.
        await take_run_lock(self.run_lock, self.session)
        await self._refuse_a_source_the_camera_lacks()
        if self.session.restore is None:
            # Captured once and kept until a restore succeeds: a run in this
            # worker that ended without a completed unstage (a halt, a failed
            # restore) must not make the armed state the "pre-run" one. A
            # killed worker's session is gone; take_run_lock names that case.
            self.session.restore = tuple(
                await asyncio.gather(
                    d.acquire.get_value(),
                    d.trigger_mode.get_value(),
                    d.trigger_source.get_value(),
                    d.image_mode.get_value(),
                )
            )
            if self.session.restore[1] == OnOff.ON:
                # The usual resting state: a camera IOC may arm its trigger at
                # boot so live view runs at the shot rate (geecs-epics'
                # cameras do). Not a sign of an unrestored run.
                logger.debug(
                    "%s: in trigger mode before the run; unstage returns it there",
                    d.name,
                )
        await d.acquire.set(False)
        # Trigger mode first and on its own: the Aravis firmware race
        # ophyd-async's AravisTriggerLogic documents (aravis#1045).
        await d.trigger_mode.set(OnOff.ON)
        await d.trigger_source.set(self.source)
        await d.image_mode.set(ADImageMode.CONTINUOUS)
        # A refused GenICam write completes the CA put anyway (#868's class):
        # wait for the readbacks, or a camera still free-running would fill
        # the stack with live-view frames and every shot would "succeed".
        refused = await unsettled_readbacks(
            d, OnOff.ON, self.source, ADImageMode.CONTINUOUS
        )
        if refused:
            raise GeecsConfigurationError(
                f"{d.name}: the camera refused its trigger setup: {', '.join(refused)}"
            )
        self.session.programmed = True

    async def _refuse_a_source_the_camera_lacks(self) -> None:
        """Refuse, before the camera is touched, a source not among its ``TriggerSource`` choices.

        The choices are the PV's (the camera's GenICam enumeration); a PV that
        reports none is not judged here — the readback wait is the backstop.
        """
        sig = self.driver.trigger_source
        choices = (await sig.describe())[sig.name].get("choices") or []
        if choices and self.source not in choices:
            raise GeecsConfigurationError(
                f"{self.driver.name}: trigger_source {self.source.name.lower()} "
                f"({self.source.value!r}) is not one of this camera's TriggerSource "
                f"choices ({', '.join(choices)}) — set its native_devices.yaml "
                "entry to the line it is wired on"
            )

    async def default_trigger_info(self) -> TriggerInfo:
        """One edge-triggered event: the strict shot."""
        return STRICT_TRIGGER_INFO


async def unsettled_readbacks(
    d: NativeDriverIO, trigger_mode: Any, trigger_source: Any, image_mode: Any
) -> list[str]:
    """The trigger readbacks that did not reach the wanted values, as ``name=got (wanted x)``.

    A refused GenICam write completes the CA put anyway (#868's class), and
    the ``_RBV`` records update through asyn callbacks that can lag the
    put's completion: waited for, never read once.
    """
    wanted = {
        "TriggerMode": (d.trigger_mode, trigger_mode),
        "TriggerSource": (d.trigger_source, trigger_source),
        "ImageMode": (d.image_mode, image_mode),
    }

    async def settled(name: str, sig: SignalR, want: Any) -> str | None:
        try:
            await wait_for_value(sig, want, timeout=READBACK_TIMEOUT_S)
        except TimeoutError:
            return f"{name}={await sig.get_value()} (wanted {want})"
        return None

    results = await asyncio.gather(
        *(settled(n, sig, want) for n, (sig, want) in wanted.items())
    )
    return [r for r in results if r]


async def take_run_lock(run_lock: RunLockIO, session: _Session) -> None:
    """Set the IOC's run lock and its owner, and wait for the lock's readback.

    Waited for, not trusted: a refused CA put can complete as success
    (#868), and a run must not go on believing the camera is protected.

    A lock already set that this session did not take means a run in
    another process never released the camera (a killed or restarted
    worker). The run takes it over, but says so, naming the previous owner:
    the camera's state saved next is the one that run left, not the
    operator's. (This session's own lock, kept by a failed restore, is not
    that case: ``session.restore`` still holds the true pre-run state.)
    """
    if not session.locked and await run_lock.lock.get_value():
        logger.warning(
            "%s: the run lock was already held (owner %r) — a previous run did "
            "not release the camera; the state it left becomes this run's "
            "pre-run state",
            run_lock.name,
            await run_lock.owner.get_value(),
        )
    await run_lock.owner.set(run_owner())
    await run_lock.lock.set(True)
    session.locked = True
    try:
        await wait_for_value(run_lock.lock, True, timeout=READBACK_TIMEOUT_S)
    except TimeoutError as exc:
        raise GeecsConfigurationError(
            f"{run_lock.name}: the IOC did not take the run lock "
            "(RUN:Lock stayed Free) — is the camera's IOC on the run-lock chain?"
        ) from exc


async def release_run_lock(run_lock: RunLockIO, session: _Session) -> None:
    """Clear the IOC's run lock (only ever after the camera is back)."""
    await run_lock.lock.set(False)
    await run_lock.owner.set("")
    session.locked = False


class NativeAcquireLogic(DetectorAcquireLogic):
    """Acquisition runs for the whole run; a software trigger is pressed per shot."""

    def __init__(
        self, driver: NativeDriverIO, session: _Session, run_lock: RunLockIO
    ) -> None:
        self.driver = driver
        self.session = session
        self.run_lock = run_lock

    async def ensure_ready(self) -> None:
        """At ``stage``: a fresh run state (a held pre-run state is kept); the camera is not touched."""
        self.session.programmed = False
        self.session.acquiring = False
        self.session.fire_pending = False
        self.session.missed = False

    async def start_acquiring(self) -> None:
        """Start the camera once (its readback, not the put: a Continuous acquire never completes), then fire if asked."""
        if not self.session.acquiring:
            # A Continuous acquire's put never completes: wait on the readback.
            await set_and_wait_for_value(
                self.driver.acquire,
                True,
                timeout=DEFAULT_TIMEOUT,
                wait_for_set_completion=False,
            )
            self.session.acquiring = True
        if self.session.fire_pending:
            self.session.fire_pending = False
            await self.driver.trigger_software.trigger()

    async def wait_for_idle(self) -> None:
        """Nothing to wait for: the frame count is the shot, the camera keeps running."""

    async def ensure_stopped(self) -> None:
        """At ``unstage``: the camera back to its pre-run state (live view between scans)."""
        restore = self.session.restore
        self.session.programmed = False
        self.session.acquiring = False
        if restore is None:
            # Nothing to restore: the camera was never touched (a failed
            # snapshot, a refused source). A lock this session took is
            # released all the same — the camera is as it was.
            if self.session.locked:
                await release_run_lock(self.run_lock, self.session)
            return
        acquire, trigger_mode, trigger_source, image_mode = restore
        d = self.driver
        await d.acquire.set(False)
        await d.trigger_mode.set(trigger_mode)
        await d.trigger_source.set(trigger_source)
        await d.image_mode.set(image_mode)
        # Checked like the arming: a silently refused restore must keep the
        # saved state and the lock, not hand back a camera still armed.
        refused = await unsettled_readbacks(d, trigger_mode, trigger_source, image_mode)
        if refused:
            raise GeecsConfigurationError(
                f"{d.name}: the camera refused its restore: {', '.join(refused)} — "
                "the run lock stays set until the camera is back"
            )
        if acquire:
            await set_and_wait_for_value(
                d.acquire, True, timeout=DEFAULT_TIMEOUT, wait_for_set_completion=False
            )
        self.session.restore = None  # only once the camera is back
        # The run's ownership ends only with the camera back; a failed
        # restore (raised above) leaves the lock set: the camera is not.
        if self.session.locked:
            await release_run_lock(self.run_lock, self.session)


#: The per-frame attributes NDFileHDF5 writes into every stack whatever the
#: layout lists — the frame's EPICS stamp pair (the join's clock) and its
#: unique id (the counter audit's) — with their dtypes in the written file.
BUILTIN_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    (EPICS_TS_SEC, "<u4"),
    (EPICS_TS_NSEC, "<u4"),
    ("NDArrayUniqueId", "<i4"),
)
#: NDFileHDF5's chunk length for an NDAttribute dataset (the stock logic's).
NDATTRIBUTE_CHUNK = 16384
HDF5_MIMETYPE = "application/x-hdf5"


@dataclass
class NativeHdfDataLogic(ADHDFDataLogic):
    """The stock HDF data logic, keeping the IOC's layout and SWMR off.

    The stock prepare clears ``XMLFileName`` (the IOC's GEECS layout, which
    writes ``/entry`` ``finalized`` on close) and switches SWMR on (a SWMR
    file cannot take that attribute at close, and GEECS never writes SWMR
    across SMB).  Both are put back after it returns — before the first
    frame, because the writer opens the file lazily at the first frame and
    a triggered camera sends none before the first trigger.
    """

    async def prepare_unbounded(self, datakey_name: str) -> StreamableDataProvider:
        """The stock prepare, then the IOC's layout and SWMR off restored.

        The stream also declares the writer's built-in per-frame attributes
        (:data:`BUILTIN_ATTRIBUTES`), under their bare names: the stock logic
        declares only the attributes an inline XML lists, and without the
        EPICS stamp pair a reader of the run in Tiled cannot time the
        camera's frames (the live s-file reads them from the stack itself).
        """
        layout = await self.writer.xml_file_name.get_value()
        provider = await super().prepare_unbounded(datakey_name)
        await asyncio.gather(
            self.writer.xml_file_name.set(layout),
            self.writer.swmr_mode.set(False),
        )
        if not isinstance(provider, StreamResourceDataProvider):
            return provider
        declared = {resource.data_key for resource in provider.resources}
        builtins = [
            StreamResourceInfo(
                data_key=name,
                shape=(),
                chunk_shape=(NDATTRIBUTE_CHUNK,),
                dtype_numpy=dtype,
                parameters={"dataset": f"{ATTRIBUTES_GROUP}/{name}"},
            )
            for name, dtype in BUILTIN_ATTRIBUTES
            if name not in declared
        ]
        return StreamResourceDataProvider(
            uri=provider.uri,
            resources=[*provider.resources, *builtins],
            mimetype=HDF5_MIMETYPE,
            collections_written_signal=provider.collections_written_signal,
            flush_signal=provider.flush_signal,
        )


class _StampProvider(ReadableDataProvider):
    """``<name>-acq_timestamp``: the frame's EPICS timestamp in the LabVIEW epoch."""

    def __init__(self, key: str, hdf: NativeHdfIO, session: _Session) -> None:
        self._key = key
        self._hdf = hdf
        self._session = session

    async def make_datakeys(self) -> dict[str, DataKey]:
        return {
            self._key: DataKey(
                source=self._hdf.epics_ts_sec.source,
                dtype="number",
                shape=[],
                units="s",
            )
        }

    async def make_readings(self) -> dict[str, Reading]:
        if self._session.missed:
            value = math.nan
        else:
            sec, nsec = await asyncio.gather(
                self._hdf.epics_ts_sec.get_value(), self._hdf.epics_ts_nsec.get_value()
            )
            value = float(epics_timestamp_to_unix(sec, nsec)) + LABVIEW_EPOCH_OFFSET
        return {
            self._key: Reading(value=value, timestamp=time.time(), alarm_severity=0)
        }


class _StampDataLogic(DetectorDataLogic):
    def __init__(self, hdf: NativeHdfIO, session: _Session) -> None:
        self._hdf = hdf
        self._session = session

    async def prepare_single(self, datakey_name: str) -> ReadableDataProvider:
        return _StampProvider(
            f"{datakey_name}-{ACQ_TIMESTAMP}", self._hdf, self._session
        )


class NativeCamera(StandardDetector):
    """A native EPICS camera as a GEECS strict detector (module docstring).

    Parameters
    ----------
    device :
        The device name (the registry key; namespace binding and stack folder).
    prefix :
        The camera's PV prefix, ``<experiment>:<device>:`` (geecs-epics
        decision 5); the driver and writer are ``cam1:`` / ``HDF1:`` under it.
    trigger_source :
        ``Line0`` / ``Line1`` (the timing system's edge on that input) or
        ``Software`` (pressed per shot).
    path_provider :
        Where the writer puts the stack, as the IOC host sees it
        (``hdf_plugin.PluginPathProvider`` with POSIX paths).
    name :
        ophyd-async device name (event keys ``<name>``, ``<name>-acq_timestamp``).
    drain_offset :
        The calibration's edge-to-frame latency, seconds.
    """

    #: Strict reads these off every detector (``plans/strict.py``).
    plugin_backed = True
    native_save = False
    #: The stock writer zeroes its count at each capture start: nothing to zero.
    count_zeroed = True

    def __init__(
        self,
        device: str,
        prefix: str,
        *,
        trigger_source: NativeTriggerSource,
        path_provider: PathProvider,
        name: str = "",
        drain_offset: float = 0.0,
    ) -> None:
        self._geecs_device_name = device
        self._source = NativeTriggerSource(trigger_source)
        self.driver = NativeDriverIO(prefix + DRIVER_SUFFIX)
        self.hdf = NativeHdfIO(prefix + HDF_SUFFIX)
        self.run_lock = RunLockIO(prefix + RUN_LOCK_SUFFIX)
        self.drain_offset = soft_signal_rw(float, float(drain_offset), units="s")
        self.connected_status = derived_signal_r(
            liveness_word, cnct=self.driver.asyn_connect
        )
        self._session = _Session()
        self._native_acquire = NativeAcquireLogic(
            self.driver, self._session, self.run_lock
        )
        self.add_detector_logics(
            NativeTriggerLogic(
                self.driver,
                self._source,
                self._session,
                self.drain_offset,
                self.run_lock,
            ),
            self._native_acquire,
            NativeHdfDataLogic(
                array_description=NDArrayDescription(
                    shape_signals=[
                        self.driver.array_size_z,
                        self.driver.array_size_y,
                        self.driver.array_size_x,
                    ],
                    data_type_signal=self.driver.data_type,
                    color_mode_signal=self.driver.color_mode,
                ),
                path_provider=path_provider,
                driver=self.driver,
                writer=self.hdf,
            ),
            _StampDataLogic(self.hdf, self._session),
        )
        super().__init__(name=name)
        self._column_headers = {
            f"{self.name}-{ACQ_TIMESTAMP}": f"{device} {ACQ_TIMESTAMP}"
        }

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        """Stop the camera, then close the stack, then restore the camera — in that order.

        The stock ``unstage`` gathers the writer's stop and the acquire
        logic's restore concurrently: a slow writer stop would let the
        restored live view append frames after the last shot (GPT review on
        #1117).  If the writer's stop fails the restore is skipped (the
        camera stays stopped, its pre-run state kept for the next run) and
        the error raised.
        """
        async with self._session.guard:  # after any prepare still in flight
            if self._session.restore is not None:
                await self.driver.acquire.set(False)  # no frame reaches the writer now
            await asyncio.gather(*(logic.stop() for logic in self._data_logics))
            await self._native_acquire.ensure_stopped()

    @property
    def trigger_source(self) -> NativeTriggerSource:
        """``Line0``, ``Line1`` or ``Software`` (the registry's choice)."""
        return self._source

    @property
    def missed_shot(self) -> bool:
        """Whether the last awaited shot's frame never arrived."""
        return self._session.missed

    async def plugin_reasons(self) -> list[str]:
        """The writer's non-empty ``WriteMessage`` (``detector.plugin_reason``, bounded)."""
        reason = await plugin_reason(self.hdf)
        return [reason] if reason else []

    @AsyncStatus.wrap
    async def prepare(self, value: TriggerInfo) -> None:
        """The stock prepare: a strict shot or an unbounded stream; a failure carries the writer's reason.

        A bounded fly prepare (a gated batch) is refused before the camera
        is touched: its quota would need a rewind the writer lacks.
        """
        if isinstance(value, FlyTriggerInfo) and value.number_of_events != 0:
            raise GeecsConfigurationError(
                f"{self._geecs_device_name} is a native EPICS camera: strict scans "
                "or an unbounded stream only for now (no gated batch)"
            )
        try:
            async with self._session.guard:  # whole, so an unstage waits for it
                await super().prepare(value)
        except Exception as exc:
            for reason in await self.plugin_reasons():
                exc.add_note(reason)
            raise

    def _frame_timeout(self) -> float:
        """The frame wait's budget: the prepared ``TriggerInfo.exposure_timeout``."""
        ctx = self._prepare_ctx
        return (ctx.trigger_info if ctx else STRICT_TRIGGER_INFO).exposure_timeout

    def _streams(self) -> bool:
        """Whether the camera is prepared as a stream (unbounded) rather than for strict shots."""
        ctx = self._prepare_ctx
        return ctx is not None and ctx.trigger_info.number_of_events == 0

    async def press(self) -> None:
        """Press ``TriggerSoftware`` once, outside ``trigger()``: the fire of a stream run.

        Refused on a hardware-source camera (its shot is the timing system's
        edge; a press would be a stray frame) and unless the camera is
        prepared as a stream and acquiring: before the Acquire readback (mid
        prepare, a failed prepare, after unstage) the press is lost or lands
        on the live view, and on a strict camera it writes a frame outside any
        row and shifts every later one.
        """
        if self._source != NativeTriggerSource.SOFTWARE:
            raise GeecsConfigurationError(
                f"{self._geecs_device_name}: trigger_source is "
                f"{self._source.name.lower()} ({self._source.value!r}) — only a "
                "Software-source native camera is pressed"
            )
        if not self._streams():
            raise GeecsConfigurationError(
                f"{self._geecs_device_name}: press() needs an unbounded prepare "
                "(the stream path); a strict shot presses inside trigger()"
            )
        if not self._session.acquiring:
            raise GeecsConfigurationError(
                f"{self._geecs_device_name}: press() before the camera is "
                "acquiring would be lost"
            )
        await self.driver.trigger_software.trigger()

    def trigger(self) -> AsyncStatus:
        """Wait for one more frame in the stack (pressing the software trigger if that is the source)."""
        if self._streams():
            return AsyncStatus(self._refuse_trigger_on_a_stream())
        self._session.missed = False
        self._session.fire_pending = self._source == NativeTriggerSource.SOFTWARE
        return AsyncStatus(self._translate_count_timeout(super().trigger()))

    async def _translate_count_timeout(self, status: AsyncStatus) -> None:
        try:
            await status
        except TimeoutError as exc:
            self._session.missed = True
            reasons = await self.plugin_reasons()
            timeout = self._frame_timeout()
            raise GeecsTriggerTimeoutError(
                self._geecs_device_name,
                timeout,
                f"{self._geecs_device_name}: no frame written within "
                f"{timeout:.1f}s" + "".join(f"; {r}" for r in reasons),
            ) from exc

    async def _refuse_trigger_on_a_stream(self) -> None:
        raise GeecsConfigurationError(
            f"{self._geecs_device_name}: prepared as a stream (unbounded) — its "
            "shots are collected, never triggered"
        )

    @AsyncStatus.wrap
    async def kickoff(self) -> None:
        """Start the stream; only after an unbounded prepare.

        A frame landing after the prepare (which starts the camera) — an
        edge while the box idles, before or during the kickoff — would make
        the stock kickoff raise ("only prepared up to …"); the stream's
        datums cover it anyway (from frame 0).  So the prepare context's
        baseline is set past any count the stock check can read: no second
        read has to agree with a first.  Nothing else reads that baseline
        on a stream (``trigger()`` is refused; the next prepare rebuilds it).
        """
        ctx = self._prepare_ctx
        if ctx is None or not self._streams():
            raise GeecsConfigurationError(
                f"{self._geecs_device_name} is a native EPICS camera: kickoff needs "
                "an unbounded prepare (the stream path)"
            )
        ctx.collections_written = sys.maxsize
        await super().kickoff()


def native_cameras(devices: Iterable[Any]) -> list[NativeCamera]:
    """The native cameras among *devices* (a ``.scalars`` view counts as its owner)."""
    return [d for d in map(owner_of, devices) if isinstance(d, NativeCamera)]


async def _native_word(cam: NativeCamera) -> str:
    """The camera's liveness word from its IOC: connect if needed, read, bounded."""
    source = cam.driver.asyn_connect
    if not is_connected(source):
        await source.connect(timeout=LIVENESS_CONNECT_TIMEOUT_S)
    cnct = await asyncio.wait_for(source.get_value(), LIVENESS_CONNECT_TIMEOUT_S)
    return liveness_word(cnct)


async def _camera_down(cam: NativeCamera) -> bool:
    return await _native_word(cam) == "Disconnected"


def down_native_cameras(devices: Iterable[Any]):
    """Plan: the native cameras among *devices* whose IOC reports or is down.

    Judged here, not by the gateway's fail-open rule: the word comes from the
    IOC's own ``AsynIO.CNCT``, so a read that fails (an IOC that never
    answered, or died after an earlier run's connect — the channel stays
    cached in a long-lived worker) means the IOC is down
    (:func:`~geecs_bluesky.devices.ca.liveness.down_natives`).  Bounded by
    ``LIVENESS_CONNECT_TIMEOUT_S`` for the connect, then the read;
    ``connect_on_demand`` would connect only the derived
    ``connected_status``, never this source.

    Returns
    -------
    list of str
        The GEECS names of the cameras reported ``Disconnected`` or unreachable.
    """
    return (yield from down_natives(native_cameras(devices), _camera_down))


def refuse_native_cameras(devices: Iterable[Any], why: str) -> None:
    """Refuse a run shape a native camera cannot take part in yet, before anything moves.

    Raises
    ------
    GeecsConfigurationError
        Naming every native camera among *devices* and *why*.
    """
    if found := native_cameras(devices):
        raise GeecsConfigurationError(
            native_refusal((c._geecs_device_name for c in found), why)
        )

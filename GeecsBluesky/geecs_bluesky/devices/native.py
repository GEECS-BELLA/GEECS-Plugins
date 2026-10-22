"""NativeCamera — a native EPICS areaDetector camera as a GEECS strict detector.

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
  — the accepted caveat), ``Line1`` waits for the timing system's edge;
- **one frame per row**: an own miss fails the step and a pause never
  replays a shot (the writer cannot rewind; Bluesky's HDF5 consolidator
  reads frames by datum count from 0);
- **the row stamp** is the frame's EPICS timestamp through
  ``scan_stack.epics_timestamp_to_unix``, in the LabVIEW epoch.

Strict runs only: fly prepares are refused here as well as at bind.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated as A
from typing import Any

import bluesky.plan_stubs as bps
from bluesky.protocols import Reading
from event_model import DataKey
from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    AsyncStatus,
    DetectorAcquireLogic,
    DetectorDataLogic,
    DetectorTriggerLogic,
    OnOff,
    PathProvider,
    ReadableDataProvider,
    SignalDict,
    SignalR,
    SignalRW,
    StandardDetector,
    StreamableDataProvider,
    StrictEnum,
    SubsetEnum,
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
from ophyd_async.epics.core import PvSuffix

from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET, epics_timestamp_to_unix

from geecs_bluesky.devices.ca._view import owner_of
from geecs_bluesky.devices.detector import (
    ACQ_TIMESTAMP,
    STRICT_TRIGGER_INFO,
    FlyTriggerInfo,
    plugin_reason,
)
from geecs_bluesky.exceptions import GeecsConfigurationError, GeecsTriggerTimeoutError
from geecs_bluesky.utils import is_connected

logger = logging.getLogger(__name__)

#: The driver and writer record prefixes under a native camera's PV prefix
#: (the areaDetector / ophyd-async defaults; geecs-epics IOCs use them).
DRIVER_SUFFIX = "cam1:"
HDF_SUFFIX = "HDF1:"
#: Seconds the trigger setup's readbacks get to settle before a refusal is declared.
READBACK_TIMEOUT_S = 2.0
#: Seconds a native camera's IOC gets to connect, then to answer, at the run's liveness gate.
LIVENESS_CONNECT_TIMEOUT_S = 5.0


class NativeTriggerSource(SubsetEnum):
    """Where a native camera's frame comes from (the registry's ``trigger_source``)."""

    LINE1 = "Line1"
    SOFTWARE = "Software"


class AsynConnect(StrictEnum):
    """The asyn record's ``CNCT`` field: the driver's connection to the camera."""

    DISCONNECT = "Disconnect"
    CONNECT = "Connect"


class NativeDriverIO(AravisDriverIO):
    """ophyd-async's ADAravis driver, with ``Software`` among the trigger sources."""

    trigger_source: A[SignalRW[NativeTriggerSource], PvSuffix.rbv("TriggerSource")]  # type: ignore[assignment]
    # Processing the command record fires it (verified on the bench camera).
    trigger_software: A[TriggerableCommand, PvSuffix("TriggerSoftware.PROC")]
    asyn_connect: A[SignalR[AsynConnect], PvSuffix("AsynIO.CNCT")]


class NativeHdfIO(NDFileHDF5IO):
    """The stock HDF writer plus what GEECS reads off it."""

    write_message: A[SignalR[str], PvSuffix("WriteMessage")]
    epics_ts_sec: A[SignalR[int], PvSuffix("EpicsTSSec_RBV")]
    epics_ts_nsec: A[SignalR[int], PvSuffix("EpicsTSNsec_RBV")]


def liveness_word(cnct: AsynConnect) -> str:
    """The liveness verdict's words: ``Disconnected`` only when the driver lost the camera."""
    return "Disconnected" if cnct == AsynConnect.DISCONNECT else "Connected"


@dataclass
class _Session:
    """Run state shared by the trigger and acquire logics (reset at ``stage``)."""

    programmed: bool = False
    acquiring: bool = False
    restore: tuple[Any, ...] | None = None
    fire_pending: bool = False
    missed: bool = False


class NativeTriggerLogic(DetectorTriggerLogic):
    """External edges (Line1) or the software command; the camera is programmed once per run.

    ``prepare_edge`` runs before the data logic starts the HDF capture, so
    this is where a free-running camera is stopped and put in trigger mode:
    no frame of its live view can reach the stack.
    """

    def __init__(
        self,
        driver: NativeDriverIO,
        source: NativeTriggerSource,
        session: _Session,
        drain_offset: SignalR[float],
    ) -> None:
        self.driver = driver
        self.source = source
        self.session = session
        self.drain_offset = drain_offset

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
        if self.session.restore is None:
            # Captured once and kept until a restore succeeds: a run that
            # ended without a completed unstage (a killed worker, a halt, a
            # failed restore) must not make the armed state the "pre-run" one.
            self.session.restore = tuple(
                await asyncio.gather(
                    d.acquire.get_value(),
                    d.trigger_mode.get_value(),
                    d.trigger_source.get_value(),
                    d.image_mode.get_value(),
                )
            )
            if self.session.restore[1] == OnOff.ON:
                logger.warning(
                    "%s: already in trigger mode before the run (a previous run "
                    "did not restore it?) — unstage will leave it in trigger mode",
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
        # Waited for, not read once: the _RBV records update through asyn
        # callbacks that can lag the put's completion.
        wanted = {
            "TriggerMode": (d.trigger_mode, OnOff.ON),
            "TriggerSource": (d.trigger_source, self.source),
            "ImageMode": (d.image_mode, ADImageMode.CONTINUOUS),
        }

        async def settled(name: str, sig: SignalR, want: Any) -> str | None:
            try:
                await wait_for_value(sig, want, timeout=READBACK_TIMEOUT_S)
            except TimeoutError:
                return f"{name}={await sig.get_value()} (wanted {want})"
            return None

        refused = [
            r
            for r in await asyncio.gather(
                *(settled(n, sig, want) for n, (sig, want) in wanted.items())
            )
            if r
        ]
        if refused:
            raise GeecsConfigurationError(
                f"{d.name}: the camera refused its trigger setup: {', '.join(refused)}"
            )
        self.session.programmed = True

    async def default_trigger_info(self) -> TriggerInfo:
        """One edge-triggered event: the strict shot."""
        return STRICT_TRIGGER_INFO


class NativeAcquireLogic(DetectorAcquireLogic):
    """Acquisition runs for the whole run; a software trigger is pressed per shot."""

    def __init__(self, driver: NativeDriverIO, session: _Session) -> None:
        self.driver = driver
        self.session = session

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
            return
        acquire, trigger_mode, trigger_source, image_mode = restore
        d = self.driver
        await d.acquire.set(False)
        await d.trigger_mode.set(trigger_mode)
        await d.trigger_source.set(trigger_source)
        await d.image_mode.set(image_mode)
        if acquire:
            await set_and_wait_for_value(
                d.acquire, True, timeout=DEFAULT_TIMEOUT, wait_for_set_completion=False
            )
        self.session.restore = None  # only once the camera is back


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
        """The stock prepare, then the IOC's layout and SWMR off restored."""
        layout = await self.writer.xml_file_name.get_value()
        provider = await super().prepare_unbounded(datakey_name)
        await asyncio.gather(
            self.writer.xml_file_name.set(layout),
            self.writer.swmr_mode.set(False),
        )
        return provider


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


class _NotRewindable:
    """What bluesky's ``all_safe_rewind`` reads (``rewindable.get()``): never safe.

    A pause after this camera's frame landed but before the row was saved
    would otherwise **replay** the trigger and the fire on resume
    (``plans/strict.py`` wraps each shot in ``rewindable_wrapper``, and a
    device without this attribute counts as safe to re-run): the stock
    writer keeps the first frame and captures a second, one frame too many
    for one row.  Not rewindable, the resume continues where it stopped.
    """

    def get(self) -> bool:
        return False


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
        ``Line1`` (the timing system's edge) or ``Software`` (pressed per shot).
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
        self.drain_offset = soft_signal_rw(float, float(drain_offset), units="s")
        self.connected_status = derived_signal_r(
            liveness_word, cnct=self.driver.asyn_connect
        )
        self._session = _Session()
        #: Read by bluesky's ``all_safe_rewind``: a shot is never replayed.
        self.rewindable = _NotRewindable()
        self._native_acquire = NativeAcquireLogic(self.driver, self._session)
        self.add_detector_logics(
            NativeTriggerLogic(
                self.driver, self._source, self._session, self.drain_offset
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
        if self._session.restore is not None:
            await self.driver.acquire.set(False)  # no frame reaches the writer now
        await asyncio.gather(*(logic.stop() for logic in self._data_logics))
        await self._native_acquire.ensure_stopped()

    @property
    def trigger_source(self) -> NativeTriggerSource:
        """``Line1`` or ``Software`` (the registry's choice)."""
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
        """The stock prepare, strict only; a failure carries the writer's reason."""
        if isinstance(value, FlyTriggerInfo):
            raise GeecsConfigurationError(
                f"{self._geecs_device_name} is a native EPICS camera: strict scans "
                "only for now (no gated batch, no non-essential stream)"
            )
        try:
            await super().prepare(value)
        except Exception as exc:
            for reason in await self.plugin_reasons():
                exc.add_note(reason)
            raise

    def _frame_timeout(self) -> float:
        """The frame wait's budget: the prepared ``TriggerInfo.exposure_timeout``."""
        ctx = self._prepare_ctx
        return (ctx.trigger_info if ctx else STRICT_TRIGGER_INFO).exposure_timeout

    def trigger(self) -> AsyncStatus:
        """Wait for one more frame in the stack (pressing the software trigger if that is the source)."""
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

    @AsyncStatus.wrap
    async def kickoff(self) -> None:
        """Refused: native cameras fly in no stream yet (strict only)."""
        raise GeecsConfigurationError(
            f"{self._geecs_device_name} is a native EPICS camera: strict scans only"
        )

    async def discard_uncollected(self) -> None:
        """Keep a frame that arrived; fail the step on this camera's own miss.

        The strict plan calls this on every plugin-backed device when any
        device missed.  A frame this camera did write is the shot's and its
        datum stays (it is the run's only external key — mixed runs with a
        gateway plugin camera are refused at bind).  Its own miss cannot be
        retaken: the stock writer cannot rewind a late frame, and a skipped
        frame shifts every later frame in Tiled (``NATIVE_DEVICES.md``).
        """
        if self._session.missed:
            raise GeecsTriggerTimeoutError(
                self._geecs_device_name,
                self._frame_timeout(),
                f"{self._geecs_device_name} missed a shot: a native camera cannot "
                "refire yet (its writer cannot rewind) — the step fails",
            )


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


def down_native_cameras(devices: Iterable[Any]):
    """Plan: the native cameras among *devices* whose IOC reports or is down.

    Judged here, not by the gateway's fail-open rule: the word comes from the
    IOC's own ``AsynIO.CNCT``, so a read that fails (an IOC that never
    answered, or died after an earlier run's connect — the channel stays
    cached in a long-lived worker) means the IOC is down.  Concurrent, one
    bounded wait (``LIVENESS_CONNECT_TIMEOUT_S`` for the connect, then the
    read); ``connect_on_demand`` would connect only the derived
    ``connected_status``, never this source.

    Returns
    -------
    list of str
        The GEECS names of the cameras reported ``Disconnected`` or unreachable.
    """
    cams = native_cameras(devices)
    if not cams:
        return []

    async def judge() -> list[str | BaseException]:
        return await asyncio.gather(
            *(_native_word(c) for c in cams), return_exceptions=True
        )

    (task,) = yield from bps.wait_for([judge])
    words = task.result()
    down: list[str] = []
    for cam, word in zip(cams, words):
        if isinstance(word, BaseException):
            logger.warning(
                "liveness: %s IOC did not answer (%s: %s) — down",
                cam._geecs_device_name,
                type(word).__name__,
                " ".join(str(word).split()),
            )
        if isinstance(word, BaseException) or word == "Disconnected":
            down.append(cam._geecs_device_name)
    return down


def refuse_native_cameras(devices: Iterable[Any], why: str) -> None:
    """Refuse a run shape a native camera cannot take part in yet, before anything moves.

    Raises
    ------
    GeecsConfigurationError
        Naming every native camera among *devices* and *why*.
    """
    if found := native_cameras(devices):
        names = ", ".join(c._geecs_device_name for c in found)
        raise GeecsConfigurationError(
            f"{names}: native EPICS cameras run in strict scans only for now — {why}"
        )

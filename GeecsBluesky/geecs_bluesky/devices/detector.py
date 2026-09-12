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
  ``PathProvider``; and, for a camera whose host serves the PVA gateway's
  file plugin (#806, ``Planning/native_bluesky/06_pva_file_plugin.md``),
  the **stock** ``ADHDFDataLogic`` over :class:`~geecs_bluesky.devices.hdf_plugin.GeecsHdfIO`
  — one per image variable, nothing of ours in the data path.

A missed shot (no frame within the timeout, the device still live) does not
void the row: the acquire logic remembers it until the next baseline and
the scalar columns of that device read ``NaN`` for that row — the partial
row the strict plan records (scalars only, no frames) before taking one
more shot for the step (design §2.1).  On a plugin-backed camera the count wait precedes the
stamp wait, so a dropped frame surfaces as the count timeout;
:meth:`GeecsDetector.trigger` translates it into the GEECS timeout the
plan's refire gate understands, and :meth:`GeecsDetector.discard_uncollected`
is the late-frame guard the plan calls before the retake.

In a **gated** batch (phase 2, ``08_gated_batch.md`` §4.2) the same
device flies: ``prepare(number_of_events=N)`` baselines the plugin's count,
``kickoff`` arms the quota and switches the acquire logic to *fly mode*
(``complete`` returns when the plugin has counted the quota — the stamp
wait is a strict-mode concept, so ``wait_for_idle`` is a no-op there;
``trigger`` switches back), :meth:`GeecsDetector.truncate_to_quota` rewinds
the extra in-flight frame after the box goes OFF, and
:meth:`GeecsDetector.rewind_to_step_baseline` throws a repeated step's
partial frames away before it is retaken.  A gated step the plan abandons
(an immediate pause, a stalled neighbour) is told so
(:meth:`GeecsDetector.abandon_step`): the pending ``complete`` then settles
quietly instead of failing into a later message.

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
import math
import numbers
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
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
    wait_for_value,
)
from ophyd_async.core._detector import _data_logic_supported
from ophyd_async.epics.adcore import ADHDFDataLogic, NDArrayDescription
from ophyd_async.epics.core import epics_signal_r, epics_signal_rw

from geecs_core.pv_naming import hdf_plugin_prefix

from geecs_bluesky.data_paths import device_server_save_path
from geecs_bluesky.devices.ca._pv import ca_pv, setpoint_pv
from geecs_bluesky.devices.ca._view import ScalarsView
from geecs_bluesky.devices.hdf_plugin import GeecsHdfIO
from geecs_bluesky.exceptions import GeecsConfigurationError, GeecsTriggerTimeoutError
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)

#: The GEECS shot stamp variable (generated inside LabVIEW, not a DB row yet).
ACQ_TIMESTAMP = "acq_timestamp"

#: Seconds a shot may take to arrive after the fire: one trigger period (the
#: single shot fires on the *next* edge) plus the device's exposure and
#: drain (§7, M1).  One constant for every device until the calibration
#: phase makes it a per-device budget.
DEFAULT_SHOT_TIMEOUT = 3.0

#: How a strict shot prepares a GeecsDetector: one externally edge-triggered
#: event.  The plan fires the box; the detector waits for its stamp — and,
#: on a plugin-backed camera, for the plugin's frame count first, bounded by
#: the same budget (``exposure_timeout``; the stock default would be 13 s).
STRICT_TRIGGER_INFO = TriggerInfo(
    trigger=DetectorTrigger.EXTERNAL_EDGE,
    number_of_events=1,
    exposure_timeout=DEFAULT_SHOT_TIMEOUT,
)


class FlyTriggerInfo(TriggerInfo):
    """A ``TriggerInfo`` that says *fly* explicitly: a batch or an unbounded stream.

    The mode cannot be read off the event count — a gated step of one shot
    (``shots_per_step=1``, the default) prepares with ``number_of_events=1``,
    the strict signature — so the plan says it with the type: a
    :class:`FlyTriggerInfo` prepare takes the streamable logic only (no
    per-event scalars, no LabVIEW-native saving) and is refused on a camera
    without a plugin; a plain :class:`TriggerInfo` is a strict shot.
    """


#: How a non-essential stream prepares a plugin-backed camera: external
#: edges, an unbounded number of events (``0`` — the plugin counts what it
#: gets for the run's duration; nothing waits on it).
UNBOUNDED_TRIGGER_INFO = FlyTriggerInfo(
    trigger=DetectorTrigger.EXTERNAL_EDGE,
    number_of_events=0,
    exposure_timeout=DEFAULT_SHOT_TIMEOUT,
)


def gated_trigger_info(
    quota: int, *, exposure_timeout: float = DEFAULT_SHOT_TIMEOUT
) -> FlyTriggerInfo:
    """How a gated step prepares a plugin-backed camera: *quota* edge-triggered events.

    ``exposure_timeout`` is **per frame** (ophyd-async passes it to
    ``observe_signals_value`` as the budget between updates): a camera that
    stops producing frames for that long fails ``complete``.
    """
    if quota < 1:
        raise ValueError(f"a gated step needs at least one shot, got {quota}")
    return FlyTriggerInfo(
        trigger=DetectorTrigger.EXTERNAL_EDGE,
        number_of_events=quota,
        exposure_timeout=exposure_timeout,
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
        shot_timeout: float = DEFAULT_SHOT_TIMEOUT,
        queue_maxsize: int | None = None,
    ) -> None:
        self._signal = acq_timestamp
        self._device = device_name
        self.shot_timeout = shot_timeout
        #: The last awaited shot never arrived (cleared by the next baseline).
        self.missed = False
        #: Fly mode (a gated batch or a non-essential stream): the plugin's
        #: count is the completion, so the stamp wait is skipped.  Set by
        #: ``GeecsDetector.kickoff``, cleared by ``trigger``.
        self.fly = False
        #: The plan abandoned the step in flight (an immediate pause, a
        #: stalled neighbour): a pending ``complete`` settles quietly.
        self.abandoned = False
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
        self.missed = False
        self._drain()

    def mark_missed(self) -> None:
        """The awaited shot never arrived: this device's row reads empty."""
        self.missed = True

    async def start_acquiring(self) -> None:
        """Nothing to start: LabVIEW is always acquiring; the plan fires the box."""

    async def wait_for_idle(self) -> None:
        """Wait for the stamp to advance past the baseline.

        Cold cache (baseline ``None``, no update since the monitor attached):
        deliberately **no** CA-get baseline — a get raced the shot itself, so
        the first positive arrival *is* the shot; ``baseline()`` already
        drained anything older.

        In fly mode the count *is* the completion (§4.2): returns at once.
        """
        if self.fly:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.shot_timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                self.missed = True
                raise GeecsTriggerTimeoutError(self._device, self.shot_timeout)
            try:
                value = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                self.missed = True
                raise GeecsTriggerTimeoutError(
                    self._device, self.shot_timeout
                ) from None
            if value != self._t0:
                logger.debug("%s: shot (%s → %s)", self._device, self._t0, value)
                return

    async def ensure_stopped(self) -> None:
        """Nothing to stop: acquisition is LabVIEW's, saving is the data logic's."""


def mask_missed_shot(readings: dict[str, Reading]) -> dict[str, Reading]:
    """The empty shot: every column blank (numbers and arrays ``NaN``, text ``""``).

    The CA monitor cache holds the *previous* shot's values when no frame
    arrived (LabVIEW's timeout event carries unchanged values and the
    gateway drops it), so reading it through would silently record the
    wrong shot.  The stamp becomes ``NaN`` too — no frame joins to this
    row.  Booleans and numpy scalars are numbers here (``numbers.Real``),
    arrays are filled with ``NaN``; anything else is left as is.
    """
    now = time.time()
    masked: dict[str, Reading] = {}
    for key, reading in readings.items():
        value = reading["value"]
        blank: Any
        if isinstance(value, str):
            blank = ""
        elif isinstance(value, numbers.Real):
            blank = math.nan
        elif isinstance(value, np.ndarray):
            blank = np.full(value.shape, np.nan)
        else:
            blank = value
        masked[key] = Reading(value=blank, timestamp=now, alarm_severity=0)
    return masked


class _SignalsProvider(ReadableDataProvider):
    """Read a fixed set of signals into the event row (masked after a missed shot)."""

    def __init__(self, signals: Sequence[SignalR], acquire: GeecsAcquireLogic) -> None:
        self._signals = tuple(signals)
        self._acquire = acquire

    async def make_datakeys(self) -> dict[str, DataKey]:
        return await merge_gathered_dicts(sig.describe() for sig in self._signals)

    async def make_readings(self) -> dict[str, Reading]:
        readings = await merge_gathered_dicts(sig.read() for sig in self._signals)
        return mask_missed_shot(readings) if self._acquire.missed else readings


class ScalarsDataLogic(DetectorDataLogic):
    """The device's own scalar variables (and its stamp) as event columns."""

    def __init__(self, signals: Sequence[SignalR], acquire: GeecsAcquireLogic) -> None:
        self._signals: list[SignalR] = list(signals)
        self._acquire = acquire

    def add(self, *signals: SignalR) -> None:
        """Add columns (a subscribed settable child's readback, at namespace build)."""
        self._signals.extend(signals)

    async def prepare_single(self, datakey_name: str) -> ReadableDataProvider:
        """One reading per event: the signals' current values."""
        return _SignalsProvider(self._signals, self._acquire)


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
        Where this run's files go.  Called with *directory_name*, not the
        datakey: the device directory inside ``ScanNNN/`` is the GEECS
        device name (``Scan065/UC_Amp4_IR_input/``), the path every
        analysis reader builds (``ScanPaths.build_device_file_map``) —
        never the lowercase ophyd name.
    directory_name :
        The GEECS device name the run folder's sub-directory is called.
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
        directory_name: str,
        device_path: Callable[[str], str] = device_server_save_path,
    ) -> None:
        self._localsavingpath = localsavingpath
        self._save = save
        self._path_provider = path_provider
        self._directory_name = directory_name
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
        info = self._path_provider(self._directory_name)
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


class GeecsDetectorScalars(ScalarsView):
    """A detector's scalars-only view: the same shot wait, no file writing.

    ``UC_Amp4_IR_input.scalars`` in a plan's detector list records the
    camera's per-shot scalars (and its stamp) **exactly** — it is
    ``Triggerable`` through the parent's acquire logic, so its row is the
    shot's frame, not whatever the monitor cache held — but never turns
    native saving on: the parent's data logics are not prepared.  This is
    how a preset says *scalars only* for a camera (``save_images: false``)
    with a stock plan signature (:class:`~geecs_bluesky.devices.ca._view.ScalarsView`).
    """

    _owner: GeecsDetector

    @property
    def connected_status(self) -> SignalR[str]:
        """The parent's gateway liveness PV (the refire gate reads it)."""
        return self._owner.connected_status

    @property
    def missed_shot(self) -> bool:
        """Whether the parent's last awaited shot never arrived."""
        return self._owner._acquire.missed

    def trigger(self) -> AsyncStatus:
        """Baseline the parent's stamp now, then wait for it to advance."""
        acquire = self._owner._acquire
        acquire.baseline()
        return AsyncStatus(acquire.wait_for_idle())

    async def read(self) -> dict[str, Reading]:
        """The parent's scalar columns (and stamp) — same keys as the parent."""
        readings = await merge_gathered_dicts(
            sig.read() for sig in self._owner._scalar_signals()
        )
        return mask_missed_shot(readings) if self._owner._acquire.missed else readings

    async def describe(self) -> dict[str, DataKey]:
        """Data keys of the parent's scalar columns."""
        return await merge_gathered_dicts(
            sig.describe() for sig in self._owner._scalar_signals()
        )


class GeecsDetector(StandardDetector):
    """One GEECS acquirer (camera, spectrometer, scope) as a StandardDetector.

    ``scalars`` (:class:`GeecsDetectorScalars`) is the scalars-only view a
    plan lists instead of the detector itself when the frames are not
    wanted this run.

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
    hdf_plugins :
        ``(image variable, path provider)`` per file plugin to capture
        (#806): each becomes a :class:`GeecsHdfIO` child (``hdf``, then
        ``hdf_<variable>``) driven by the stock ``ADHDFDataLogic``; the
        first writes the ``<name>`` stream key, the others
        ``<name>-<variable>``.  The namespace passes the camera's primary
        image variable only (a secondary one is pushed only when an
        operation produces it, so its plugin would never arm).  With a *path_provider* as well the
        camera also writes its native files (dual-write, until PNG
        retirement #738); without one a stale ``save=on`` is still cleared.
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
        hdf_plugins: Sequence[tuple[str, PathProvider]] = (),
        shot_timeout: float = DEFAULT_SHOT_TIMEOUT,
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
        self._scalars_logic = ScalarsDataLogic(
            (*scalars, self.acq_timestamp), self._acquire
        )
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
                LvNativeFileDataLogic(
                    self.localsavingpath,
                    self.save,
                    path_provider,
                    directory_name=device,
                )
            )
        self._hdf_ios: list[GeecsHdfIO] = []
        for index, (variable, provider) in enumerate(hdf_plugins):
            io = GeecsHdfIO(
                f"pva://{hdf_plugin_prefix(experiment or '', device, variable)}"
            )
            setattr(self, "hdf" if index == 0 else f"hdf_{safe_name(variable)}", io)
            self._hdf_ios.append(io)
            logics.append(
                ADHDFDataLogic(
                    array_description=NDArrayDescription(
                        shape_signals=[
                            io.array_size_z,
                            io.array_size_y,
                            io.array_size_x,
                        ],
                        data_type_signal=io.data_type,
                        color_mode_signal=io.color_mode,
                    ),
                    path_provider=provider,
                    driver=io,
                    writer=io,
                    datakey_suffix="" if index == 0 else f"-{safe_name(variable)}",
                )
            )
        self.add_detector_logics(*logics)
        # The scalars-only view (``X.scalars`` in a plan's detector list).
        self.scalars = GeecsDetectorScalars(self)
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

    @property
    def plugin_backed(self) -> bool:
        """Whether the camera's frames are written by the gateway's file plugin."""
        return bool(self._hdf_ios)

    @property
    def missed_shot(self) -> bool:
        """Whether the last awaited shot never arrived (the row reads empty)."""
        return self._acquire.missed

    def _scalar_signals(self) -> tuple[Any, ...]:
        """Every scalar column plus the stamp — what ``scalars`` reads."""
        return (*self._scalars, self.acq_timestamp)

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

    @staticmethod
    def _is_fly_prepare(value: TriggerInfo) -> bool:
        """A :class:`FlyTriggerInfo` (a batch, an unbounded stream) — never a strict shot."""
        return isinstance(value, FlyTriggerInfo)

    async def _update_prepare_context(self, trigger_info: TriggerInfo) -> None:
        """The stock context; in a fly prepare only the streamable logics take part.

        A :class:`FlyTriggerInfo` prepare — a batch (the gated step, any
        quota, one included) or an unbounded stream (a non-essential
        detector) — produces data through the plugin's stream only: the
        per-event readables — the scalar columns
        (a gated run's rows come from the sampler, ``08`` §4.7) and
        LabVIEW-native saving (the plugin counts the frames; native saving
        would write every edge's frame unbounded) — are left out, where the
        stock logic would refuse ("Multiple collections not supported") or
        switch native saving on.  Switching mode invalidates a context the
        stock code would otherwise reuse (it keys reuse on
        ``collections_per_event`` alone).
        """
        fly = self._is_fly_prepare(trigger_info)
        if (
            self._prepare_ctx is not None
            and self._is_fly_prepare(self._prepare_ctx.trigger_info) != fly
        ):
            self._prepare_ctx = None
        if not fly:
            await super()._update_prepare_context(trigger_info)
            return
        saved = self._data_logics
        self._data_logics = tuple(
            dl for dl in saved if _data_logic_supported(dl.prepare_unbounded)
        )
        try:
            await super()._update_prepare_context(trigger_info)
        finally:
            self._data_logics = saved

    @AsyncStatus.wrap
    async def prepare(self, value: TriggerInfo) -> None:
        """The stock prepare; a failure on a plugin-backed camera carries the plugin's reason.

        The stock logic completes ``Capture=1`` on ``Capture_RBV`` alone and
        never awaits the put, so the plugin's ``op.done(error=…)`` (no
        frame while arming, a missing directory) is lost and the failure
        reads as a bare timeout on the PV.  ``WriteMessage`` holds the
        reason; it is attached to the exception as a note.  A fly prepare
        (a batch or an unbounded stream) on a camera without a plugin is
        refused here, before any move: nothing of it can count.
        """
        if self._is_fly_prepare(value) and not self._hdf_ios:
            raise GeecsConfigurationError(
                f"{self._geecs_device_name} has no file plugin: it cannot count a "
                "batch or stream frames (a LabVIEW-native camera in a gated run "
                "or a non-essential list) — use acquisition='strict', or list "
                "its scalars only"
            )
        try:
            await super().prepare(value)
        except Exception as exc:
            for io in self._hdf_ios:
                try:
                    message = await asyncio.wait_for(io.write_message.get_value(), 2.0)
                except Exception:  # noqa: BLE001 - the note is best effort
                    continue
                if message:
                    exc.add_note(f"file plugin {io.name}: {message}")
            raise

    def trigger(self) -> AsyncStatus:
        """Baseline the stamp **now**, then wait for it to advance.

        The baseline must happen before this returns: the plan's very next
        message is the fire, and a stamp that lands before an asynchronous
        baseline would be counted as the pre-shot value and the real shot
        missed.  On a plugin-backed camera the stock trigger waits for the
        plugin's count first (``exposure_timeout``); that timeout is
        translated into the GEECS one so the plan's refire gate sees one
        kind of miss.  A trigger is strict by definition: it leaves fly
        mode (the stamp wait is back on).
        """
        self._acquire.fly = False
        self._acquire.baseline()
        status = super().trigger()
        if not self._hdf_ios:
            return status
        return AsyncStatus(self._translate_count_timeout(status))

    @AsyncStatus.wrap
    async def kickoff(self) -> None:
        """The stock kickoff in **fly mode**: the count is the completion.

        The mode cannot be read off the prepare (a gated step with one shot
        prepares with ``number_of_events=1``, the strict signature), so it
        is explicit: ``kickoff`` sets it, ``trigger`` clears it (§4.2).  A
        stale ``missed`` from a strict run's last shot is cleared too — no
        row of this stream is a shot the plan fired.
        """
        self._acquire.fly = True
        self._acquire.abandoned = False
        self._acquire.missed = False
        await super().kickoff()

    def complete(self) -> AsyncStatus:
        """The stock complete; a count timeout carries the GEECS error, an abandoned step settles.

        The stock wait raises a bare ``TimeoutError`` when the plugin counts
        nothing for ``exposure_timeout`` — translated into
        :exc:`~geecs_bluesky.exceptions.GeecsTriggerTimeoutError` (one kind
        of miss for the plan) unless the plan abandoned the step meanwhile
        (:meth:`abandon_step`): the box is OFF then and no frame is coming,
        so the pending status completes quietly instead of failing into
        whatever message the plan is at by then (the RunEngine throws any
        failed status into the plan at its next message).
        """
        inner = super().complete()
        status = AsyncStatus(self._guarded_complete(inner))
        self._step_status = status
        return status

    async def _guarded_complete(self, inner: AsyncStatus) -> None:
        try:
            await inner
        except TimeoutError as exc:
            if self._acquire.abandoned:
                logger.info(
                    "%s: gated step abandoned; the pending complete settled",
                    self._geecs_device_name,
                )
                return
            raise GeecsTriggerTimeoutError(
                self._geecs_device_name,
                self._acquire.shot_timeout,
                f"{self._geecs_device_name}: the file plugin counted no frame "
                f"for {self._acquire.shot_timeout:.1f}s while the box ran",
            ) from exc

    def mark_abandoned(self) -> None:
        """Synchronously: the step is over; a pending ``complete`` settles quietly.

        The plan calls this the moment its interrupted wait returns — before
        it yields another message — so a count timeout landing in the next
        loop iteration is already pardoned (the RunEngine throws any failed
        status into the plan at its next message).
        """
        self._acquire.abandoned = True

    async def abandon_step(self) -> None:
        """Wait for a pending gated ``complete`` to settle (after :meth:`mark_abandoned`).

        Called by the plan after it drove the box OFF on an interrupted or
        failed step, before it rewinds and retakes (or fails) the step —
        so no status of the abandoned step fails into a later message.
        """
        self.mark_abandoned()
        status = getattr(self, "_step_status", None)
        if status is None or status.done:
            return
        try:
            await status
        except Exception:  # noqa: BLE001 - a failure of the abandoned step is settled here
            logger.debug(
                "%s: abandoned complete finished with an error (settled)",
                self._geecs_device_name,
                exc_info=True,
            )

    async def _translate_count_timeout(self, status: AsyncStatus) -> None:
        try:
            await status
        except TimeoutError as exc:  # observe_signals_value's exposure_timeout
            self._acquire.mark_missed()
            raise GeecsTriggerTimeoutError(
                self._geecs_device_name,
                self._acquire.shot_timeout,
                f"{self._geecs_device_name}: no frame counted by the file plugin "
                f"within {self._acquire.shot_timeout:.1f}s",
            ) from exc

    @property
    def step_baseline(self) -> int | None:
        """The plugin count the current prepare baselined (``None`` outside ``prepare``)."""
        ctx = self._prepare_ctx
        return None if ctx is None else int(ctx.collections_written)

    async def truncate_to_quota(self) -> None:
        """Rewind every plugin to ``baseline + quota``: the step's frames, exactly.

        The gated step's trim (§4.2): after ``complete`` returned and the box
        went OFF, at most one more edge was in flight; once it has landed
        (the plan waits one period plus the drain offset), every frame past
        the quota is truncated and a later arrival is stale to the plugin.
        A no-op without a plugin or outside ``prepare``.
        """
        ctx = self._prepare_ctx
        if ctx is None or not self._hdf_ios:
            return
        keep = int(ctx.collections_written + ctx.trigger_info.number_of_collections)
        await self._rewind_plugins(keep, "quota")

    async def rewind_to_step_baseline(self) -> None:
        """Rewind every plugin to the count the step's prepare baselined.

        The repeated-step path (§4.2, Sam 2026-09-12): after an immediate
        pause the step is retaken from its first shot, so the partial frames
        (and any edge that slipped in between the resume and the OFF) leave
        the stack first.  A no-op without a plugin or outside ``prepare``.
        """
        ctx = self._prepare_ctx
        if ctx is None or not self._hdf_ios:
            return
        await self._rewind_plugins(int(ctx.collections_written), "step baseline")

    async def _rewind_plugins(self, keep: int | None, what: str) -> None:
        """Rewind every plugin to *keep* frames (``None``: each provider's ``last_emitted``)."""
        ctx = self._prepare_ctx
        assert ctx is not None
        for provider in ctx.streamable_data_providers:
            io = next(
                (
                    io
                    for io in self._hdf_ios
                    if provider.collections_written_signal is io.num_captured
                ),
                None,
            )
            if io is None:
                continue
            target = int(getattr(provider, "last_emitted", 0)) if keep is None else keep
            written = int(await io.num_captured.get_value())
            if written < target:
                raise GeecsTriggerTimeoutError(
                    self._geecs_device_name,
                    self._acquire.shot_timeout,
                    f"{self._geecs_device_name}: {written} frame(s) in the stack "
                    f"but the {what} is {target} — frames vanished after complete",
                )
            # Always put, even at the target: the plugin's Rewind also sets its
            # stale watermark, so a frame arriving later (the missed shot's,
            # the in-flight edge's) is dropped rather than appended.
            await io.rewind.set(target)
            await wait_for_value(io.num_captured, target, timeout=DEFAULT_TIMEOUT)
            logger.info(
                "%s: rewound %d → %d frame(s) (%s)",
                self._geecs_device_name,
                written,
                target,
                what,
            )

    async def discard_uncollected(self) -> None:
        """Rewind every plugin to the last frame a document referenced.

        The late-frame guard (design §2.1): called by the plan on every
        plugin-backed device of a partial shot, before the retake fires.  A
        delivered frame no row referenced and a late frame of the missed shot
        are truncated alike, and one that arrives later is stale to the
        plugin.  Nothing a row references is reachable, because the retake
        has not fired yet.  A no-op without a plugin or outside ``prepare``.
        """
        ctx = self._prepare_ctx
        if ctx is None or not self._hdf_ios:
            return
        await self._rewind_plugins(None, "last datum")

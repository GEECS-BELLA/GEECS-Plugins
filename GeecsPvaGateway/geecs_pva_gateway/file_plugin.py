"""The areaDetector-shaped HDF5 file plugin: one per served image variable.

The second consumer of the frame the gateway already receives
(``Planning/native_bluesky/06_pva_file_plugin.md``): it branches off the
push callback **before** the latest-wins slot, so intake is lossless within
a capture session, and writes one ``<device>.h5`` per device per scan in
the NDFileHDF5 layout (``/entry/data/data`` and
``/entry/instrument/NDAttributes/<name>``: the frame's stamps and, since
0.9.0, the device's subscribed scalars as pushed with the frame —
``Planning/native_bluesky/08_gated_batch.md`` §4.4).  The PV contract is the
``NDFileHDF5IO`` set ophyd-async 0.19.3 connects (every annotated suffix
must exist, served under ``<experiment>:<device>:<variable>:hdf1:``) plus
three GEECS PVs: ``Rewind`` (drop the frames past a count and any later
frame stamped before now — the late-frame guard of a refire), and
``WriteStatus`` / ``WriteMessage`` (the last writer error).

Threads: puts arrive on p4p worker threads and frames on the gateway's
event loop; both only enqueue.  One writer thread owns every piece of
session state and the file handle, so nothing is shared.

Session semantics (§4 of the design):

- ``Capture=1`` validates the parameters, retains the variable's GEECS
  subscription (the same refcount a PVA client holds), and completes the
  put only once one frame has been decoded — that is where the geometry
  the stream resource describes comes from; LabVIEW's 1 Hz idle re-push
  is enough.  A camera that pushes nothing fails the put.
- A frame is written iff its stamp is unseen this session (LabVIEW
  re-pushes its last frame with an unchanged stamp when idle) and not
  older than the stale watermark set at ``Capture=1`` (and moved by
  ``Rewind``).  The first accepted frame opens the file (``LazyOpen``):
  a session that accepts nothing leaves no file.
- ``NumCaptured_RBV`` posts after each frame is flushed to disk.
- ``Capture=0`` stamps the reconciliation counters, closes the file and
  releases the subscription.

The plugin never creates a directory: ``CreateDirectory`` is accepted and
ignored, and ``FilePathExists_RBV`` answers for the directory the worker
claimed (root ``CLAUDE.md``, the scan-folder invariant).  HDF5 SWMR is
never used across SMB: ``SWMRMode`` is accepted and ignored, the writer
flushes per frame with file locking off, and readers open the closed
``libver="latest"`` file afterwards (§5 of the design).
"""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from xml.sax.saxutils import quoteattr

import numpy as np
from p4p.nt import NTEnum, NTScalar
from p4p.server.thread import SharedPV

from geecs_core.pv_naming import (
    HDF_PLUGIN_SUFFIX,
    hdf_plugin_prefix,
    normalize_component,
)
from geecs_data_utils.io import decode_imaq_image_string
from geecs_data_utils.io.scan_stack import ATTRIBUTES_GROUP, FRAMES_DATASET

from geecs_pva_gateway import __version__

logger = logging.getLogger(__name__)


def available() -> bool:
    """Whether the writer's container library is installed on this host.

    ``h5py`` reaches a camera server as a fleet pin installed on restart
    (``deploy/requirements-fleet.txt`` from the share's wheel cache), so a
    box whose restart could not reach the cache serves no plugin PVs at all
    — the worker's host list keeps such cameras on LabVIEW-native saving —
    rather than PVs whose ``Capture`` can only fail.
    """
    try:
        import h5py  # noqa: F401
    except ImportError:
        return False
    return True


#: PV suffix under the image variable's PV name (the areaDetector ``HDF1:``),
#: owned by the naming contract so the worker's ``GeecsHdfIO`` cannot drift.
PLUGIN_SUFFIX = HDF_PLUGIN_SUFFIX
#: The dataset paths (``FRAMES_DATASET``, ``ATTRIBUTES_GROUP``) are the read
#: side's (``geecs_data_utils.io.scan_stack``): the NDFileHDF5 layout
#: ophyd-async's ``ADHDFDataLogic`` describes and Tiled's HDF5 adapter reads.
#: The per-frame attribute datasets.  Each plugin names them
#: ``<device>-hdf-<variable>-<suffix>`` (:func:`attribute_names`, every
#: part through ``normalize_component``, the worker's ophyd-name rule):
#: the two frame stamps (:data:`ATTRIBUTE_SUFFIXES`) and then one per
#: subscribed scalar of the device (``CameraSpec.scalar_variables``, the
#: DB ``get='yes'`` list — the same columns a strict row carries for that
#: device, so a gated row is the same row; ``08_gated_batch.md`` §4.4).
#: The stock ``ADHDFDataLogic`` turns attribute names into stream data
#: keys verbatim, so the names must be **unique across the cameras of one
#: run** (bare ``acq_timestamp`` collided on the second camera,
#: GEECS-Plugins#829) and **disjoint from every event column** of the
#: detector (``<name>-acq_timestamp`` is the camera's own CA stamp column;
#: a stream key of the same name overwrote its description and broke
#: Tiled's ingestion).  ``-hdf-<variable>-`` names the plugin child, so
#: ``<name>-hdf-image-maxcounts`` never spells the event column
#: ``<name>-maxcounts``; the ``frame_`` suffixes never spell a variable.
ATTRIBUTE_SUFFIXES = ("frame_acq_timestamp", "frame_recv_timestamp")
_ATTRIBUTE_DESCRIPTIONS = {
    "frame_acq_timestamp": "GEECS acquisition stamp of the frame, Unix s (the shot join key)",
    "frame_recv_timestamp": "gateway receive time of the frame, Unix s (delivery diagnostics)",
}


def attribute_prefix(device: str, variable: str) -> str:
    """``<device>-hdf-<variable>``: the attribute-name prefix of one image variable."""
    return f"{normalize_component(device)}-hdf-{normalize_component(variable)}"


def attribute_names(
    device: str, variable: str, scalars: Sequence[str] = ()
) -> tuple[str, ...]:
    """The attribute dataset (and stream data key) names for one image variable.

    The two frame stamps first, then one per scalar in *scalars* (the
    device's subscribed scalar variables, in their DB order).
    """
    prefix = attribute_prefix(device, variable)
    return tuple(
        f"{prefix}-{suffix}"
        for suffix in (*ATTRIBUTE_SUFFIXES, *(normalize_component(s) for s in scalars))
    )


def attributes_xml(device: str, variable: str, scalars: Sequence[str] = ()) -> str:
    """The ``NDAttributesFile`` document declaring the plugin's attribute datasets."""
    descriptions = [
        *(_ATTRIBUTE_DESCRIPTIONS[suffix] for suffix in ATTRIBUTE_SUFFIXES),
        *(
            f"GEECS {device} {scalar} as pushed with the frame (NaN if absent)"
            for scalar in scalars
        ),
    ]
    return (
        "<Attributes>"
        + "".join(
            f'<Attribute name="{name}" type="PARAM" source="{name}" '
            f'datatype="DOUBLE" description={quoteattr(description)}/>'
            for name, description in zip(
                attribute_names(device, variable, scalars), descriptions, strict=True
            )
        )
        + "</Attributes>"
    )


def scalar_value(value: object) -> float:
    """The ``DOUBLE`` a pushed scalar is written as: the number, else ``NaN``.

    The subscriber already coerces a non-text variable's value to ``int`` /
    ``float`` when it parses as one; anything else (a text enum value, a
    variable the device did not send this shot) is ``NaN`` — the plugin
    never invents a value and never changes an attribute's dtype.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value)


#: The chunk shape ophyd-async assumes for attribute datasets.
ATTRIBUTE_CHUNK = 16384
#: Seconds a frame may be stamped before the watermark and still count as
#: fresh (clock skew between the camera server and the stamp's domain time).
STALE_MARGIN_S = 0.1
#: How long ``Capture=1`` waits for the first frame before failing the put
#: (below ophyd-async's 10 s default put timeout).
ARM_TIMEOUT_S = 8.0
#: Raw blobs the writer may fall behind by before intake counts drops.
QUEUE_DEPTH = 64

DATA_TYPES = (
    "Int8",
    "UInt8",
    "Int16",
    "UInt16",
    "Int32",
    "UInt32",
    "Int64",
    "UInt64",
    "Float32",
    "Float64",
)
_NUMPY_TO_AD = {
    "int8": "Int8",
    "uint8": "UInt8",
    "int16": "Int16",
    "uint16": "UInt16",
    "int32": "Int32",
    "uint32": "UInt32",
    "int64": "Int64",
    "uint64": "UInt64",
    "float32": "Float32",
    "float64": "Float64",
}
COLOR_MODES = ("Mono",)
FILE_WRITE_MODES = ("Single", "Capture", "Stream")
COMPRESSIONS = ("None", "N-bit", "szip", "zlib", "Blosc", "BSLZ4", "LZ4", "JPEG")
ENABLE_DISABLE = ("Enable", "Disable")
WRITE_STATUS = ("Write OK", "Write Error")


@dataclass(frozen=True)
class _Param:
    """One row of the PV table."""

    suffix: str
    kind: str  # "s", "i", "?", "enum"
    initial: Any
    rbv: bool = False  # also serve <suffix>_RBV echoing the value
    choices: Sequence[str] = ()


#: Every suffix ``NDFileHDF5IO`` connects, plus the three GEECS PVs.
PV_TABLE: tuple[_Param, ...] = (
    # NDArrayBaseIO
    _Param("PortName_RBV", "s", "HDF1"),
    _Param("UniqueId_RBV", "i", 0),
    _Param(
        "NDAttributesFile", "s", ""
    ),  # per instance: attributes_xml(device, variable)
    _Param("ArraySizeX_RBV", "i", 0),
    _Param("ArraySizeY_RBV", "i", 0),
    _Param("ArraySizeZ_RBV", "i", 0),
    _Param("ArraySize0_RBV", "i", 0),
    _Param("ArraySize1_RBV", "i", 0),
    _Param("ArraySize2_RBV", "i", 0),
    _Param("ColorMode_RBV", "enum", "Mono", choices=COLOR_MODES),
    _Param("DataType_RBV", "enum", "UInt16", choices=DATA_TYPES),
    _Param("ArrayCounter", "i", 0, rbv=True),
    _Param("ADCoreVersion_RBV", "s", f"geecs-pva-gateway {__version__}"),
    _Param("DriverVersion_RBV", "s", __version__),
    # NDPluginBaseIO
    _Param("NDArrayPort", "s", "", rbv=True),
    _Param("EnableCallbacks", "enum", "Enable", rbv=True, choices=ENABLE_DISABLE),
    _Param("NDArrayAddress", "i", 0, rbv=True),
    _Param("QueueSize", "i", QUEUE_DEPTH, rbv=True),
    # NDFileIO
    _Param("FilePath", "s", "", rbv=True),
    _Param("FileName", "s", "", rbv=True),
    _Param("FilePathExists_RBV", "?", False),
    _Param("FileTemplate", "s", "%s%s.h5", rbv=True),
    _Param("FullFileName_RBV", "s", ""),
    _Param("FileNumber", "i", 0),
    _Param("AutoIncrement", "?", True),
    _Param("FileWriteMode", "enum", "Stream", rbv=True, choices=FILE_WRITE_MODES),
    _Param("NumCapture", "i", 0, rbv=True),
    _Param("NumCaptured_RBV", "i", 0),
    _Param("Capture", "?", False, rbv=True),
    _Param("ArraySize0", "i", 0),
    _Param("ArraySize1", "i", 0),
    _Param("CreateDirectory", "i", 0),
    # NDFileHDF5IO
    _Param("PositionMode", "?", False, rbv=True),
    _Param("Compression", "enum", "None", rbv=True, choices=COMPRESSIONS),
    _Param("NumExtraDims", "i", 0, rbv=True),
    _Param("SWMRMode", "?", False, rbv=True),
    _Param("FlushNow", "?", False),
    _Param("XMLFileName", "s", "", rbv=True),
    _Param("NumFramesChunks", "i", 1, rbv=True),
    _Param("ChunkSizeAuto", "?", True, rbv=True),
    _Param("LazyOpen", "?", True, rbv=True),
    # GEECS
    _Param("Rewind", "i", 0),
    _Param("WriteStatus", "enum", "Write OK", choices=WRITE_STATUS),
    _Param("WriteMessage", "s", ""),
)

#: Puts the writer thread handles (everything else is store-and-echo).
_COMMANDS = frozenset({"Capture", "Rewind", "FlushNow"})
#: Puts accepted and ignored: the value is the plugin's, not the client's.
_IGNORED_PUTS = frozenset({"NDAttributesFile", "CreateDirectory"})


def _nt(kind: str) -> Any:
    return NTEnum() if kind == "enum" else NTScalar(kind)


def _wrap(kind: str, value: Any, choices: Sequence[str]) -> Any:
    if kind == "enum":
        return {"index": list(choices).index(value), "choices": list(choices)}
    return value


def _unwrap_put(op: Any, kind: str, choices: Sequence[str]) -> Any:
    """The Python value of a put, whatever form the client sent it in."""
    value = op.value()
    raw = getattr(value, "raw", value)
    if kind == "enum":
        index = int(raw["value.index"])
        sent = list(raw["value.choices"]) or list(choices)
        try:
            return sent[index]
        except IndexError as exc:
            raise ValueError(f"enum index {index} outside {sent}") from exc
    scalar = raw["value"]
    if kind == "?":
        return bool(scalar)
    if kind == "i":
        return int(scalar)
    return str(scalar)


@dataclass
class _Counters:
    """Per-session reconciliation: every received frame lands in one bucket.

    ``frames_received == frames_written + duplicates_dropped + stale_skipped
    + shape_errors + decode_errors + open_failures + append_failures
    + callbacks_disabled``;
    ``queue_drops`` are frames that never reached the writer and ``rewound``
    frames were written and then discarded by ``Rewind``.
    """

    frames_received: int = 0
    frames_written: int = 0
    duplicates_dropped: int = 0
    stale_skipped: int = 0
    shape_errors: int = 0
    decode_errors: int = 0
    queue_drops: int = 0
    rewound: int = 0
    append_failures: int = 0
    open_failures: int = 0
    callbacks_disabled: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))


@dataclass
class _Session:
    """One capture window (``Capture=1`` → ``Capture=0``); writer-thread only."""

    directory: str
    filename: str
    stale_before: float
    frames_per_chunk: int
    compression: str
    seen: set[float] = field(default_factory=set)
    counters: _Counters = field(default_factory=_Counters)
    count: int = 0
    file: Any = None  # h5py.File once opened
    shape: tuple[int, ...] | None = None
    path: str = ""


class _PutHandler:
    """p4p handler for one settable PV."""

    def __init__(self, plugin: HdfFilePlugin, param: _Param) -> None:
        self._plugin = plugin
        self._param = param

    def put(self, pv: SharedPV, op: Any) -> None:
        try:
            value = _unwrap_put(op, self._param.kind, self._param.choices)
        except Exception as exc:  # noqa: BLE001 - reported to the client
            op.done(error=f"{self._param.suffix}: {exc}")
            return
        self._plugin.on_put(self._param, value, op)


class HdfFilePlugin:
    """One image variable's file writer and its areaDetector-shaped PVs.

    Parameters
    ----------
    device, variable, experiment :
        The camera and the image variable this plugin writes.
    retain, release :
        The worker's per-variable subscription refcount (thread-safe): the
        plugin holds the GEECS subscription for the length of a session.
    scalar_variables :
        The device's subscribed scalars, written per frame as ``DOUBLE``
        attributes after the two stamps (``CameraSpec.scalar_variables``).
    """

    def __init__(
        self,
        *,
        device: str,
        variable: str,
        experiment: str,
        retain: Callable[[str], None],
        release: Callable[[str], None],
        scalar_variables: Sequence[str] = (),
    ) -> None:
        self.device = device
        self.variable = variable
        self.experiment = experiment
        self.prefix = hdf_plugin_prefix(experiment, device, variable)
        self.scalar_variables = tuple(scalar_variables)
        self.attributes = attribute_names(device, variable, self.scalar_variables)
        if len(set(self.attributes)) != len(self.attributes):
            # Two scalars normalizing to one name would write one dataset
            # twice and describe one stream key twice — refuse at build.
            raise ValueError(
                f"{device} {variable}: attribute names collide after "
                f"normalization: {self.attributes}"
            )
        self._retain = retain
        self._release = release
        self._lock = threading.Lock()
        self._values: dict[str, Any] = {}
        self._pvs: dict[str, SharedPV] = {}
        self._params: dict[str, _Param] = {}
        for param in PV_TABLE:
            initial = param.initial
            if param.suffix == "NDArrayPort":
                initial = variable
            elif param.suffix == "NDAttributesFile":
                initial = attributes_xml(device, variable, self.scalar_variables)
            self._params[param.suffix] = param
            self._values[param.suffix] = initial
            wrapped = _wrap(param.kind, initial, param.choices)
            settable = not param.suffix.endswith("_RBV")
            self._pvs[param.suffix] = SharedPV(
                handler=_PutHandler(self, param) if settable else None,
                nt=_nt(param.kind),
                initial=wrapped,
            )
            if param.rbv:
                self._pvs[param.suffix + "_RBV"] = SharedPV(
                    nt=_nt(param.kind), initial=wrapped
                )
        self._queue: queue.Queue[tuple[Any, ...]] = queue.Queue(maxsize=QUEUE_DEPTH)
        self._session: _Session | None = None
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name=f"hdf[{device}:{variable}]", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------ PV surface
    def provider_entries(self) -> list[tuple[str, str, SharedPV]]:
        """``[(pv_name, source_label, SharedPV), ...]`` for the server's provider map."""
        return [
            (self.prefix + suffix, f"{self.variable}{PLUGIN_SUFFIX}{suffix}", pv)
            for suffix, pv in self._pvs.items()
        ]

    def value(self, suffix: str) -> Any:
        """Current value of a parameter (tests and the writer thread)."""
        with self._lock:
            return self._values[suffix]

    @property
    def capturing(self) -> bool:
        """Whether a session is open (writer-thread state, read for diagnostics)."""
        return self._session is not None

    def _post(self, suffix: str, value: Any) -> None:
        param = self._params.get(suffix) or self._params[suffix.removesuffix("_RBV")]
        with self._lock:
            self._values[suffix] = value
        self._pvs[suffix].post(_wrap(param.kind, value, param.choices))

    def _store(self, param: _Param, value: Any) -> None:
        self._post(param.suffix, value)
        if param.rbv:
            self._post(param.suffix + "_RBV", value)

    def on_put(self, param: _Param, value: Any, op: Any) -> None:
        """Route a put (p4p worker thread): store-and-echo, or a writer command."""
        if param.suffix in _IGNORED_PUTS:
            op.done()
            return
        if param.suffix in _COMMANDS:
            self._post(param.suffix, value)
            try:
                self._queue.put_nowait(("command", param.suffix, value, op))
            except queue.Full:
                op.done(error=f"{param.suffix}: the writer is not keeping up")
            return
        self._store(param, value)
        if param.suffix == "FilePath":
            # A new run starts with its path: the status reflects this run.
            self._post("FilePathExists_RBV", bool(value) and os.path.isdir(value))
            self._post("WriteStatus", "Write OK")
            self._post("WriteMessage", "")
        op.done()

    # -------------------------------------------------------------- intake
    def offer(
        self,
        blob: str,
        stamp: float,
        recv_time: float,
        scalars: Mapping[str, object] | None = None,
    ) -> None:
        """Hand a raw push frame to the writer (event loop; never blocks).

        *scalars* is the push's ``{variable: value}`` for the plugin's
        ``scalar_variables`` — the same TCP message the frame came in, so
        the attribute row is positionally exact; a missing key is ``NaN``.
        """
        try:
            self._queue.put_nowait(("frame", blob, stamp, recv_time, scalars))
        except queue.Full:
            session = self._session
            if session is not None:
                session.counters.queue_drops += 1
                self._error("frame queue full: the writer is wedged")

    def stop(self) -> None:
        """Close any open session and stop the writer thread (gateway shutdown)."""
        self._stopping = True
        self._queue.put(("stop",), timeout=5)
        self._thread.join(timeout=10)

    # -------------------------------------------------------- writer thread
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item[0] == "stop":
                if self._session is not None:
                    self._close_session()
                return
            try:
                self._dispatch(item)
            except Exception as exc:  # noqa: BLE001 - the writer must survive
                logger.exception("%s %s: writer error", self.device, self.variable)
                self._error(str(exc))

    def _dispatch(self, item: tuple[Any, ...]) -> None:
        if item[0] == "frame":
            self._on_frame(*item[1:])
            return
        _, suffix, value, op = item
        if suffix == "Capture":
            if value:
                self._capture_on(op)
            else:
                self._capture_off(op)
        elif suffix == "Rewind":
            self._rewind(int(value), op)
        elif suffix == "FlushNow":
            if self._session is not None and self._session.file is not None:
                self._session.file.flush()
            self._post("FlushNow", False)
            op.done()

    def _error(self, message: str) -> None:
        logger.warning("%s %s: %s", self.device, self.variable, message)
        self._post("WriteStatus", "Write Error")
        self._post("WriteMessage", message[:255])

    def _capture_on(self, op: Any) -> None:
        if self._session is not None:
            op.done()  # idempotent: the stock logic may repeat the put
            return
        directory = self.value("FilePath")
        problems = []
        if self.value("FileWriteMode") != "Stream":
            problems.append(
                f"FileWriteMode {self.value('FileWriteMode')!r}: only Stream"
            )
        if self.value("Compression") not in ("None", "zlib"):
            problems.append(f"Compression {self.value('Compression')!r}: None or zlib")
        if self.value("EnableCallbacks") != "Enable":
            problems.append("EnableCallbacks is Disable")
        if not directory or not os.path.isdir(directory):
            problems.append(
                f"FilePath {directory!r} does not exist (the plugin never creates it)"
            )
        if not self.value("FileName"):
            problems.append("FileName is empty")
        if self.value("FileTemplate") != "%s%s.h5":
            problems.append(
                f"FileTemplate {self.value('FileTemplate')!r}: only %s%s.h5"
            )
        if problems:
            message = "; ".join(problems)
            self._error(message)
            self._post("Capture", False)
            op.done(error=message)
            return
        self._post("WriteStatus", "Write OK")
        self._post("WriteMessage", "")
        session = _Session(
            directory=directory,
            filename=self.value("FileName"),
            stale_before=time.time(),
            frames_per_chunk=max(1, int(self.value("NumFramesChunks"))),
            compression=self.value("Compression"),
        )
        self._retain(self.variable)
        # Arm: the geometry the worker describes the stream with comes from
        # the first decoded frame — LabVIEW's idle re-push is enough.
        deadline = time.monotonic() + ARM_TIMEOUT_S
        while True:
            try:
                item = self._queue.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                self._release(self.variable)
                message = f"no frame from {self.device} {self.variable} within {ARM_TIMEOUT_S:.0f} s"
                self._error(message)
                self._post("Capture", False)
                op.done(error=message)
                return
            if item[0] == "frame":
                try:
                    frame = decode_imaq_image_string(item[1])
                except Exception as exc:  # noqa: BLE001 - reported, keep arming
                    logger.warning("%s: arming frame undecodable: %s", self.device, exc)
                    continue
                self._post_geometry(frame)
                self._session = session
                try:
                    self._on_frame(*item[1:])
                    failed = self.value("WriteStatus") == "Write Error"
                except Exception as exc:  # noqa: BLE001 - the put must complete
                    self._error(f"arming frame: {exc}")
                    failed = True
                if failed:
                    # A fresh arming frame that could not be written (the
                    # stack could not be opened): this run cannot record,
                    # so the put fails with the reason and nothing leaks.
                    self._session = None
                    self._release(self.variable)
                    self._post("Capture", False)
                    self._post("Capture_RBV", False)
                    op.done(error=self.value("WriteMessage"))
                    return
                break
            if item[0] == "stop":
                self._release(self.variable)
                self._queue.put(item)
                op.done(error="gateway stopping")
                return
            if item[0] == "command" and item[1] == "Capture":
                if item[2]:  # a repeated Capture=1 while arming: one session
                    item[3].done()
                    continue
                self._release(self.variable)
                self._post("Capture_RBV", False)
                item[3].done()
                op.done(error="capture cancelled")
                return
            self._dispatch(item)
        self._post("Capture_RBV", True)
        op.done()
        logger.info("%s %s: capturing → %s", self.device, self.variable, directory)

    def _post_geometry(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        for suffix, value in (
            ("ArraySizeX_RBV", width),
            ("ArraySizeY_RBV", height),
            ("ArraySizeZ_RBV", 0),
            ("ArraySize0_RBV", width),
            ("ArraySize1_RBV", height),
            ("ArraySize2_RBV", 0),
            ("ArraySize0", width),
            ("ArraySize1", height),
        ):
            self._post(suffix, int(value))
        ad_type = _NUMPY_TO_AD.get(frame.dtype.name)
        if ad_type is not None:
            self._post("DataType_RBV", ad_type)

    def _on_frame(
        self,
        blob: str,
        stamp: float,
        recv_time: float,
        scalars: Mapping[str, object] | None = None,
    ) -> None:
        session = self._session
        if session is None:
            return  # a PVA client holds the subscription; nothing to write
        counters = session.counters
        counters.frames_received += 1
        self._post("UniqueId_RBV", counters.frames_received)
        if stamp in session.seen:
            counters.duplicates_dropped += 1
            return
        if stamp < session.stale_before - STALE_MARGIN_S:
            counters.stale_skipped += 1
            return
        if self.value("EnableCallbacks") != "Enable":
            counters.callbacks_disabled += 1
            return
        try:
            frame = decode_imaq_image_string(blob)
        except Exception as exc:  # noqa: BLE001 - counted, never fatal
            counters.decode_errors += 1
            self._error(f"undecodable frame ({len(blob)} bytes): {exc}")
            return
        if frame.ndim != 2:
            counters.shape_errors += 1
            self._error(f"frame of shape {frame.shape}: only 2-D (Mono) frames")
            return
        if session.file is None:
            try:
                self._open_file(session, frame)
            except Exception as exc:  # noqa: BLE001 - counted; the count never advances
                counters.open_failures += 1
                self._error(f"could not open the stack in {session.directory}: {exc}")
                return
        elif frame.shape != session.shape:
            counters.shape_errors += 1
            self._error(f"frame shape {frame.shape} != stack shape {session.shape}")
            return
        session.seen.add(stamp)
        try:
            self._append(session, frame, stamp, recv_time, scalars or {})
        except Exception as exc:  # noqa: BLE001 - counted; the stack tail stays valid
            counters.append_failures += 1
            self._error(f"append failed: {exc}")
            return
        counters.frames_written += 1
        session.count += 1
        self._post("NumCaptured_RBV", session.count)
        self._post("ArrayCounter_RBV", session.count)

    def _open_file(self, session: _Session, frame: np.ndarray) -> None:
        import h5py

        session.path = os.path.join(session.directory, f"{session.filename}.h5")
        h5 = h5py.File(session.path, "w", libver="latest", locking=False)
        h5.attrs["device"] = self.device
        h5.attrs["variable"] = self.variable
        h5.attrs["experiment"] = self.experiment
        h5.attrs["source_pv"] = self.prefix
        h5.attrs["writer"] = f"geecs-pva-gateway {__version__}"
        h5.attrs["created"] = time.time()
        filters: dict[str, Any] = {}
        if session.compression == "zlib":
            filters = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
        h5.create_dataset(
            FRAMES_DATASET,
            shape=(0, *frame.shape),
            maxshape=(None, *frame.shape),
            chunks=(session.frames_per_chunk, *frame.shape),
            dtype=frame.dtype,
            **filters,
        )
        for name in self.attributes:
            h5.create_dataset(
                f"{ATTRIBUTES_GROUP}/{name}",
                shape=(0,),
                maxshape=(None,),
                chunks=(ATTRIBUTE_CHUNK,),
                dtype="f8",
            )
        session.file = h5
        session.shape = frame.shape
        self._post("FullFileName_RBV", session.path)
        logger.info("%s %s: opened %s", self.device, self.variable, session.path)

    def _append(
        self,
        session: _Session,
        frame: np.ndarray,
        stamp: float,
        recv: float,
        scalars: Mapping[str, object],
    ) -> None:
        n = session.count
        h5 = session.file
        frames = h5[FRAMES_DATASET]
        frames.resize(n + 1, axis=0)
        frames[n] = frame
        values = (
            stamp,
            recv,
            *(scalar_value(scalars.get(var)) for var in self.scalar_variables),
        )
        for name, value in zip(self.attributes, values, strict=True):
            ds = h5[f"{ATTRIBUTES_GROUP}/{name}"]
            ds.resize(n + 1, axis=0)
            ds[n] = value
        h5.flush()

    def _rewind(self, n: int, op: Any) -> None:
        session = self._session
        if session is None:
            op.done(error="Rewind outside a capture session")
            return
        if n < 0 or n > session.count:
            op.done(error=f"Rewind {n} outside 0..{session.count}")
            return
        if session.file is not None and n < session.count:
            session.file[FRAMES_DATASET].resize(n, axis=0)
            for name in self.attributes:
                session.file[f"{ATTRIBUTES_GROUP}/{name}"].resize(n, axis=0)
            session.file.flush()
        session.counters.rewound += session.count - n
        session.count = n
        session.stale_before = time.time()
        self._post("NumCaptured_RBV", n)
        self._post("ArrayCounter_RBV", n)
        op.done()
        logger.info("%s %s: rewound to %d", self.device, self.variable, n)

    def _capture_off(self, op: Any) -> None:
        if self._session is not None:
            self._close_session()
        self._post("Capture_RBV", False)
        op.done()

    def _close_session(self) -> None:
        session = self._session
        assert session is not None
        self._session = None
        try:
            if session.file is not None:
                for key, value in session.counters.as_dict().items():
                    session.file.attrs[key] = value
                session.file.attrs["finalized"] = True
                session.file.close()
        finally:
            self._release(self.variable)
        logger.info(
            "%s %s: session closed, %d frame(s) written (%s)",
            self.device,
            self.variable,
            session.count,
            session.counters.as_dict(),
        )

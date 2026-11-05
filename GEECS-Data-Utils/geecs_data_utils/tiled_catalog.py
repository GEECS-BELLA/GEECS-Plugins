"""Scan-shaped navigation of a Tiled catalog: the ``ScanCatalog`` seam.

The Tiled analogue of :class:`~geecs_data_utils.scan_paths.ScanPaths` /
``ScanData``: day → scan → data, over the Bluesky runs a GEECS scan
records to the lab Tiled server.  GUI consumers (the scan browser) and
batch consumers (future ScanAnalysis Tiled readers) depend on the
:class:`ScanCatalog` protocol — never on ``tiled`` directly.

Two implementations live here:

- :class:`StubCatalog` — the offline default: no runs, a "not connected"
  probe, returns instantly.
- :class:`TiledScanCatalog` — the real client.  ``tiled`` is imported
  lazily inside the methods (module import-safe without the ``tiled``
  extra, matching :mod:`geecs_data_utils.tiled_export`); the day listing
  is one metadata-only search on the ``start.time`` range (+ experiment
  key), and a run's event table is the primary stream's **scalar table
  only** (:func:`read_primary_scalars` — never ``run["primary"].read()``,
  which downloads every array part; see ``GeecsBluesky/TILED_SETUP.md``).

Connection details are constructor arguments (pure);
:meth:`TiledScanCatalog.from_config` reads ``[tiled]`` from the shared
config.ini directly — never import ``geecs_bluesky`` here (it depends on
this package).

Every catalog method may block on the network — interactive callers must
dispatch them off the GUI thread.
"""

from __future__ import annotations

import configparser
import functools
import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from geecs_data_utils import tiled_schema

logger = logging.getLogger(__name__)

#: How long a running run's (or a failed) shot-table build is kept, seconds.
_SHOTS_TTL_S = 30.0
#: Runs whose shot table is kept.
_SHOTS_CACHE_RUNS = 64

#: Bounded budget for the connection probe (seconds).
_PROBE_TIMEOUT_S = 2.5

#: The shared config file's location (same source ``tiled_export`` reads).
_CONFIG_PATH = Path("~/.config/geecs_python_api/config.ini")


def read_tiled_config(
    config_path: Optional[Path] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Read Tiled ``uri``/``api_key`` from the shared GEECS config file.

    Parameters
    ----------
    config_path : Path, optional
        The config file; defaults to
        ``~/.config/geecs_python_api/config.ini`` (tests pass a tmp path).

    Returns
    -------
    tuple of (str or None, str or None)
        ``(uri, api_key)``, either ``None`` when the file, the ``[tiled]``
        section, or the option is absent.
    """
    path = (config_path if config_path is not None else _CONFIG_PATH).expanduser()
    if not path.exists():
        return None, None
    cfg = configparser.ConfigParser()
    cfg.read(path)
    if "tiled" not in cfg:
        return None, None
    return cfg["tiled"].get("uri") or None, cfg["tiled"].get("api_key") or None


@dataclass(frozen=True)
class RunSummary:
    """One run's listing row — built from run metadata only (no event data).

    Attributes
    ----------
    uid : str
        The Bluesky run uid (the real key; ``scan_number`` is day-scoped).
    scan_number : int or None
        Day-scoped GEECS scan number (``None`` when never claimed).
    start_time : float
        Start-document ``time`` (epoch seconds).
    mode : str
        Scan-shape chip (``NOSCAN`` / ``1D`` / ``GRID`` / ``OPT``).
    shots : int or None
        Planned shot total from the loop dimensions.
    exit_status : str or None
        Stop-document ``exit_status`` (``success`` / ``abort`` / ``fail``),
        ``None`` when the run has no stop document.
    experiment : str
        GEECS experiment name from the start doc.
    description : str
        Free-text description (filterable).
    save_sets : tuple of str
        The named save sets recorded in the start doc.
    """

    uid: str
    scan_number: Optional[int]
    start_time: float
    mode: str
    shots: Optional[int]
    exit_status: Optional[str]
    experiment: str = ""
    description: str = ""
    save_sets: tuple[str, ...] = ()

    def filter_text(self) -> str:
        """Return the lowercase haystack a metadata filter box searches.

        Returns
        -------
        str
            Scan number, mode, exit status, description, and save sets,
            joined and lowercased.
        """
        number = f"scan {self.scan_number:03d}" if self.scan_number else ""
        return " ".join(
            [
                number,
                self.mode,
                self.exit_status or "",
                self.description,
                *self.save_sets,
            ]
        ).lower()


@dataclass(frozen=True)
class RunDetail:
    """One loaded run: metadata plus its primary event table.

    Attributes
    ----------
    summary : RunSummary
        The listing row this detail belongs to.
    start_doc : dict
        The full start document (identity display, ``tiled_schema``
        lookups).
    stop_doc : dict
        The full stop document (may be empty).
    data : object
        The run's rows as a pandas ``DataFrame`` (one per shot,
        ``Shotnumber`` = position + 1), or ``None`` when it has none
        (typed loosely so this module never imports pandas at import
        time).  An old strict run's ``primary`` as recorded; an old gated
        run's ``shots`` and a shot-table run's ``primary`` with every
        device's scalars joined (:func:`geecs_data_utils.shot_table.shot_table`).
    row_stream : str
        The stream the rows came from (``""``: not a Tiled run's).
    shot_source : callable, optional
        Builds the run's :class:`RunShots` (cached); see :meth:`shots`.
    scalar_source : callable, optional
        Joins every device's scalars onto the rows (cached); see :meth:`scalars`.
    """

    summary: RunSummary
    start_doc: dict = field(default_factory=dict)
    stop_doc: dict = field(default_factory=dict)
    data: Any = None  # pandas.DataFrame — loose so pandas stays lazy
    row_stream: str = ""
    shot_source: Optional[Callable[[], Optional["RunShots"]]] = field(
        default=None, repr=False, compare=False
    )
    scalar_source: Optional[Callable[[], Any]] = field(
        default=None, repr=False, compare=False
    )

    @property
    def rows_are_frames(self) -> bool:
        """Row *k* is frame *k* of every device in it: an old strict run (or no Tiled run).

        Otherwise (gated, shot-table) a device's frame for a shot comes only
        from :meth:`shots`, never from its ordinal.
        """
        start = self.start_doc
        return (
            self.row_stream in ("", "primary")
            and start.get("layout") is None
            and start.get("acquisition") != "gated"
        )

    def shots(self) -> Optional["RunShots"]:
        """The run's shot table, built on first use; ``None`` when it has none.

        Never raises: a build that fails is logged and reads ``None``.
        """
        return None if self.shot_source is None else self.shot_source()

    def scalars(self) -> Any:
        """:attr:`data` with every device's scalars joined (gated, shot-table), read on first use."""
        return self.data if self.scalar_source is None else self.scalar_source()


@dataclass(frozen=True)
class RunShots:
    """A run's shot table and the device records it assigned.

    Attributes
    ----------
    table : geecs_data_utils.shot_table.ShotTable
        The join.
    records : dict
        :func:`tiled_schema.normalize_token` of the device → its
        :class:`~geecs_data_utils.shot_table.DeviceRecords`: the devices
        recorded outside the rows (an old strict run's in-row devices are
        not here; their row is their frame), stamps only.
    rows : DataFrame
        The rows the table was built from.
    """

    table: Any
    records: dict
    rows: Any = field(default=None, repr=False, compare=False)

    def covers(self, device: str) -> bool:
        """Whether *device* (object or folder name) is one of the joined devices."""
        return tiled_schema.normalize_token(device) in self.records

    def stamp_for(self, device: str, shot: int) -> Optional[float]:
        """The stamp (LabVIEW epoch) of *device*'s record for 1-based *shot*, or ``None`` (a gap)."""
        from geecs_data_utils.shot_table import frame_for

        source = self.records.get(tiled_schema.normalize_token(device))
        hit = None if source is None else frame_for(self.table, source.device, shot)
        if hit is None:
            return None
        stamp = float(source.records.stamps[hit[1]])
        return stamp if math.isfinite(stamp) and stamp > 0 else None


@dataclass(frozen=True)
class CatalogStatus:
    """A connection probe's outcome.

    Attributes
    ----------
    ok : bool
        Whether the catalog answered.
    label : str
        Human text (address, or the failure).
    """

    ok: bool
    label: str


@runtime_checkable
class ScanCatalog(Protocol):
    """Anything that can list a day's runs and load one of them.

    Every method may block (network) — interactive callers dispatch off
    the GUI thread.
    """

    def probe(self) -> CatalogStatus:
        """Return the connection state; must never raise."""
        ...

    def list_runs(self, experiment: str, day: date) -> list[RunSummary]:
        """Return *day*'s runs for *experiment*, newest first (metadata only)."""
        ...

    def load_run(self, uid: str) -> RunDetail:
        """Load one run's documents and primary event table.

        Raises ``KeyError`` for an unknown *uid* — the not-found signal
        front-ends map to 404; any other exception means the catalog
        itself is unavailable (front-ends report it as an outage, not
        as run-not-found). Implementations must preserve this split.
        """
        ...


class StubCatalog:
    """The offline default: no runs, an honest "not connected" probe."""

    def probe(self) -> CatalogStatus:
        """Return the offline probe state.

        Returns
        -------
        CatalogStatus
            Never-connected: ``ok=False``.
        """
        return CatalogStatus(ok=False, label="tiled: not connected")

    def list_runs(self, experiment: str, day: date) -> list[RunSummary]:
        """Return no runs.

        Parameters
        ----------
        experiment : str
            Ignored.
        day : datetime.date
            Ignored.

        Returns
        -------
        list of RunSummary
            Always empty.
        """
        return []

    def load_run(self, uid: str) -> RunDetail:
        """Refuse — the stub holds no runs.

        Parameters
        ----------
        uid : str
            The requested run uid.

        Raises
        ------
        KeyError
            Always.
        """
        raise KeyError(f"StubCatalog has no run {uid!r}")


def summary_from_metadata(
    uid: str, start_doc: dict, stop_doc: Optional[dict]
) -> RunSummary:
    """Build a listing row from run documents (shared by real + fake catalogs).

    Parameters
    ----------
    uid : str
        The run uid.
    start_doc : dict
        The start document.
    stop_doc : dict, optional
        The stop document (``None``/empty for an unfinished run).

    Returns
    -------
    RunSummary
        The metadata-only row.
    """
    stop_doc = stop_doc or {}
    save_sets = start_doc.get("save_sets") or []
    if isinstance(save_sets, str):
        save_sets = [save_sets]
    scan_number = start_doc.get("scan_number")
    return RunSummary(
        uid=uid,
        scan_number=int(scan_number) if scan_number is not None else None,
        start_time=float(start_doc.get("time") or 0.0),
        mode=tiled_schema.scan_mode(start_doc),
        shots=tiled_schema.total_shots(start_doc),
        exit_status=stop_doc.get("exit_status"),
        experiment=str(start_doc.get("experiment") or ""),
        description=str(
            start_doc.get("additional_description")
            or start_doc.get("description")
            or ""
        ),
        save_sets=tuple(str(s) for s in save_sets),
    )


class TiledScanCatalog:
    """The real catalog: one Tiled client, day queries on ``start.time``.

    Parameters
    ----------
    uri : str, optional
        Tiled root URI.  ``None`` leaves the catalog unconfigured: the
        probe reports it and the query methods raise a clear error.
    api_key : str, optional
        Tiled API key.

    Notes
    -----
    All ``tiled`` imports happen inside the methods, so this module stays
    import-safe without the ``tiled`` extra (the
    :mod:`geecs_data_utils.tiled_export` pattern).  The client is created
    once and cached; a connection failure surfaces from ``list_runs`` /
    ``load_run`` as an exception for the caller to report, while
    :meth:`probe` never raises.
    """

    def __init__(
        self, uri: Optional[str] = None, api_key: Optional[str] = None
    ) -> None:
        self.uri = uri
        self._api_key = api_key
        self._client: Any = None
        self._clock = time.monotonic
        self._shots: OrderedDict[str, tuple[float, bool, Optional[RunShots]]] = (
            OrderedDict()
        )
        self._shots_lock = threading.Lock()
        self._building: dict[str, threading.Lock] = {}

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> "TiledScanCatalog":
        """Build a catalog from the shared GEECS config file.

        Parameters
        ----------
        config_path : Path, optional
            Config file location; defaults to
            ``~/.config/geecs_python_api/config.ini``.

        Returns
        -------
        TiledScanCatalog
            Configured from the ``[tiled]`` section (unconfigured — and
            honest about it in :meth:`probe` — when the section is
            absent).
        """
        uri, api_key = read_tiled_config(config_path)
        return cls(uri=uri, api_key=api_key)

    def _connect(self) -> Any:
        """Return the cached Tiled client, creating it on first use."""
        if self._client is not None:
            return self._client
        if not self.uri:
            raise RuntimeError(
                "No Tiled URI configured "
                "(~/.config/geecs_python_api/config.ini [tiled] uri)"
            )
        from tiled.client import from_uri

        self._client = from_uri(self.uri, api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # ScanCatalog implementation
    # ------------------------------------------------------------------

    def probe(self) -> CatalogStatus:
        """HTTP-check the Tiled root; never raises.

        Returns
        -------
        CatalogStatus
            ``ok=True`` with the host label on a 2xx root response;
            ``ok=False`` with the failure otherwise.
        """
        if not self.uri:
            return CatalogStatus(ok=False, label="tiled: not configured")
        host = self.uri.split("//", 1)[-1].rstrip("/")
        try:
            import requests

            response = requests.get(self.uri, timeout=_PROBE_TIMEOUT_S)
            if 200 <= response.status_code < 300:
                return CatalogStatus(ok=True, label=f"tiled: {host}")
            return CatalogStatus(
                ok=False, label=f"tiled: {host} (HTTP {response.status_code})"
            )
        except Exception as exc:  # noqa: BLE001 — any failure is a down probe
            logger.debug("tiled probe failed: %s", exc)
            return CatalogStatus(ok=False, label=f"tiled: {host} unreachable")

    def list_runs(self, experiment: str, day: date) -> list[RunSummary]:
        """Search the catalog for *day*'s runs — metadata only, newest first.

        Parameters
        ----------
        experiment : str
            GEECS experiment name (matched on ``start.experiment`` when
            non-empty).
        day : datetime.date
            The local calendar day; converted to a ``start.time`` epoch
            range.

        Returns
        -------
        list of RunSummary
            The day's runs sorted newest first.
        """
        from tiled.queries import Key

        client = self._connect()
        day_start = datetime(day.year, day.month, day.day).timestamp()
        day_end = (
            datetime(day.year, day.month, day.day) + timedelta(days=1)
        ).timestamp()
        results = client.search(Key("start.time") >= day_start).search(
            Key("start.time") < day_end
        )
        if experiment:
            results = results.search(Key("start.experiment") == experiment)
        summaries: list[RunSummary] = []
        for uid, run in results.items():
            metadata = run.metadata
            start_doc = dict(metadata.get("start") or {})
            stop_doc = dict(metadata.get("stop") or {})
            summaries.append(summary_from_metadata(str(uid), start_doc, stop_doc))
        summaries.sort(key=lambda s: s.start_time, reverse=True)
        return summaries

    def load_run(self, uid: str) -> RunDetail:
        """Load one run: documents plus its rows as a DataFrame.

        Parameters
        ----------
        uid : str
            The run uid.

        Returns
        -------
        RunDetail
            Start/stop docs and the rows' **scalar** columns
            (:func:`read_primary_scalars` — array parts are never
            downloaded): ``primary``, else a gated run's ``shots``, joined
            through the shot table unless they are an old strict run's
            (:attr:`RunDetail.data`); ``data=None`` when the run has no
            event rows.  The shot table is built lazily for a strict run.

        Raises
        ------
        KeyError
            Unknown *uid* (``client[uid]`` — tiled's Mapping semantics).
            The protocol's not-found contract: keep it a ``KeyError``
            (never wrap it) so front-ends can distinguish run-not-found
            from catalog-unavailable.
        """
        client = self._connect()
        run = client[uid]
        metadata = run.metadata
        start_doc = dict(metadata.get("start") or {})
        stop_doc = dict(metadata.get("stop") or {})
        from geecs_data_utils.tiled_export import read_run_rows

        rows, stream = read_run_rows(run)
        data, source = None, None
        if not len(rows):
            logger.info("run %s has no event rows", uid)
        else:
            data = rows
            source = functools.partial(
                self._shots_for, uid, run, start_doc, rows, stream, bool(stop_doc)
            )
        detail = RunDetail(
            summary=summary_from_metadata(uid, start_doc, stop_doc),
            start_doc=start_doc,
            stop_doc=stop_doc,
            data=data,
            row_stream=stream,
            shot_source=source,
        )
        if source is not None and not detail.rows_are_frames:
            shots = source()  # stamps only: one read per device
            if shots is not None:
                joined = functools.partial(_joined_rows, run, start_doc, shots, stream)
                detail = replace(
                    detail,
                    data=shots.table.rows,
                    scalar_source=functools.cache(joined),
                )
        return detail

    def _shots_for(
        self, uid: str, run: Any, start: dict, rows: Any, stream: str, complete: bool
    ) -> Optional[RunShots]:
        """*uid*'s :class:`RunShots`, one build at a time per run.

        Kept for good once a finished run's build succeeds; a running run's
        (or a failed build, as ``None``) for :data:`_SHOTS_TTL_S`, so a
        running run's rows lag the catalog by up to that long.
        """
        with self._shots_lock:
            lock = self._building.setdefault(uid, threading.Lock())
        with lock:
            with self._shots_lock:
                hit = self._shots.get(uid)
            if hit is not None and (hit[1] or self._clock() - hit[0] < _SHOTS_TTL_S):
                return hit[2]
            shots = _run_shots(run, start, rows, stream)
            with self._shots_lock:
                self._shots[uid] = (
                    self._clock(),
                    complete and shots is not None,
                    shots,
                )
                self._shots.move_to_end(uid)
                while len(self._shots) > _SHOTS_CACHE_RUNS:
                    self._building.pop(self._shots.popitem(last=False)[0], None)
            return shots


def _run_shots(run: Any, start: dict, rows: Any, stream: str) -> Optional[RunShots]:
    """A Tiled run's :class:`RunShots` from its devices' stamps, or ``None`` (logged).

    Its devices are every stream but the rows' (:func:`_stamp_records`), with
    the run's drain offsets.
    """
    from geecs_data_utils.shot_table import shot_table
    from geecs_data_utils.tiled_export import read_drain_offsets

    try:
        devices = _stamp_records(run, start, stream)
        offsets = read_drain_offsets(run)
        table = shot_table(
            start, rows, devices, row_stream=stream, drain_offsets=offsets
        )
    except Exception:  # a run that cannot be joined still loads; its images refuse
        logger.exception("run %s: no shot table", start.get("uid"))
        return None
    return RunShots(
        table=table,
        records={tiled_schema.normalize_token(d.device): d for d in devices},
        rows=rows,
    )


def _stamp_records(run: Any, start: dict, row_stream: str) -> list:
    """One stamps-only ``DeviceRecords`` per device outside the rows, in its own stream.

    The one Tiled read behind a frame lookup.  Counters come only from bare
    stream parts no plan publishes yet, so a portal frame lookup has none
    until W3c (``tiled_export.read_device_records``).
    """
    from geecs_data_utils.tiled_export import read_device_records

    return read_device_records(run, row_stream, stamps_only=True)


def _joined_rows(run: Any, start: dict, shots: RunShots, stream: str) -> Any:
    """*shots*' rows with every device's scalars joined, read on first use."""
    from geecs_data_utils.shot_table import shot_table
    from geecs_data_utils.tiled_export import read_device_records, read_drain_offsets

    try:
        devices = read_device_records(run, stream)
        offsets = read_drain_offsets(run)
        table = shot_table(
            start, shots.rows, devices, row_stream=stream, drain_offsets=offsets
        )
    except Exception:  # the stamps-only rows still serve
        logger.exception("run %s: device scalars not joined", start.get("uid"))
        return shots.table.rows
    return table.rows


def read_primary_scalars(primary: Any) -> Optional[Any]:
    """The primary stream's scalar table(s) only — its array parts are never read.

    A Bluesky run's ``primary`` node is a Tiled composite: one ``internal``
    table (the event rows' scalar columns) plus one array node per external
    data key (camera stacks, per-frame attributes).  ``primary.read()``
    downloads every part and ``to_dataframe()`` then takes the outer
    product of every array's dimensions — one 5×600×600 stack survives,
    two stacks of different shapes multiply into billions of rows (a
    two-camera plugin run took the worker host down, 2026-09-11).  So the
    table parts are read by name through ``primary.base`` and the arrays
    are left where they are.

    Nothing about it is specific to ``primary``: any composite stream node
    works the same way, and :func:`geecs_data_utils.tiled_export.read_run_rows`
    passes a gated run's ``shots`` node through here.

    Parameters
    ----------
    primary :
        The run's ``primary`` node (a Tiled ``CompositeClient``), or any
        other stream node of the run.

    Returns
    -------
    pandas.DataFrame or None
        The scalar columns, one row per event; ``None`` for a stream with
        no table part or no rows (an aborted or legacy run).  Typed loosely
        so pandas stays a lazy import of this module.
    """
    tables = [
        part
        for part, item in primary.get_contents().items()
        if item["attributes"]["structure_family"] == "table"
    ]
    if not tables:
        return None
    frames = []
    for part in tables:
        frame = primary.base[part].read()
        if hasattr(frame, "compute"):
            frame = frame.compute()
        # Positional rows: a partitioned table's index labels repeat per
        # partition, so the tables are aligned by row order, never by label.
        frames.append(frame.reset_index(drop=True))
    if len(frames) == 1:
        data = frames[0]
    else:
        import pandas as pd

        data = pd.concat(frames, axis=1)
    return data if len(data) else None


def resolve_scan_folder(detail: RunDetail, day: date) -> Optional[Path]:
    """Resolve a run's scan folder on disk — strictly read-only.

    Shared by every front-end over the catalog (the data portal's resource
    endpoints; the Qt console's scan browser before its deletion).
    Prefers the run's own ``scan_folder`` start-doc path; falls back to
    building the daily ``scans/ScanNNN`` path for the given date via
    :func:`geecs_data_utils.scan_paths.daily_scan_folder` (pure path
    construction).  Only an *existing* directory is returned — nothing
    on the scans path is ever created (repo scan-folder invariant;
    pinned by a tree-untouched test).

    Parameters
    ----------
    detail : RunDetail
        The selected run (start doc + summary).
    day : datetime.date
        The date the run was listed under.

    Returns
    -------
    Path or None
        The existing scan folder, or ``None`` when it is absent or
        unresolvable.
    """
    metadata_folder = detail.start_doc.get("scan_folder")
    if metadata_folder:
        candidate = Path(str(metadata_folder))
        if candidate.is_dir():
            return candidate
        # The recorded path is host-specific (the scanner host's mount,
        # e.g. /mnt/... recorded by the Linux worker, read from a client
        # whose share mounts at /Volumes/... or Z:) — fall through and
        # re-base via this host's own daily path instead of giving up.
    scan_number = detail.summary.scan_number
    if scan_number is None:
        return None
    # An experiment-less run must not re-base: daily_scan_folder would
    # substitute the host config's default experiment, and that tree's
    # same-numbered ScanNNN is a different scan's data.
    if not detail.summary.experiment:
        return None
    from geecs_data_utils.scan_paths import daily_scan_folder

    daily = daily_scan_folder(detail.summary.experiment, day=day)
    if daily is None:
        return None
    candidate = daily / f"Scan{scan_number:03d}"
    return candidate if candidate.is_dir() else None


def _fmt_datetime(epoch: float) -> str:
    """Format epoch seconds as local ``YYYY-MM-DD HH:MM:SS`` ("" if invalid)."""
    if not epoch or not math.isfinite(epoch):
        return ""
    try:
        return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return ""


def fmt_time_of_day(epoch: float) -> str:
    """Format epoch seconds as local ``HH:MM`` ("" for 0/invalid).

    The shared run-list time-cell formatter (console scan browser B1,
    portal day view) — the HH:MM sibling of :func:`_fmt_datetime`, moved
    down so the two front-ends render times identically.

    Parameters
    ----------
    epoch : float
        Unix epoch seconds (a run summary's ``start_time``).

    Returns
    -------
    str
        ``HH:MM`` in the host's local time, or ``""``.
    """
    if not epoch or not math.isfinite(epoch):
        return ""
    try:
        return datetime.fromtimestamp(epoch).strftime("%H:%M")
    except (OverflowError, OSError, ValueError):
        return ""


def metadata_rows(detail: RunDetail) -> list[tuple[str, str]]:  # noqa: C901, PLR0912, PLR0915
    """Compose display field/value rows from a loaded run's metadata.

    Pure — reads only the already-loaded :class:`RunDetail` (summary +
    start/stop documents), never the catalog.  Rows whose source key is
    absent or empty are omitted, so legacy or aborted runs render a
    shorter list rather than blank cells.  Shared by every front-end's
    run-detail view (the data portal today).

    Parameters
    ----------
    detail : RunDetail
        The selected run.

    Returns
    -------
    list of (str, str)
        Ordered ``(field, value)`` pairs.
    """
    summary = detail.summary
    start = detail.start_doc
    stop = detail.stop_doc
    rows: list[tuple[str, str]] = []

    if summary.scan_number is not None:
        rows.append(("Scan", f"Scan {summary.scan_number:03d}"))
    rows.append(("uid", summary.uid))
    if summary.experiment:
        rows.append(("Experiment", summary.experiment))
    if summary.description:
        rows.append(("Description", summary.description))
    # Both spellings: the native scanner writes `acquisition` (strict/gated),
    # the retired funnel wrote `acquisition_mode`.
    acquisition = str(start.get("acquisition") or start.get("acquisition_mode") or "")
    rows.append(
        ("Mode", f"{summary.mode} · {acquisition}" if acquisition else summary.mode)
    )
    plan = str(start.get("plan_name") or "")
    if plan:
        rows.append(("Plan", plan))

    axes = start.get("scan_axes") or []
    if axes:
        rows.append(("Scan axes", ", ".join(str(a) for a in axes)))
        shape = start.get("grid_shape") or []
        points = start.get("num_grid_points")
        if shape and points:
            shape_text = " × ".join(str(s) for s in shape)
            rows.append(("Grid", f"{shape_text} = {points} steps"))
    elif tiled_schema.scan_motors(start):
        # Plural `motors` (stock bluesky) or the funnel's singular `motor`;
        # several entries for a correlated or grid trajectory.
        rows.append(("Scan variable", ", ".join(tiled_schema.scan_motors(start))))

    num_points = start.get("num_points")
    shots_per_step = start.get("shots_per_step")
    if num_points and shots_per_step:
        rows.append(
            (
                "Planned shots",
                f"{num_points} steps × {shots_per_step} = {summary.shots}",
            )
        )
    if summary.save_sets:
        rows.append(("Save sets", ", ".join(summary.save_sets)))
    reference = str(start.get("reference_device") or "")
    if reference:
        rows.append(("Reference device", reference))
    folder = str(start.get("scan_folder") or "")
    if folder:
        rows.append(("Scan folder", folder))

    started = _fmt_datetime(float(start.get("time") or 0.0))
    if started:
        rows.append(("Started", started))
    if stop:
        if stop.get("time") and start.get("time"):
            rows.append(("Duration", f"{stop['time'] - start['time']:.0f} s"))
        rows.append(("Exit status", str(stop.get("exit_status") or "unknown")))
        reason = str(stop.get("reason") or "")
        if reason:
            rows.append(("Reason", reason))
        # The rows' stream: ``primary`` (strict, shot-table), else gated's ``shots``.
        events = stop.get("num_events")
        events = events if isinstance(events, dict) else {}
        recorded = events.get("primary", events.get("shots"))
        if recorded is not None:
            rows.append(("Recorded rows", str(recorded)))
    else:
        rows.append(("Exit status", "no stop document"))
    return rows

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
  key), and a run's event table is read with the repo-blessed pattern
  ``run["primary"].read()`` (see ``GeecsBluesky/TILED_SETUP.md``).

Connection details are constructor arguments (pure);
:meth:`TiledScanCatalog.from_config` reads ``[tiled]`` from the shared
config.ini directly — never import ``geecs_bluesky`` here (it depends on
this package).

Every catalog method may block on the network — interactive callers must
dispatch them off the GUI thread.
"""

from __future__ import annotations

import configparser
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from geecs_data_utils import tiled_schema

logger = logging.getLogger(__name__)

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
        The primary event stream as a pandas ``DataFrame``, or ``None``
        when the run has no primary stream (typed loosely so this module
        never imports pandas at import time).
    """

    summary: RunSummary
    start_doc: dict = field(default_factory=dict)
    stop_doc: dict = field(default_factory=dict)
    data: Any = None  # pandas.DataFrame — loose so pandas stays lazy


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
        """Load one run's documents and primary event table."""
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
        """Load one run: documents plus the primary table as a DataFrame.

        Parameters
        ----------
        uid : str
            The run uid.

        Returns
        -------
        RunDetail
            Start/stop docs and the primary event stream (the repo-blessed
            ``run["primary"].read()`` composite-container pattern,
            flattened to a pandas DataFrame; ``data=None`` when the run
            has no primary stream).
        """
        client = self._connect()
        run = client[uid]
        metadata = run.metadata
        start_doc = dict(metadata.get("start") or {})
        stop_doc = dict(metadata.get("stop") or {})
        data = None
        try:
            dataset = run["primary"].read()
            if dataset.sizes:
                data = dataset.to_dataframe().reset_index()
            else:
                # A dimensionless dataset — an aborted or legacy run whose
                # stream holds no event rows — has no index for a frame;
                # ``to_dataframe`` raises. Same contract as "no stream".
                logger.info("run %s primary stream has no event rows", uid)
        except KeyError:
            logger.info("run %s has no primary stream", uid)
        return RunDetail(
            summary=summary_from_metadata(uid, start_doc, stop_doc),
            start_doc=start_doc,
            stop_doc=stop_doc,
            data=data,
        )


def resolve_scan_folder(detail: RunDetail, day: date) -> Optional[Path]:
    """Resolve a run's scan folder on disk — strictly read-only.

    Shared by every front-end over the catalog (the console scan
    browser's Open button, the data portal's resource endpoints).
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


def metadata_rows(detail: RunDetail) -> list[tuple[str, str]]:
    """Compose display field/value rows from a loaded run's metadata.

    Pure — reads only the already-loaded :class:`RunDetail` (summary +
    start/stop documents), never the catalog.  Rows whose source key is
    absent or empty are omitted, so legacy or aborted runs render a
    shorter list rather than blank cells.  Shared by the console scan
    browser (B7 table) and the data portal's run-detail view.

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
    acquisition = str(start.get("acquisition_mode") or "")
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
    elif start.get("motor"):
        rows.append(("Scan variable", str(start["motor"])))

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
        events = stop.get("num_events")
        primary = events.get("primary") if isinstance(events, dict) else None
        if primary is not None:
            rows.append(("Recorded rows", str(primary)))
    else:
        rows.append(("Exit status", "no stop document"))
    return rows

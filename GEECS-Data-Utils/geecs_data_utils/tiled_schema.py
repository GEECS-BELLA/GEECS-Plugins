"""Event-schema column semantics for Tiled-recorded GEECS runs — schema v1.

Everywhere a consumer *interprets* an event-stream column name or a
start/stop-document key (the scan browser, drift analysis, future
ScanAnalysis Tiled readers), the rule lives here, sourced from
``GeecsBluesky/EVENT_SCHEMA.md`` (the canonical data contract) — never
from ad-hoc string guessing spread across consumers.  When the event
schema evolves, this is the single file to touch.

Column families (schema v1):

- Row identity: ``scan_event_index`` / ``bin_number`` / ``shot_index_in_bin``.
- Per synchronous device ``<dev>-<variable>`` data columns plus companion
  columns (``-acq_timestamp``, ``-t0_acq_timestamp``, ``-shot_id``,
  ``-shot_offset``, ``-valid``, ``-nonscalar_save_path``).
- Background telemetry: every column of the ``telemetry_<device>`` ophyd
  device, keyed ``telemetry_<device>-<safe_var>`` (dtype-tolerant — may be
  float *or* string; never assume numeric).
- ``geecs_scalar_headers`` in the start doc maps scalar data keys back to
  their legacy ``Device Variable`` headers (``safe_name()`` mangling is
  irreversible; this map is the only recovery path) — used here for
  display-name prettification.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

#: The ``geecs_event_schema`` version this module targets.
TARGET_SCHEMA_VERSION = 1

#: Row-identity columns set by the plan on every mode, every row.
ID_COLUMNS: tuple[str, ...] = ("scan_event_index", "bin_number", "shot_index_in_bin")

#: The 1-based step index — groups rows into steps for per-step statistics.
BIN_COLUMN = "bin_number"

#: The 1-based global row index — the default plot X axis ("shot #").
SHOT_INDEX_COLUMN = "scan_event_index"

#: Ophyd device-name prefix marking Tier-2 background-telemetry columns.
TELEMETRY_PREFIX = "telemetry_"

#: Companion-column suffixes contributed per synchronous device
#: (schema v1 "Per synchronous device" table).  These are matching
#: machinery / diagnostics, not measurements — hidden from the default
#: Y-column offering and from drift analysis.
COMPANION_SUFFIXES: tuple[str, ...] = (
    "-acq_timestamp",
    "-t0_acq_timestamp",
    "-shot_id",
    "-shot_offset",
    "-valid",
    "-nonscalar_save_path",
)

_ACQ_TIMESTAMP_SUFFIX = "-acq_timestamp"


def is_telemetry_column(column: str) -> bool:
    """Return whether *column* is a Tier-2 background-telemetry column.

    Parameters
    ----------
    column : str
        An event-stream column name.

    Returns
    -------
    bool
        True for ``telemetry_<device>-<safe_var>`` columns.
    """
    return column.startswith(TELEMETRY_PREFIX)


def is_companion_column(column: str) -> bool:
    """Return whether *column* is a schema-v1 companion (machinery) column.

    Parameters
    ----------
    column : str
        An event-stream column name.

    Returns
    -------
    bool
        True for ``-acq_timestamp`` / ``-t0_acq_timestamp`` / ``-shot_id`` /
        ``-shot_offset`` / ``-valid`` / ``-nonscalar_save_path`` columns.
    """
    return column.endswith(COMPANION_SUFFIXES)


def is_id_column(column: str) -> bool:
    """Return whether *column* is one of the row-identity columns.

    Parameters
    ----------
    column : str
        An event-stream column name.

    Returns
    -------
    bool
        True for ``scan_event_index`` / ``bin_number`` / ``shot_index_in_bin``.
    """
    return column in ID_COLUMNS


def is_acq_timestamp_column(column: str) -> bool:
    """Return whether *column* is a device ``acq_timestamp`` column.

    Parameters
    ----------
    column : str
        An event-stream column name.

    Returns
    -------
    bool
        True for ``<dev>-acq_timestamp`` columns (the file-join key).
    """
    return column.endswith(_ACQ_TIMESTAMP_SUFFIX)


def data_columns(columns: Sequence[str]) -> list[str]:
    """Return the measurement columns — data variables, machinery excluded.

    Parameters
    ----------
    columns : sequence of str
        All event-stream column names.

    Returns
    -------
    list of str
        Columns that are neither row identity nor companion machinery —
        device data variables, the scan-device readback, and telemetry.
        Order preserved.
    """
    return [c for c in columns if not is_id_column(c) and not is_companion_column(c)]


def telemetry_columns(columns: Sequence[str]) -> list[str]:
    """Return the Tier-2 telemetry data columns (drift-analysis candidates).

    Parameters
    ----------
    columns : sequence of str
        All event-stream column names.

    Returns
    -------
    list of str
        ``telemetry_<device>-<safe_var>`` data columns, companion columns
        excluded.  Telemetry is dtype-tolerant: callers must still guard
        against string-typed columns.
    """
    return [c for c in data_columns(columns) if is_telemetry_column(c)]


def display_name(
    column: str, scalar_headers: Optional[Mapping[str, str]] = None
) -> str:
    """Prettify a column name for display.

    Parameters
    ----------
    column : str
        The event-stream column name.
    scalar_headers : mapping, optional
        The start-doc ``geecs_scalar_headers`` map (data key → legacy
        ``Device Variable`` header).  ``safe_name()`` mangling is
        irreversible, so this map is the only faithful recovery; when a
        column is absent from it (telemetry, machinery, id columns) a
        readable fallback is derived from the key itself.

    Returns
    -------
    str
        The legacy header when known, else ``device : variable`` split on
        the schema's device/variable dash (telemetry prefix stripped, with
        a ``[t]`` marker so Tier-2 provenance stays visible).
    """
    if scalar_headers:
        header = scalar_headers.get(column)
        if header:
            return str(header)
    name = column
    telemetry = is_telemetry_column(name)
    if telemetry:
        name = name[len(TELEMETRY_PREFIX) :]
    device, dash, variable = name.partition("-")
    if dash:
        pretty = f"{device} : {variable}"
    else:
        pretty = name
    return f"{pretty} [t]" if telemetry else pretty


def reference_acq_timestamp_column(
    columns: Sequence[str], start_doc: Mapping[str, Any]
) -> Optional[str]:
    """Pick the row-time column: the reference device's ``acq_timestamp``.

    Parameters
    ----------
    columns : sequence of str
        All event-stream column names.
    start_doc : mapping
        The run start document.  Free-run mode names the pacemaker under
        ``reference_device``; strict mode has no reference, so the first
        ``-acq_timestamp`` column stands in.

    Returns
    -------
    str or None
        The chosen ``<dev>-acq_timestamp`` column, or ``None`` when the
        run has no synchronous device.
    """
    candidates = [c for c in columns if is_acq_timestamp_column(c)]
    if not candidates:
        return None
    reference = start_doc.get("reference_device")
    if reference:
        wanted = f"{reference}{_ACQ_TIMESTAMP_SUFFIX}"
        if wanted in candidates:
            return wanted
    return candidates[0]


def pinned_columns(columns: Sequence[str], start_doc: Mapping[str, Any]) -> list[str]:
    """Return the always-shown table columns: shot sequence + row time.

    Parameters
    ----------
    columns : sequence of str
        All event-stream column names.
    start_doc : mapping
        The run start document (reference-device lookup).

    Returns
    -------
    list of str
        ``scan_event_index`` (when present) followed by the reference
        ``acq_timestamp`` column (when present).
    """
    pinned: list[str] = []
    if SHOT_INDEX_COLUMN in columns:
        pinned.append(SHOT_INDEX_COLUMN)
    time_column = reference_acq_timestamp_column(columns, start_doc)
    if time_column is not None:
        pinned.append(time_column)
    return pinned


def scan_variable_columns(
    columns: Sequence[str], start_doc: Mapping[str, Any]
) -> list[str]:
    """Return the scan-device readback columns (stepped-scan X candidates).

    Parameters
    ----------
    columns : sequence of str
        All event-stream column names.
    start_doc : mapping
        The run start document; ``motor`` is the scan-device ophyd name
        (``None`` for statistics collection).

    Returns
    -------
    list of str
        Data columns contributed by the scan device (its readback), empty
        for motorless runs.  Multi-axis grids record every axis readback,
        so several columns may return.
    """
    motor = start_doc.get("motor")
    if not motor:
        return []
    motors = [motor] if isinstance(motor, str) else [str(m) for m in motor]
    matches: list[str] = []
    for column in data_columns(columns):
        if column in motors or any(column.startswith(f"{m}-") for m in motors):
            matches.append(column)
    return matches


def scan_mode(start_doc: Mapping[str, Any]) -> str:
    """Classify a run's scan shape (mode chip / grouping).

    Parameters
    ----------
    start_doc : mapping
        The run start document.

    Returns
    -------
    str
        ``"OPT"`` for adaptive-scan plans, ``"NOSCAN"`` for motorless runs
        (statistics collection), ``"GRID"`` for multi-axis runs, else
        ``"1D"``.
    """
    plan_name = str(start_doc.get("plan_name") or "")
    if "adaptive" in plan_name or "optimize" in plan_name:
        return "OPT"
    motor = start_doc.get("motor")
    if not motor:
        return "NOSCAN"
    if not isinstance(motor, str) and isinstance(motor, Sequence) and len(motor) > 1:
        return "GRID"
    if start_doc.get("grid_shape") or (
        isinstance(start_doc.get("scan_axes"), Sequence)
        and len(start_doc.get("scan_axes") or []) > 1
    ):
        return "GRID"
    return "1D"


def total_shots(start_doc: Mapping[str, Any]) -> Optional[int]:
    """Return the planned shot total (``num_points × shots_per_step``).

    Parameters
    ----------
    start_doc : mapping
        The run start document.

    Returns
    -------
    int or None
        The planned total, or ``None`` when the loop dimensions are absent.
    """
    num_points = start_doc.get("num_points")
    shots_per_step = start_doc.get("shots_per_step")
    if num_points is None or shots_per_step is None:
        return None
    try:
        return int(num_points) * int(shots_per_step)
    except (TypeError, ValueError):
        return None


def is_stepped_scan(start_doc: Mapping[str, Any]) -> bool:
    """Return whether the run stepped a scan variable (per-step stats apply).

    Parameters
    ----------
    start_doc : mapping
        The run start document.

    Returns
    -------
    bool
        True when a motor was moved (``motor`` non-null).
    """
    return bool(start_doc.get("motor"))

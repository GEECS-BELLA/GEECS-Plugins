"""Export legacy GEECS scalar files from a Bluesky run recorded in Tiled.

A Bluesky scan persists its per-shot data to a Tiled catalog only.  Downstream
GEECS analysis (ScanAnalysis, optimization, log tooling) still consumes the
legacy on-disk scalar files written by the original scanner:

- ``scans/ScanNNN/ScanDataScanNNN.txt`` — tab-separated scalar summary.
- ``analysis/sNNN.txt`` — a copy of the same table; the *mutable* one analysis
  code appends to.

:func:`write_scalar_files_from_tiled` reads a run back from Tiled by ``uid`` and
writes both files in the legacy format.  The Bluesky event stream names scalar
columns ``<ophyd>-<safe_var>`` (e.g. ``uc_wavemeter-wavelength_nm``), which is an
irreversible mangling of the original GEECS ``Device Variable``; the run's
start-document carries a ``geecs_scalar_headers`` map (event key →
``Device Variable``) recorded at scan time so the original headers can be
recovered.  See ``GeecsBluesky/EVENT_SCHEMA.md``.

This module is a **consumer** of scan folders: it writes into an
already-claimed ``scans/ScanNNN/`` folder but never creates one (the
cross-package "analysis code never creates scan folders" invariant).
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Derived companion columns (event-schema v1) that must not appear in the legacy
# scalar file.  Any event-stream column not named in ``geecs_scalar_headers`` is
# dropped anyway; this set documents the intent.
_COMPANION_SUFFIXES = (
    "-acq_timestamp",
    "-t0_acq_timestamp",
    "-shot_id",
    "-shot_offset",
    "-valid",
    "-nonscalar_save_path",
)


def _read_tiled_config() -> tuple[Optional[str], Optional[str]]:
    """Read Tiled ``uri``/``api_key`` from ``~/.config/geecs_python_api/config.ini``.

    Returns ``(uri, api_key)``, either of which may be ``None`` if absent.  Same
    config source ``BlueskyScanner`` uses to subscribe its ``TiledWriter``.
    """
    config_path = Path.home() / ".config" / "geecs_python_api" / "config.ini"
    if not config_path.exists():
        return None, None
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    if "tiled" not in cfg:
        return None, None
    return cfg["tiled"].get("uri") or None, cfg["tiled"].get("api_key") or None


def build_legacy_scalar_dataframe(
    start_doc: dict[str, Any], primary_df: pd.DataFrame
) -> pd.DataFrame:
    """Build a legacy-format scalar DataFrame from a run's start doc + events.

    Pure transform (no I/O) so it is unit-testable without a live Tiled server.

    Parameters
    ----------
    start_doc:
        The run's start-document metadata.  Must contain ``geecs_scalar_headers``
        (event key → ``"Device Variable"``); ``scan_number`` is used for the
        ``scan`` column when present.
    primary_df:
        The ``primary`` event stream as a DataFrame, with Bluesky ophyd-named
        columns (``<ophyd>-<safe_var>``, companion columns, ``bin_number``, …).

    Returns
    -------
    pandas.DataFrame
        Columns ``Bin #``, ``scan``, ``<Device Variable>…`` (in
        ``geecs_scalar_headers`` order), ``Shotnumber``.  The legacy
        ``Elapsed Time`` column is intentionally omitted.
    """
    headers: dict[str, str] = dict(start_doc.get("geecs_scalar_headers") or {})
    n_rows = len(primary_df)
    out = pd.DataFrame(index=range(n_rows))

    # Row-identity columns.
    if "bin_number" in primary_df.columns:
        out["Bin #"] = primary_df["bin_number"].to_numpy()
    else:
        out["Bin #"] = 1
    out["scan"] = start_doc.get("scan_number", 0)

    # Device data columns: rename via the header map, preserving its order.
    # Only keys actually present in the event stream are emitted; everything not
    # in the map (companion columns, row-identity columns) is dropped.
    for event_key, legacy_header in headers.items():
        if event_key in primary_df.columns:
            out[legacy_header] = primary_df[event_key].to_numpy()
        else:
            logger.debug(
                "geecs_scalar_headers key %r absent from primary stream", event_key
            )

    out["Shotnumber"] = range(1, n_rows + 1)
    return out


def _resolve_output_paths(start_doc: dict[str, Any]) -> Optional[tuple[Path, Path]]:
    """Resolve ``(ScanDataScanNNN.txt, sNNN.txt)`` paths from the start doc.

    Returns ``None`` (and logs) when the scan folder is missing from the start
    doc or does not exist on disk — this module never creates scan folders.
    """
    scan_folder = start_doc.get("scan_folder")
    if not scan_folder:
        logger.warning("Run has no scan_folder in start doc; cannot write s-file")
        return None
    folder = Path(scan_folder)
    if not folder.is_dir():
        logger.warning(
            "Scan folder %s does not exist; refusing to create it (invariant)",
            folder,
        )
        return None

    scan_number = start_doc.get("scan_number")
    if scan_number is None:
        # Fall back to the trailing digits of the folder name (ScanNNN).
        scan_number = int("".join(ch for ch in folder.name if ch.isdigit()) or "0")

    scan_txt = folder / f"ScanData{folder.name}.txt"
    # analysis/ is the sibling of scans/ under the day folder: replace the
    # "scans" path component with "analysis" and drop the ScanNNN leaf.
    parts = list(folder.parts)
    parts[-2] = "analysis"
    analysis_dir = Path(*parts[:-1])
    analysis_dir.mkdir(exist_ok=True)  # day folder exists; only analysis/ is new
    sfile_txt = analysis_dir / f"s{int(scan_number)}.txt"
    return scan_txt, sfile_txt


def _fetch_run(uid: str, tiled_uri: str, tiled_api_key: Optional[str]):
    """Return ``(start_doc, primary_df)`` for *uid* from the Tiled catalog."""
    try:
        from tiled.client import from_uri
    except ImportError as exc:  # pragma: no cover - exercised only without tiled
        raise RuntimeError(
            "tiled is not installed; install the 'tiled' extra "
            "(pip install 'geecs-data-utils[tiled]') to export scalar files"
        ) from exc

    client = from_uri(tiled_uri, api_key=tiled_api_key)
    run = client[uid]
    start_doc = dict(run.metadata.get("start") or {})
    primary_df = run["primary"].read().to_dataframe().reset_index()
    return start_doc, primary_df


def write_scalar_files_from_tiled(
    uid: str,
    *,
    tiled_uri: Optional[str] = None,
    tiled_api_key: Optional[str] = None,
) -> Optional[tuple[Path, Path]]:
    """Read a Bluesky run from Tiled and write the legacy scalar files.

    Parameters
    ----------
    uid:
        The Bluesky run uid (the start-document ``uid``).
    tiled_uri, tiled_api_key:
        Tiled connection details.  When omitted, read from the ``[tiled]``
        section of ``~/.config/geecs_python_api/config.ini``.

    Returns
    -------
    tuple[Path, Path] or None
        ``(scan_data_txt_path, sfile_txt_path)`` on success, or ``None`` when
        the scan folder is absent (see :func:`_resolve_output_paths`).

    Raises
    ------
    RuntimeError
        If ``tiled`` is not installed or no Tiled URI can be resolved.
    """
    if tiled_uri is None:
        tiled_uri, tiled_api_key = _read_tiled_config()
    if not tiled_uri:
        raise RuntimeError(
            "No Tiled URI given and none found in "
            "~/.config/geecs_python_api/config.ini [tiled]"
        )

    start_doc, primary_df = _fetch_run(uid, tiled_uri, tiled_api_key)
    paths = _resolve_output_paths(start_doc)
    if paths is None:
        return None
    scan_txt, sfile_txt = paths

    df = build_legacy_scalar_dataframe(start_doc, primary_df)
    if df.empty:
        logger.warning("Run %s produced no scalar rows; nothing written", uid)
        return None

    df.to_csv(scan_txt, sep="\t", index=False)
    df.to_csv(sfile_txt, sep="\t", index=False)
    logger.info("Wrote legacy scalar files: %s and %s", scan_txt, sfile_txt)
    return scan_txt, sfile_txt

"""
Provides utilities to construct partitioned parquet 'scans scans_database'.

Structured metadata records (`ScanEntry`) are created for each scan.
Supports streaming scan metadata to Parquet format for persistent
storage and querying.

Classes
-------
ScanDatabaseBuilder
    Static methods for building and streaming scan metadata from directory structure.

See Also
--------
ScanEntry : Represents an individual scan's metadata and file references.
ScanDatabase : In-memory collection of ScanEntry records.
"""

import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Literal, Union
from datetime import date, datetime, timedelta
import re
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from pydantic import BaseModel, ConfigDict

from geecs_data_utils import ScanPaths
from geecs_data_utils.scan_data import ScanData
from geecs_data_utils.type_defs import ScanTag
from geecs_data_utils.scans_database.entries import (
    ScanEntry,
    ScanMetadata,
    collect_dtypes,
    get_pyarrow_schema,
)
from geecs_data_utils.type_defs import parse_ecs_dump

logger = logging.getLogger(__name__)


class _UpdateLogEntry(BaseModel):
    """Internal model for update log entries."""

    model_config = ConfigDict(extra="forbid")
    experiment: str
    update_date: date
    date_range_added: Tuple[date, date]


class ScanDatabaseBuilder:
    """Builder class for constructing a full ScanDatabase from GEECS scan folders."""

    _LOG_NAME = "_update_log.json"

    @staticmethod
    def build_scan_entry(scan_data: ScanData) -> ScanEntry:
        """Construct a ScanEntry from a given ScanData object."""
        folder = scan_data.paths.get_folder()
        tag = scan_data.paths.get_tag()

        files_and_folders = scan_data.paths.get_folders_and_files()
        devices = files_and_folders.get("devices", [])
        files = files_and_folders.get("files", [])

        scalar_txt = next(
            (f for f in files if f.endswith(".txt") and f.startswith("ScanData")), None
        )
        tdms_file = next((f for f in files if f.endswith(".tdms")), None)

        scalar_data_file = str(folder / scalar_txt) if scalar_txt else None
        tdms_file_path = str(folder / tdms_file) if tdms_file else None

        ini_dict = scan_data.paths.load_scan_info()
        scan_metadata = ScanMetadata.from_ini_dict(ini_dict)

        has_analysis = scan_data.paths.get_analysis_folder().exists()

        ecs_folder = folder.parent.parent / "ECS Live dumps"
        ecs_path = ecs_folder / f"Scan{tag.number}.txt"
        ecs_dump = parse_ecs_dump(ecs_path)

        return ScanEntry(
            year=tag.year,
            month=tag.month,
            day=tag.day,
            number=tag.number,
            experiment=tag.experiment,
            scalar_data_file=scalar_data_file,
            tdms_file=tdms_file_path,
            non_scalar_devices=devices,
            scan_metadata=scan_metadata,
            ecs_dump=ecs_dump,
            has_analysis_dir=has_analysis,
        )

    @staticmethod
    def _scan_data_iterator(
        data_root: Union[str, Path],
        experiment: str,
        date_range: Optional[Tuple[date, date]] = None,
    ):
        """
        Generator that yields ScanData objects from a directory tree.

        Parameters
        ----------
        data_root : str or Path
            Root path containing all experiment directories.
        experiment : str
            Name of the experiment directory to search.
        date_range : tuple of (date, date), optional
            If specified, only scans within the inclusive date range will be returned.

        Yields
        ------
        ScanData
            Parsed object for each scan that could be read without error.
        """
        data_root = Path(data_root)
        exp_root = data_root / experiment

        for year_dir in sorted(exp_root.glob("Y*/")):
            if not year_dir.is_dir():
                continue

            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue

                for day_dir in sorted(month_dir.iterdir()):
                    try:
                        tag_date = datetime.strptime(day_dir.name, "%y_%m%d").date()
                    except ValueError:
                        continue
                    if date_range:
                        start, end = date_range

                        if not (start <= tag_date <= end):
                            continue
                    scans_dir = day_dir / "scans"
                    if not scans_dir.is_dir():
                        continue

                    for scan_folder in sorted(scans_dir.glob("Scan*")):
                        if not scan_folder.is_dir():
                            continue

                        match = re.match(r"Scan(\d+)", scan_folder.name)
                        if not match:
                            continue

                        scan_number = int(match.group(1))
                        tag = ScanTag(
                            year=tag_date.year,
                            month=tag_date.month,
                            day=tag_date.day,
                            number=scan_number,
                            experiment=experiment,
                        )

                        try:
                            sp = ScanPaths(tag=tag, read_mode=True)
                            yield ScanData(paths=sp)
                        except Exception as e:
                            logger.warning(
                                f"[SKIPPED] Scan {scan_folder} - {type(e).__name__}: {e}"
                            )

    @staticmethod
    def _generate_scan_entries(
        data_root: Path, experiment: str, date_range: Tuple[date, date]
    ):
        """
        Generator that builds ScanEntry objects from a folder structure.

        Parameters
        ----------
        data_root : Path
            Root folder containing all experiment data.
        experiment : str
            Name of the experiment.
        date_range : tuple of date
            Inclusive range for filtering scan folders.

        Yields
        ------
        ScanEntry
            Successfully parsed scan entry object.
        """
        for scan_data in ScanDatabaseBuilder._scan_data_iterator(
            data_root, experiment, date_range
        ):
            try:
                yield ScanDatabaseBuilder.build_scan_entry(scan_data)
            except Exception as e:
                logger.warning(
                    f"[SKIPPED] {scan_data.paths.get_tag()} - {type(e).__name__}: {e}"
                )

    @staticmethod
    def stream_to_parquet(
        data_root: Path,
        experiment: str,
        output_path: Path,
        date_range: Optional[Tuple[date, date]],
        buffer_size: int = 100,
        max_scans: Optional[int] = None,
        mode: Literal["overwrite", "append"] = "append",
    ):
        """
        Stream scan metadata to a Hive-partitioned Parquet dataset.

        This method is the only write path for the scans database and supports both
        full rebuilds and incremental appends.

        Behavior by mode
        ----------------
        - overwrite:
            Deletes the existing Parquet dataset (including the sidecar update log),
            then writes all scans in the specified date range.
            If ``date_range`` is None, uses a broad default range from 2000-01-01 to today.
            After writing, the sidecar log is initialized with a single entry for the
            written date range.

        - append:
            Adds only new scans to the existing dataset.
            If ``date_range`` is None, the method reads the sidecar log to determine
            the most recent ingested date, then resumes from the next day up to today.
            The log is updated with the exact range of dates written in this run.

        Parameters
        ----------
        data_root : Path
            Root directory containing the GEECS data hierarchy.
        experiment : str
            Name of the experiment to search under the data root.
        output_path : Path
            Destination directory for the Parquet dataset (root of Hive partitions).
        date_range : tuple of date, optional
            Inclusive date range of scans to include. If None, behavior depends on mode.
        buffer_size : int, optional
            Number of scan entries to buffer before writing to Parquet. Default is 100.
        max_scans : int, optional
            Maximum number of scan entries to write (useful for testing).
        mode : {'overwrite', 'append'}, optional
            Writing mode for the Parquet dataset. Default is 'append'.

        Notes
        -----
        - The dataset is written as Hive-style partitions by year and month.
        - The sidecar log (``_update_log.json``) resides in the root of ``output_path``
          and tracks the date ranges written in each run.
        - This method does not perform duplicate checking beyond the append logic;
          use a separate deduplication process if needed.
        """
        # --- Resolve date_range ---
        if mode == "overwrite":
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            # If not specified, scan broadly (you can replace with an inferred earliest date)
            if date_range is None:
                date_range = (date(2000, 1, 1), date.today())
            # Also reset the log now (empty)
            ScanDatabaseBuilder._write_update_log(output_path, [])
        elif mode == "append":
            if date_range is None:
                last_date = ScanDatabaseBuilder._last_ingested_date(
                    output_path, experiment
                )
                start_date = (
                    (last_date + timedelta(days=1)) if last_date else date(2000, 1, 1)
                )
                end_date = date.today()
                date_range = (start_date, end_date)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        start_date, end_date = date_range
        if start_date > end_date:
            return  # nothing to do

        # --- Stream & write ---
        buffer: List[dict] = []
        written_any = False
        scan_counter = 0

        # Track actual written min/max day (for precise log update)
        written_start: Optional[date] = None
        written_end: Optional[date] = None

        def _maybe_flush():
            nonlocal buffer, written_any
            if buffer:
                ScanDatabaseBuilder._write_parquet_buffer(buffer, output_path)
                written_any = True
                buffer.clear()

        for entry in ScanDatabaseBuilder._generate_scan_entries(
            data_root=data_root,
            experiment=experiment,
            date_range=(start_date, end_date),
        ):
            entry_date = date(entry.year, entry.month, entry.day)
            # Update written range (based on entries we actually append)
            if written_start is None or entry_date < written_start:
                written_start = entry_date
            if written_end is None or entry_date > written_end:
                written_end = entry_date

            buffer.append(entry)
            scan_counter += 1

            if max_scans and scan_counter >= max_scans:
                _maybe_flush()
                break

            if len(buffer) >= buffer_size:
                _maybe_flush()

        _maybe_flush()

        # --- Update sidecar log only if we actually wrote something ---
        if written_any and written_start is not None and written_end is not None:
            ScanDatabaseBuilder._append_update_log(
                output_path, experiment, written_start, written_end
            )

    @staticmethod
    def _buffer_to_dataframe(buffer: List[ScanEntry]) -> pd.DataFrame:
        """Convert a list of scan entry dictionaries to a normalized DataFrame."""
        records = ScanDatabaseBuilder.get_parquet_records_from_entries(buffer)
        schema = collect_dtypes(buffer[0])
        df = pd.DataFrame(records).astype(
            schema, errors="ignore"
        )  # `schema` from collect_dtypes()

        return df

    @staticmethod
    def get_parquet_records_from_entries(entries: list[ScanEntry]) -> list[dict]:
        """
        Flatten a list of ScanEntry objects into a list of dictionaries suitable for writing to Parquet.

        Returns
        -------
        list of dict
            Each dict has flat keys including nested metadata fields.
        """
        return [entry.flatten() for entry in entries]

    @staticmethod
    def _write_parquet_buffer(buffer: List[ScanEntry], output_path: Path) -> None:
        """
        Write a buffered list of scan entry dicts to a Hive-partitioned Parquet dataset.

        Parameters
        ----------
        buffer : list of dict
            List of scan entry dictionaries, typically from a scan streaming process.
        output_path : Path
            Root directory for the Hive-partitioned Parquet dataset.

        Notes
        -----
        - Partitions are by ``year`` and ``month`` for efficient filtering.
        - All structured ``ScanMetadata`` fields are extracted for top-level filtering.
        - Complex structures (e.g., ``ecs_dump``) are stored as JSON strings.
        """
        if not buffer:
            return
        df = ScanDatabaseBuilder._buffer_to_dataframe(buffer)
        arrow_schema = get_pyarrow_schema(buffer[0])  # new function
        table = pa.Table.from_pandas(df, schema=arrow_schema, preserve_index=False)

        pq.write_to_dataset(
            table, root_path=str(output_path), partition_cols=["year", "month"]
        )

    @staticmethod
    def _log_path(root: Path) -> Path:
        return Path(root) / ScanDatabaseBuilder._LOG_NAME

    @staticmethod
    def _read_update_log(root: Path) -> List[_UpdateLogEntry]:
        p = ScanDatabaseBuilder._log_path(root)
        if not p.exists():
            return []
        data = json.loads(p.read_text())
        return [_UpdateLogEntry.model_validate(x) for x in data]

    @staticmethod
    def _write_update_log(root: Path, entries: List[_UpdateLogEntry]) -> None:
        final_path = ScanDatabaseBuilder._log_path(root)
        tmp = final_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([e.model_dump() for e in entries], indent=2, default=str)
        )
        tmp.replace(final_path)

    @staticmethod
    def _append_update_log(
        root: Path, experiment: str, start_date: date, end_date: date
    ) -> None:
        log = ScanDatabaseBuilder._read_update_log(root)
        log.append(
            _UpdateLogEntry(
                experiment=experiment,
                update_date=date.today(),
                date_range_added=(start_date, end_date),
            )
        )
        ScanDatabaseBuilder._write_update_log(root, log)

    @staticmethod
    def _last_ingested_date(root: Path, experiment: str) -> Optional[date]:
        log = ScanDatabaseBuilder._read_update_log(root)
        ends = [e.date_range_added[1] for e in log if e.experiment == experiment]
        return max(ends) if ends else None

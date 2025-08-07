"""
rovides utilities for scanning GEECS experiment directories.

Provides utilities for scanning GEECS experiment directories and constructing
structured metadata records (`ScanEntry`) for each scan. Supports streaming
scan metadata to Parquet format for persistent storage and querying.

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
from datetime import date, datetime
import re
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging

from geecs_data_utils.scan_data import ScanData
from geecs_data_utils.scan_paths import ScanPaths
from geecs_data_utils.utils import ScanTag
from geecs_data_utils.database.entries import ScanEntry, ScanMetadata

logger = logging.getLogger(__name__)


class ScanDatabaseBuilder:
    """Builder class for constructing a full ScanDatabase from GEECS scan folders."""

    @staticmethod
    def build_scan_entry(scan_data: ScanData) -> ScanEntry:
        """
        Construct a ScanEntry from a given ScanData object.

        Parameters
        ----------
        scan_data : ScanData
            Object containing paths and parsed content for a single scan.

        Returns
        -------
        ScanEntry
            Fully populated metadata entry for the scan.
        """
        folder = scan_data.get_folder()
        tag = scan_data.get_tag()

        files_and_folders = scan_data.get_folders_and_files()
        devices = files_and_folders.get("devices", [])
        files = files_and_folders.get("files", [])

        scalar_txt = next(
            (f for f in files if f.endswith(".txt") and f.startswith("ScanData")), None
        )
        tdms_file = next((f for f in files if f.endswith(".tdms")), None)

        scalar_data_file = str(folder / scalar_txt) if scalar_txt else None
        tdms_file_path = str(folder / tdms_file) if tdms_file else None

        ini_dict = scan_data.load_scan_info()
        scan_metadata = ScanMetadata.from_ini_dict(ini_dict)

        has_analysis = scan_data.get_analysis_folder().exists()

        ecs_folder = folder.parent.parent / "ECS Live dumps"
        ecs_path = ecs_folder / f"Scan{tag.number}.txt"
        ecs_dump = ScanPaths.parse_ecs_dump(ecs_path)

        return ScanEntry(
            scan_tag=tag,
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
                            yield ScanData(tag=tag, read_mode=True)
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
                    f"[SKIPPED] {scan_data.get_tag()} - {type(e).__name__}: {e}"
                )

    @staticmethod
    def stream_to_parquet(
        data_root: Path,
        experiment: str,
        output_path: Path,
        date_range: Tuple[date, date],
        buffer_size: int = 100,
        max_scans: Optional[int] = None,
        mode: Literal["overwrite", "append", "smart_append"] = "overwrite",
    ):
        """
        Stream scan metadata to a partitioned Parquet dataset.

        Parameters
        ----------
        data_root : Path
            Root directory containing the GEECS data hierarchy.
        experiment : str
            Experiment name to search under the data root.
        output_path : Path
            Destination directory for the Parquet dataset.
        date_range : tuple of date
            Inclusive date range for scans to include.
        buffer_size : int, optional
            Number of scan entries to buffer before writing. Default is 100.
        max_scans : int, optional
            Maximum number of scan entries to write. Useful for testing.
        mode : {'overwrite', 'append', 'smart_append'}, optional
            Writing mode for the Parquet dataset. Default is 'overwrite'.

            - 'overwrite': deletes existing dataset and rewrites everything.
            - 'append': adds all new entries to the dataset.
            - 'smart_append': skips duplicates based on scan_tag.
        """
        if mode == "overwrite":
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

        existing_keys = set()
        if mode in {"append", "smart_append"} and output_path.exists():
            try:
                df_existing = pd.read_parquet(output_path)
                existing_keys = {
                    (row.year, row.month, row.day, row.number, row.experiment)
                    for _, row in df_existing.iterrows()
                }
            except Exception as e:
                logger.warning(
                    f"Could not load existing database for deduplication: {e}"
                )

        buffer = []
        scan_counter = 0
        for entry in ScanDatabaseBuilder._generate_scan_entries(
            data_root=data_root, experiment=experiment, date_range=date_range
        ):
            tag = entry.scan_tag
            key = (tag.year, tag.month, tag.day, tag.number, tag.experiment)
            if key in existing_keys:
                continue

            buffer.append(entry.model_dump())
            scan_counter += 1

            if max_scans and scan_counter >= max_scans:
                break

            if len(buffer) >= buffer_size:
                ScanDatabaseBuilder._write_parquet_buffer(buffer, output_path)
                buffer.clear()

        if buffer:
            ScanDatabaseBuilder._write_parquet_buffer(buffer, output_path)

    @staticmethod
    def _write_parquet_buffer(buffer: List[dict], output_path: Path):
        """
        Write a list of scan entry dictionaries to Parquet with partitioning.

        Parameters
        ----------
        buffer : list of dict
            List of scan entry dicts produced by model_dump().
        output_path : Path
            Target directory for the Parquet partitioned dataset.
        """
        df = pd.DataFrame(buffer)

        # Extract partition columns from scan_tag
        if "scan_tag" in df.columns:
            for field in ["year", "month", "day", "number", "experiment"]:
                df[field] = df["scan_tag"].apply(lambda tag: tag[field])

        # 🔁 Serialize all variable-structure columns to JSON strings
        for col in ["scan_tag", "scan_metadata", "ecs_dump", "non_scalar_devices"]:
            if col in df.columns:
                df[col] = df[col].apply(json.dumps)

        # Convert to Arrow table and write to partitioned Parquet dataset
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(table, root_path=output_path, partition_cols=["year"])

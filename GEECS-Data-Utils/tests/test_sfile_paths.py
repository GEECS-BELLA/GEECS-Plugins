"""The scan folder's standard files, by path construction only (``data/sfile.py``)."""

from pathlib import Path

import pytest

from geecs_data_utils.data.sfile import (
    scan_data_txt_path_for,
    stream_table_parquet_path_for,
)


def test_the_stream_table_is_the_sfiles_sibling_named_by_stream(tmp_path: Path):
    scan = tmp_path / "scans" / "Scan007"
    assert scan_data_txt_path_for(scan) == scan / "ScanDataScan007.txt"
    assert (
        stream_table_parquet_path_for(scan, "primary")
        == scan / "ScanDataScan007-primary.parquet"
    )
    assert (
        stream_table_parquet_path_for(scan, "shots").name
        == "ScanDataScan007-shots.parquet"
    )
    assert not scan.exists()  # nothing touched on disk


@pytest.mark.parametrize("bad", ["", "a/b", "a\\b", ".hidden"])
def test_a_stream_name_is_a_plain_name(tmp_path: Path, bad: str):
    with pytest.raises(ValueError):
        stream_table_parquet_path_for(tmp_path / "scans" / "Scan007", bad)

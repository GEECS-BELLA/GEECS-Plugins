"""
Unit tests for validating the behavior of the ScanDatabaseBuilder class.

This test assumes the presence of a known directory layout at TEST_DATA_ROOT
containing a scan from the "Undulator" experiment on 2025-08-05.

Functions
---------
test_scan_database_builder_runs()
    Verifies that ScanDatabaseBuilder.build_from_directory returns a valid
    ScanDatabase and that the resulting ScanEntry matches expected properties.
"""

from pathlib import Path
from datetime import date

from geecs_data_utils.scans_database.builder import ScanDatabaseBuilder
from geecs_data_utils.scans_database.database import ScanDatabase
from geecs_data_utils.scans_database.entries import ScanEntry


TEST_DATA_ROOT = Path("/Volumes/hdna2/data")


def test_scan_database_builder_runs():
    """
    Test building a ScanDatabase from a single scan directory.

    This test verifies that:
    - The scans_database is built without error.
    - A single scan entry is returned.
    - The scan entry contains valid metadata, file paths, and devices.

    Assumes:
    - TEST_DATA_ROOT contains an "Undulator" experiment folder.
    - The scan date is 2025-08-05.
    - The first scan has scan number 1 and includes the UC_HiResMagCam device.

    Raises
    ------
    AssertionError
        If any property of the built ScanDatabase or its entries is invalid.
    """
    db: ScanDatabase = ScanDatabaseBuilder.build_from_directory(
        data_root=TEST_DATA_ROOT,
        experiment="Undulator",
        date_range=(date(2025, 8, 5), date(2025, 8, 5)),
    )

    assert isinstance(db, ScanDatabase)
    assert len(db.scans) == 1

    entry = db.scans[0]
    assert isinstance(entry, ScanEntry)
    assert entry.scan_tag.number == 1
    assert entry.scan_metadata.scan_parameter is not None
    assert entry.scalar_data_file.endswith(".txt")
    assert "UC_HiResMagCam" in entry.non_scalar_devices
    assert entry.ecs_dump is not None
    assert entry.has_analysis_dir is True  # assuming not created for test

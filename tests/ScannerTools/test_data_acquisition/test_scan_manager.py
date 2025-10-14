"""
System test to try the basic functionality of scan manager.

Uses the `get_default_scan_manager` tool to get a basic
instance of Scan Manager that can
"""

import pytest
from pathlib import Path
import shutil

from geecs_scanner.data_acquisition.default_scan_manager import get_default_scan_manager


def test_noscan_from_default_scan_manager():
    """Run a basic noscan scan."""
    manager = get_default_scan_manager("Undulator")
    config_filename = (
        Path(__file__).parents[1]
        / "test_configs"
        / "Test"
        / "aux_configs"
        / "test_noscan.yaml"
    )
    success = manager.reinitialize(config_path=config_filename)

    if success is False:
        raise AssertionError("Scan Manager unable to reinitialize")

    scan_config = {
        "device_var": "noscan",
        "start": 0,
        "end": 0,
        "step": 1,
        "wait_time": 2.5,
        "additional_description": "System test for a noscan",
    }

    manager.start_scan_thread(scan_config=scan_config)
    manager.data_logger.sound_player.stop()  # Turn off sounds
    test_scan_data = None

    while manager.is_scanning_active():
        if test_scan_data is None:
            test_scan_data = manager.scan_data_manager.scan_paths
        pass

    if test_scan_data is None:
        raise AssertionError("Scan Data was never initialized during scan")

    data_folder = test_scan_data.get_folder()
    scan_number = test_scan_data.get_tag().number
    analysis_folder = test_scan_data.get_analysis_folder()

    scan_info_file = data_folder / f"ScanInfoScan{scan_number:03d}.ini"
    sfile = analysis_folder.parent / f"s{scan_number}.txt"

    print(sfile)
    print(scan_info_file)
    print(data_folder)
    print(analysis_folder)

    assert data_folder.exists() is True
    assert scan_info_file.exists() is True
    assert analysis_folder.exists() is True
    assert sfile.exists() is True

    # Cleanup files created from test scan
    if int(scan_number) > 0:
        if data_folder.exists() and data_folder.is_dir():
            shutil.rmtree(data_folder)
        if analysis_folder.exists() and analysis_folder.is_dir():
            shutil.rmtree(analysis_folder)
        if sfile.exists() and sfile.is_file():
            sfile.unlink()


if __name__ == "__main__":
    pytest.main()

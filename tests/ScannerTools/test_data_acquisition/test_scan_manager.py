"""
System test to try the basic functionality of scan manager.  Uses the `get_default_scan_manager` tool to get a basic
instance of Scan Manager that can

-Chris
"""

import pytest
from pathlib import Path

from geecs_scanner.data_acquisition.default_scan_manager import get_default_scan_manager


def test_noscan_from_default_scan_manager():
    """
     Run a basic noscan scan
    """

    manager = get_default_scan_manager("Undulator")
    config_filename = Path(__file__).parents[1] / "test_configs" / "Test" / "aux_configs" / "test_noscan.yaml"
    success = manager.reinitialize(config_path=config_filename)

    if success is False:
        raise AssertionError("Scan Manager unable to reinitialize")

    scan_config = {
        'device_var': 'noscan',
        'start': 0,
        'end': 0,
        'step': 1,
        'wait_time': 5.5,
        'additional_description': 'System test for a noscan'}

    manager.start_scan_thread(scan_config=scan_config)
    manager.data_logger.sound_player.stop()
    test_scan_data = manager.scan_data_manager.scan_data

    while manager.is_scanning_active():
        pass

    data_folder = test_scan_data.get_folder()
    scan_number = test_scan_data.get_tag().number
    analysis_folder = test_scan_data.get_analysis_folder()

    assert data_folder.exists() is True
    assert (data_folder / f"ScanInfoScan{scan_number:03d}.ini").exists() is True

    assert analysis_folder.exists() is True
    assert (analysis_folder.parent / f"s{scan_number}.txt)").exists() is True


if __name__ == "__main__":
    pytest.main()

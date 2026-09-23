"""The ScanAnalysis adapter preserves shared resolution and queue outcomes."""

from types import SimpleNamespace

import pandas as pd

from geecs_data_utils import ScanPaths, ScanTag
from scan_analysis.analyzers.common.single_device_scan_analyzer import (
    SingleDeviceScanAnalyzer,
)
from scan_analysis.task_queue import read_statuses, run_worklist


def test_missing_required_stack_records_no_data(tmp_path):
    """Execute the queue: absent captures must not become a successful empty run."""
    tag = ScanTag(year=2025, month=1, day=1, number=1, experiment="Test")
    scan = ScanPaths.get_scan_folder_path(tag=tag, base_directory=tmp_path)
    directory = scan / "scope"
    directory.mkdir(parents=True)  # fixture acquisition, not analysis
    (directory / "scope_3866137959.524.txt").write_text("0\t1\n")
    analyzer = SingleDeviceScanAnalyzer.__new__(SingleDeviceScanAnalyzer)
    analyzer.device_name = "scope"
    analyzer.file_tail = ".txt"
    analyzer.path_dict = {"data": directory}
    analyzer.data_format = "device_hdf5"
    analyzer.auxiliary_data = pd.DataFrame(
        {"Shotnumber": [1], "scope acq_timestamp": [3866137959.524]}
    )
    analyzer.image_analyzer = SimpleNamespace(
        line_config=SimpleNamespace(data_loading=SimpleNamespace(data_type="pva_stack"))
    )
    task = SimpleNamespace(
        id="scope",
        priority=1,
        run_analysis=lambda tag: analyzer._build_data_file_map(),
        cleanup=lambda: None,
    )
    run_worklist([(1, tag, task)], base_directory=tmp_path)
    (status,) = read_statuses(scan)
    assert status.state == "no_data"
    assert analyzer._data_file_map == {}
    assert not list(scan.rglob("*.claim"))


def test_adapter_preserves_filename_device_override(tmp_path):
    path = tmp_path / "other_3866137959.524.png"
    path.touch()
    analyzer = SingleDeviceScanAnalyzer.__new__(SingleDeviceScanAnalyzer)
    analyzer.device_name = "camera"
    analyzer.data_device_name = "other"
    analyzer.file_tail = ".png"
    analyzer.path_dict = {"data": tmp_path}
    analyzer.auxiliary_data = pd.DataFrame(
        {"Shotnumber": [7], "camera acq_timestamp": [3866137959.524]}
    )
    analyzer._build_data_file_map()
    assert analyzer._data_file_map == {7: path}

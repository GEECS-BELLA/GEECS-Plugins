"""Behavioral tests for DataLogger (Decompose step D1).

All tests are network-free.  DataLogger is constructed via object.__new__ to
bypass the DeviceManager network dependency; only the attributes touched by the
methods under test are populated.  FakeGeecsDevice duck-types GeecsDevice for
the subset of the interface that DataLogger uses.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock


from geecs_scanner.engine.data_logger import DataLogger


# ---------------------------------------------------------------------------
# Fake device
# ---------------------------------------------------------------------------


class FakeGeecsDevice:
    """Minimal duck-type for GeecsDevice used by DataLogger."""

    def __init__(self, name: str, state: Optional[dict] = None):
        self.name = name
        self.state = state or {}
        self.is_composite = False

    def get_name(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# DataLogger factory
# ---------------------------------------------------------------------------


def _make_logger(
    bin_num: int = 0,
    scan_number: int = 1,
    event_driven_observables: Optional[list] = None,
    synchronous_device_names: Optional[list] = None,
    device_save_paths_mapping: Optional[dict] = None,
) -> DataLogger:
    """Build a DataLogger with only the attributes the tested methods need."""
    dl = object.__new__(DataLogger)
    dl.lock = threading.Lock()
    dl.log_entries = {}
    dl.bin_num = bin_num
    dl.scan_number = scan_number
    dl.virtual_variable_name = None
    dl.virtual_variable_value = 0
    dl.shot_index = 0
    dl.event_driven_observables = event_driven_observables or []
    dl.device_save_paths_mapping = device_save_paths_mapping or {}
    dl.file_mover = MagicMock()
    dl.sound_player = MagicMock()
    dl.standby_mode_device_status = {}
    dl.initial_timestamps = {}
    dl.synced_timestamps = {}
    dl.synchronous_device_names = synchronous_device_names or []
    dl.last_timestamps = {}
    # async_observables are empty by default; _update_async_observables won't fire
    dl.device_manager = MagicMock()
    dl.device_manager.async_observables = []
    return dl


# ---------------------------------------------------------------------------
# _log_device_data — entry creation and shot index
# ---------------------------------------------------------------------------


class TestLogDeviceData:
    def test_first_call_creates_entry(self):
        dl = _make_logger(bin_num=3, scan_number=7, event_driven_observables=["D:Var"])
        dev = FakeGeecsDevice("D", {"Var": 42.0})

        dl._log_device_data(dev, 1.0)

        assert 1.0 in dl.log_entries
        entry = dl.log_entries[1.0]
        assert entry["Elapsed Time"] == 1.0
        assert entry["Bin #"] == 3
        assert entry["scan"] == 7
        assert entry["D:Var"] == 42.0

    def test_shot_index_increments_once_per_elapsed_time(self):
        dl = _make_logger(event_driven_observables=["D:V"])
        dev = FakeGeecsDevice("D", {"V": 0})

        assert dl.shot_index == 0
        dl._log_device_data(dev, 1.0)
        assert dl.shot_index == 1
        dl._log_device_data(dev, 2.0)
        assert dl.shot_index == 2

    def test_second_device_merges_without_incrementing_shot_index(self):
        """Two devices arriving for the same elapsed_time must share one log entry."""
        dl = _make_logger(event_driven_observables=["D1:V1", "D2:V2"])
        dev1 = FakeGeecsDevice("D1", {"V1": 1.0})
        dev2 = FakeGeecsDevice("D2", {"V2": 2.0})

        dl._log_device_data(dev1, 5.0)
        dl._log_device_data(dev2, 5.0)

        assert dl.shot_index == 1  # incremented only once
        entry = dl.log_entries[5.0]
        assert entry["D1:V1"] == 1.0
        assert entry["D2:V2"] == 2.0

    def test_virtual_variable_written_when_set(self):
        dl = _make_logger(event_driven_observables=["D:V"])
        dl.virtual_variable_name = "ScanVar"
        dl.virtual_variable_value = 99.5
        dev = FakeGeecsDevice("D", {"V": 0})

        dl._log_device_data(dev, 1.0)

        assert dl.log_entries[1.0]["ScanVar"] == 99.5

    def test_virtual_variable_absent_when_not_set(self):
        dl = _make_logger(event_driven_observables=["D:V"])
        dev = FakeGeecsDevice("D", {"V": 0})

        dl._log_device_data(dev, 1.0)

        assert "ScanVar" not in dl.log_entries[1.0]


# ---------------------------------------------------------------------------
# get_current_shot
# ---------------------------------------------------------------------------


class TestGetCurrentShot:
    def test_returns_zero_before_any_logging(self):
        dl = _make_logger()
        assert dl.get_current_shot() == 0

    def test_tracks_shot_index_after_logging(self):
        dl = _make_logger(event_driven_observables=["D:V"])
        dev = FakeGeecsDevice("D", {"V": 0})

        dl._log_device_data(dev, 1.0)
        assert dl.get_current_shot() == 1

        dl._log_device_data(dev, 2.0)
        assert dl.get_current_shot() == 2


# ---------------------------------------------------------------------------
# FileMover task queuing
# ---------------------------------------------------------------------------


class TestFileMoverTaskQueuing:
    def _save_paths(self) -> dict:
        return {
            "D": {
                "source_dir": Path("/tmp/src"),
                "target_dir": Path("/tmp/dst"),
                "device_type": "Camera",
            }
        }

    def test_task_queued_when_device_in_save_paths_mapping(self):
        dl = _make_logger(
            event_driven_observables=["D:acq_timestamp"],
            device_save_paths_mapping=self._save_paths(),
        )
        dev = FakeGeecsDevice("D", {"acq_timestamp": 1_000_000.5})

        dl._log_device_data(dev, 1.0)

        dl.file_mover.move_files_by_timestamp.assert_called_once()
        task = dl.file_mover.move_files_by_timestamp.call_args[0][0]
        assert task.device_name == "D"
        assert task.expected_timestamp == 1_000_000.5
        assert task.shot_index == 1

    def test_no_task_when_device_not_in_save_paths_mapping(self):
        dl = _make_logger(event_driven_observables=["D:acq_timestamp"])
        dev = FakeGeecsDevice("D", {"acq_timestamp": 1_000_000.5})

        dl._log_device_data(dev, 1.0)

        dl.file_mover.move_files_by_timestamp.assert_not_called()

    def test_task_queued_for_correct_shot_index(self):
        """Task shot_index must match the incremented shot_index, not the pre-call value."""
        dl = _make_logger(
            event_driven_observables=["D:acq_timestamp"],
            device_save_paths_mapping=self._save_paths(),
        )
        dev = FakeGeecsDevice("D", {"acq_timestamp": 1.0})

        dl._log_device_data(dev, 0.5)  # shot 1
        dl.file_mover.reset_mock()
        dl._log_device_data(dev, 1.5)  # shot 2

        task = dl.file_mover.move_files_by_timestamp.call_args[0][0]
        assert task.shot_index == 2


# ---------------------------------------------------------------------------
# Standby mode detection
# ---------------------------------------------------------------------------


class TestStandbyDetection:
    def test_standby_true_when_timestamp_unchanged(self):
        dl = _make_logger(synchronous_device_names=["D"])
        dev = FakeGeecsDevice("D")

        dl._check_device_standby_mode_status(dev, 1000.0)  # sets t0; status = None
        assert dl.standby_mode_device_status.get("D") is None

        dl._check_device_standby_mode_status(dev, 1000.0)  # same ts → standby
        assert dl.standby_mode_device_status["D"] is True

    def test_standby_false_when_timestamp_advances(self):
        dl = _make_logger(synchronous_device_names=["D"])
        dev = FakeGeecsDevice("D")

        dl._check_device_standby_mode_status(dev, 1000.0)
        dl._check_device_standby_mode_status(dev, 1001.0)  # different → exited standby
        assert dl.standby_mode_device_status["D"] is False

    def test_false_status_is_sticky(self):
        """Once a device exits standby, further calls with unchanged ts must not reset it."""
        dl = _make_logger(synchronous_device_names=["D"])
        dev = FakeGeecsDevice("D")

        dl._check_device_standby_mode_status(dev, 1000.0)
        dl._check_device_standby_mode_status(dev, 1001.0)  # → False
        dl._check_device_standby_mode_status(dev, 1001.0)  # same ts again → still False
        assert dl.standby_mode_device_status["D"] is False

    def test_check_all_standby_false_when_some_devices_missing(self):
        dl = _make_logger(synchronous_device_names=["D1", "D2"])
        dl.standby_mode_device_status = {"D1": True}  # D2 absent
        assert not dl._check_all_standby_status()

    def test_check_all_standby_false_when_not_all_true(self):
        dl = _make_logger(synchronous_device_names=["D1", "D2"])
        dl.standby_mode_device_status = {"D1": True, "D2": None}
        assert not dl._check_all_standby_status()

    def test_check_all_standby_true_when_all_devices_standby(self):
        dl = _make_logger(synchronous_device_names=["D1", "D2"])
        dl.standby_mode_device_status = {"D1": True, "D2": True}
        assert dl._check_all_standby_status()

    def test_all_exited_standby_true_when_all_false(self):
        dl = _make_logger(synchronous_device_names=["D1", "D2"])
        dl.standby_mode_device_status = {"D1": False, "D2": False}
        assert dl._check_all_exited_standby_status()

    def test_all_exited_standby_false_when_some_still_in_standby(self):
        dl = _make_logger(synchronous_device_names=["D1", "D2"])
        dl.standby_mode_device_status = {"D1": False, "D2": True}
        assert not dl._check_all_exited_standby_status()

"""
Unit tests for Data Logger and File Mover.  This tries to isolate as much of the functionality as possible without
performing a full scan and saving data.

-Chris
"""

import pytest
import time
from pathlib import Path
from geecs_scanner.data_acquisition.data_logger import DataLogger, FileMoveTask, FileMover


TEST_DEVICE_NAME = "TEST_DEVICE"  # Update if not valid in your test setup
TEST_FLOAT = 12345.678
TEST_SHOT_NUMBER = 1


def test_file_mover_functionality():
    """
    Unit test for the File Mover, generates a single file of the correct formatting and moves it
    """

    # Create a FileMover with default settings
    file_mover = FileMover()
    assert file_mover.task_queue.empty() is True

    # Create the test file
    test_filename = f"{TEST_DEVICE_NAME}_{TEST_FLOAT:.3f}.txt"
    test_filepath = Path(__file__).parent / "data_test_bin" / "start" / TEST_DEVICE_NAME / test_filename
    test_filepath.parent.mkdir(parents=True, exist_ok=True)
    test_filepath.touch()

    # Generate the filename for the expected resulting file (scan number is never set, so it is `None`)
    expected_file = (test_filepath.parents[2] / "end" / TEST_DEVICE_NAME /
                     f"ScanNone_{TEST_DEVICE_NAME}_{TEST_SHOT_NUMBER:03d}.txt")

    # Assert that we've created the test file and that the expected file does not exist yet
    assert test_filepath.exists() is True
    if expected_file.exists():
        expected_file.unlink()

    # Create the FileMoveTask dataclass instance and add to the FileMover queue
    test_filemove_task = FileMoveTask(
        source_dir=test_filepath.parent,
        target_dir=test_filepath.parents[2] / "end" / TEST_DEVICE_NAME,
        device_name=TEST_DEVICE_NAME,
        device_type="Does not matter",
        expected_timestamp=TEST_FLOAT,
        shot_index=TEST_SHOT_NUMBER,
    )
    file_mover.move_files_by_timestamp(test_filemove_task)

    # Wait for a little bit for the worker to move the file
    tries = 0
    while test_filepath.exists():
        time.sleep(0.01)
        tries += 1
        if tries > 20:
            raise AssertionError("Maximum waiting time occurred, File Mover test failed")
        pass

    # Assert that the new file exists and the file mover has completed its task
    assert expected_file.exists() is True
    assert file_mover.task_queue.empty() is True

    # Cleanup the test
    expected_file.unlink()
    file_mover.shutdown()


#def test_data_logger_functionality():


if __name__ == "__main__":
    pytest.main()

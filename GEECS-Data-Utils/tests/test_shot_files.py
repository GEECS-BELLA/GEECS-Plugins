"""Public mapping API guards and the read-only data-layer boundary."""

import subprocess
import sys

import pandas as pd
import pytest

from geecs_data_utils.shot_files import map_shot_files


def test_missing_scan_is_not_created_and_rows_are_unchanged(tmp_path):
    directory = tmp_path / "scans" / "Scan001" / "camera"
    rows = pd.DataFrame({"Shotnumber": [1], "camera acq_timestamp": [3866137959.524]})
    before = rows.copy(deep=True)
    assert map_shot_files(directory, rows, device="camera", file_tail=".png") == {}
    pd.testing.assert_frame_equal(rows, before)
    assert not (tmp_path / "scans").exists()


def test_public_mapper_has_no_analysis_dependency(tmp_path):
    source = r"""
import importlib.abc
import sys
from pathlib import Path
class BlockAnalysis(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'image_analysis', 'scan_analysis', 'geecs_analysis', 'geecs_schemas'}:
            raise AssertionError('upward dependency: ' + fullname)
sys.meta_path.insert(0, BlockAnalysis())
import pandas as pd
from geecs_data_utils.shot_files import map_shot_files
root = Path(sys.argv[1])
path = root / 'Scan001_camera_002.png'
path.touch()
rows = pd.DataFrame({'Shotnumber': [2, 3]})
assert map_shot_files(root, rows, device='camera', file_tail='.png') == {2: path}
"""
    subprocess.run([sys.executable, "-c", source, str(tmp_path)], check=True)


def test_invalid_source_selection_fails_before_discovery(tmp_path):
    with pytest.raises(ValueError, match="prefer_stack"):
        map_shot_files(
            tmp_path,
            pd.DataFrame({"Shotnumber": [1]}),
            device="scope",
            file_tail=".txt",
            stacks_only=True,
        )
    with pytest.raises(ValueError, match="Shotnumber"):
        map_shot_files(
            tmp_path, pd.DataFrame({"other": [1]}), device="camera", file_tail=".png"
        )

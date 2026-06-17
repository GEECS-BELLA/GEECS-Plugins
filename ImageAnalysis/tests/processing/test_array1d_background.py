"""Tests for image_analysis.processing.array1d.background.save_background_to_file.

The function previously did ``file_path.parent.mkdir(parents=True, exist_ok=True)``
which could silently re-create a missing scan folder if the caller's destination
resolved into ``scans/ScanXXX/``. The patched version requires the parent to exist
and raises ``FileNotFoundError`` otherwise — fail-loud, never silent-create.
"""

from pathlib import Path

import numpy as np
import pytest

from image_analysis.processing.array1d.background import save_background_to_file


def _valid_background() -> np.ndarray:
    """Minimal Nx2 background array accepted by the validator."""
    return np.column_stack([np.linspace(0.0, 1.0, 5), np.linspace(2.0, 3.0, 5)])


class TestSaveBackgroundParentDirInvariant:
    def test_refuses_when_parent_missing(self, tmp_path: Path):
        # File path inside a non-existent ScanXXX folder — the failure mode
        # we guard against.
        dest = tmp_path / "scans" / "Scan015" / "bg.npy"
        assert not dest.parent.exists(), "precondition: parent dir missing"

        with pytest.raises(FileNotFoundError, match="does not exist"):
            save_background_to_file(_valid_background(), dest)

        # Critical: the function must not have created the scan folder as a
        # side effect.
        assert not dest.parent.exists()
        assert not dest.exists()

    def test_writes_when_parent_exists(self, tmp_path: Path):
        dest = tmp_path / "bg.npy"
        assert dest.parent.is_dir()

        save_background_to_file(_valid_background(), dest)

        assert dest.is_file()
        loaded = np.load(dest)
        assert loaded.shape == (5, 2)

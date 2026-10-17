"""The ``expected`` fixture's update mode never rewrites what a test only reads."""

import numpy as np
import pytest


@pytest.fixture
def expected_root(tmp_path):
    return tmp_path


def test_a_read_only_call_leaves_the_stored_file_byte_identical(
    expected, tmp_path, monkeypatch
):
    folder = tmp_path / "test_expected_fixture"
    folder.mkdir()
    path = folder / "reference.npz"
    np.savez_compressed(path, value=np.array([1.0, 2.0, 3.0]))
    before = path.read_bytes()
    monkeypatch.setenv("GEECS_UPDATE_GOLDENS", "1")
    stored = expected("reference")
    np.testing.assert_array_equal(stored, [1.0, 2.0, 3.0])
    assert path.read_bytes() == before


def test_update_refuses_an_object_array(expected, monkeypatch):
    monkeypatch.setenv("GEECS_UPDATE_GOLDENS", "1")
    with pytest.raises(TypeError, match="object"):
        expected("bad", {"value": np.array([None, 1], dtype=object)})

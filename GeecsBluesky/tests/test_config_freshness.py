"""The scan-variable catalog follows its file (GeecsScanner PR 5a).

The resolver lives as long as its process — the web scanner's, the
worker's environment — so a lifetime cache made every catalog edit wait
for a restart.  The contract now: an unchanged file is served from the
cache (one ``stat`` per call, no re-parse); a file whose mtime or size
changed is re-read on the next call; a file that disappears raises.
"""

from __future__ import annotations

import os

import pytest
import yaml

from geecs_bluesky.config_resolver import ConfigsRepoResolver
from geecs_bluesky.exceptions import GeecsConfigurationError


def _catalog(*names: str) -> dict:
    return {
        "schema_version": 1,
        "variables": {n: {"target": f"U_Dev:{n}", "kind": "setpoint"} for n in names},
    }


@pytest.fixture
def repo(tmp_path):
    folder = tmp_path / "TestExp" / ConfigsRepoResolver.SCAN_VARIABLES_FOLDER
    folder.mkdir(parents=True)
    path = folder / "scan_variables.yaml"
    path.write_text(yaml.safe_dump(_catalog("jet_x")))
    return tmp_path, path


def _touch_later(path, seconds: float = 2.0) -> None:
    """Move the file's mtime forward, so the change is visible even on a
    coarse-mtime filesystem and within the same second as the write."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + int(seconds * 1e9)))


def test_unchanged_file_is_served_from_the_cache(repo):
    root, _ = repo
    resolver = ConfigsRepoResolver("TestExp", experiments_root=root)
    first = resolver.scan_variable_catalog()
    assert resolver.scan_variable_catalog() is first


def test_edited_file_is_reread_on_the_next_call(repo):
    root, path = repo
    resolver = ConfigsRepoResolver("TestExp", experiments_root=root)
    assert sorted(resolver.scan_variable_catalog().variables) == ["jet_x"]
    path.write_text(yaml.safe_dump(_catalog("jet_x", "jet_z")))
    _touch_later(path)
    assert sorted(resolver.scan_variable_catalog().variables) == ["jet_x", "jet_z"]
    assert resolver.resolve_scan_variable("jet_z").target == "U_Dev:jet_z"


def test_same_size_edit_with_new_mtime_is_reread(repo):
    root, path = repo
    resolver = ConfigsRepoResolver("TestExp", experiments_root=root)
    resolver.scan_variable_catalog()
    path.write_text(yaml.safe_dump(_catalog("jet_y")))  # same length as jet_x
    _touch_later(path)
    assert sorted(resolver.scan_variable_catalog().variables) == ["jet_y"]


def test_removed_file_raises_instead_of_serving_the_cache(repo):
    root, path = repo
    resolver = ConfigsRepoResolver("TestExp", experiments_root=root)
    resolver.scan_variable_catalog()
    path.unlink()
    with pytest.raises(GeecsConfigurationError, match="no scan-variable catalog"):
        resolver.scan_variable_catalog()

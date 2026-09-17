"""The file-reading layer must install and import without the access library."""

import subprocess
import sys
import tomllib
from pathlib import Path


def test_no_intra_repo_runtime_dependencies():
    package = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((package / "pyproject.toml").read_text())
    dependencies = metadata["tool"]["poetry"]["dependencies"]
    assert not any(
        isinstance(dep, dict) and "path" in dep for dep in dependencies.values()
    )
    locked = tomllib.loads((package / "poetry.lock").read_text())
    assert not {"geecs-core", "mysql-connector-python"} & {
        dep["name"] for dep in locked["package"]
    }


def test_scan_stack_import_without_access_library():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class NoAccessLibrary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"geecs_core", "mysql"}:
            raise ImportError("access library forbidden: " + fullname)
sys.meta_path.insert(0, NoAccessLibrary())
from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET
assert LABVIEW_EPOCH_OFFSET == 2_082_844_800
assert not any(name.split(".")[0] in {"geecs_core", "mysql"} for name in sys.modules)
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

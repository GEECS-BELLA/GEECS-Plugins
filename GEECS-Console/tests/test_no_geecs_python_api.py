"""The binding architecture rule: geecs_python_api is NEVER imported.

Two pins: a source grep over every file in the package, and a sys.modules
check after importing every geecs_console module.
"""

import importlib
import pkgutil
import sys
from pathlib import Path

import geecs_console

FORBIDDEN = "geecs_python_api"

PACKAGE_DIR = Path(geecs_console.__file__).parent


def test_source_never_mentions_geecs_python_api():
    offenders = []
    for path in PACKAGE_DIR.rglob("*"):
        if path.suffix in (".py", ".ui") and FORBIDDEN in path.read_text():
            offenders.append(str(path))
    assert not offenders, f"{FORBIDDEN} referenced in: {offenders}"


def test_importing_every_module_never_loads_geecs_python_api():
    for module_info in pkgutil.walk_packages(
        geecs_console.__path__, prefix="geecs_console."
    ):
        importlib.import_module(module_info.name)
    loaded = [name for name in sys.modules if name.startswith(FORBIDDEN)]
    assert not loaded, f"{FORBIDDEN} was imported: {loaded}"

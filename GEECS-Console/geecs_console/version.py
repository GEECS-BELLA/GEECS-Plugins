"""Console version resolution — the source tree wins over installed metadata.

``importlib.metadata`` reflects whatever the last ``poetry install`` recorded,
not the code actually running — a dev checkout that bumped ``pyproject.toml``
without reinstalling would show a stale number in the status bar.  So the
status-bar version prefers the adjacent ``pyproject.toml`` (present in a dev
checkout, absent in a built wheel), then falls back to the installed
distribution metadata, then to ``"unknown"``.
"""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path
from typing import Optional

#: The dev-checkout pyproject adjacent to the package directory.
_PYPROJECT_PATH = Path(__file__).parent.parent / "pyproject.toml"


def console_version(pyproject: Optional[Path] = None) -> str:
    """Return the running console's version string.

    Parameters
    ----------
    pyproject : Path, optional
        The ``pyproject.toml`` to read first; defaults to the one adjacent
        to the ``geecs_console`` package (a dev checkout).  Tests pass a
        tmp path.

    Returns
    -------
    str
        The ``[tool.poetry] version`` from *pyproject* when readable, else
        the installed ``geecs-console`` distribution version, else
        ``"unknown"``.
    """
    path = pyproject if pyproject is not None else _PYPROJECT_PATH
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        version = data["tool"]["poetry"]["version"]
        if isinstance(version, str) and version:
            return version
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError):
        pass
    try:
        return importlib.metadata.version("geecs-console")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

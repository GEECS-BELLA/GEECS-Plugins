"""Type aliases and lightweight primitives shared across the geecs_python_api package."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from threading import Event, Thread
from typing import Any, Optional, Union

VarDict = dict[str, dict[str, Any]]
ExpDict = dict[str, dict[str, dict[str, Any]]]
SysPath = Union[str, bytes, PathLike, Path]
ThreadInfo = tuple[Optional[Thread], Optional[Event]]
AsyncResult = tuple[bool, str, ThreadInfo]


class VarAlias(str):
    """Typed string used as a device variable alias key in state dicts."""

    pass

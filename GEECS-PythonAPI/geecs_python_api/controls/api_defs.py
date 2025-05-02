# from __future__ import annotations
from os import PathLike
from pathlib import Path
from dateutil.parser import parse as dateparse
from threading import Thread, Event
from typing import Optional, Any, Union, NamedTuple


# if TYPE_CHECKING:
VarDict = dict[str, dict[str, Any]]
ExpDict = dict[str, dict[str, dict[str, Any]]]
SysPath = Union[str, bytes, PathLike, Path]
ThreadInfo = tuple[Optional[Thread], Optional[Event]]
AsyncResult = tuple[bool, str, ThreadInfo]


def exec_async(fct, args=(), kwargs=None) -> AsyncResult:
    if kwargs is None:
        kwargs = {}
    return fct(*args, **kwargs, sync=False)

class VarAlias(str):
    pass

import warnings
from geecs_paths_utils.scan_paths import ScanTag as _ScanTag

warnings.warn(
    "geecs_python_api.controls.api_defs.ScanTag has moved to "
    "geecs_paths_utils.geecs_paths; please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

class ScanTag(_ScanTag):
    """Stub for backward compatibility."""
    pass
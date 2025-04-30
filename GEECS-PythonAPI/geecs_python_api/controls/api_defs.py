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
from geecs_paths_utils.geecs_paths import ScanTag as _ScanTag

warnings.warn(
    "geecs_python_api.controls.api_defs.ScanTag has moved to "
    "geecs_paths_utils.geecs_paths; please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

class ScanTag(_ScanTag):
    """Stub for backward compatibility."""
    pass

# class ScanTag(NamedTuple):
#     year: int
#     month: int
#     day: int
#     number: int
#     experiment: Optional[str] = None
#
#
# def month_to_int(month: Union[str, int]) -> int:
#     """ :return: an integer representing the given month """
#     try:
#         month_int = int(month)
#         if 1 <= month_int <= 12:
#             return month_int
#     except ValueError:
#         pass
#
#     if isinstance(month, str):
#         return dateparse(month).month
#     else:
#         raise ValueError(f"'{month}' is not a valid month")

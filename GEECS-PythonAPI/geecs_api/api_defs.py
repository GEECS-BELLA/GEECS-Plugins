# from __future__ import annotations
from os import PathLike
from pathlib import Path
from threading import Thread, Event
from typing import Optional, Any, Union, TYPE_CHECKING


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

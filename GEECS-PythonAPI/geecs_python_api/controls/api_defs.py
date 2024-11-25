# from __future__ import annotations
from os import PathLike
from pathlib import Path
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


class ScanTag(NamedTuple):
    year: int
    month: int
    day: int
    number: int


def month_to_int(month: Union[str, int]) -> int:
    """ :return: an integer representing the given month """
    month_map = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }
    try:
        month_int = int(month)
        if 1 <= month_int <= 12:
            return month_int
    except ValueError:
        pass

    month_lower = month.lower()
    if month_lower in month_map:
        return month_map[month_lower]

    raise ValueError(f"Invalid month: {month}")

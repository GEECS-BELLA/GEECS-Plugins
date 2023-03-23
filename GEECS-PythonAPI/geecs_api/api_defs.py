# from typing import NewType
from threading import Thread, Event
from typing import Optional, Any, Union

# VarAlias = NewType('VarAlias', str)
# VarName = NewType('VarName', str)


VarDict = dict[str, dict[str, Any]]
ExpDict = dict[str, dict[str, dict[str, Any]]]
AsyncResult = tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]


def exec_async(fct, args=(), kwargs=None) -> Union[Any, AsyncResult]:
    if kwargs is None:
        kwargs = {}
    return fct(*args, **kwargs, sync=False)


class VarAlias(str):
    pass

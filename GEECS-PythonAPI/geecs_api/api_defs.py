# from typing import NewType
from threading import Thread, Event
from typing import Optional

# VarAlias = NewType('VarAlias', str)
# VarName = NewType('VarName', str)


AsyncResult = tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]


def exec_async(fun, args=(), kwargs=None) -> AsyncResult:
    return fun(*args, **kwargs, sync=False)


class VarAlias(str):
    pass

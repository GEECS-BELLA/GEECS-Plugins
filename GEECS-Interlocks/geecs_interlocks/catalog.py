"""What the builder offers: the experiment's devices and the variables a check can watch.

:class:`DbCatalog` asks the experiment DB (``geecs_core.db.GeecsDb``); off
the lab network that lookup can block for a long time, so callers run it off
the window's thread. :class:`StaticCatalog` answers from a fixed table: the
demo's devices, or (offline) the devices the current file already names.
Both list only variables a check can read as one value — never images,
arrays or paths.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Protocol

from geecs_schemas import (
    AlignmentCondition,
    ConfigChangeCondition,
    InterlockSet,
    StaleCondition,
    ThresholdCondition,
)

logger = logging.getLogger(__name__)

#: DB variable types a check can read as one value (``None``: untyped, kept).
SCALAR_TYPES = frozenset({"numeric", "choice", "string", "bool", "boolean"})


class Catalog(Protocol):
    """The devices a check can watch, and each one's variables."""

    def devices(self) -> list[str]:
        """Every device, sorted."""

    def variables(self, device: str) -> list[str]:
        """The variables of *device* a check can read, sorted (empty if unknown)."""


class StaticCatalog:
    """A catalog from a fixed ``device → variables`` table."""

    def __init__(self, table: Mapping[str, Iterable[str]]) -> None:
        self._table = {d: sorted(set(v)) for d, v in table.items()}

    def devices(self) -> list[str]:
        """Every device, sorted."""
        return sorted(self._table)

    def variables(self, device: str) -> list[str]:
        """*device*'s variables, sorted."""
        return list(self._table.get(device, ()))

    def merged(self, other: Mapping[str, Iterable[str]]) -> StaticCatalog:
        """A catalog with *other*'s devices and variables added to these."""
        table = {d: set(v) for d, v in self._table.items()}
        for device, variables in other.items():
            table.setdefault(device, set()).update(variables)
        return StaticCatalog(table)


def named_in(interlocks: InterlockSet) -> dict[str, set[str]]:
    """``device → variables`` every check in *interlocks* names (armed or not)."""
    out: dict[str, set[str]] = {}
    for interlock in interlocks.interlocks:
        for c in interlock.conditions:
            if isinstance(c, ThresholdCondition):
                out.setdefault(c.device, set()).add(c.variable)
            elif isinstance(c, AlignmentCondition):
                out.setdefault(c.device, set()).add(c.variable)
                target = c.target_device or c.device
                out.setdefault(target, set()).add(c.target_variable)
            elif isinstance(c, ConfigChangeCondition):
                out.setdefault(c.device, set()).update(c.variables)
            elif isinstance(c, StaleCondition):
                out.setdefault(c.device, set())
    return out


class DbCatalog:
    """The experiment's devices and their scalar variables, from the GEECS DB.

    Every answer is cached; the first call of each kind queries the DB and
    may block (run it off the window's thread).
    """

    def __init__(self, experiment: str) -> None:
        self.experiment = experiment
        self._devices: list[str] | None = None
        self._variables: dict[str, list[str]] = {}

    def devices(self) -> list[str]:
        """The experiment's enabled devices, sorted."""
        if self._devices is None:
            from geecs_core.db import GeecsDb

            self._devices = sorted(GeecsDb.get_experiment_devices(self.experiment))
        return list(self._devices)

    def variables(self, device: str) -> list[str]:
        """*device*'s variables a check can read, sorted."""
        if device not in self._variables:
            from geecs_core.db import GeecsDb

            rows = GeecsDb.get_device_variables(device)
            self._variables[device] = sorted(
                r["name"]
                for r in rows
                if r.get("variabletype") is None
                or str(r["variabletype"]).lower() in SCALAR_TYPES
            )
        return list(self._variables[device])


__all__ = ["SCALAR_TYPES", "Catalog", "DbCatalog", "StaticCatalog", "named_in"]

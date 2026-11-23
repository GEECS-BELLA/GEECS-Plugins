"""``patch_geecs_db``: answer the experiment-wide ``GeecsDb`` queries in a test.

Test code only. It replaces the batched queries every DB-derived roster
issues (devices, their variables, the ``get='yes'`` subscriptions, the
device types) with canned answers, so a suite pins its derivation without
a database.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from geecs_core.db.geecs_db import GeecsDb


class _Patcher(Protocol):
    """The one ``pytest.MonkeyPatch`` method used (pytest stays a test dependency)."""

    def setattr(self, target: object, name: str, value: object) -> None:
        """Set ``target.name`` to ``value`` until the test ends."""


#: The four experiment-wide queries, keyed by this helper's keyword names.
_QUERIES = {
    "devices": "get_experiment_devices",
    "variables": "get_experiment_device_variables",
    "subscribed": "get_subscribed_variables",
    "types": "get_experiment_device_types",
}


def patch_geecs_db(
    monkeypatch: _Patcher,
    *,
    devices: object = None,
    variables: object = None,
    subscribed: object = None,
    types: object = None,
    before: Callable[[], None] | None = None,
) -> None:
    """Replace the experiment-wide ``GeecsDb`` queries with canned answers.

    Each keyword left ``None`` leaves its query untouched (a test that must
    prove a query is never issued patches it to raise, itself). A patched
    query accepts any experiment and keyword arguments and returns its
    answer object as given.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The test's monkeypatch fixture; the patches end with the test.
    devices : dict, optional
        ``get_experiment_devices``: device name -> ``(host, port)``.
    variables : dict, optional
        ``get_experiment_device_variables``: device name -> variable rows.
    subscribed : dict, optional
        ``get_subscribed_variables``: device name -> ``get='yes'`` names.
    types : dict, optional
        ``get_experiment_device_types``: device name -> device type.
    before : callable, optional
        Called before each patched query answers; raise from it to play an
        unreachable DB.
    """
    answers = {
        "devices": devices,
        "variables": variables,
        "subscribed": subscribed,
        "types": types,
    }
    for key, answer in answers.items():
        if answer is None:
            continue

        def query(
            cls: type, experiment: str, *_args: object, _answer=answer, **_kw: object
        ) -> object:
            if before is not None:
                before()
            return _answer

        monkeypatch.setattr(GeecsDb, _QUERIES[key], classmethod(query))

"""Device/variable completion source for editor forms (offline-safe).

The scan-variable editor's device and variable fields carry ``QCompleter``
popups.  Their word lists come from a :class:`CompletionsProvider` — one
blocking call returning ``{device: [variable, ...]}`` — which the editor
dispatches to a short-lived daemon thread (the package's no-QThread rule)
and marshals back through a queued Qt signal, so a slow or unreachable DB
never blocks the GUI thread.

- :class:`EmptyCompletions` — the offline/test default: no suggestions,
  returns instantly.
- :class:`GeecsDbCompletions` — the real provider: one
  ``GeecsDb.get_experiment_device_variables`` query (imported lazily inside
  the call, keeping this module import-safe offline), cached after the first
  fetch; any failure — no MySQL, no credentials, unreachable host — degrades
  to empty with a log line, never an exception.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class CompletionsProvider(Protocol):
    """Anything that can supply device → variable-name completions."""

    def device_variables(self) -> dict[str, list[str]]:
        """Return ``{device: [variable name, ...]}``; may block, never raises."""
        ...


class EmptyCompletions:
    """No completions — the offline and hermetic-test default."""

    def device_variables(self) -> dict[str, list[str]]:
        """Return no completions.

        Returns
        -------
        dict
            Always empty.
        """
        return {}


class GeecsDbCompletions:
    """DB-backed completions for one experiment (cached, failure-tolerant).

    Parameters
    ----------
    experiment : str
        GEECS experiment name to enumerate devices/variables for.
    """

    def __init__(self, experiment: str) -> None:
        self._experiment = experiment
        self._cache: Optional[dict[str, list[str]]] = None

    def device_variables(self) -> dict[str, list[str]]:
        """Fetch (once) ``{device: [variable name, ...]}`` from ``GeecsDb``.

        Returns
        -------
        dict
            Sorted variable names per device; empty when the experiment is
            unset or the DB is unreachable (logged, never raised).  Cached
            after the first successful or failed attempt.
        """
        if self._cache is not None:
            return self._cache
        self._cache = {}
        if not self._experiment:
            return self._cache
        try:
            from geecs_ca_gateway.db.geecs_db import GeecsDb

            per_device = GeecsDb.get_experiment_device_variables(self._experiment)
            self._cache = {
                device: sorted({str(row["name"]) for row in rows if "name" in row})
                for device, rows in per_device.items()
            }
        except Exception as exc:  # noqa: BLE001 — degrade to empty offline
            logger.info(
                "device completions unavailable for %r: %s", self._experiment, exc
            )
        return self._cache

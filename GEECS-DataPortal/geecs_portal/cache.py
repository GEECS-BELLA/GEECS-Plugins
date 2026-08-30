"""In-memory caches: seamless navigation instead of per-shot round trips.

The owner's doctrine amendment (2026-08-29): a diagnostic's full data for
one scan is ~100s of MB — trivial against server RAM, while every NAS/
Tiled round trip is the real cost.  So the portal now loads *within-scan*
data eagerly and keeps it: the scope doc's "lazy loading is a hard rule"
still governs *across-scan* eagerness (never thumbnail whole days), but
navigating one run's shots must not re-read the share per click.

Two caches, both process-local and thread-safe (FastAPI threadpool):

- :class:`CachingScanCatalog` — a delegating ``ScanCatalog`` that caches
  ``load_run``.  A **completed** run (stop doc present) is immutable, so
  its detail is kept until LRU eviction; a still-running run is cached
  for a short TTL only.  Kills the full-event-table Tiled read that every
  ``plot.png``/``image.png`` request used to repeat.
- :class:`ShotDataCache` — per ``(uid, device)`` pixel data for
  **completed** runs only: a stack device's whole ``/frames`` array in
  one HDF5 read on first touch; a native device's decoded per-shot
  arrays, warmed by a background thread walking the run's event rows.
  Rendering to PNG then happens from memory.  Bytes-bounded LRU.

Invalidation is deliberately trivial: completed runs never change (the
same property the immutable cache headers rely on); running runs are
never entered into ``ShotDataCache`` and expire from the detail cache in
seconds.  Everything here is read-only over the scans path (repo
scan-folder invariant).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Detail-cache size (completed runs kept, LRU).
_DETAIL_CACHE_RUNS = 8

#: Seconds a still-running run's detail stays fresh.
_RUNNING_TTL_S = 5.0

#: Byte budget for cached pixel data across all (uid, device) entries.
_PIXEL_BUDGET_BYTES = 1_500_000_000

#: No single (uid, device) entry may exceed this fraction of the budget —
#: the budget must bound the cache even while one entry is warming, and
#: two doctrine-sized diagnostics must coexist without evicting each other.
_ENTRY_CAP_FRACTION = 3

#: At most this many background warm threads at once (the NAS path is the
#: bottleneck — warms must not compete with a live experiment's writes).
_MAX_WARM_THREADS = 2


class CachingScanCatalog:
    """A delegating ``ScanCatalog`` that caches ``load_run`` results.

    Wraps any catalog implementation; every method routes to the wrapped
    instance (never back through a registry).  ``list_runs`` and
    ``probe`` pass straight through — the day listing stays live.

    Parameters
    ----------
    catalog : ScanCatalog
        The wrapped implementation.
    clock : callable, optional
        Monotonic time source (tests inject a fake).
    """

    def __init__(self, catalog, clock: Callable[[], float] = time.monotonic):
        self._catalog = catalog
        self._clock = clock
        self._lock = threading.Lock()
        self._details: OrderedDict[str, tuple[object, float]] = OrderedDict()

    def probe(self):
        """Delegate the liveness probe (never cached)."""
        return self._catalog.probe()

    def list_runs(self, experiment: str, day):
        """Delegate the day listing (never cached — new scans must appear)."""
        return self._catalog.list_runs(experiment, day)

    def load_run(self, uid: str):
        """Load one run, serving completed runs from the cache.

        Raises ``KeyError`` for an unknown uid (the protocol's not-found
        contract — misses are never cached, so a run that appears later
        loads normally).
        """
        now = self._clock()
        with self._lock:
            hit = self._details.get(uid)
            if hit is not None:
                detail, fetched_at = hit
                complete = bool(detail.summary.exit_status)
                if complete or (now - fetched_at) < _RUNNING_TTL_S:
                    self._details.move_to_end(uid)
                    return detail
                del self._details[uid]
        detail = self._catalog.load_run(uid)
        with self._lock:
            self._details[uid] = (detail, now)
            self._details.move_to_end(uid)
            while len(self._details) > _DETAIL_CACHE_RUNS:
                self._details.popitem(last=False)
        return detail


class ShotDataCache:
    """Bytes-bounded LRU of per-``(uid, device)`` pixel data.

    Entries exist only for completed runs.  A stack entry holds the whole
    frames array plus its millisecond-key index map (one HDF5 read); a
    native entry accumulates decoded per-shot arrays — served shots
    immediately, the rest by :meth:`warm_native` on a background thread.

    Parameters
    ----------
    budget_bytes : int, optional
        Total pixel-byte budget; least-recently-used entries evict.
    """

    def __init__(self, budget_bytes: int = _PIXEL_BUDGET_BYTES):
        self._budget = budget_bytes
        self._entry_cap = max(1, budget_bytes // _ENTRY_CAP_FRACTION)
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._warming: set[tuple[str, str]] = set()
        self._warmed: set[tuple[str, str]] = set()

    def _entry_bytes(self, entry: dict) -> int:
        frames = entry.get("frames")
        if frames is not None:
            return int(frames.nbytes)
        return int(sum(a.nbytes for a in entry.get("shots", {}).values()))

    def _evict_over_budget(self) -> None:
        total = sum(self._entry_bytes(e) for e in self._entries.values())
        while total > self._budget and len(self._entries) > 1:
            _, evicted = self._entries.popitem(last=False)
            total -= self._entry_bytes(evicted)

    def stack_frames(
        self, key: tuple[str, str], stack_path: Path
    ) -> Optional[tuple[dict, np.ndarray]]:
        """The stack's ``(index_map, frames)`` — whole array, one read.

        First touch reads every frame (the owner's eager-within-scan
        doctrine: ~100s MB, one open); navigation then never reopens the
        file.  Admission is gated: the stack must be **finalized** (the
        daemon's stop-doc handler stamps ``finalized=True`` *after* the
        Tiled stop document lands, so an un-finalized file may still be
        missing tail frames — caching it would 404 those shots from
        memory forever) and must fit the per-entry cap (the budget must
        bound the cache even mid-warm).  ``None`` means "don't cache" —
        the caller serves from disk per shot as before.

        Parameters
        ----------
        key : tuple of (str, str)
            ``(uid, device)``.
        stack_path : Path
            The validated stack file.

        Returns
        -------
        tuple of (dict, numpy.ndarray) or None
            The keep-first millisecond-key → index map and the frames,
            or ``None`` when the stack is not admissible.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and "frames" in entry:
                self._entries.move_to_end(key)
                return entry["index_map"], entry["frames"]
        import h5py

        from geecs_data_utils.io.scan_stack import (
            LABVIEW_EPOCH_OFFSET,
            stack_frame_index_map,
        )

        with h5py.File(stack_path, "r") as f:
            if not bool(f.attrs.get("finalized", False)):
                return None
            dataset = f["frames"]
            size = int(np.prod(dataset.shape)) * dataset.dtype.itemsize
            if size > self._entry_cap:
                logger.info(
                    "stack %s (%d bytes) exceeds the per-entry cap — serving "
                    "per shot from disk",
                    stack_path,
                    size,
                )
                return None
            stamps = np.asarray(f["acq_timestamp"][:], dtype=float)
            stamps = stamps + LABVIEW_EPOCH_OFFSET
            frames = np.asarray(dataset[:])
        index_map = stack_frame_index_map(stamps)
        with self._lock:
            self._entries[key] = {"index_map": index_map, "frames": frames}
            self._entries.move_to_end(key)
            self._evict_over_budget()
        return index_map, frames

    def stack_entry(self, key: tuple[str, str]) -> Optional[tuple[dict, np.ndarray]]:
        """The cached ``(index_map, frames)`` for *key*, or ``None``.

        A hit serves shots with zero filesystem access (no listing, no
        open) — the whole point of the eager load.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or "frames" not in entry:
                return None
            self._entries.move_to_end(key)
            return entry["index_map"], entry["frames"]

    def native_shot(self, key: tuple[str, str], shot: int) -> Optional[np.ndarray]:
        """A cached decoded array for *shot*, or ``None`` on a miss."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry.get("shots", {}).get(shot)

    def store_native_shot(
        self, key: tuple[str, str], shot: int, array: np.ndarray
    ) -> bool:
        """Keep one decoded shot array (creates the entry as needed).

        Returns
        -------
        bool
            False when the entry is at its per-entry cap (the shot is not
            stored — the budget bounds the cache even mid-warm).
        """
        with self._lock:
            entry = self._entries.setdefault(key, {"shots": {}})
            if self._entry_bytes(entry) + array.nbytes > self._entry_cap:
                return False
            entry.setdefault("shots", {})[shot] = array
            self._entries.move_to_end(key)
            self._evict_over_budget()
            return True

    def _entry_at_cap(self, key: tuple[str, str]) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            return entry is not None and self._entry_bytes(entry) >= self._entry_cap

    def warm_native(
        self,
        key: tuple[str, str],
        load_shot: Callable[[int], None],
        shots: list[int],
        *,
        synchronous: bool = False,
    ) -> None:
        """Background-load a native device's remaining shots, once per key.

        *load_shot* performs its own storing (the normal loader path does,
        via :meth:`store_native_shot`); already-cached shots are skipped.
        Best-effort: individual failures are logged at debug and skipped.

        Parameters
        ----------
        key : tuple of (str, str)
            ``(uid, device)``.
        load_shot : callable
            Loads (and stores) one 1-based shot.
        shots : list of int
            The run's shot numbers to warm.
        synchronous : bool, optional
            Run inline instead of on a daemon thread (tests only).
        """
        with self._lock:
            if key in self._warming or key in self._warmed:
                # Once per key per process: a device with genuinely
                # missing shot files must not re-probe every hole on
                # every page view.
                return
            if not synchronous and len(self._warming) >= _MAX_WARM_THREADS:
                return  # NAS throttle; a later view reschedules
            self._warming.add(key)

        def _run() -> None:
            try:
                for shot in shots:
                    if self._entry_at_cap(key):
                        logger.info("warm %s stopped at the per-entry cap", key)
                        break
                    if self.native_shot(key, shot) is not None:
                        continue
                    try:
                        load_shot(shot)
                    except Exception as exc:  # noqa: BLE001 — warming is best-effort
                        logger.debug("warm %s shot %d failed: %s", key, shot, exc)
            finally:
                with self._lock:
                    self._warming.discard(key)
                    self._warmed.add(key)

        if synchronous:
            _run()
            return
        threading.Thread(target=_run, name=f"warm-{key[1]}", daemon=True).start()

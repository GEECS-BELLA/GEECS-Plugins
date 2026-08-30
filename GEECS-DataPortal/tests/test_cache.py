"""Cache tests — the eager within-scan doctrine, pinned hermetically.

The strongest pins delete the underlying file after the first (caching)
access and assert the shot still serves — proving zero filesystem access
on the hit path, not just a faster one.
"""

from __future__ import annotations


import pytest
from fastapi.testclient import TestClient

from geecs_portal import resources
from geecs_portal.app import create_app
from geecs_portal.cache import CachingScanCatalog, ShotDataCache

from test_app import _LV, TEST_DAY, FakeCatalog, _detail
from test_resources import scan_folder  # noqa: F401 — fixture import


class _CountingCatalog(FakeCatalog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_calls: list[str] = []

    def load_run(self, uid: str):
        self.load_calls.append(uid)
        return super().load_run(uid)


class TestCachingScanCatalog:
    def test_completed_run_loads_once(self):
        inner = _CountingCatalog()
        catalog = CachingScanCatalog(inner)
        first = catalog.load_run("uid-002")
        second = catalog.load_run("uid-002")
        assert first is second
        assert inner.load_calls == ["uid-002"]

    def test_running_run_expires_after_ttl(self):
        from dataclasses import replace

        inner = _CountingCatalog()
        detail = _detail(9)
        running = type(detail)(
            summary=replace(detail.summary, exit_status=None),
            start_doc=detail.start_doc,
            stop_doc=None,
            data=detail.data,
        )
        inner.details["uid-009"] = running
        now = [0.0]
        catalog = CachingScanCatalog(inner, clock=lambda: now[0])
        catalog.load_run("uid-009")
        catalog.load_run("uid-009")  # within TTL: cached
        assert inner.load_calls == ["uid-009"]
        now[0] = 10.0  # past the TTL: refetch (new shots may exist)
        catalog.load_run("uid-009")
        assert inner.load_calls == ["uid-009", "uid-009"]

    def test_unknown_uid_raises_and_is_not_cached(self):
        inner = _CountingCatalog()
        catalog = CachingScanCatalog(inner)
        with pytest.raises(KeyError):
            catalog.load_run("nope")
        with pytest.raises(KeyError):
            catalog.load_run("nope")
        assert inner.load_calls == ["nope", "nope"]

    def test_listing_and_probe_pass_through_uncached(self):
        inner = _CountingCatalog()
        catalog = CachingScanCatalog(inner)
        catalog.list_runs("Undulator", TEST_DAY)
        assert inner.listed == ("Undulator", TEST_DAY)
        assert catalog.probe().ok is True


class TestShotDataCache:
    def test_stack_hit_needs_no_filesystem(self, scan_folder):  # noqa: F811
        cache = ShotDataCache()
        key = ("uid-002", "UC_StackCam")
        stack = scan_folder / "UC_StackCam" / "UC_StackCam.h5"
        first = resources.load_shot_image(
            scan_folder,
            "UC_StackCam",
            2,
            acq_timestamp=_LV + 2.0,
            data_cache=cache,
            cache_key=key,
        )
        assert first.kind == "stack"
        stack.unlink()  # the proof: a hit never touches the share
        again = resources.load_shot_image(
            scan_folder,
            "UC_StackCam",
            2,
            acq_timestamp=_LV + 2.0,
            data_cache=cache,
            cache_key=key,
        )
        assert again.kind == "stack"
        assert again.png == first.png
        # exact-key refusal still enforced from memory
        miss = resources.load_shot_image(
            scan_folder,
            "UC_StackCam",
            2,
            acq_timestamp=_LV + 9.0,
            data_cache=cache,
            cache_key=key,
        )
        assert miss.kind == "missing"

    def test_native_timestamp_shot_cached_ordinal_never(self, scan_folder):  # noqa: F811
        cache = ShotDataCache()
        key = ("uid-002", "cam")
        joined = resources.load_shot_image(
            scan_folder,
            "cam",
            2,
            acq_timestamp=_LV + 2.0,
            data_cache=cache,
            cache_key=key,
        )
        assert joined.kind == "native"
        assert cache.native_shot(key, 2) is not None
        # ordinal (listing-order) resolution must never enter the cache
        ordinal = resources.load_shot_image(
            scan_folder, "cam", 3, data_cache=cache, cache_key=key
        )
        assert ordinal.kind == "native"
        assert cache.native_shot(key, 3) is None
        # and the cached shot serves after the file disappears
        (scan_folder / "cam" / f"cam_{_LV + 2.0:.3f}.png").unlink()
        again = resources.load_shot_image(
            scan_folder,
            "cam",
            2,
            acq_timestamp=_LV + 2.0,
            data_cache=cache,
            cache_key=key,
        )
        assert again.kind == "native" and again.png == joined.png

    def test_warm_native_loads_every_joined_shot(self, scan_folder):  # noqa: F811
        cache = ShotDataCache()
        key = ("uid-002", "cam")

        def _load(shot: int) -> None:
            resources.load_shot_image(
                scan_folder,
                "cam",
                shot,
                acq_timestamp=_LV + float(shot),
                data_cache=cache,
                cache_key=key,
            )

        cache.warm_native(key, _load, [1, 2, 3], synchronous=True)
        assert all(cache.native_shot(key, s) is not None for s in (1, 2, 3))

    def test_budget_evicts_least_recently_used(self):
        import numpy as np

        # budget for ~4 shots; per-entry cap = budget // 3 ≈ 1.3 shots
        shot_bytes = np.zeros((16, 16)).nbytes
        cache = ShotDataCache(budget_bytes=4 * shot_bytes)
        assert cache.store_native_shot(("u", "a"), 1, np.zeros((16, 16)))
        assert cache.store_native_shot(("u", "b"), 1, np.zeros((16, 16)))
        assert cache.store_native_shot(("u", "c"), 1, np.zeros((16, 16)))
        assert cache.store_native_shot(("u", "d"), 1, np.zeros((16, 16)))
        assert cache.store_native_shot(("u", "e"), 1, np.zeros((16, 16)))
        # over budget: the oldest evicted, the newest kept
        assert cache.native_shot(("u", "a"), 1) is None
        assert cache.native_shot(("u", "e"), 1) is not None

    def test_per_entry_cap_bounds_a_single_warming_entry(self):
        import numpy as np

        shot_bytes = np.zeros((16, 16)).nbytes
        cache = ShotDataCache(budget_bytes=6 * shot_bytes)  # cap = 2 shots
        key = ("u", "big")
        assert cache.store_native_shot(key, 1, np.zeros((16, 16)))
        assert cache.store_native_shot(key, 2, np.zeros((16, 16)))
        # the third shot would exceed the per-entry cap: refused, and the
        # warm loop's cap check would stop the thread here too
        assert cache.store_native_shot(key, 3, np.zeros((16, 16))) is False
        assert cache.native_shot(key, 3) is None
        assert cache._entry_at_cap(key)

    def test_unfinalized_stack_is_never_cached(self, scan_folder):  # noqa: F811
        # The stop doc lands BEFORE the daemon finalizes the stack (a
        # seconds-wide race): an un-finalized file may miss tail frames,
        # so it must serve from disk per shot, never enter the cache.
        import h5py

        stack = scan_folder / "UC_StackCam" / "UC_StackCam.h5"
        with h5py.File(stack, "a") as handle:
            del handle.attrs["finalized"]
        cache = ShotDataCache()
        key = ("uid-002", "UC_StackCam")
        result = resources.load_shot_image(
            scan_folder,
            "UC_StackCam",
            2,
            acq_timestamp=_LV + 2.0,
            data_cache=cache,
            cache_key=key,
        )
        assert result.kind == "stack"  # still serves — from disk
        assert cache.stack_entry(key) is None  # but never cached

    def test_oversize_stack_serves_from_disk_uncached(self, scan_folder):  # noqa: F811
        cache = ShotDataCache(budget_bytes=8)  # cap far below the stack
        key = ("uid-002", "UC_StackCam")
        result = resources.load_shot_image(
            scan_folder,
            "UC_StackCam",
            2,
            acq_timestamp=_LV + 2.0,
            data_cache=cache,
            cache_key=key,
        )
        assert result.kind == "stack"
        assert cache.stack_entry(key) is None

    def test_warm_runs_once_per_key_per_process(self, scan_folder):  # noqa: F811
        cache = ShotDataCache()
        key = ("uid-002", "cam")
        calls: list[int] = []

        def _load(shot: int) -> None:
            calls.append(shot)

        cache.warm_native(key, _load, [1, 2], synchronous=True)
        cache.warm_native(key, _load, [1, 2], synchronous=True)  # no re-probe
        assert calls == [1, 2]

    def test_threaded_warm_completes(self, scan_folder):  # noqa: F811
        import time as _time

        cache = ShotDataCache()
        key = ("uid-002", "cam")

        def _load(shot: int) -> None:
            resources.load_shot_image(
                scan_folder,
                "cam",
                shot,
                acq_timestamp=_LV + float(shot),
                data_cache=cache,
                cache_key=key,
            )

        cache.warm_native(key, _load, [1, 2, 3])  # real daemon thread
        deadline = _time.monotonic() + 5.0
        while _time.monotonic() < deadline:
            if all(cache.native_shot(key, s) is not None for s in (1, 2, 3)):
                break
            _time.sleep(0.02)
        assert all(cache.native_shot(key, s) is not None for s in (1, 2, 3))


class TestCachedRoutes:
    def _client(self, scan_folder, exit_status="success"):  # noqa: F811
        catalog = FakeCatalog()
        detail = _detail(2)
        if exit_status is None:
            from geecs_data_utils.tiled_catalog import RunDetail, summary_from_metadata

            detail = RunDetail(
                summary=summary_from_metadata(
                    detail.start_doc["uid"], detail.start_doc, None
                ),
                start_doc=detail.start_doc,
                stop_doc=None,
                data=detail.data,
            )
        detail.start_doc["scan_folder"] = str(scan_folder)
        catalog.details["uid-002"] = detail
        return TestClient(create_app(catalog))

    def test_completed_run_serves_after_file_vanishes(self, scan_folder):  # noqa: F811
        client = self._client(scan_folder)
        first = client.get("/run/uid-002/image.png?device=UC_StackCam&shot=2")
        assert first.status_code == 200
        (scan_folder / "UC_StackCam" / "UC_StackCam.h5").unlink()
        again = client.get("/run/uid-002/image.png?device=UC_StackCam&shot=2")
        assert again.status_code == 200
        assert again.content == first.content

    def test_running_run_never_caches(self, scan_folder):  # noqa: F811
        client = self._client(scan_folder, exit_status=None)
        first = client.get("/run/uid-002/image.png?device=UC_StackCam&shot=2")
        assert first.status_code == 200
        (scan_folder / "UC_StackCam" / "UC_StackCam.h5").unlink()
        again = client.get("/run/uid-002/image.png?device=UC_StackCam&shot=2")
        assert again.status_code == 404  # live runs always read the share

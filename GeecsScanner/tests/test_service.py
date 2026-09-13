"""The service's locking: a slow verb never delays a read."""

from __future__ import annotations

import threading
import time

from geecs_scanner.service import ProgressCache, ScannerService
from geecs_scanner.service.demo import DemoQueueClient, DemoResolver, demo_preflight
from geecs_scanner.service.models import SubmitIn, VerbIn


class _SlowStop(DemoQueueClient):
    """A manager whose graceful stop takes a while, like the real one (up to 120 s)."""

    def stop_scan(self) -> tuple[bool, str]:
        """Sleep inside the verb, then stop."""
        time.sleep(0.6)
        return super().stop_scan()


def _service(manager: DemoQueueClient, streams: ProgressCache) -> ScannerService:
    return ScannerService(
        manager,
        DemoResolver(),
        experiment="Demo",
        identity="t",
        streams=streams,
        preflight=demo_preflight,
    )


def test_a_slow_stop_does_not_block_status_or_queue() -> None:
    streams = ProgressCache()
    manager = _SlowStop(streams, period=0.0, user="t")
    service = _service(manager, streams)
    preset = DemoResolver().resolve_preset("eb_align_1hz").model_dump(mode="json")
    service.submit(SubmitIn(preset=preset, acknowledged=["gateway_liveness"]))
    done = threading.Event()
    result: dict = {}

    def stop() -> None:
        result["out"] = service.stop(VerbIn())
        done.set()

    threading.Thread(target=stop, daemon=True).start()
    time.sleep(0.05)  # the stop is now inside the client, holding the verb lock
    t0 = time.monotonic()
    st = service.status()
    q = service.queue()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.2, f"a read waited {elapsed:.2f}s behind a stop"
    assert st.connected and q.running is not None
    assert done.wait(2.0) and result["out"].ok is True


def test_verbs_hold_the_one_lock() -> None:
    streams = ProgressCache()
    manager = _SlowStop(streams, period=0.0, user="t")
    service = _service(manager, streams)
    preset = DemoResolver().resolve_preset("eb_align_1hz").model_dump(mode="json")
    service.submit(SubmitIn(preset=preset, acknowledged=["gateway_liveness"]))
    threading.Thread(target=lambda: service.stop(VerbIn()), daemon=True).start()
    time.sleep(0.05)
    t0 = time.monotonic()
    service.clear()  # a second verb waits for the first
    assert time.monotonic() - t0 > 0.3

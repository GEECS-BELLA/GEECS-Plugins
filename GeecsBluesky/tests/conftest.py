"""Shared pytest configuration for GeecsBluesky tests."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Bound fake-server tests so socket/thread leaks fail fast."""
    timeout = pytest.mark.timeout(30)
    for item in items:
        if item.get_closest_marker("fake_server") and not item.get_closest_marker(
            "timeout"
        ):
            item.add_marker(timeout)


@pytest.fixture(autouse=True)
def _stop_run_engine_loops_left_by_the_test(request: pytest.FixtureRequest):
    """Stop the event-loop threads a test's ``RunEngine()`` instances leave behind.

    Every ``RunEngine()`` starts a daemon thread running its own asyncio loop
    forever; nothing in bluesky stops it, so a suite with ~60 inline
    constructions accumulates 100+ live loops
    and the whole process crawls (#812: a 1.4 s test taking the full 180 s
    timeout at the midpoint of the run, on macOS).  bluesky records
    loop → thread in ``_ensure_event_loop_running.loop_to_thread``; this
    stops every loop that appeared during the test, drains what it still had
    pending (a cancelled pacer, a paused RunEngine) the way
    ``asyncio.Runner.close`` does — so nothing is destroyed while pending and
    no stray traceback lands after the pytest summary — then closes it.
    Loops created before the test (module- or session-level engines) are
    left alone.  A loop whose thread will not stop within 5 s is left
    running, as before this fixture, and reported so the leak is visible.
    """
    import asyncio
    import warnings

    from bluesky.run_engine import _ensure_event_loop_running

    registry = _ensure_event_loop_running.loop_to_thread
    before = set(registry.keys())
    yield
    for loop in list(registry.keys()):
        if loop in before:
            continue
        thread = registry.get(loop)
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        if loop.is_running():
            warnings.warn(
                f"{request.node.nodeid}: a RunEngine loop thread did not stop "
                "within 5 s and was left running (a blocking callback on the "
                "loop?)",
                RuntimeWarning,
                stacklevel=1,
            )
            continue
        if loop.is_closed():
            continue
        # Drain before closing: cancel what is still pending and let it finish
        # cancelling (the test thread may drive a stopped loop).
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

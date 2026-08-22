"""Pin the one-persistent-loop contract of ``devices/ca/oneshot``.

The module exists to close the aioca loop-cache leak: a per-call
``asyncio.run()`` gives every read a fresh event loop, re-creating the CA
channel each time and stranding the previous loop's cache entries.  The
fix is only real if every read runs on the SAME loop — that is what these
tests pin (a revert to per-call ``asyncio.run`` fails them).
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types

import pytest


@pytest.fixture
def fake_aioca(monkeypatch):
    """Install a recording ``aioca`` whose caget notes its loop and thread.

    The recorded entry keeps a strong reference to the **loop object**
    (never its ``id()`` — a freed loop's address can be reused by the
    next ``asyncio.run()`` loop, which made an id-based assertion pass
    with the bug present; reviewer-caught on the extraction PR).
    """
    module = types.ModuleType("aioca")
    calls: list[tuple[asyncio.AbstractEventLoop, str]] = []

    async def caget(pv, timeout=None, datatype=None):
        calls.append((asyncio.get_running_loop(), threading.current_thread().name))
        if pv == "dead:pv":
            raise RuntimeError("no channel")
        return "Connected" if datatype is str else 42

    module.caget = caget
    monkeypatch.setitem(sys.modules, "aioca", module)
    return calls


def test_reads_share_one_persistent_loop(fake_aioca):
    from geecs_bluesky.devices.ca.oneshot import caget_once

    assert caget_once("some:pv", timeout=1.0) == 42
    assert caget_once("other:pv", timeout=1.0) == 42
    # THE contract: both reads ran on the same loop object, on the shared
    # reader thread.  A per-call asyncio.run() necessarily allocates a
    # second loop (the recorded strong references keep the first alive,
    # so no address reuse can alias them) and runs on the caller's
    # thread, failing both assertions.
    loops = {loop for loop, _thread in fake_aioca}
    threads = {thread for _loop, thread in fake_aioca}
    assert len(fake_aioca) == 2
    assert len(loops) == 1
    assert threads == {"geecs-ca-oneshot"}


def test_datatype_is_forwarded(fake_aioca):
    from geecs_bluesky.devices.ca.oneshot import caget_once

    assert caget_once("enum:pv", timeout=1.0, datatype=str) == "Connected"


def test_caget_once_raises_and_try_variant_fails_open(fake_aioca):
    from geecs_bluesky.devices.ca.oneshot import caget_once, try_caget_once

    with pytest.raises(RuntimeError, match="no channel"):
        caget_once("dead:pv", timeout=1.0)
    assert try_caget_once("dead:pv", timeout=1.0) is None

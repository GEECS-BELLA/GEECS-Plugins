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
import types

import pytest


@pytest.fixture
def fake_aioca(monkeypatch):
    """Install a recording ``aioca`` whose caget notes its running loop."""
    module = types.ModuleType("aioca")
    loop_ids: list[int] = []

    async def caget(pv, timeout=None, datatype=None):
        loop_ids.append(id(asyncio.get_running_loop()))
        if pv == "dead:pv":
            raise RuntimeError("no channel")
        return "Connected" if datatype is str else 42

    module.caget = caget
    monkeypatch.setitem(sys.modules, "aioca", module)
    return loop_ids


def test_reads_share_one_persistent_loop(fake_aioca):
    from geecs_bluesky.devices.ca.oneshot import caget_once

    assert caget_once("some:pv", timeout=1.0) == 42
    assert caget_once("other:pv", timeout=1.0) == 42
    # THE contract: both reads ran on the same loop — a per-call
    # asyncio.run() would record two distinct loop ids here.
    assert len(fake_aioca) == 2
    assert len(set(fake_aioca)) == 1


def test_datatype_is_forwarded(fake_aioca):
    from geecs_bluesky.devices.ca.oneshot import caget_once

    assert caget_once("enum:pv", timeout=1.0, datatype=str) == "Connected"


def test_caget_once_raises_and_try_variant_fails_open(fake_aioca):
    from geecs_bluesky.devices.ca.oneshot import caget_once, try_caget_once

    with pytest.raises(RuntimeError, match="no channel"):
        caget_once("dead:pv", timeout=1.0)
    assert try_caget_once("dead:pv", timeout=1.0) is None

"""The test doubles may not outlive the seams they stand in for.

0.9.0 was written because four MCP tools called ``qs_client`` methods the
native-Bluesky rebuild had removed — and the suite was green throughout,
because ``_FakeClient`` still defined all four.  The review of that PR
then found the identical drift a second time, one file over:
``list_scan_configs("save_sets")`` routed to
``ConfigsRepoResolver.list_save_sets``, also removed by the rebuild, also
still on ``_FakeResolver``.  Two independent copies of one bug class.

So the assertions live here, once, rather than beside each fake: every
public method a double promises must exist on the real collaborator, and
every field a fabricated ``status()`` exposes must exist on
``QueueStatus``.  A removal upstream then fails *here*, naming the
double, instead of at an agent's first tool call.

**Known gap** (review finding 2, waived deliberately): this pins *names*,
not signatures or return shapes.  A ``request_pause()`` that returned a
``SubmitResult`` instead of ``(ok, message)`` would pass and still break
``_pause_scan_impl``'s tuple unpack.  Pinning shapes wants a typed
conformance helper next to the protocol itself in GeecsBluesky (so the
scanner's ``DemoQueueClient`` gets it too) — a cross-package change, not
this PR's.
"""

from __future__ import annotations

import dataclasses

import pytest

from geecs_bluesky.config_resolver import ConfigsRepoResolver
from geecs_bluesky.qs_client import QueueClient, QueueStatus
from test_control_tools import _FakeClient as _ControlFakeClient
from test_read_tools import _FakeClient as _ReadFakeClient, _FakeResolver


def _public_methods(cls: type) -> set[str]:
    return {
        name
        for name in vars(cls)
        if callable(getattr(cls, name)) and not name.startswith("_")
    }


@pytest.mark.parametrize(
    "double, real, label",
    [
        (_ControlFakeClient, QueueClient, "test_control_tools._FakeClient"),
        (_ReadFakeClient, QueueClient, "test_read_tools._FakeClient"),
        (_FakeResolver, ConfigsRepoResolver, "test_read_tools._FakeResolver"),
    ],
)
def test_double_promises_no_method_the_real_collaborator_lacks(double, real, label):
    missing = sorted(n for n in _public_methods(double) if not hasattr(real, n))
    assert not missing, (
        f"{label} promises {missing}, which {real.__name__} does not have — "
        "either the real collaborator lost something the tools still call, or "
        "the double grew a method of its own"
    )


@pytest.mark.parametrize(
    "double, label",
    [
        (_ControlFakeClient, "test_control_tools._FakeClient"),
        (_ReadFakeClient, "test_read_tools._FakeClient"),
    ],
)
def test_fabricated_status_promises_no_field_queue_status_lacks(double, label):
    real = {f.name for f in dataclasses.fields(QueueStatus)}
    fabricated = set(vars(double().status()))
    extra = sorted(fabricated - real)
    assert not extra, (
        f"{label}.status() fabricates {extra}, which QueueStatus does not have — "
        "a renamed field leaves the suite green and every tool reading it "
        "raising AttributeError in production"
    )


def test_the_tool_groups_partition_every_registered_tool():
    """Every registered tool sits in exactly one safety group.

    The two registration tests assert *group ⊆ registered*, so a tool
    added without a ``tool_names`` group entry appears in neither
    ``allow`` nor ``ask``/``write_tools`` — it would ship ungated and
    unlisted.  Assert the other direction too, and that the groups do
    not overlap (a tool in both READ_TOOLS and QUEUE_TOOLS has no
    single safety class).
    """
    import anyio

    from geecs_mcp import tool_names
    from geecs_mcp.server import create_server

    registered = {tool.name for tool in anyio.run(create_server().list_tools)}
    groups = {
        "READ_TOOLS": set(tool_names.READ_TOOLS),
        "QUEUE_TOOLS": set(tool_names.QUEUE_TOOLS),
        "STOP_TOOLS": set(tool_names.STOP_TOOLS),
    }
    classified = set().union(*groups.values())

    assert not (registered - classified), (
        f"registered but in no safety group: {sorted(registered - classified)} — "
        "such a tool is absent from both the allow and the ask/write_tools lists"
    )
    assert not (classified - registered), (
        f"in a safety group but not registered: {sorted(classified - registered)}"
    )
    overlaps = {
        f"{a}&{b}": sorted(groups[a] & groups[b])
        for a, b in (
            ("READ_TOOLS", "QUEUE_TOOLS"),
            ("READ_TOOLS", "STOP_TOOLS"),
            ("QUEUE_TOOLS", "STOP_TOOLS"),
        )
        if groups[a] & groups[b]
    }
    assert not overlaps, f"tools with two safety classes: {overlaps}"

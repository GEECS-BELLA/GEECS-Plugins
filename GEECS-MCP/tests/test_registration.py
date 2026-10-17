"""The server's tool set and the contract the profile imports."""

from __future__ import annotations

import anyio
import pytest

from geecs_mcp import __version__, tool_names
from geecs_mcp.identity import InvalidArgument, identity_for
from geecs_mcp.server import create_server


def test_the_tool_groups_partition_every_registered_tool():
    """Every registered tool sits in exactly one safety group.

    A tool added without a ``tool_names`` group entry would appear in
    neither ``allow`` nor ``write_tools`` — shipped ungated and unlisted —
    and a tool in two groups has no single safety class.
    """
    registered = {tool.name for tool in anyio.run(create_server().list_tools)}
    groups = {
        "READ_TOOLS": set(tool_names.READ_TOOLS),
        "WRITE_TOOLS": set(tool_names.WRITE_TOOLS),
        "STOP_TOOLS": set(tool_names.STOP_TOOLS),
    }
    classified = set().union(*groups.values())
    assert not (registered - classified), sorted(registered - classified)
    assert not (classified - registered), sorted(classified - registered)
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not (groups[a] & groups[b]), (a, b, groups[a] & groups[b])


def test_the_retired_names_are_gone():
    assert not hasattr(tool_names, "GET_SCAN_RESULT")
    assert not hasattr(tool_names, "QUEUE_TOOLS")


def test_version_is_the_installed_package_version():
    from importlib.metadata import version

    assert __version__ == version("geecs-mcp")


class TestIdentity:
    def test_the_string_shape(self):
        assert identity_for("Ada Lovelace") == "agent, admin supervision (Ada Lovelace)"
        assert (
            identity_for("  Ada   Lovelace ")
            == "agent, admin supervision (Ada Lovelace)"
        )

    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_a_blank_operator_is_refused(self, bad):
        with pytest.raises(InvalidArgument, match="operator is required"):
            identity_for(bad)

    def test_the_logbook_author_cap_holds(self):
        with pytest.raises(ValueError, match="too long"):
            identity_for("x" * 120)

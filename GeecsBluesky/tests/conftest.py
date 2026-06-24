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

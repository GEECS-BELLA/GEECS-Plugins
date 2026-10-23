"""GeecsBluesky applies the config.ini EPICS addressing at package import.

The behaviour of the hook itself is tested in GEECS-Core
(``tests/test_epics_env.py``); the import order is pinned by
``_qserver_startup_probe.py``.
"""

from __future__ import annotations


def test_package_import_applies_config():
    """geecs_bluesky/__init__ calls the hook before any device import."""
    import geecs_bluesky
    from geecs_core.epics_env import apply_epics_address_config

    assert geecs_bluesky.apply_epics_address_config is apply_epics_address_config

"""Root conftest: --hardware flag and session-level DB initialisation."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--hardware",
        action="store_true",
        default=False,
        help="Run tests that require a live GEECS hardware connection.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--hardware"):
        return
    skip = pytest.mark.skip(reason="requires --hardware flag and live GEECS hardware")
    for item in items:
        if item.get_closest_marker("hardware"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def geecs_exp_info():
    """Connect to the GEECS MySQL database and populate GeecsDevice.exp_info.

    Requires a valid ~/.config/geecs_python_api/config.ini pointing at the
    Undulator experiment database.  Tests that use this fixture should be
    marked @pytest.mark.hardware.
    """
    from geecs_python_api.controls.devices.geecs_device import GeecsDevice
    from geecs_python_api.controls.interface.geecs_database import GeecsDatabase

    exp_info = GeecsDatabase.collect_exp_info("Undulator")
    GeecsDevice.exp_info = exp_info
    return exp_info

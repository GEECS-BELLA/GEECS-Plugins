"""Asset-registry coverage for real GEECS device types (needs the lab DB)."""

from __future__ import annotations

import pytest

from geecs_bluesky.assets import POINTGREY_CAMERA_DEVICE_TYPE, supports_device_type
from geecs_ca_gateway.db import geecs_db
from geecs_ca_gateway.db.geecs_db import GeecsDb


@pytest.mark.integration
def test_uc_topview_device_type_matches_real_database(monkeypatch) -> None:
    """Real DB lookup should match the registered camera device-type string."""
    mysql_connector = pytest.importorskip("mysql.connector")
    monkeypatch.setattr(geecs_db, "_credentials", None)

    try:
        device_type = GeecsDb.get_device_type("UC_TopView")
    except (FileNotFoundError, KeyError, OSError, mysql_connector.Error) as exc:
        pytest.skip(f"Real GEECS database is unavailable: {exc}")

    assert device_type == POINTGREY_CAMERA_DEVICE_TYPE
    assert supports_device_type(device_type)

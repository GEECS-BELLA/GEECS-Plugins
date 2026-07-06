"""Client-side EPICS addressing from the shared GEECS config.

Clients resolve the gateway host from ``config.ini [epics] ca_addr_list``
the same way they resolve the database — no per-shell env vars.  Explicitly
exported variables always win.
"""

from __future__ import annotations

from pathlib import Path

from geecs_bluesky.epics_env import apply_epics_address_config


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.ini"
    path.write_text(body)
    return path


def test_addr_list_applied_from_config(tmp_path):
    path = _write_config(tmp_path, "[epics]\nca_addr_list = 192.168.6.14\n")
    env: dict[str, str] = {}
    applied = apply_epics_address_config(env=env, config_path=path)
    assert env["EPICS_CA_ADDR_LIST"] == "192.168.6.14"
    # Directed addressing implies no broadcast unless configured otherwise.
    assert env["EPICS_CA_AUTO_ADDR_LIST"] == "NO"
    assert applied == {
        "EPICS_CA_ADDR_LIST": "192.168.6.14",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
    }


def test_explicit_env_var_wins(tmp_path):
    path = _write_config(tmp_path, "[epics]\nca_addr_list = 192.168.6.14\n")
    env = {"EPICS_CA_ADDR_LIST": "10.0.0.1"}
    applied = apply_epics_address_config(env=env, config_path=path)
    assert env["EPICS_CA_ADDR_LIST"] == "10.0.0.1"
    assert "EPICS_CA_AUTO_ADDR_LIST" not in env
    assert applied == {}


def test_configured_auto_addr_list_respected(tmp_path):
    path = _write_config(
        tmp_path, "[epics]\nca_addr_list = 192.168.6.14\nca_auto_addr_list = YES\n"
    )
    env: dict[str, str] = {}
    apply_epics_address_config(env=env, config_path=path)
    assert env["EPICS_CA_AUTO_ADDR_LIST"] == "YES"


def test_missing_file_and_section_are_noops(tmp_path):
    env: dict[str, str] = {}
    assert apply_epics_address_config(env=env, config_path=tmp_path / "nope.ini") == {}
    path = _write_config(tmp_path, "[Paths]\ngeecs_data = /data\n")
    assert apply_epics_address_config(env=env, config_path=path) == {}
    assert env == {}


def test_package_import_applies_config():
    """geecs_bluesky/__init__ calls the hook before any device import."""
    import geecs_bluesky

    assert callable(geecs_bluesky.apply_epics_address_config)

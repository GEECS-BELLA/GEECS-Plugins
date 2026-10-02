"""Resolution of the appliance URL, the experiment and the configs-repo files."""

from pathlib import Path

import pytest
from geecs_schemas import ArchivePolicy

from geecs_archiver import config


def write_ini(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.ini"
    p.write_text(text, encoding="utf-8")
    return p


def test_archiver_url_from_config_strips_the_slash(tmp_path, monkeypatch):
    monkeypatch.delenv("GEECS_ARCHIVER_URL", raising=False)
    ini = write_ini(tmp_path, "[archiver]\nurl = http://host:17665/\n")
    assert config.archiver_url(ini) == "http://host:17665"
    assert config.mgmt_url("http://host:17665") == "http://host:17665/mgmt/bpl"
    assert config.retrieval_url("http://host:17665/") == "http://host:17665/retrieval"


def test_environment_overrides_the_file(tmp_path, monkeypatch):
    ini = write_ini(tmp_path, "[archiver]\nurl = http://file:17665\n")
    monkeypatch.setenv("GEECS_ARCHIVER_URL", "http://env:17665/")
    assert config.archiver_url(ini) == "http://env:17665"


def test_missing_values_are_none(tmp_path, monkeypatch):
    monkeypatch.delenv("GEECS_ARCHIVER_URL", raising=False)
    assert config.archiver_url(tmp_path / "absent.ini") is None
    ini = write_ini(tmp_path, "[Experiment]\nexpt = Undulator\n[archiver]\nurl =\n")
    assert config.archiver_url(ini) is None
    assert config.experiment_name(ini) == "Undulator"


def test_configs_repo_resolution_order(tmp_path, monkeypatch):
    repo = tmp_path / "configs"
    exp = repo / "scanner_configs" / "experiments" / "Undulator"
    (exp / "archiver").mkdir(parents=True)
    (exp / "gateway").mkdir(parents=True)
    (exp / "archiver" / "archive_policy.yaml").write_text(
        "schema_version: 1\nexclude: ['*:noise']\n"
    )
    (exp / "gateway" / "derived_channels.yaml").write_text(
        "schema_version: 1\nderived_channels: []\n"
    )
    monkeypatch.delenv("GEECS_SCANNER_CONFIG_DIR", raising=False)
    monkeypatch.setenv("GEECS_PLUGINS_CONFIGS", str(repo))
    base = config.scanner_configs_base()
    assert base == (repo / "scanner_configs" / "experiments").resolve()
    assert (
        config.policy_path("Undulator", base)
        == exp / "archiver" / "archive_policy.yaml"
    )
    assert (
        config.derived_channels_path("Undulator", base)
        == exp / "gateway" / "derived_channels.yaml"
    )
    assert config.policy_path("Other", base) is None
    monkeypatch.setenv("GEECS_SCANNER_CONFIG_DIR", str(tmp_path / "direct"))
    assert config.scanner_configs_base() == (tmp_path / "direct").resolve()
    monkeypatch.delenv("GEECS_SCANNER_CONFIG_DIR")
    monkeypatch.delenv("GEECS_PLUGINS_CONFIGS")
    ini = write_ini(tmp_path, f"[Paths]\nscanner_config_root_path = {repo}\n")
    assert (
        config.scanner_configs_base(ini)
        == (repo / "scanner_configs" / "experiments").resolve()
    )
    assert config.scanner_configs_base(tmp_path / "absent.ini") is None


def test_load_policy_defaults_and_file(tmp_path):
    assert config.load_policy(None) == ArchivePolicy()
    f = tmp_path / "p.yaml"
    f.write_text(
        "schema_version: 1\ninclude_setpoints: false\nsampling_overrides:\n  - match: 'a:*'\n    policy: Slow\n"
    )
    policy = config.load_policy(f)
    assert (
        policy.include_setpoints is False
        and policy.sampling_overrides[0].policy == "Slow"
    )
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert config.load_policy(empty) == ArchivePolicy()
    bad = tmp_path / "bad.yaml"
    bad.write_text("unknown_key: 1\n")
    with pytest.raises(Exception):
        config.load_policy(bad)


def test_load_derived_channels(tmp_path):
    assert config.load_derived_channels(None) is None
    f = tmp_path / "d.yaml"
    f.write_text("schema_version: 1\nderived_channels: []\n")
    assert config.load_derived_channels(f).derived_channels == []

"""The one configs-repository resolver."""

from pathlib import Path

from geecs_core import configs_repo


def test_read_config_entry(tmp_path: Path) -> None:
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Paths]\nscanner_config_root_path = /x\nblank =\n", encoding="utf-8"
    )
    assert (
        configs_repo.read_config_entry("Paths", "scanner_config_root_path", ini) == "/x"
    )
    assert configs_repo.read_config_entry("Paths", "blank", ini) is None
    assert configs_repo.read_config_entry("Nope", "x", ini) is None
    assert configs_repo.read_config_entry("Paths", "x", tmp_path / "absent.ini") is None


def test_resolution_order(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "GEECS-Plugins-Configs"
    (repo / "scanner_configs" / "experiments" / "Exp" / "gateway").mkdir(parents=True)
    target = (
        repo
        / "scanner_configs"
        / "experiments"
        / "Exp"
        / "gateway"
        / "derived_channels.yaml"
    )
    target.write_text("schema_version: 1\n", encoding="utf-8")
    ini = tmp_path / "config.ini"
    ini.write_text(f"[Paths]\nscanner_config_root_path = {repo}\n", encoding="utf-8")

    monkeypatch.setenv("GEECS_SCANNER_CONFIG_DIR", str(tmp_path / "direct"))
    monkeypatch.setenv("GEECS_PLUGINS_CONFIGS", str(repo))
    assert configs_repo.scanner_configs_base(ini) == (tmp_path / "direct").resolve()

    monkeypatch.delenv("GEECS_SCANNER_CONFIG_DIR")
    base = configs_repo.scanner_configs_base(ini)
    assert base == (repo / "scanner_configs" / "experiments").resolve()

    monkeypatch.delenv("GEECS_PLUGINS_CONFIGS")
    assert configs_repo.scanner_configs_base(ini) == base
    assert configs_repo.scanner_configs_base(tmp_path / "absent.ini") is None

    assert (
        configs_repo.experiment_config_path(
            "Exp", "gateway", "derived_channels.yaml", config_path=ini
        )
        == target
    )
    assert (
        configs_repo.experiment_config_path(
            "Exp", "archiver", "archive_policy.yaml", config_path=ini
        )
        is None
    )
    assert (
        configs_repo.experiment_config_path(
            "Exp", "gateway", "derived_channels.yaml", base=base
        )
        == target
    )
    assert (
        configs_repo.experiment_config_path(
            "Exp", "gateway", "x.yaml", config_path=tmp_path / "absent.ini"
        )
        is None
    )

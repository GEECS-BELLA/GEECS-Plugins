"""A new experiment's configs folder: empty, valid, and never over an existing one."""

from __future__ import annotations

from pathlib import Path

import pytest

from geecs_bluesky.config_resolver import ConfigsRepoResolver
from geecs_bluesky.config_scaffold import main, scaffold_experiment


def test_the_scaffold_opens_as_an_empty_experiment(tmp_path: Path) -> None:
    created = scaffold_experiment("Bench stand", tmp_path)
    assert created[0] == tmp_path / "Bench stand"
    resolver = ConfigsRepoResolver("Bench stand", experiments_root=tmp_path)
    assert resolver.resolve_experiment_defaults() is not None
    assert dict(resolver.scan_variable_catalog().variables) == {}
    assert resolver.list_presets() == []
    assert resolver.list_trigger_profiles() == []
    assert (tmp_path / "Bench stand" / "presets").is_dir()
    assert (tmp_path / "Bench stand" / "shot_control_configurations").is_dir()


def test_an_existing_folder_is_left_alone(tmp_path: Path) -> None:
    (tmp_path / "Main").mkdir()
    (tmp_path / "Main" / "mine.yaml").write_text("x: 1\n")
    assert scaffold_experiment("Main", tmp_path) == []
    assert sorted(p.name for p in (tmp_path / "Main").iterdir()) == ["mine.yaml"]


def test_a_missing_root_is_never_created(tmp_path: Path) -> None:
    root = tmp_path / "no" / "such"
    with pytest.raises(FileNotFoundError):
        scaffold_experiment("Bench", root)
    assert not root.exists()


@pytest.mark.parametrize("name", ["", "a/b", "..", "../x"])
def test_a_name_that_is_not_one_folder_is_refused(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        scaffold_experiment(name, tmp_path)


def test_cli(tmp_path: Path, capsys) -> None:
    assert main(["Bench", "--experiments-root", str(tmp_path)]) == 0
    assert "created" in capsys.readouterr().out
    assert main(["Bench", "--experiments-root", str(tmp_path)]) == 0
    assert "left as it is" in capsys.readouterr().out
    assert main(["Bench", "--experiments-root", str(tmp_path / "x")]) == 2

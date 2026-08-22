"""Pin the resolver's config-listing surface (#666).

Non-GUI clients (the scan MCP, notebooks) need to enumerate the
experiment's save sets / trigger profiles / presets without importing
console code — the listing lives beside the resolution it feeds, with the
console-matching semantics: sorted YAML stems, and every missing layer
(configs root, experiment folder, kind folder) reads as an empty list,
never an exception.
"""

from __future__ import annotations

import pytest

from geecs_bluesky.config_resolver import ConfigsRepoResolver


@pytest.fixture
def repo(tmp_path):
    """A minimal configs-repo experiments root with one experiment."""
    exp = tmp_path / "TestExp"
    for folder, names in {
        ConfigsRepoResolver.SAVE_SET_FOLDER: ["Amp4In", "BCave"],
        ConfigsRepoResolver.TRIGGER_FOLDER: ["HTU-LaserOFF"],
        ConfigsRepoResolver.PRESET_FOLDER: ["basic test"],
        ConfigsRepoResolver.OPTIMIZER_FOLDER: ["bayes_jet"],
    }.items():
        d = exp / folder
        d.mkdir(parents=True)
        for name in names:
            (d / f"{name}.yaml").write_text("{}\n")
    return tmp_path


def test_listings_return_sorted_stems(repo):
    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    assert resolver.list_save_sets() == ["Amp4In", "BCave"]
    assert resolver.list_trigger_profiles() == ["HTU-LaserOFF"]
    assert resolver.list_presets() == ["basic test"]
    assert resolver.list_optimizer_configs() == ["bayes_jet"]


def test_yml_suffix_counts_and_others_do_not(repo):
    folder = repo / "TestExp" / ConfigsRepoResolver.SAVE_SET_FOLDER
    (folder / "Extra.yml").write_text("{}\n")
    (folder / "notes.txt").write_text("not a config\n")
    (folder / "README.md").write_text("docs\n")
    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    assert resolver.list_save_sets() == ["Amp4In", "BCave", "Extra"]


def test_missing_kind_folder_is_empty(repo):
    import shutil

    shutil.rmtree(repo / "TestExp" / ConfigsRepoResolver.PRESET_FOLDER)
    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    assert resolver.list_presets() == []


def test_missing_experiment_is_empty(repo):
    resolver = ConfigsRepoResolver("NoSuchExp", experiments_root=repo)
    assert resolver.list_save_sets() == []


def test_listed_yml_names_round_trip_through_resolution(repo):
    # Review finding: the listings count .yml files, so resolution must
    # accept them too (console NamedConfigStore parity) — a listed name
    # that resolve_* refuses on spelling alone is a client-facing trap.
    exp = repo / "TestExp"
    (exp / ConfigsRepoResolver.SAVE_SET_FOLDER / "YmlSet.yml").write_text(
        "Devices:\n  UC_Cam:\n    variable_list: [MeanCounts]\n    synchronous: true\n"
    )
    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    assert "YmlSet" in resolver.list_save_sets()
    save_set = resolver.resolve_save_set("YmlSet")
    assert [entry.device for entry in save_set.entries] == ["UC_Cam"]


def test_io_failure_mid_scan_is_empty(repo, monkeypatch):
    # The never-raises contract covers the scan itself, not just root
    # resolution — an SMB blip / permissions failure during iterdir must
    # read as empty (review finding).
    from pathlib import Path

    def boom(self):
        raise PermissionError("share blipped")

    monkeypatch.setattr(Path, "iterdir", boom)
    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    assert resolver.list_save_sets() == []


def test_unresolvable_configs_root_is_empty(monkeypatch):
    # No experiments_root override and the production resolution raises
    # (no env var, no config.ini entry) — listing reads empty, never raises.
    monkeypatch.setattr(
        "geecs_bluesky.config_resolver.scanner_configs_base",
        lambda: (_ for _ in ()).throw(RuntimeError("unconfigured")),
    )
    resolver = ConfigsRepoResolver("TestExp")
    assert resolver.list_save_sets() == []
    assert resolver.list_trigger_profiles() == []

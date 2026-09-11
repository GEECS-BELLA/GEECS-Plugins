"""Pin the resolver's config-listing surface (#666).

Non-GUI clients (the scan MCP, notebooks) need to enumerate the
experiment's presets / trigger profiles / optimizer configs without
importing console code — the listing lives beside the resolution it
feeds, with the console-matching semantics: sorted YAML stems, and every
missing layer (configs root, experiment folder, kind folder) reads as an
empty list, never an exception.
"""

from __future__ import annotations

import pytest
import yaml

from geecs_bluesky.config_resolver import ConfigsRepoResolver

PRESET = {
    "schema_version": 1,
    "name": "smoke",
    "devices": [{"device": "UC_Cam", "save_images": True}],
    "plan": {"name": "count", "kwargs": {"num": 3}},
}


@pytest.fixture
def repo(tmp_path):
    """A minimal configs-repo experiments root with one experiment."""
    exp = tmp_path / "TestExp"
    for folder, names in {
        ConfigsRepoResolver.PRESET_FOLDER: ["Amp4In", "BCave"],
        ConfigsRepoResolver.TRIGGER_FOLDER: ["HTU-LaserOFF"],
        ConfigsRepoResolver.OPTIMIZER_FOLDER: ["bayes_jet"],
    }.items():
        d = exp / folder
        d.mkdir(parents=True)
        for name in names:
            (d / f"{name}.yaml").write_text("{}\n")
    return tmp_path


def test_listings_return_sorted_stems(repo):
    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    assert resolver.list_presets() == ["Amp4In", "BCave"]
    assert resolver.list_trigger_profiles() == ["HTU-LaserOFF"]
    assert resolver.list_optimizer_configs() == ["bayes_jet"]


def test_yml_suffix_counts_and_others_do_not(repo):
    folder = repo / "TestExp" / ConfigsRepoResolver.PRESET_FOLDER
    (folder / "Extra.yml").write_text("{}\n")
    (folder / "notes.txt").write_text("not a config\n")
    (folder / "README.md").write_text("docs\n")
    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    assert resolver.list_presets() == ["Amp4In", "BCave", "Extra"]


def test_missing_kind_folder_is_empty(repo):
    import shutil

    shutil.rmtree(repo / "TestExp" / ConfigsRepoResolver.PRESET_FOLDER)
    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    assert resolver.list_presets() == []


def test_missing_experiment_is_empty(repo):
    resolver = ConfigsRepoResolver("NoSuchExp", experiments_root=repo)
    assert resolver.list_presets() == []


def test_listed_yml_names_round_trip_through_resolution(repo):
    # The listings count .yml files, so resolution must accept them too
    # (console NamedConfigStore parity) — a listed name that resolve_*
    # refuses on spelling alone is a client-facing trap.
    exp = repo / "TestExp"
    (exp / ConfigsRepoResolver.PRESET_FOLDER / "YmlSet.yml").write_text(
        yaml.safe_dump(PRESET)
    )
    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    assert "YmlSet" in resolver.list_presets()
    preset = resolver.resolve_preset("YmlSet")
    assert [d.device for d in preset.devices] == ["UC_Cam"]
    assert preset.plan.name == "count" and preset.plan.kwargs == {"num": 3}


def test_io_failure_mid_scan_is_empty(repo, monkeypatch):
    # The never-raises contract covers the scan itself, not just root
    # resolution — an SMB blip / permissions failure during iterdir must
    # read as empty (review finding).
    from pathlib import Path

    def boom(self):
        raise PermissionError("share blipped")

    monkeypatch.setattr(Path, "iterdir", boom)
    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    assert resolver.list_presets() == []


def test_unresolvable_configs_root_is_empty(monkeypatch):
    # No experiments_root override and the production resolution raises
    # (no env var, no config.ini entry) — listing reads empty, never raises.
    monkeypatch.setattr(
        "geecs_bluesky.config_resolver.scanner_configs_base",
        lambda: (_ for _ in ()).throw(RuntimeError("unconfigured")),
    )
    resolver = ConfigsRepoResolver("TestExp")
    assert resolver.list_presets() == []
    assert resolver.list_trigger_profiles() == []


def test_resolve_preset_refuses_a_legacy_document(repo):
    from pydantic import ValidationError

    folder = repo / "TestExp" / ConfigsRepoResolver.PRESET_FOLDER
    (folder / "old.yaml").write_text(
        yaml.safe_dump({"mode": "noscan", "shots_per_step": 3, "save_sets": ["A"]})
    )
    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    with pytest.raises(ValidationError):
        resolver.resolve_preset("old")


def test_resolve_preset_missing_raises_with_kind(repo):
    from geecs_bluesky.exceptions import GeecsConfigurationError

    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    with pytest.raises(GeecsConfigurationError, match="preset 'nope'"):
        resolver.resolve_preset("nope")

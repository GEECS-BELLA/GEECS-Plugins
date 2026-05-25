"""Tests for the analysis-group loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scan_analysis.config.analysis_group_loader import (
    LoadedAnalysisGroup,
    discover_analyzers,
    discover_groups,
    load_analysis_group,
    resolve_group,
)
from scan_analysis.config.diagnostic_models import (
    AnalysisGroupConfig,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_diagnostic(path: Path, name: str, *, priority: int = 100) -> None:
    """Write a minimal diagnostic YAML at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "name": name,
                "image_analyzer": (
                    "image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer"
                ),
                "image": {"bit_depth": 16},
                "scan": {"priority": priority},
            }
        )
    )


def _write_group(path: Path, name: str, analyzers: list) -> None:
    """Write a minimal group YAML at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"name": name, "analyzers": analyzers}))


@pytest.fixture
def configs_tree(tmp_path: Path) -> Path:
    """Build a small config tree with two namespaces and a few diagnostics."""
    _write_diagnostic(tmp_path / "analyzers" / "HTU" / "GaiaMode.yaml", "UC_GaiaMode")
    _write_diagnostic(
        tmp_path / "analyzers" / "HTU" / "OAPin2.yaml", "UC_OAPin2", priority=50
    )
    _write_diagnostic(
        tmp_path / "analyzers" / "PW" / "FROG.yaml", "PW_FROG", priority=30
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscoverAnalyzers:
    """Recursive analyzer discovery, stem-based indexing, duplicate detection."""

    def test_indexes_yaml_files_by_stem(self, configs_tree):
        idx = discover_analyzers(configs_tree)
        assert set(idx) == {"GaiaMode", "OAPin2", "FROG"}
        assert idx["GaiaMode"].name == "GaiaMode.yaml"

    def test_missing_analyzers_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Analyzer directory not found"):
            discover_analyzers(tmp_path)

    def test_duplicate_stems_rejected(self, tmp_path):
        _write_diagnostic(tmp_path / "analyzers" / "A" / "GaiaMode.yaml", "x")
        _write_diagnostic(tmp_path / "analyzers" / "B" / "GaiaMode.yaml", "x")
        with pytest.raises(ValueError, match="Duplicate diagnostic ID"):
            discover_analyzers(tmp_path)


class TestDiscoverGroups:
    """Group discovery indexes both stem and path-like names."""

    def test_stem_and_path_like_keys(self, tmp_path):
        _write_group(tmp_path / "groups" / "HTU" / "laser.yaml", "HTU_laser", [])
        idx = discover_groups(tmp_path)
        assert "laser" in idx
        assert "HTU/laser" in idx
        assert idx["laser"] == idx["HTU/laser"]

    def test_ambiguous_stem_drops_stem_key(self, tmp_path):
        _write_group(tmp_path / "groups" / "HTU" / "shared.yaml", "HTU_shared", [])
        _write_group(tmp_path / "groups" / "PW" / "shared.yaml", "PW_shared", [])
        idx = discover_groups(tmp_path)
        assert "shared" not in idx, "ambiguous stem must be removed"
        assert "HTU/shared" in idx
        assert "PW/shared" in idx

    def test_missing_groups_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Group directory not found"):
            discover_groups(tmp_path)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


class TestResolveGroup:
    """Resolve analyzer refs, apply per-group overrides, sort by priority."""

    def test_resolves_string_refs_in_priority_order(self, configs_tree):
        idx = discover_analyzers(configs_tree)
        group = AnalysisGroupConfig(
            name="test", analyzers=["GaiaMode", "OAPin2", "FROG"]
        )
        result = resolve_group(group, idx)

        # Priorities: FROG=30, OAPin2=50, GaiaMode=100
        assert [r.id for r in result.analyzers] == ["FROG", "OAPin2", "GaiaMode"]
        assert [r.priority for r in result.analyzers] == [30, 50, 100]

    def test_group_priority_override_applies(self, configs_tree):
        idx = discover_analyzers(configs_tree)
        group = AnalysisGroupConfig(
            name="test",
            analyzers=[{"ref": "GaiaMode", "priority": 5}, "OAPin2"],
        )
        result = resolve_group(group, idx)

        # GaiaMode's effective priority is 5 (override), beating OAPin2's 50
        assert [r.id for r in result.analyzers] == ["GaiaMode", "OAPin2"]
        assert result.analyzers[0].priority == 5

    def test_disabled_entries_excluded(self, configs_tree):
        idx = discover_analyzers(configs_tree)
        group = AnalysisGroupConfig(
            name="test",
            analyzers=["GaiaMode", {"ref": "OAPin2", "enabled": False}],
        )
        result = resolve_group(group, idx)

        assert [r.id for r in result.analyzers] == ["GaiaMode"]

    def test_unknown_ref_raises_with_known_ids(self, configs_tree):
        idx = discover_analyzers(configs_tree)
        group = AnalysisGroupConfig(name="test", analyzers=["NotARealAnalyzer"])
        with pytest.raises(ValueError, match="unknown analyzer 'NotARealAnalyzer'"):
            resolve_group(group, idx)

    def test_duplicate_refs_rejected(self, configs_tree):
        idx = discover_analyzers(configs_tree)
        group = AnalysisGroupConfig(name="test", analyzers=["GaiaMode", "GaiaMode"])
        with pytest.raises(ValueError, match="duplicate analyzer reference"):
            resolve_group(group, idx)

    def test_invalid_diagnostic_yaml_surfaces_path(self, configs_tree):
        # Corrupt a diagnostic by omitting the required ``name`` field.
        (configs_tree / "analyzers" / "HTU" / "GaiaMode.yaml").write_text(
            yaml.safe_dump(
                {
                    "image_analyzer": (
                        "image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer"
                    )
                }
            )
        )
        idx = discover_analyzers(configs_tree)
        group = AnalysisGroupConfig(name="test", analyzers=["GaiaMode"])
        with pytest.raises(ValueError, match="Invalid diagnostic config"):
            resolve_group(group, idx)


# ---------------------------------------------------------------------------
# Entry point: load_analysis_group
# ---------------------------------------------------------------------------


class TestLoadAnalysisGroup:
    """End-to-end: name-based and path-based loading."""

    def test_load_by_name_with_config_dir(self, configs_tree):
        _write_group(
            configs_tree / "groups" / "HTU" / "laser.yaml",
            "HTU_laser",
            ["GaiaMode", "OAPin2"],
        )
        result = load_analysis_group("laser", config_dir=configs_tree)

        assert isinstance(result, LoadedAnalysisGroup)
        assert result.name == "HTU_laser"
        assert [r.id for r in result.analyzers] == ["OAPin2", "GaiaMode"]

    def test_load_by_path_infers_config_root(self, configs_tree):
        group_path = configs_tree / "groups" / "PW" / "frog.yaml"
        _write_group(group_path, "PW_frog", ["FROG"])
        result = load_analysis_group(group_path)

        assert result.name == "PW_frog"
        assert [r.id for r in result.analyzers] == ["FROG"]

    def test_load_by_path_like_name(self, configs_tree):
        _write_group(configs_tree / "groups" / "PW" / "frog.yaml", "PW_frog", ["FROG"])
        result = load_analysis_group("PW/frog", config_dir=configs_tree)

        assert result.name == "PW_frog"

    def test_load_by_name_missing_config_dir_raises(self, configs_tree):
        with pytest.raises(ValueError, match="config_dir is required"):
            load_analysis_group("laser")

    def test_ambiguous_name_raises_with_disambiguation_hint(self, configs_tree):
        _write_group(configs_tree / "groups" / "HTU" / "shared.yaml", "HTU_shared", [])
        _write_group(configs_tree / "groups" / "PW" / "shared.yaml", "PW_shared", [])
        with pytest.raises(ValueError, match="ambiguous; matches"):
            load_analysis_group("shared", config_dir=configs_tree)

    def test_missing_group_lists_known_groups(self, configs_tree):
        _write_group(configs_tree / "groups" / "HTU" / "laser.yaml", "HTU_laser", [])
        with pytest.raises(FileNotFoundError, match="Known groups"):
            load_analysis_group("nope", config_dir=configs_tree)


# NB: single-diagnostic loading moved to ``image_analysis.config.load_diagnostic``
# (returns ``DiagnosticAnalysisConfig`` directly). See
# ``ImageAnalysis/tests/test_config_factory.py::TestLoadDiagnostic``.

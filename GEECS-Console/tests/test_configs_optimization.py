"""ConsoleConfigs optimizer-config listing and loading, against a tmp tree."""

import pytest

from geecs_console.services import configs as configs_module
from geecs_console.services.configs import (
    OPTIMIZATION_FOLDER,
    ConsoleConfigs,
    ConsoleConfigsError,
)
from geecs_schemas import OptimizationSpec

NEW_SCHEMA_YAML = """\
variables:
  jet_x: [0.0, 1.0]
objectives:
  counts: MAXIMIZE
evaluator:
  module: my.evaluators
  class: BeamCounts
generator:
  name: random
"""

LEGACY_YAML = """\
vocs:
  variables:
    U_Hexapod:ypos: [17.0, 19.0]
  objectives:
    f: MINIMIZE
evaluator:
  module: my.evaluators
  class: Legacy
  kwargs: {}
generator:
  name: bayes_default
"""


@pytest.fixture
def configs_tree(tmp_path, monkeypatch):
    """A tmp experiments root with one experiment's optimizer configs."""
    folder = tmp_path / "HTU" / OPTIMIZATION_FOLDER
    folder.mkdir(parents=True)
    (folder / "new_style.yaml").write_text(NEW_SCHEMA_YAML, encoding="utf-8")
    (folder / "legacy_style.yaml").write_text(LEGACY_YAML, encoding="utf-8")
    (folder / "broken.yaml").write_text("just a string", encoding="utf-8")
    monkeypatch.setenv("GEECS_SCANNER_CONFIG_DIR", str(tmp_path))
    return tmp_path


class TestListing:
    def test_listing_includes_optimizer_config_stems(self, configs_tree):
        listing = ConsoleConfigs("HTU").listing()
        assert listing.optimization_configs == [
            "broken",
            "legacy_style",
            "new_style",
        ]

    def test_listing_empty_without_optimizer_folder(self, configs_tree):
        (configs_tree / "Bella").mkdir()
        listing = ConsoleConfigs("Bella").listing()
        assert listing.optimization_configs == []


class TestOptimizationSpecLoading:
    def test_new_schema_document_loads_directly(self, configs_tree):
        spec = ConsoleConfigs("HTU").optimization_spec("new_style")
        assert isinstance(spec, OptimizationSpec)
        assert spec.variables == {"jet_x": (0.0, 1.0)}
        assert spec.objectives == {"counts": "MAXIMIZE"}
        assert spec.generator.name == "random"

    def test_legacy_vocs_document_converts(self, configs_tree):
        spec = ConsoleConfigs("HTU").optimization_spec("legacy_style")
        assert isinstance(spec, OptimizationSpec)
        assert spec.variables == {"U_Hexapod:ypos": (17.0, 19.0)}
        assert spec.objectives == {"f": "MINIMIZE"}
        assert spec.generator.name == "bayes_default"

    def test_non_mapping_document_raises(self, configs_tree):
        with pytest.raises(ConsoleConfigsError, match="YAML mapping"):
            ConsoleConfigs("HTU").optimization_spec("broken")

    def test_missing_config_raises(self, configs_tree):
        with pytest.raises(ConsoleConfigsError, match="not found"):
            ConsoleConfigs("HTU").optimization_spec("nope")

    def test_invalid_new_schema_document_raises(self, configs_tree):
        folder = configs_tree / "HTU" / OPTIMIZATION_FOLDER
        (folder / "bad.yaml").write_text("variables: {}\n", encoding="utf-8")
        with pytest.raises(ConsoleConfigsError, match="not a valid"):
            ConsoleConfigs("HTU").optimization_spec("bad")

    def test_offline_raises_with_actionable_message(self, monkeypatch):
        monkeypatch.setattr(configs_module, "_configs_base", lambda: None)
        with pytest.raises(ConsoleConfigsError, match="Configs repo not found"):
            ConsoleConfigs("HTU").optimization_spec("anything")

    def test_no_experiment_selected_raises(self, configs_tree):
        with pytest.raises(ConsoleConfigsError, match="No experiment"):
            ConsoleConfigs("").optimization_spec("new_style")

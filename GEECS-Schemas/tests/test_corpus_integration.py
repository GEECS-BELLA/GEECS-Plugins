"""Corpus walk: convert EVERY real config in the sibling configs checkout.

The corpus walk (``TestFullCorpus``) is marked ``integration``: it auto-skips
when the sibling ``GEECS-Plugins-Configs`` checkout is absent (e.g. in CI).
Locally this is the proof that the converters cover the real world, file by
file. Empty/deviceless shot-control configs legitimately convert to "no
trigger profile". Retired optimizer dialects are excluded while their separate
corpus migration is pending; every deployed native optimizer is validated.

Corpus layout (regenerated 2026-09-10, GEECS-Plugins#807 phase 1 PR 2)::

    scanner_configs/experiments/<Experiment>/
      presets/                       # Preset documents (new schema only; no converter)
      scan_devices/                  # scan_variables.yaml (new schema only; no converter)
      shot_control_configurations/   # trigger configs (incl. laser-on/off pairs)
      action_library/                # actions.yaml (ActionPlanLibrary, new schema only; no converter)
      optimizer_configs/             # Xopt optimizer configs (Undulator only)
      aux_configs/                   # visa plunger lookup (app data; no converter)
"""

import os
import sys
from pathlib import Path

import pytest
import yaml

from geecs_schemas import ActionPlanLibrary, Preset, ScanVariables
from geecs_schemas.convert import (
    convert_shot_control,
)


def find_configs_repo() -> Path | None:
    """Locate the sibling GEECS-Plugins-Configs checkout, if present.

    Honours the ``GEECS_PLUGINS_CONFIGS`` env var, then searches each
    ancestor of this file for a ``GEECS-Plugins-Configs`` sibling containing
    ``scanner_configs/`` (works from the main checkout and from nested
    ``.claude/worktrees/`` worktrees alike).
    """
    override = os.environ.get("GEECS_PLUGINS_CONFIGS")
    if override:
        path = Path(override)
        return path if (path / "scanner_configs").is_dir() else None
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / "GEECS-Plugins-Configs"
        if (candidate / "scanner_configs").is_dir():
            return candidate
    return None


CONFIGS = find_configs_repo()
skip_without_corpus = pytest.mark.skipif(
    CONFIGS is None,
    reason="sibling GEECS-Plugins-Configs checkout not found",
)


def experiments() -> list[Path]:
    return sorted((CONFIGS / "scanner_configs" / "experiments").iterdir())


@pytest.mark.integration
@skip_without_corpus
class TestFullCorpus:
    def test_every_preset_validates(self):
        """Every deployed preset is a valid new-schema ``Preset`` (no converter)."""
        validated = 0
        for experiment in experiments():
            for path in sorted(experiment.glob("presets/*.yaml")):
                document = yaml.safe_load(path.read_text())
                preset = Preset.model_validate(document)
                assert preset.name, path
                validated += 1
        assert validated > 0

    def test_every_scan_variable_catalog_validates(self):
        """Every deployed catalog is a valid new-schema ``ScanVariables``.

        There is no scan-variable converter any more (0.18.0), so this
        replaces the old conversion pin with a validation pin.  It also pins
        the #779 policy: every entry carries an explicit ``kind`` (pseudo
        entries necessarily do; plain ones must too) — the deployed corpus
        never relies on the schema default, which is what silently opted
        every axis out of readback confirmation.
        """
        validated = 0
        for experiment in experiments():
            path = experiment / "scan_devices" / "scan_variables.yaml"
            if not path.exists():
                continue
            document = yaml.safe_load(path.read_text())
            catalog = ScanVariables.model_validate(document)
            assert catalog.variables, experiment.name
            for name, raw in document["variables"].items():
                assert "kind" in raw, (
                    f"{experiment.name}: {name!r} relies on the default kind"
                )
            validated += 1
        assert validated >= 2  # Undulator + Thomson

    def test_every_shot_control_converts(self):
        converted, no_device = 0, 0
        for experiment in experiments():
            for path in sorted(experiment.glob("shot_control_configurations/*.yaml")):
                profile = convert_shot_control(path)
                if profile is None:
                    # Documented empty/deviceless configs: Bella Normal,
                    # Undulator "No Device" ("no shot control configured").
                    no_device += 1
                else:
                    converted += 1
        assert converted >= 8 and no_device >= 2

    def test_every_action_library_validates(self):
        """Every deployed action library is a new-schema ``ActionPlanLibrary``.

        There is no action-library converter any more (0.22.0): the corpus
        was regenerated once, and the legacy ``assigned_actions.yaml``
        (the old GUI's pinned-button list) went with it.
        """
        libraries = {}
        for experiment in experiments():
            actions = experiment / "action_library" / "actions.yaml"
            if actions.exists():
                document = yaml.safe_load(actions.read_text())
                assert "actions" not in document, f"{experiment.name}: legacy dialect"
                libraries[experiment.name] = ActionPlanLibrary.model_validate(document)
            assert not (
                experiment / "action_library" / "assigned_actions.yaml"
            ).exists()
        assert set(libraries) >= {"Undulator", "Thomson"}
        assert libraries["Undulator"].plans

    def test_every_optimizer_config_validates(self):
        from geecs_schemas import OptimizerConfig

        validated = 0
        for experiment in experiments():
            for path in sorted(experiment.glob("optimizer_configs/*.yaml")):
                text = path.read_text()
                if text.startswith("# LEGACY"):
                    continue
                document = yaml.safe_load(text)
                if isinstance(document, dict) and (
                    "evaluator" in document or "device_requirements" in document
                ):
                    continue  # Retired dialect: unavailable to the resolver/UI.
                config = OptimizerConfig.model_validate(document)
                assert config.vocs.variables, path
                validated += 1
        assert validated >= 6, (
            f"optimizer corpus migration incomplete: {validated} native configs; "
            "deploy all six keepers before rollout acceptance"
        )


@pytest.mark.parametrize("native_count", [0, 1, 5, 6])
def test_optimizer_rollout_requires_all_keepers(tmp_path, monkeypatch, native_count):
    """Exercise the real corpus walk against absent, partial and complete rollouts."""
    from geecs_schemas import OptimizerConfig

    folder = tmp_path / "scanner_configs/experiments/Test/optimizer_configs"
    folder.mkdir(parents=True)
    (folder / "old.yaml").write_text("evaluator: {}")
    document = OptimizerConfig(
        vocs={
            "variables": {"Motor:Current": [-1, 1]},
            "objectives": {"score": "MINIMIZE"},
        },
        measurements={"score": {"signal": "Meter:Value"}},
        generator={"name": "random"},
    ).model_dump(mode="json")
    for index in range(native_count):
        (folder / f"native{index}.yaml").write_text(yaml.safe_dump(document))
    monkeypatch.setattr(sys.modules[__name__], "CONFIGS", tmp_path)
    if native_count < 6:
        with pytest.raises(AssertionError, match=f"incomplete: {native_count} native"):
            TestFullCorpus().test_every_optimizer_config_validates()
    else:
        TestFullCorpus().test_every_optimizer_config_validates()

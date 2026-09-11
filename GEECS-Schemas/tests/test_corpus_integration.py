"""Corpus walk: convert EVERY real config in the sibling configs checkout.

The corpus walk (``TestFullCorpus``) is marked ``integration``: it auto-skips
when the sibling ``GEECS-Plugins-Configs`` checkout is absent (e.g. in CI).
Locally this is the proof that the converters cover the real world, file by
file, with zero skips beyond the documented empty/deviceless shot-control
configs (which legitimately convert to "no trigger profile").

Corpus layout (regenerated 2026-09-10, GEECS-Plugins#807 phase 1 PR 2)::

    scanner_configs/experiments/<Experiment>/
      presets/                       # Preset documents (new schema only; no converter)
      scan_devices/                  # scan_variables.yaml (new schema only; no converter)
      shot_control_configurations/   # trigger configs (incl. laser-on/off pairs)
      action_library/                # actions.yaml + assigned_actions.yaml
      optimizer_configs/             # Xopt optimizer configs (Undulator only)
      aux_configs/                   # visa plunger lookup (app data; no converter)
"""

import os
from pathlib import Path

import pytest
import yaml

from geecs_schemas import Preset, ScanVariables
from geecs_schemas.convert import (
    convert_action_library,
    convert_assigned_actions,
    convert_optimizer_config,
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

    def test_every_action_library_converts(self):
        libraries = {}
        for experiment in experiments():
            actions = experiment / "action_library" / "actions.yaml"
            if actions.exists():
                libraries[experiment.name] = convert_action_library(actions)
        assert set(libraries) >= {"Undulator", "Thomson"}
        for experiment in experiments():
            assigned = experiment / "action_library" / "assigned_actions.yaml"
            if assigned.exists():
                convert_assigned_actions(
                    assigned, library=libraries.get(experiment.name)
                )

    def test_every_optimizer_config_converts(self):
        converted = 0
        for experiment in experiments():
            for path in sorted(experiment.glob("optimizer_configs/*.yaml")):
                conversion = convert_optimizer_config(path)
                assert conversion.optimization.variables, path
                converted += 1
        assert converted >= 11

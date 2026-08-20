"""Corpus walk: convert EVERY real config in the sibling configs checkout.

The corpus walk (``TestFullCorpus``) is marked ``integration``: it auto-skips
when the sibling ``GEECS-Plugins-Configs`` checkout is absent (e.g. in CI).
Locally this is the proof that the converters cover the real world, file by
file, with zero skips beyond the documented empty/deviceless shot-control
configs (which legitimately convert to "no trigger profile").

``save_devices/`` folders may mix legacy save elements with new-schema
``SaveSet`` files (top-level ``schema_version``); ``load_save_set`` branches
between the two exactly as geecs_bluesky's ``ConfigsRepoResolver`` does, and
``TestSaveSetLoadDispatch`` pins that dispatch hermetically (no corpus
needed, runs in CI).

Corpus layout (as found 2026-07-07)::

    scanner_configs/experiments/<Experiment>/
      save_devices/                  # save elements
      scan_devices/                  # scan_devices.yaml + composite_variables.yaml
      shot_control_configurations/   # trigger configs (incl. laser-on/off pairs)
      action_library/                # actions.yaml + assigned_actions.yaml
      scan_presets/                  # legacy presets
      optimizer_configs/             # Xopt optimizer configs (Undulator only)
      multiscan_presets/             # GUI queue presets (front-end state; no converter)
      aux_configs/                   # visa plunger lookup (app data; no converter)
"""

import os
from pathlib import Path

import pytest
import yaml

from geecs_schemas import SaveSet
from geecs_schemas.convert import (
    SchemaConversionError,
    convert_action_library,
    convert_assigned_actions,
    convert_optimizer_config,
    convert_save_element,
    convert_scan_preset,
    convert_scan_variables,
    convert_shot_control,
    merge_trigger_variant,
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


def load_save_set(path: Path):
    """Load one ``save_devices/`` file, whichever schema it uses.

    The corpus mixes legacy save elements with new-schema ``SaveSet`` files
    (marked by a top-level ``schema_version``), mirroring how geecs_bluesky's
    ``ConfigsRepoResolver`` branches between the two.  Empty files normalize
    to ``{}`` (as ``load_legacy`` and the resolver's ``_load_yaml`` both do)
    so they fail with the converter's typed error, not a bare ``TypeError``.

    Returns ``(save_set, conversion)``: *conversion* is ``None`` for
    new-schema files, and the full ``SaveElementConversion`` for legacy ones
    (whose *save_set* is ``None`` for action-only elements).
    """
    document = yaml.safe_load(path.read_text()) or {}
    if "schema_version" in document:
        return SaveSet.model_validate(document), None
    result = convert_save_element(document, name=path.stem)
    return result.save_set, result


class TestSaveSetLoadDispatch:
    """Hermetic pin of the dual-schema dispatch (runs without the corpus)."""

    def test_new_schema_file_validates_as_save_set(self, tmp_path):
        path = tmp_path / "new.yaml"
        path.write_text(
            "schema_version: 1\n"
            "name: new\n"
            "entries:\n"
            "- device: UC_Device\n"
            "  images: true\n"
        )
        save_set, conversion = load_save_set(path)
        assert conversion is None
        assert [entry.device for entry in save_set.entries] == ["UC_Device"]

    def test_legacy_file_goes_through_the_converter(self, tmp_path):
        path = tmp_path / "legacy.yaml"
        path.write_text(
            "Devices:\n"
            "  UC_Device:\n"
            "    variable_list: []\n"
            "    save_nonscalar_data: true\n"
        )
        save_set, conversion = load_save_set(path)
        assert conversion is not None
        assert [entry.device for entry in save_set.entries] == ["UC_Device"]
        assert save_set.entries[0].db_scalars is False

    def test_empty_file_fails_with_the_converter_error(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(SchemaConversionError, match="empty"):
            load_save_set(path)


@pytest.mark.integration
@skip_without_corpus
class TestFullCorpus:
    def test_every_save_element_converts(self):
        converted = 0
        for experiment in experiments():
            for path in sorted(experiment.glob("save_devices/*.yaml")):
                save_set, conversion = load_save_set(path)
                if conversion is None:
                    # new-schema file living in the legacy folder: already
                    # validated by load_save_set, nothing to convert
                    assert save_set.entries, path
                    continue
                assert save_set is not None or conversion.actions, path
                # converted legacy elements preserve exact legacy behavior:
                # explicit db_scalars=False on EVERY entry (the DB-first
                # True default is for new configs only), start/end override
                # maps untouched
                for entry in save_set.entries if save_set else []:
                    assert entry.db_scalars is False, path
                    assert entry.at_scan_start == {}, path
                    assert entry.at_scan_end == {}, path
                converted += 1
        assert converted >= 70  # 71 legacy files at the time of writing

    def test_every_scan_variable_catalog_converts(self):
        converted = 0
        for experiment in experiments():
            scan_devices = experiment / "scan_devices" / "scan_devices.yaml"
            composites = experiment / "scan_devices" / "composite_variables.yaml"
            if not scan_devices.exists() and not composites.exists():
                continue
            catalog = convert_scan_variables(
                scan_devices if scan_devices.exists() else None,
                composites if composites.exists() else None,
            )
            assert catalog.variables, experiment.name
            converted += 1
        assert converted >= 2  # Undulator + Thomson

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

    def test_laser_pairs_fold_into_variants(self):
        pairs = [
            ("Undulator", "HTU-Normal", "HTU-LaserOFF"),
            ("Thomson", "HTT-Normal", "HTT-LaserOFF"),
        ]
        for experiment, base_name, off_name in pairs:
            directory = (
                CONFIGS
                / "scanner_configs/experiments"
                / experiment
                / "shot_control_configurations"
            )
            base = convert_shot_control(directory / f"{base_name}.yaml")
            off = convert_shot_control(directory / f"{off_name}.yaml")
            merged = merge_trigger_variant(base, off, "laser_off")
            for state in ("OFF", "STANDBY", "SCAN", "SINGLESHOT", "ARMED"):
                # set comparison: write order within a transition may differ
                # when the variant appends writes the base lacked
                resolved = {
                    (w.device, w.variable, w.value)
                    for w in merged.writes_for(state, variant="laser_off")
                }
                expected = {
                    (w.device, w.variable, w.value) for w in off.writes_for(state)
                }
                assert resolved == expected, (experiment, state)

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

    def test_every_scan_preset_converts_and_composes(self):
        converted = 0
        for experiment in experiments():
            save_sets = {
                path.stem: load_save_set(path)[0]
                for path in experiment.glob("save_devices/*.yaml")
            }
            for path in sorted(experiment.glob("scan_presets/*.yaml")):
                known = {
                    name: save_sets[name]
                    for name in (convert_scan_preset(path).element_names)
                    if name in save_sets
                }
                conversion = convert_scan_preset(
                    path,
                    save_sets=known
                    if set(known) >= set(convert_scan_preset(path).element_names)
                    else None,
                )
                assert conversion.scan_request.save_sets == conversion.element_names
                converted += 1
        assert converted >= 12

    def test_every_optimizer_config_converts(self):
        converted = 0
        for experiment in experiments():
            for path in sorted(experiment.glob("optimizer_configs/*.yaml")):
                conversion = convert_optimizer_config(path)
                assert conversion.optimization.variables, path
                converted += 1
        assert converted >= 11

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


def test_action_library_in_the_legacy_dialect_is_refused(repo):
    """No converter any more (GEECS-Schemas 0.22.0): the schema names the regeneration.

    Both the resolution and the registry (the MCP's listing) raise — a
    listing that read empty would hide the regeneration the file needs.
    """
    from pydantic import ValidationError

    folder = repo / "TestExp" / ConfigsRepoResolver.ACTION_FOLDER
    folder.mkdir(exist_ok=True)
    (folder / "actions.yaml").write_text(
        yaml.safe_dump({"actions": {"x": {"steps": [{"action": "wait", "wait": 1}]}}})
    )
    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    with pytest.raises(ValidationError, match="legacy 'actions:' dialect"):
        resolver.resolve_action_plan("x")
    with pytest.raises(ValidationError, match="legacy 'actions:' dialect"):
        resolver.action_plan_registry()


def test_action_library_absent_or_empty(repo):
    """No file → an empty registry; an empty file → an empty library (the Console's rule)."""
    from geecs_bluesky.exceptions import GeecsConfigurationError

    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    assert resolver.action_plan_registry() == {}
    with pytest.raises(GeecsConfigurationError, match="not found"):
        resolver.resolve_action_plan("x")
    folder = repo / "TestExp" / ConfigsRepoResolver.ACTION_FOLDER
    folder.mkdir(exist_ok=True)
    (folder / "actions.yaml").write_text("")
    assert resolver.action_plan_registry() == {}
    with pytest.raises(GeecsConfigurationError, match="not in the"):
        resolver.resolve_action_plan("x")
    (folder / "actions.yaml").write_text("{}\n")  # _load_yaml reads None and {} alike
    assert resolver.action_plan_registry() == {}


def test_resolve_preset_missing_raises_with_kind(repo):
    from geecs_bluesky.exceptions import GeecsConfigurationError

    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    with pytest.raises(GeecsConfigurationError, match="preset 'nope'"):
        resolver.resolve_preset("nope")


# ------------------------------------------------- measured drain offsets


OFFSETS = {
    "schema_version": 1,
    "reference": "uc_amp3_ir_input",
    "devices": {
        "uc_amp3_ir_input": {"offset_s": 0.0, "scatter_s": 0.004, "shots": 10},
        "uc_amp4_ir_input": {"offset_s": 0.036, "scatter_s": 0.009, "shots": 10},
    },
    "measured_at": "2026-09-13T18:22:04-07:00",
    "trigger_profile": "HTU-LaserOFF",
}


def test_shot_offsets_absent_reads_as_never_measured(repo):
    """Every experiment's state until the calibration is first run."""
    assert ConfigsRepoResolver("TestExp", repo).resolve_shot_offsets() is None


def test_shot_offsets_round_trip(repo):
    resolver = ConfigsRepoResolver("TestExp", repo)
    resolver.shot_offsets_path.write_text(yaml.safe_dump(OFFSETS))
    document = resolver.resolve_shot_offsets()
    assert document.reference == "uc_amp3_ir_input"
    assert document.offset_for("uc_amp4_ir_input") == pytest.approx(0.036)
    assert document.trigger_profile == "HTU-LaserOFF"


def test_an_invalid_shot_offsets_document_is_loud(repo):
    """Never a silent fall back to zeros.

    A calibration that quietly reverted to 0.0 would misjoin rows at a tight
    rep rate with nothing in the log to say why — the failure this whole
    phase exists to prevent. The worker's startup catches this and warns;
    the resolver itself must raise so there is something to catch.
    """
    resolver = ConfigsRepoResolver("TestExp", repo)
    broken = dict(OFFSETS, reference="a_device_it_does_not_list")
    resolver.shot_offsets_path.write_text(yaml.safe_dump(broken))
    with pytest.raises(Exception):
        resolver.resolve_shot_offsets()


def test_write_shot_offsets_replaces_the_previous_measurement(repo):
    from geecs_schemas import DeviceOffset, ShotOffsets

    resolver = ConfigsRepoResolver("TestExp", repo)
    resolver.shot_offsets_path.write_text(yaml.safe_dump(OFFSETS))
    fresh = ShotOffsets(
        reference="uc_amp4_ir_input",
        devices={
            "uc_amp4_ir_input": DeviceOffset(offset_s=0.0),
            "uc_amp3_ir_input": DeviceOffset(offset_s=0.012),
        },
    )
    path = resolver.write_shot_offsets(fresh)
    assert path == resolver.shot_offsets_path
    again = resolver.resolve_shot_offsets()
    assert again.reference == "uc_amp4_ir_input"
    assert again.offset_for("uc_amp3_ir_input") == pytest.approx(0.012)
    # Atomic: no temporary file left behind for the next listing to trip on.
    assert [p.name for p in path.parent.glob(".*tmp")] == []


def test_write_shot_offsets_refuses_a_missing_experiment_folder(tmp_path):
    """A misconfigured or unmounted configs root must not be papered over.

    Creating the tree here would plant an experiment folder wherever the
    misconfiguration pointed, and the next read would find a calibration
    that no reviewer ever saw.
    """
    from geecs_bluesky.exceptions import GeecsConfigurationError
    from geecs_schemas import DeviceOffset, ShotOffsets

    resolver = ConfigsRepoResolver("NoSuchExp", tmp_path)
    document = ShotOffsets(reference="a", devices={"a": DeviceOffset(offset_s=0.0)})
    with pytest.raises(GeecsConfigurationError, match="no configs folder"):
        resolver.write_shot_offsets(document)
    assert not (tmp_path / "NoSuchExp").exists()


def test_write_shot_offsets_keeps_the_file_readable(repo):
    """A shared git checkout: 0600 would lock out the reviewer and git.

    NamedTemporaryFile creates 0600 and os.replace keeps the temp inode's
    mode, so without an explicit chmod one write makes the calibration
    unreadable to the operator who has to commit it, to another host's
    resolver, and to git run as anyone else (review of #861, finding 6).
    """
    import os
    import stat

    from geecs_schemas import DeviceOffset, ShotOffsets

    resolver = ConfigsRepoResolver("TestExp", repo)
    document = ShotOffsets(reference="a", devices={"a": DeviceOffset(offset_s=0.0)})
    # A new file gets a sane default.
    path = resolver.write_shot_offsets(document)
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    # An existing file keeps whatever the repo gave it.
    os.chmod(path, 0o664)
    resolver.write_shot_offsets(document)
    assert stat.S_IMODE(path.stat().st_mode) == 0o664


def test_a_failed_write_leaves_no_temp_file_and_keeps_the_calibration(
    repo, monkeypatch
):
    """The realistic failure is a full or disconnected share, mid-write.

    Only the replace was guarded before, so a write that raised left a
    dot-prefixed temp file in the configs git working tree (review of #861,
    finding 7). The previous calibration must also survive intact.
    """
    import io
    import tempfile

    from geecs_schemas import DeviceOffset, ShotOffsets

    resolver = ConfigsRepoResolver("TestExp", repo)
    resolver.shot_offsets_path.write_text(yaml.safe_dump(OFFSETS))
    before = resolver.shot_offsets_path.read_text()

    real_named_temporary = tempfile.NamedTemporaryFile

    class FullDisk(io.StringIO):
        """A real temp file on disk whose `write` fails, as a full share does."""

        def __init__(self, handle):
            super().__init__()
            self._handle = handle
            self.name = handle.name

        def write(self, _data):
            raise OSError(28, "No space left on device")

        def flush(self):
            pass

        def fileno(self):
            return self._handle.fileno()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()
            return False

    monkeypatch.setattr(
        "geecs_bluesky.config_resolver.tempfile.NamedTemporaryFile",
        lambda *a, **k: FullDisk(real_named_temporary(*a, **k)),
    )
    document = ShotOffsets(reference="a", devices={"a": DeviceOffset(offset_s=0.0)})
    with pytest.raises(OSError):
        resolver.write_shot_offsets(document)
    assert resolver.shot_offsets_path.read_text() == before
    assert list(resolver.shot_offsets_path.parent.glob(".*tmp")) == []


def test_a_share_that_refuses_chmod_still_gets_the_calibration(repo, monkeypatch):
    """A mode we cannot set must never cost the ten shots the document took.

    The configs repo usually lives on the data share, and CIFS mounts reject
    `chmod` with EPERM unless mounted with unix extensions. The chmod sits
    between the fsync and the replace, inside the try — so an EPERM there
    would unlink the temp and propagate, failing every `write=True` run on
    such a mount. A cosmetic problem must not destroy a measurement (review
    of #861, finding 6).
    """
    from geecs_schemas import DeviceOffset, ShotOffsets

    resolver = ConfigsRepoResolver("TestExp", repo)

    def refuse(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr("geecs_bluesky.config_resolver.os.chmod", refuse)
    document = ShotOffsets(reference="a", devices={"a": DeviceOffset(offset_s=0.0)})
    path = resolver.write_shot_offsets(document)
    assert path.exists()
    assert resolver.resolve_shot_offsets().reference == "a"
    assert list(path.parent.glob(".*tmp")) == []


def _preset(name: str = "s1h_steer", **overrides):
    from geecs_schemas import Preset

    doc = {
        "name": name,
        "description": "steer S1H",
        "trigger_profile": "HTU-LaserOFF",
        "devices": [{"device": "UC_Amp4_IR_input", "save_images": True}],
        "plan": {
            "name": "scan",
            "args": ["S1H current", 0.0, 0.2, 3],
            "kwargs": {"shots_per_step": 3, "acquisition": "strict"},
        },
    }
    doc.update(overrides)
    return Preset.model_validate(doc)


def test_write_preset_round_trips_through_resolve_and_listing(repo):
    resolver = ConfigsRepoResolver("TestExp", repo)
    path = resolver.write_preset(_preset())
    assert (
        path == repo / "TestExp" / ConfigsRepoResolver.PRESET_FOLDER / "s1h_steer.yaml"
    )
    assert "s1h_steer" in resolver.list_presets()
    again = resolver.resolve_preset("s1h_steer")
    assert again.plan.args == ["S1H current", 0.0, 0.2, 3]
    assert again.trigger_profile == "HTU-LaserOFF"
    assert yaml.safe_load(path.read_text())["schema_version"] == 1
    assert [p.name for p in path.parent.glob(".*tmp")] == []


def test_write_preset_refuses_to_replace_unless_asked(repo):
    from geecs_bluesky.exceptions import GeecsConfigurationError

    resolver = ConfigsRepoResolver("TestExp", repo)
    # Amp4In exists in the fixture (as an empty document).
    before = resolver.preset_path("Amp4In").read_text()
    with pytest.raises(GeecsConfigurationError, match="already exists"):
        resolver.write_preset(_preset("Amp4In"))
    assert resolver.preset_path("Amp4In").read_text() == before
    resolver.write_preset(_preset("Amp4In"), overwrite=True)
    assert resolver.resolve_preset("Amp4In").description == "steer S1H"


def test_write_preset_refuses_a_name_that_is_not_a_file_stem(repo):
    from geecs_bluesky.exceptions import GeecsConfigurationError

    resolver = ConfigsRepoResolver("TestExp", repo)
    for bad in ("../escape", "a/b", ".hidden", "", "with space"):
        with pytest.raises(GeecsConfigurationError, match="not a file name"):
            resolver.write_preset(_preset(bad))
    assert not (repo / "escape.yaml").exists()


def test_write_preset_refuses_a_missing_experiment_folder(tmp_path):
    """Same rule as write_shot_offsets: never plant an experiment folder."""
    from geecs_bluesky.exceptions import GeecsConfigurationError

    resolver = ConfigsRepoResolver("NoSuchExp", tmp_path)
    with pytest.raises(GeecsConfigurationError, match="no configs folder"):
        resolver.write_preset(_preset())
    assert not (tmp_path / "NoSuchExp").exists()


def test_write_preset_creates_only_the_presets_folder(tmp_path):
    """A fresh experiment has the folder but no presets/ yet; that one is fine to add."""
    (tmp_path / "Fresh").mkdir()
    resolver = ConfigsRepoResolver("Fresh", tmp_path)
    path = resolver.write_preset(_preset())
    assert path.parent == tmp_path / "Fresh" / "presets"
    assert resolver.list_presets() == ["s1h_steer"]


def test_write_preset_strips_a_yaml_suffix_into_the_document(repo):
    resolver = ConfigsRepoResolver("TestExp", repo)
    path = resolver.write_preset(_preset("jet.yaml.yaml"))
    assert path.name == "jet.yaml"
    assert (
        "jet" in resolver.list_presets() and "jet.yaml" not in resolver.list_presets()
    )
    assert resolver.resolve_preset("jet").name == "jet"

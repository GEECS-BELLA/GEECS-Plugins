"""Pin the resolver's config-listing surface (#666).

Non-GUI clients (the scan MCP, notebooks) need to enumerate the
experiment's presets / trigger profiles / optimizer configs — the
listing lives beside the resolution it feeds: sorted YAML stems, and every
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
            doc = (
                {
                    "vocs": {
                        "variables": {"Motor:Current": [-1, 1]},
                        "objectives": {"score": "MINIMIZE"},
                    },
                    "measurements": {"score": {"signal": "Meter:Value"}},
                    "generator": {"name": "random"},
                }
                if folder == ConfigsRepoResolver.OPTIMIZER_FOLDER
                else {}
            )
            (d / f"{name}.yaml").write_text(yaml.safe_dump(doc))
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
    # The listings count .yml files, so resolution must accept them too:
    # a listed name that resolve_* refuses on spelling alone is a
    # client-facing trap.
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
    with pytest.raises(Exception):  # noqa: B017 — asserts any failure
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
        "geecs_bluesky._fs.tempfile.NamedTemporaryFile",
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

    monkeypatch.setattr("geecs_bluesky._fs.os.chmod", refuse)
    document = ShotOffsets(reference="a", devices={"a": DeviceOffset(offset_s=0.0)})
    path = resolver.write_shot_offsets(document)
    assert path.exists()
    assert resolver.resolve_shot_offsets().reference == "a"
    assert list(path.parent.glob(".*tmp")) == []


def test_the_measurement_lands_under_the_write_root_never_the_read_root(tmp_path):
    """On the box the read root is the configurator's live checkout, which is never written.

    The write root mirrors the configs tree (the configurator's inbox); the
    read path is untouched, so the live checkout stays clean for its
    fast-forwards and the worker reads the committed file back from it.
    """
    from geecs_schemas import DeviceOffset, ShotOffsets

    (tmp_path / "live" / "TestExp").mkdir(parents=True)
    (tmp_path / "inbox" / "TestExp").mkdir(parents=True)
    resolver = ConfigsRepoResolver(
        "TestExp",
        tmp_path / "live",
        experiments_write_root=tmp_path / "inbox",
    )
    document = ShotOffsets(reference="a", devices={"a": DeviceOffset(offset_s=0.0)})
    path = resolver.write_shot_offsets(document)
    assert path == tmp_path / "inbox" / "TestExp" / "shot_offsets.yaml"
    assert not resolver.shot_offsets_path.exists()
    assert resolver.resolve_shot_offsets() is None  # read from the live checkout


def test_the_write_root_comes_from_the_site_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GEECS_SCANNER_CONFIG_DIR", str(tmp_path / "live"))
    monkeypatch.setenv("GEECS_CONFIGS_WRITE_ROOT", str(tmp_path / "inbox"))
    resolver = ConfigsRepoResolver("TestExp")
    expected = tmp_path / "inbox" / "scanner_configs" / "experiments" / "TestExp"
    assert resolver.shot_offsets_write_path == expected.resolve() / "shot_offsets.yaml"
    assert (
        resolver.shot_offsets_path.parent == (tmp_path / "live" / "TestExp").resolve()
    )


def test_a_missing_inbox_folder_is_refused_not_created(tmp_path):
    """The worker never creates the inbox's experiment folder (the configurator does, at start)."""
    from geecs_bluesky.exceptions import GeecsConfigurationError
    from geecs_schemas import DeviceOffset, ShotOffsets

    (tmp_path / "live" / "TestExp").mkdir(parents=True)
    resolver = ConfigsRepoResolver(
        "TestExp", tmp_path / "live", experiments_write_root=tmp_path / "inbox"
    )
    document = ShotOffsets(reference="a", devices={"a": DeviceOffset(offset_s=0.0)})
    with pytest.raises(GeecsConfigurationError, match="no configs folder"):
        resolver.write_shot_offsets(document)
    assert not (tmp_path / "inbox").exists()


# ------------------------------------------------------------------- setups


def _write_setup(repo, name, steps, *, suffix=".yaml"):
    folder = repo / "TestExp" / "setups"
    folder.mkdir(exist_ok=True)
    (folder / f"{name}{suffix}").write_text(
        yaml.safe_dump({"schema_version": 1, "name": name, "steps": steps})
    )


def test_setups_are_read_fresh_and_resolved_by_name(repo):
    from geecs_bluesky.exceptions import GeecsConfigurationError

    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    assert resolver.setups() == {}  # no folder: none, and none created
    assert not (repo / "TestExp" / "setups").exists()
    _write_setup(repo, "low_power", [{"do": "wait", "seconds": 1}], suffix=".yml")
    setup, setups = resolver.resolve_setup("low_power")
    assert setup.name == "low_power" and set(setups) == {"low_power"}
    assert resolver.setup_repo_path("low_power") == (
        "scanner_configs/experiments/TestExp/setups/low_power.yml"
    )
    with pytest.raises(GeecsConfigurationError, match=r"Known setups: \['low_power'\]"):
        resolver.resolve_setup("high_power")
    (repo / "TestExp" / "setups" / "broken.yaml").write_text("name: other\nsteps: []\n")
    with pytest.raises(GeecsConfigurationError, match="cannot be read"):
        resolver.setups()


def test_the_policy_and_panel_devices_absent_are_empty_and_invalid_refuse(repo):
    from geecs_bluesky.exceptions import GeecsConfigurationError

    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    assert resolver.resolve_operator_policy().hazards == []
    assert resolver.resolve_panel_devices().devices == {}
    (repo / "TestExp" / "operator_policy.yaml").write_text("hazards: [{roles: []}]\n")
    with pytest.raises(GeecsConfigurationError, match="operator policy"):
        resolver.resolve_operator_policy()
    (repo / "TestExp" / "panel_devices.yaml").write_text(
        "devices: {X: {family: nope}}\n"
    )
    with pytest.raises(GeecsConfigurationError, match="panel devices"):
        resolver.resolve_panel_devices()


def test_configs_commit_is_the_checkouts_head_and_none_outside_one(repo):
    import shutil
    import subprocess

    resolver = ConfigsRepoResolver("TestExp", experiments_root=repo)
    assert resolver.configs_commit() is None
    vcs = shutil.which("git")
    if vcs is None:
        pytest.skip("no git on this host")
    ident = ["-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([vcs, "init", "-q", str(repo)], check=True)
    subprocess.run([vcs, "-C", str(repo), "add", "-A"], check=True)
    subprocess.run([vcs, "-C", str(repo), *ident, "commit", "-qm", "x"], check=True)
    head = subprocess.run(
        [vcs, "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert resolver.configs_commit() == head

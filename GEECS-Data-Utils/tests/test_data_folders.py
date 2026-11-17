"""Pin the ``[data_folders]`` config section: an experiment's data folder.

The GEECS convention puts an experiment's scans under
``{base}/{experiment}/Y{YYYY}/...``. A site whose experiment keeps its
``Y{YYYY}`` folders directly in the data root (or under a folder named
otherwise) says so in ``config.ini``::

    [data_folders]
    Flat =
    Renamed = OtherFolder

These tests pin both directions — building a path from a tag and parsing a
tag back from a path — and that an experiment without an entry, or a config
without the section, behaves exactly as before.
"""

from __future__ import annotations

import logging
import textwrap
from datetime import date
from pathlib import Path

import pytest

from geecs_data_utils.geecs_paths_config import GeecsPathsConfig
from geecs_data_utils.folder_catalog import FolderScanCatalog, parse_folder_uid
from geecs_data_utils.scan_paths import (
    ScanPaths,
    daily_scan_folder,
    experiment_dir_present,
)

DAY = (2026, 11, 17)
DAY_PARTS = ("Y2026", "11-Nov", "26_1117", "scans")


@pytest.fixture
def propagate_paths_config_logger():
    """The module sets propagate=False; flip it so caplog can capture."""
    pcfg_logger = logging.getLogger("geecs_data_utils.geecs_paths_config")
    original = pcfg_logger.propagate
    pcfg_logger.propagate = True
    try:
        yield
    finally:
        pcfg_logger.propagate = original


def _config(tmp_path: Path, data_folders: str | None) -> GeecsPathsConfig:
    """A GeecsPathsConfig over ``tmp_path/data`` with an optional [data_folders] body."""
    base = tmp_path / "data"
    base.mkdir(exist_ok=True)
    body = f"""
    [Paths]
    GEECS_DATA_LOCAL_BASE_PATH = {base}

    [Experiment]
    expt = Conventional
    """
    if data_folders is not None:
        body += "\n[data_folders]\n" + textwrap.dedent(data_folders)
    ini = tmp_path / "config.ini"
    ini.write_text(textwrap.dedent(body).lstrip())
    return GeecsPathsConfig(config_path=ini)


@pytest.fixture
def mapped(tmp_path, monkeypatch) -> GeecsPathsConfig:
    """ScanPaths configured with one flat and one renamed experiment."""
    config = _config(tmp_path, "Flat =\nRenamed = OtherFolder\n")
    monkeypatch.setattr(ScanPaths, "paths_config", config)
    return config


def _tag(experiment: str, number: int = 1):
    return ScanPaths.get_scan_tag(*DAY, number, experiment=experiment)


class TestReadingTheSection:
    def test_entries_keep_the_experiment_names_case(self, mapped):
        assert mapped.data_folders == {"Flat": "", "Renamed": "OtherFolder"}

    def test_no_section_means_no_entries(self, tmp_path):
        assert _config(tmp_path, None).data_folders == {}

    @pytest.mark.parametrize("value", ["..", ".", "a/b", "a\\b"])
    def test_a_value_that_is_not_one_folder_name_is_ignored(
        self, tmp_path, value, propagate_paths_config_logger, caplog
    ):
        config = _config(tmp_path, f"Bad = {value}\n")
        assert config.data_folders == {}
        assert config.data_folder_for("Bad") == "Bad"
        assert "not a single folder name" in caplog.text


class TestBuildingPaths:
    def test_a_flat_experiment_has_no_experiment_folder(self, mapped):
        folder = ScanPaths.get_daily_scan_folder(tag=_tag("Flat"))
        assert folder == mapped.base_path.joinpath(*DAY_PARTS)

    def test_a_renamed_experiment_uses_its_folder(self, mapped):
        folder = ScanPaths.get_daily_scan_folder(tag=_tag("Renamed"))
        assert folder == mapped.base_path.joinpath("OtherFolder", *DAY_PARTS)

    def test_an_unlisted_experiment_keeps_the_convention(self, mapped):
        folder = ScanPaths.get_daily_scan_folder(tag=_tag("Conventional"))
        assert folder == mapped.base_path.joinpath("Conventional", *DAY_PARTS)

    def test_the_module_level_daily_folder_follows(self, mapped):
        assert daily_scan_folder("Flat", day=date(*DAY)) == mapped.base_path.joinpath(
            *DAY_PARTS
        )


class TestParsingPaths:
    def test_a_scan_in_the_data_root_belongs_to_the_flat_experiment(self, mapped):
        folder = mapped.base_path.joinpath(*DAY_PARTS, "Scan004")
        folder.mkdir(parents=True)
        paths = ScanPaths(folder=folder)
        assert paths.get_tag().experiment == "Flat"
        assert ScanPaths.get_scan_folder_path(paths.get_tag()) == folder

    def test_a_scan_in_a_renamed_folder_belongs_to_its_experiment(self, mapped):
        folder = mapped.base_path.joinpath("OtherFolder", *DAY_PARTS, "Scan002")
        folder.mkdir(parents=True)
        paths = ScanPaths(folder=folder)
        assert paths.get_tag().experiment == "Renamed"
        assert ScanPaths.get_scan_folder_path(paths.get_tag()) == folder

    def test_a_conventional_scan_is_unchanged(self, mapped):
        folder = mapped.base_path.joinpath("Conventional", *DAY_PARTS, "Scan001")
        folder.mkdir(parents=True)
        assert ScanPaths(folder=folder).get_tag().experiment == "Conventional"

    def test_an_explicit_base_directory_is_the_root_compared(self, mapped, tmp_path):
        other_root = tmp_path / "mirror"
        folder = other_root.joinpath(*DAY_PARTS, "Scan001")
        folder.mkdir(parents=True)
        paths = ScanPaths(tag=_tag("Flat"), base_directory=other_root)
        assert paths.get_folder() == folder
        assert paths.get_tag().experiment == "Flat"

    def test_two_flat_experiments_cannot_claim_the_root(
        self, tmp_path, monkeypatch, propagate_paths_config_logger, caplog
    ):
        config = _config(tmp_path, "One =\nTwo =\n")
        monkeypatch.setattr(ScanPaths, "paths_config", config)
        folder = config.base_path.joinpath(*DAY_PARTS, "Scan001")
        folder.mkdir(parents=True)
        assert ScanPaths(folder=folder).get_tag().experiment == "data"
        assert "several experiments" in caplog.text


class TestWithoutTheSection:
    def test_paths_and_tags_are_exactly_as_before(self, tmp_path, monkeypatch):
        config = _config(tmp_path, None)
        monkeypatch.setattr(ScanPaths, "paths_config", config)
        folder = config.base_path.joinpath("Undulator", *DAY_PARTS, "Scan003")
        folder.mkdir(parents=True)
        tag = ScanPaths(folder=folder).get_tag()
        assert tag.experiment == "Undulator"
        assert ScanPaths.get_scan_folder_path(tag) == folder

    def test_a_scan_in_the_root_still_reads_the_root_name(self, tmp_path, monkeypatch):
        config = _config(tmp_path, None)
        monkeypatch.setattr(ScanPaths, "paths_config", config)
        folder = config.base_path.joinpath(*DAY_PARTS, "Scan001")
        folder.mkdir(parents=True)
        assert ScanPaths(folder=folder).get_tag().experiment == "data"


class TestScannerSide:
    """The claim's path: next number in the flat day, created there (read_mode=False)."""

    def test_the_next_scan_numbers_after_existing_flat_scans(self, mapped):
        day = mapped.base_path.joinpath(*DAY_PARTS)
        for n in (1, 2):
            (day / f"Scan{n:03d}").mkdir(parents=True)
        tag = ScanPaths.get_next_scan_tag(
            experiment="Flat", year=DAY[0], month=DAY[1], day=DAY[2]
        )
        assert (tag.experiment, tag.number) == ("Flat", 3)

        created = ScanPaths(tag=tag, read_mode=False).get_folder()
        assert created == day / "Scan003"
        assert created.is_dir()
        assert not (mapped.base_path / "Flat").exists()


class TestFolderCatalog:
    """The portal's folder catalog names the experiment, not the folder above Y{YYYY}."""

    def _scan(self, root: Path, number: int) -> None:
        folder = root.joinpath(*DAY_PARTS, f"Scan{number:03d}")
        folder.mkdir(parents=True)
        (folder / f"ScanInfoScan{number:03d}.ini").write_text(
            '[Scan Info]\nScan No = "1"\nScan Parameter = "Shotnumber"\n'
        )

    @pytest.mark.parametrize(
        ("experiment", "folder"), [("Flat", ""), ("Renamed", "OtherFolder")]
    )
    def test_listed_scans_carry_the_experiment_and_load(
        self, mapped, experiment, folder
    ):
        root = mapped.base_path / folder if folder else mapped.base_path
        self._scan(root, 1)
        catalog = FolderScanCatalog()
        runs = catalog.list_runs(experiment, date(*DAY))
        assert [parse_folder_uid(r.uid)[0] for r in runs] == [experiment]
        detail = catalog.load_run(runs[0].uid)
        assert detail.summary.scan_number == 1


class TestExperimentDirPresent:
    """The "share is mounted" sentinel: a flat experiment's folder is the data root."""

    def test_an_empty_root_is_not_a_flat_experiments_share(self, mapped):
        assert not experiment_dir_present(mapped.base_path, "Flat")
        (mapped.base_path / "Y2026").mkdir()
        assert experiment_dir_present(mapped.base_path, "Flat")

    def test_a_conventional_folder_needs_only_to_exist(self, mapped):
        folder = mapped.base_path / "Conventional"
        assert not experiment_dir_present(folder, "Conventional")
        folder.mkdir()
        assert experiment_dir_present(folder, "Conventional")

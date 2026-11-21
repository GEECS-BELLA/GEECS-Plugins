"""Pin ``[Paths] data_root_experiment``: the experiment that owns the data root.

The GEECS convention puts an experiment's scans under
``{base}/{experiment}/Y{YYYY}/...``. A site whose experiment keeps its
``Y{YYYY}`` folders directly in the data root (a data tree that predates
the convention) names that one experiment in ``config.ini``::

    [Paths]
    data_root_experiment = Flat

These tests pin both directions — building a path from a tag and parsing a
tag back from a path — and that every other experiment, and a config
without the key, behaves exactly as before.
"""

from __future__ import annotations

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


def _config(tmp_path: Path, data_root_experiment: str | None) -> GeecsPathsConfig:
    """A GeecsPathsConfig over ``tmp_path/data``, with the key when given."""
    base = tmp_path / "data"
    base.mkdir(exist_ok=True)
    lines = ["[Paths]", f"GEECS_DATA_LOCAL_BASE_PATH = {base}"]
    if data_root_experiment is not None:
        lines.append(f"data_root_experiment = {data_root_experiment}")
    lines += ["", "[Experiment]", "expt = Conventional", ""]
    ini = tmp_path / "config.ini"
    ini.write_text("\n".join(lines))
    return GeecsPathsConfig(config_path=ini)


@pytest.fixture
def mapped(tmp_path, monkeypatch) -> GeecsPathsConfig:
    """ScanPaths configured with ``Flat`` owning the data root."""
    config = _config(tmp_path, "Flat")
    monkeypatch.setattr(ScanPaths, "paths_config", config)
    return config


def _tag(experiment: str, number: int = 1):
    return ScanPaths.get_scan_tag(*DAY, number, experiment=experiment)


class TestReadingTheKey:
    def test_the_value_keeps_its_case(self, mapped):
        assert mapped.data_root_experiment == "Flat"

    @pytest.mark.parametrize("value", [None, ""])
    def test_absent_or_empty_means_none(self, tmp_path, value):
        assert _config(tmp_path, value).data_root_experiment is None


class TestBuildingPaths:
    def test_the_root_experiment_has_no_experiment_folder(self, mapped):
        folder = ScanPaths.get_daily_scan_folder(tag=_tag("Flat"))
        assert folder == mapped.base_path.joinpath(*DAY_PARTS)

    def test_another_experiment_keeps_the_convention(self, mapped):
        folder = ScanPaths.get_daily_scan_folder(tag=_tag("Conventional"))
        assert folder == mapped.base_path.joinpath("Conventional", *DAY_PARTS)

    def test_the_module_level_daily_folder_follows(self, mapped):
        assert daily_scan_folder("Flat", day=date(*DAY)) == mapped.base_path.joinpath(
            *DAY_PARTS
        )

    def test_without_a_config_the_convention_holds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ScanPaths, "paths_config", None)
        folder = ScanPaths.get_daily_scan_folder(
            tag=_tag("Flat"), base_directory=tmp_path
        )
        assert folder == tmp_path.joinpath("Flat", *DAY_PARTS)


class TestParsingPaths:
    def test_a_scan_in_the_data_root_belongs_to_the_root_experiment(self, mapped):
        folder = mapped.base_path.joinpath(*DAY_PARTS, "Scan004")
        folder.mkdir(parents=True)
        paths = ScanPaths(folder=folder)
        assert paths.get_tag().experiment == "Flat"
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


class TestWithoutTheKey:
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
    """The claim's path: next number in the root's day, created there (read_mode=False)."""

    def test_the_next_scan_numbers_after_existing_root_scans(self, mapped):
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

    def test_listed_scans_carry_the_experiment_and_load(self, mapped):
        folder = mapped.base_path.joinpath(*DAY_PARTS, "Scan001")
        folder.mkdir(parents=True)
        (folder / "ScanInfoScan001.ini").write_text(
            '[Scan Info]\nScan No = "1"\nScan Parameter = "Shotnumber"\n'
        )
        catalog = FolderScanCatalog()
        runs = catalog.list_runs("Flat", date(*DAY))
        assert [parse_folder_uid(r.uid)[0] for r in runs] == ["Flat"]
        detail = catalog.load_run(runs[0].uid)
        assert detail.summary.scan_number == 1

    def test_a_bare_mount_point_is_an_outage_not_an_empty_day(self, mapped):
        """The root experiment's folder is the data root: empty, the share is down."""
        catalog = FolderScanCatalog()
        with pytest.raises(OSError, match="unreachable"):
            catalog.list_runs("Flat", date(*DAY))
        mapped.experiment = "Flat"  # the default experiment, named by ""
        with pytest.raises(OSError, match="unreachable"):
            catalog.list_runs("", date(*DAY))
        (mapped.base_path / "Y2020").mkdir()  # the share is back; the day has no scans
        assert catalog.list_runs("Flat", date(*DAY)) == []
        assert catalog.list_runs("", date(*DAY)) == []


class TestExperimentDirPresent:
    """The "share is mounted" sentinel: the root experiment's folder is the data root."""

    def test_an_empty_root_is_not_the_root_experiments_share(self, mapped):
        assert not experiment_dir_present(mapped.base_path, "Flat")
        (mapped.base_path / "Y2026").mkdir()
        assert experiment_dir_present(mapped.base_path, "Flat")

    def test_a_conventional_folder_needs_only_to_exist(self, mapped):
        folder = mapped.base_path / "Conventional"
        assert not experiment_dir_present(folder, "Conventional")
        folder.mkdir()
        assert experiment_dir_present(folder, "Conventional")

    def test_without_a_config_a_folder_needs_only_to_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ScanPaths, "paths_config", None)
        assert experiment_dir_present(tmp_path, "Flat")

"""Ops-menu path resolvers: pure functions against tmp trees, no Finder.

The load-bearing pin here is TestTodaysScanFolder.test_never_creates_*:
resolving today's scans folder must NEVER create directories — GUI code is
a consumer of scan folders, never a producer (repo-wide invariant).
"""

from datetime import date
from pathlib import Path

from geecs_console.services import ops_paths


def tree_snapshot(root: Path) -> set:
    """Every path under *root*, for asserting a tree is untouched."""
    return set(root.rglob("*"))


class TestExperimentConfigsFolder:
    def test_existing_experiment_dir_resolves(self, tmp_path):
        (tmp_path / "HTU").mkdir()
        assert (
            ops_paths.experiment_configs_folder("HTU", base=tmp_path)
            == tmp_path / "HTU"
        )

    def test_missing_experiment_dir_is_none(self, tmp_path):
        assert ops_paths.experiment_configs_folder("Ghost", base=tmp_path) is None

    def test_empty_experiment_falls_back_to_base(self, tmp_path):
        assert ops_paths.experiment_configs_folder("", base=tmp_path) == tmp_path

    def test_unresolvable_base_is_none(self, monkeypatch):
        # base=None consults the configs root; force it unresolvable.
        from geecs_console.services import configs

        monkeypatch.setattr(configs, "_configs_base", lambda: None)
        assert ops_paths.experiment_configs_folder("HTU") is None

    def test_nothing_created(self, tmp_path):
        before = tree_snapshot(tmp_path)
        ops_paths.experiment_configs_folder("Ghost", base=tmp_path)
        assert tree_snapshot(tmp_path) == before


class TestUserConfigTarget:
    def test_existing_file_resolves_to_the_file(self, tmp_path):
        config = tmp_path / "config.ini"
        config.write_text("[Paths]\n")
        assert ops_paths.user_config_target(config) == config

    def test_missing_file_with_existing_dir_resolves_to_the_dir(self, tmp_path):
        assert ops_paths.user_config_target(tmp_path / "config.ini") == tmp_path

    def test_missing_dir_is_none(self, tmp_path):
        assert ops_paths.user_config_target(tmp_path / "nope" / "config.ini") is None

    def test_default_points_at_geecs_python_api_config(self):
        # The console opens the shared config file by path only — it must
        # never import geecs_python_api (pinned by test_no_geecs_python_api).
        assert ops_paths.USER_CONFIG_PATH == Path(
            "~/.config/geecs_python_api/config.ini"
        )


class TestTodaysScanFolder:
    DAY = date(2026, 7, 11)

    def expected(self, base: Path) -> Path:
        return base / "TestExp" / "Y2026" / "07-Jul" / "26_0711" / "scans"

    def test_resolves_existing_daily_folder(self, tmp_path):
        target = self.expected(tmp_path)
        target.mkdir(parents=True)
        resolved = ops_paths.todays_scan_folder(
            "TestExp", base_path=tmp_path, today=self.DAY
        )
        assert resolved == target
        assert resolved.is_dir()

    def test_missing_daily_folder_returns_candidate_path(self, tmp_path):
        resolved = ops_paths.todays_scan_folder(
            "TestExp", base_path=tmp_path, today=self.DAY
        )
        assert resolved == self.expected(tmp_path)
        assert not resolved.exists()  # the caller reports "no scans today"

    def test_never_creates_directories(self, tmp_path):
        """The invariant pin: resolving with a missing folder changes nothing."""
        before = tree_snapshot(tmp_path)
        assert before == set()
        ops_paths.todays_scan_folder("TestExp", base_path=tmp_path, today=self.DAY)
        assert tree_snapshot(tmp_path) == set()

    def test_never_creates_even_when_partially_present(self, tmp_path):
        (tmp_path / "TestExp" / "Y2026").mkdir(parents=True)
        before = tree_snapshot(tmp_path)
        ops_paths.todays_scan_folder("TestExp", base_path=tmp_path, today=self.DAY)
        assert tree_snapshot(tmp_path) == before

    def test_unresolvable_without_config_is_none(self, monkeypatch):
        from geecs_data_utils import ScanPaths

        monkeypatch.setattr(ScanPaths, "paths_config", None)
        assert ops_paths.todays_scan_folder("TestExp") is None  # no base path
        assert ops_paths.todays_scan_folder("", base_path=Path("/tmp")) is None

    def test_experiment_falls_back_to_paths_config(self, tmp_path, monkeypatch):
        from geecs_data_utils import ScanPaths

        class FakePathsConfig:
            base_path = tmp_path
            experiment = "TestExp"

        monkeypatch.setattr(ScanPaths, "paths_config", FakePathsConfig())
        resolved = ops_paths.todays_scan_folder("", today=self.DAY)
        assert resolved == self.expected(tmp_path)


class TestHighestScanNumber:
    def make_scans(self, tmp_path: Path, *names: str) -> Path:
        scans = tmp_path / "scans"
        scans.mkdir()
        for name in names:
            (scans / name).mkdir()
        return scans

    def test_highest_of_several(self, tmp_path):
        scans = self.make_scans(tmp_path, "Scan001", "Scan002", "Scan017")
        assert ops_paths.highest_scan_number(scans) == 17

    def test_ignores_non_scan_entries(self, tmp_path):
        scans = self.make_scans(tmp_path, "Scan003", "ScanData", "Scan01", "notes")
        (scans / "Scan999.txt").write_text("a file, not a scan folder")
        assert ops_paths.highest_scan_number(scans) == 3

    def test_four_digit_scan_numbers(self, tmp_path):
        scans = self.make_scans(tmp_path, "Scan0999", "Scan1000")
        assert ops_paths.highest_scan_number(scans) == 1000

    def test_empty_folder_is_none(self, tmp_path):
        scans = self.make_scans(tmp_path)
        assert ops_paths.highest_scan_number(scans) is None

    def test_missing_folder_is_none(self, tmp_path):
        assert ops_paths.highest_scan_number(tmp_path / "scans") is None

    def test_none_input_is_none(self):
        assert ops_paths.highest_scan_number(None) is None

    def test_never_creates_or_modifies_anything(self, tmp_path):
        """The invariant pin: the peek is resolution + listdir only.

        Neither a present nor an absent daily folder may be touched — GUI
        code is a consumer of scan folders, never a producer.
        """
        scans = self.make_scans(tmp_path, "Scan001")
        before = tree_snapshot(tmp_path)
        assert ops_paths.highest_scan_number(scans) == 1
        assert ops_paths.highest_scan_number(tmp_path / "missing" / "scans") is None
        assert tree_snapshot(tmp_path) == before

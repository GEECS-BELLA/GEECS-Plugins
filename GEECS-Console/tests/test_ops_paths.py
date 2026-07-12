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

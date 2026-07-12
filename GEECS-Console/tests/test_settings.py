"""ConsoleSettings: QSettings-backed, isolated to a tmp path by conftest."""

from geecs_console.services.settings import ConsoleSettings


class TestConsoleSettings:
    def test_defaults_to_empty(self):
        assert ConsoleSettings().last_experiment == ""

    def test_last_experiment_round_trips_across_instances(self):
        ConsoleSettings().last_experiment = "HTU"
        assert ConsoleSettings().last_experiment == "HTU"

    def test_injected_qsettings_is_used(self, tmp_path):
        from PySide6.QtCore import QSettings

        backing = QSettings(str(tmp_path / "own.ini"), QSettings.Format.IniFormat)
        settings = ConsoleSettings(backing)
        settings.last_experiment = "Bella"
        assert (tmp_path / "own.ini").is_file()
        reread = ConsoleSettings(
            QSettings(str(tmp_path / "own.ini"), QSettings.Format.IniFormat)
        )
        assert reread.last_experiment == "Bella"

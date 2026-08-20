"""ConsoleSettings: QSettings-backed, isolated to a tmp path by conftest."""

from geecs_console.services.settings import ConsoleSettings


class TestConsoleSettings:
    def test_defaults_to_empty(self):
        assert ConsoleSettings().last_experiment == ""

    def test_last_experiment_round_trips_across_instances(self):
        ConsoleSettings().last_experiment = "HTU"
        assert ConsoleSettings().last_experiment == "HTU"

    def test_beep_options_default_off(self):
        settings = ConsoleSettings()
        assert settings.per_shot_beep is False
        assert settings.randomized_beeps is False

    def test_beep_options_round_trip_across_instances(self):
        settings = ConsoleSettings()
        settings.per_shot_beep = True
        settings.randomized_beeps = True
        reread = ConsoleSettings()
        assert reread.per_shot_beep is True
        assert reread.randomized_beeps is True
        reread.per_shot_beep = False
        assert ConsoleSettings().per_shot_beep is False
        assert ConsoleSettings().randomized_beeps is True

    def test_show_tooltips_defaults_on(self):
        assert ConsoleSettings().show_tooltips is True

    def test_show_tooltips_round_trips_across_instances(self):
        ConsoleSettings().show_tooltips = False
        assert ConsoleSettings().show_tooltips is False
        ConsoleSettings().show_tooltips = True
        assert ConsoleSettings().show_tooltips is True

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

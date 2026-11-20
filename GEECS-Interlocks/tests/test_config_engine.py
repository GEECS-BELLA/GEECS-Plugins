"""Settings from config.ini, loading the file, and the engine's reload."""

import socket
import time

import pytest
import yaml
from conftest import FakeSource, threshold
from pydantic import ValidationError

from geecs_interlocks.config import (
    LOCAL_HOST,
    Settings,
    configuration_names,
    interlocks_path,
    load_interlocks,
)
from geecs_interlocks.engine import Engine, device_variables
from geecs_interlocks.wire import read_frame


def write(path, *entries):
    path.write_text(yaml.safe_dump({"schema_version": 1, "interlocks": list(entries)}))
    return path


class TestSettings:
    def test_defaults_listen_locally(self, tmp_path):
        settings = Settings.from_config_ini(tmp_path / "missing.ini")
        assert settings.status_host == settings.web_host == LOCAL_HOST
        assert settings.experiment is None

    def test_reads_config_ini(self, tmp_path):
        ini = tmp_path / "config.ini"
        ini.write_text(
            "[Experiment]\nexpt = Demo\n[interlocks]\nstatus_host = 0.0.0.0\nstatus_port = 6001\n"
        )
        settings = Settings.from_config_ini(ini)
        assert (settings.experiment, settings.status_host, settings.status_port) == (
            "Demo",
            "0.0.0.0",
            6001,
        )

    def test_a_bad_port_is_an_error(self, tmp_path):
        ini = tmp_path / "config.ini"
        ini.write_text("[interlocks]\nstatus_port = lots\n")
        with pytest.raises(ValidationError):
            Settings.from_config_ini(ini)


class TestLoad:
    def test_missing_file_is_empty(self, tmp_path):
        interlocks, origin = load_interlocks(tmp_path / "interlocks.yaml")
        assert interlocks.interlocks == [] and "no interlocks file" in origin

    def test_valid_file(self, tmp_path):
        interlocks, _ = load_interlocks(write(tmp_path / "i.yaml", threshold()))
        assert interlocks.interlocks[0].name == "vac"

    def test_invalid_file_raises(self, tmp_path):
        path = tmp_path / "i.yaml"
        path.write_text("interlocks: [{name: x}]\n")
        with pytest.raises(ValidationError):
            load_interlocks(path)

    def test_the_example_validates(self):
        from pathlib import Path

        example = Path(__file__).parents[1] / "examples" / "demo_interlocks.yaml"
        assert len(load_interlocks(example)[0].interlocks) == 7


def test_device_variables_skips_disabled():
    from conftest import interlock_set

    doc = interlock_set(
        threshold(), threshold("x", enabled=False, condition={"device": "Other"})
    )
    assert device_variables(doc) == {"Gauge": {"P"}}


class TestEngine:
    @pytest.fixture
    def engine(self, tmp_path):
        path = write(tmp_path / "i.yaml", threshold())
        sources: list[FakeSource] = []

        def make(_doc):
            sources.append(FakeSource())
            return sources[-1]

        eng = Engine(path, make, Settings(status_port=0))
        eng.sources = sources
        eng.start()
        yield eng
        eng.stop()

    def test_stream_serves_the_engine(self, engine):
        with socket.create_connection(engine.stream.address, timeout=2) as sock:
            assert read_frame(sock).startswith("WARNING! TRIPPED | TRIPPED: vac")

    def test_reload_swaps_and_keeps_the_stream(self, engine):
        write(engine.path, threshold("vac"), threshold("second"))
        with socket.create_connection(engine.stream.address, timeout=2) as sock:
            read_frame(sock)
            assert engine.reload().startswith("reloaded 2")
            assert engine.sources[0].stopped
            deadline = time.time() + 2
            while "second" not in read_frame(sock) and time.time() < deadline:
                pass
        assert [s.name for s in engine.snapshot().statuses] == ["vac", "second"]
        assert all(s.state == "waiting" for s in engine.snapshot().statuses)

    def test_a_bad_reload_keeps_the_old_set(self, engine):
        engine.path.write_text("interlocks: [{name: broken}]\n")
        assert engine.reload().startswith("reload refused")
        assert [s.name for s in engine.snapshot().statuses] == ["vac"]


class TestNamedConfigurations:
    @pytest.fixture
    def repo(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEECS_SCANNER_CONFIG_DIR", str(tmp_path))
        folder = tmp_path / "Demo" / "interlocks"
        folder.mkdir(parents=True)
        return folder

    def test_path_of_a_name(self, repo):
        assert interlocks_path("Demo", "main") == repo / "main.yaml"

    def test_names_are_the_saved_files(self, repo):
        for name in ("tape", "main"):
            write(repo / f"{name}.yaml", threshold())
        (repo / "notes.txt").write_text("not a configuration")
        assert configuration_names("Demo") == ["main", "tape"]

    def test_no_folder_no_names(self, repo):
        assert configuration_names("Other") == []

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


class TestSaveTarget:
    def test_outside_the_configs_repo_is_written_in_place(self, tmp_path, monkeypatch):
        from geecs_interlocks.config import save_target

        monkeypatch.delenv("GEECS_SCANNER_CONFIG_DIR", raising=False)
        monkeypatch.delenv("GEECS_PLUGINS_CONFIGS", raising=False)
        path = tmp_path / "mine.yaml"
        assert save_target(path, tmp_path / "no.ini") == path

    def test_inside_goes_to_the_write_root(self, tmp_path, monkeypatch):
        from geecs_interlocks.config import save_target

        read, write = tmp_path / "read", tmp_path / "write"
        monkeypatch.setenv("GEECS_SCANNER_CONFIG_DIR", str(read))
        monkeypatch.setenv("GEECS_CONFIGS_WRITE_ROOT", str(write))
        path = read / "Demo" / "interlocks" / "main.yaml"
        target = save_target(path)
        assert (
            target
            == write
            / "scanner_configs"
            / "experiments"
            / "Demo"
            / "interlocks"
            / "main.yaml"
        )

    def test_never_into_a_read_checkout_without_a_write_root(
        self, tmp_path, monkeypatch
    ):
        from geecs_interlocks.config import SaveRefused, save_target

        repo = tmp_path / "configs"
        (repo / ".git").mkdir(parents=True)
        read = repo / "scanner_configs" / "experiments"
        monkeypatch.setenv("GEECS_SCANNER_CONFIG_DIR", str(read))
        monkeypatch.delenv("GEECS_CONFIGS_WRITE_ROOT", raising=False)
        with pytest.raises(SaveRefused, match="only the configurator writes"):
            save_target(read / "Demo" / "interlocks" / "main.yaml")

    def test_a_plain_read_folder_is_written_in_place(self, tmp_path, monkeypatch):
        """A configs folder that is no git checkout (a laptop's copy) is the operator's own."""
        from geecs_interlocks.config import save_target

        read = tmp_path / "experiments"
        monkeypatch.setenv("GEECS_SCANNER_CONFIG_DIR", str(read))
        monkeypatch.delenv("GEECS_CONFIGS_WRITE_ROOT", raising=False)
        path = read / "Demo" / "interlocks" / "main.yaml"
        assert save_target(path) == path.resolve()


class TestReloadRollback:
    def test_a_failed_start_closes_the_ports_it_opened(self, tmp_path):
        import socket as _socket

        with _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        path = write(tmp_path / "i.yaml", threshold())
        calls = {"n": 0}

        class Broken(FakeSource):
            def start(self):
                raise RuntimeError("cannot start")

        def make(_doc):
            calls["n"] += 1
            return FakeSource() if calls["n"] == 1 else Broken()

        eng = Engine(path, make, Settings(status_port=0, external_host="127.0.0.1"))
        eng.start()
        try:
            write(
                path,
                threshold(),
                {"name": "ext", "conditions": [{"kind": "external", "port": port}]},
            )
            assert eng.reload().startswith("reload refused")
            assert eng.signals.ports() == []
            assert [s.name for s in eng.snapshot().statuses] == ["vac"]
        finally:
            eng.stop()


def test_a_reload_onto_the_inbox_keeps_the_home_path(tmp_path):
    home = write(tmp_path / "repo.yaml", threshold())
    inbox = write(tmp_path / "inbox.yaml", threshold(), threshold("second"))
    eng = Engine(home, lambda _doc: FakeSource(), Settings(status_port=0))
    eng.start()
    try:
        assert eng.reload(inbox, home=home).startswith("reloaded 2")
        assert eng.path == home
        assert [s.name for s in eng.snapshot().statuses] == ["vac", "second"]
    finally:
        eng.stop()

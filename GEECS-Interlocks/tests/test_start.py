"""What a start refuses (exit 2, which the unit never restarts) and what it accepts."""

import dataclasses
import time

import yaml
from conftest import FakeSource, threshold
from fastapi.testclient import TestClient

from geecs_interlocks import __main__ as cli
from geecs_interlocks.config import Settings
from geecs_interlocks.engine import Engine
from geecs_interlocks.web import create_app


def test_a_missing_file_is_refused(tmp_path):
    args = ["serve", "--interlocks", str(tmp_path / "nope.yaml"), "--no-web"]
    assert cli.main(args) == cli.EXIT_REFUSED


def test_an_invalid_file_is_refused(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\ninterlocks: [{name: x}]\n")
    assert (
        cli.main(["serve", "--interlocks", str(path), "--no-web"]) == cli.EXIT_REFUSED
    )


def test_a_blank_configuration_name_is_refused():
    args = ["serve", "--experiment", "Demo", "--config", "", "--no-web"]
    assert cli.main(args) == cli.EXIT_REFUSED


def test_log_level_is_accepted_after_the_subcommand():
    parse = cli._parser().parse_args
    assert parse(["serve", "--log-level", "debug"]).log_level == "debug"
    assert parse(["--log-level", "warning", "serve"]).log_level == "warning"
    assert parse(["serve"]).log_level == "INFO"


def test_a_blank_host_is_no_value():
    args = cli._parser().parse_args(
        ["serve", "--status-host", "", "--web-host", "", "--external-host", ""]
    )
    default = Settings.from_config_ini()
    settings = cli._settings(args)
    assert settings.status_host == default.status_host
    assert settings.web_host == default.web_host
    assert settings.external_host == default.external_host


def _engine(tmp_path, interlocks):
    path = tmp_path / "main.yaml"
    doc = {"schema_version": 1, "stale_after_s": None, "interlocks": interlocks}
    path.write_text(yaml.safe_dump(doc))

    def make(_doc):
        source = FakeSource()
        source.push("Gauge", P=1.0)
        return source

    eng = Engine(path, make, Settings(status_port=0))
    eng.start()
    eng.monitor.step()
    return eng


class TestHealthz:
    def test_ok_while_it_evaluates(self, tmp_path):
        eng = _engine(tmp_path, [threshold()])
        try:
            r = TestClient(create_app(eng)).get("/healthz")
            assert r.status_code == 200 and r.json() == {"status": "ok"}
        finally:
            eng.stop()

    def test_no_interlocks_is_503(self, tmp_path):
        eng = _engine(tmp_path, [])
        try:
            r = TestClient(create_app(eng)).get("/healthz")
            assert r.status_code == 503 and "no interlocks" in r.json()["reason"]
        finally:
            eng.stop()

    def test_a_stalled_loop_is_503(self, tmp_path, monkeypatch):
        eng = _engine(tmp_path, [threshold()])
        try:
            old = dataclasses.replace(eng.snapshot(), mono=time.monotonic() - 5)
            monkeypatch.setattr(eng, "snapshot", lambda: old)
            r = TestClient(create_app(eng)).get("/healthz")
            assert r.status_code == 503 and "stopped" in r.json()["reason"]
        finally:
            eng.stop()

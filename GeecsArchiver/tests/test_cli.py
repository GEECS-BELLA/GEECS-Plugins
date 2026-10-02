"""The command line, with the database and the appliance faked."""

import httpx
import pytest

from geecs_archiver import cli
from geecs_archiver.archive_set import ArchiveCandidate


def fake_build(experiment, *, policy, derived=None, enabled_only=True):
    return [
        ArchiveCandidate(
            "undulator:u_s1h:current", "U_S1H", "Current", "readback", "float"
        ),
        ArchiveCandidate(
            "undulator:u_s1h:connected", "U_S1H", "connected", "status", "enum"
        ),
    ]


class Appliance:
    def __init__(self):
        self.archived = {"undulator:u_s1h:connected", "undulator:old:pv"}
        self.requests = []
        self.paused = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/getAllPVs"):
            return httpx.Response(200, json=sorted(self.archived))
        if path.endswith("/getPVStatus"):
            rows = []
            for pv in request.url.params.get_list("pv"):
                if pv in self.archived:
                    rows.append(
                        {
                            "pvName": pv,
                            "status": "Being archived",
                            "connectionState": "true",
                            "samplingPeriod": "1.0",
                        }
                    )
                else:
                    rows.append({"pvName": pv, "status": "Not being archived"})
            return httpx.Response(200, json=rows)
        if path.endswith("/archivePV"):
            import json

            body = json.loads(request.content)
            self.requests.extend(body)
            for r in body:
                self.archived.add(r["pv"])
            return httpx.Response(
                200,
                json=[
                    {"pvName": r["pv"], "status": "Archive request submitted"}
                    for r in body
                ],
            )
        if path.endswith("/pauseArchivingPV"):
            self.paused.append(request.url.params["pv"])
            return httpx.Response(200, json={"status": "ok"})
        if path.endswith("/getVersions"):
            return httpx.Response(
                200, json={"mgmt_version": "Archiver Appliance Version 2.4.1"}
            )
        if path.endswith("/getApplianceMetrics"):
            return httpx.Response(
                200, json=[{"pvCount": str(len(self.archived)), "status": "Working"}]
            )
        if path.endswith("/getCurrentlyDisconnectedPVs"):
            return httpx.Response(200, json=[])
        if path.endswith("/exportConfig"):
            return httpx.Response(
                200, json=[{"pvName": pv} for pv in sorted(self.archived)]
            )
        return httpx.Response(404)


@pytest.fixture
def appliance(monkeypatch):
    app = Appliance()
    monkeypatch.setattr(cli, "build_archive_set", fake_build)
    real_init = cli.MgmtClient.__init__

    def patched_init(self, base_url, *, timeout=30.0, transport=None):
        real_init(
            self, base_url, timeout=timeout, transport=httpx.MockTransport(app.handler)
        )

    monkeypatch.setattr(cli.MgmtClient, "__init__", patched_init)
    monkeypatch.setenv("GEECS_ARCHIVER_URL", "http://fake:17665")
    monkeypatch.delenv("GEECS_SCANNER_CONFIG_DIR", raising=False)
    monkeypatch.setenv("GEECS_PLUGINS_CONFIGS", "/nonexistent")
    return app


def test_list_prints_the_derived_set(appliance, capsys):
    assert cli.main(["list", "--experiment", "Undulator"]) == 0
    out = capsys.readouterr().out
    assert "undulator:u_s1h:current" in out and "readback" in out


def test_onboard_dry_run_sends_nothing(appliance, capsys):
    assert cli.main(["onboard", "--experiment", "Undulator", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "+ undulator:u_s1h:current" in out
    assert "- pause  undulator:old:pv" in out  # planned, under this experiment's prefix
    assert appliance.requests == [] and appliance.paused == []


def test_onboard_applies_and_verifies(appliance, capsys):
    rc = cli.main(["onboard", "--experiment", "Undulator", "--wait", "1"])
    assert rc == 0
    assert [r["pv"] for r in appliance.requests] == ["undulator:u_s1h:current"]
    assert appliance.paused == ["undulator:old:pv"]
    assert "archived+connected 1" in capsys.readouterr().out


def test_onboard_no_pause_keeps_strays(appliance):
    assert (
        cli.main(["onboard", "--experiment", "Undulator", "--no-pause", "--wait", "0"])
        == 0
    )
    assert appliance.paused == []


def test_status_and_export(appliance, capsys, tmp_path):
    assert cli.main(["status", "--experiment", "Undulator"]) == 0
    assert "2.4.1" in capsys.readouterr().out
    out = tmp_path / "cfg.json"
    assert cli.main(["export-config", "--out", str(out)]) == 0
    assert out.read_text().count("pvName") == 2


def test_missing_url_is_a_usage_error(monkeypatch):
    monkeypatch.delenv("GEECS_ARCHIVER_URL", raising=False)
    monkeypatch.setattr(cli.config, "archiver_url", lambda config_path=None: None)
    with pytest.raises(SystemExit, match="no appliance URL"):
        cli.main(["status", "--experiment", "Undulator"])

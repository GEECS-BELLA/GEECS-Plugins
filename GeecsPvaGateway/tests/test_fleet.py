"""Fleet roster: DB-derived hosts, the config.ini deployed mark, the screen rows."""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from geecs_pva_gateway import fleet
from geecs_pva_gateway.fleet import FleetHost, fleet_roster

ENDPOINTS = {
    "UC_CamA": ("192.168.6.100", 65186),
    "UC_CamB": ("192.168.6.100", 65199),
    "U_TimingBox": ("192.168.6.100", 64804),
    "UC_LoneCam": ("192.168.6.66", 65001),
    "UC_OtherSubnet": ("192.168.7.161", 65002),
    "U_NoImages": ("192.168.8.207", 65003),
}

VAR_MAP = {
    "UC_CamA": [{"name": "image", "variabletype": "image", "choices": None}],
    "UC_CamB": [{"name": "image", "variabletype": "choice", "choices": "image"}],
    "U_TimingBox": [{"name": "delay", "variabletype": "numeric", "choices": None}],
    "UC_LoneCam": [{"name": "image", "variabletype": "image", "choices": None}],
    "UC_OtherSubnet": [{"name": "image", "variabletype": "image", "choices": None}],
    "U_NoImages": [{"name": "exposure", "variabletype": "numeric", "choices": None}],
}


@pytest.fixture
def fake_db(monkeypatch):
    from geecs_core.db.geecs_db import GeecsDb

    monkeypatch.setattr(
        GeecsDb, "get_experiment_devices", classmethod(lambda cls, e, **kw: ENDPOINTS)
    )
    monkeypatch.setattr(
        GeecsDb,
        "get_experiment_device_variables",
        classmethod(lambda cls, e, **kw: VAR_MAP),
    )


def _config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.ini"
    path.write_text(body)
    return path


def test_roster_is_image_hosts_sorted_by_ip(fake_db, tmp_path):
    """Only endpoints with image devices; numeric IP order; cameras listed."""
    hosts = fleet_roster("Undulator", config_path=_config(tmp_path, ""))
    assert [h.ip for h in hosts] == ["192.168.6.66", "192.168.6.100", "192.168.7.161"]
    assert hosts[1].cameras == ["UC_CamA", "UC_CamB"]  # the timing box is not a camera


def test_no_addr_list_means_all_deployed(fake_db, tmp_path):
    """Without [pva] addr_list every roster host is deployed (pre-key behaviour)."""
    hosts = fleet_roster(
        "Undulator", config_path=_config(tmp_path, "[epics]\nca_addr_list = 1.2.3.4\n")
    )
    assert all(h.deployed for h in hosts)
    hosts = fleet_roster("Undulator", config_path=tmp_path / "missing.ini")
    assert all(h.deployed for h in hosts)


def test_addr_list_marks_not_deployed_and_flags_stale(fake_db, tmp_path, caplog):
    """Roster minus addr_list = not deployed; a listed host with no cameras is kept + warned."""
    cfg = _config(
        tmp_path, "[pva]\naddr_list = 192.168.6.100, 192.168.7.161 192.168.9.9\n"
    )
    with caplog.at_level("WARNING", logger="geecs_pva_gateway.fleet"):
        hosts = fleet_roster("Undulator", config_path=cfg)
    by_ip = {h.ip: h for h in hosts}
    assert not by_ip["192.168.6.66"].deployed
    assert by_ip["192.168.6.100"].deployed and by_ip["192.168.7.161"].deployed
    assert by_ip["192.168.9.9"].deployed and by_ip["192.168.9.9"].cameras == []
    assert "192.168.9.9" in caplog.text
    assert [h.ip for h in hosts][-1] == "192.168.9.9"


def test_instance_pv_follows_naming_contract():
    host = FleetHost(ip="192.168.6.100")
    assert (
        host.instance_pv("Undulator", "version")
        == "undulator:pvagateway:192_168_6_100:version"
    )


def test_default_experiment_reads_config(tmp_path):
    assert (
        fleet.default_experiment(_config(tmp_path, "[Experiment]\nexpt = HTU\n"))
        == "HTU"
    )
    assert (
        fleet.default_experiment(_config(tmp_path, "[Experiment]\nexp_name = Thales\n"))
        == "Thales"
    )
    assert fleet.default_experiment(tmp_path / "missing.ini") == ""


def test_addr_list_tolerates_inline_comment_port_suffix_and_hostname(
    fake_db, tmp_path, caplog
):
    """The documented config.ini shapes parse: a '#' comment, host:port, a bare hostname."""
    cfg = _config(
        tmp_path,
        "[pva]\naddr_list = 192.168.6.100:5076 camserver7  # the deployed servers\n",
    )
    with caplog.at_level("WARNING", logger="geecs_pva_gateway.fleet"):
        hosts = fleet_roster("Undulator", config_path=cfg)
    by_ip = {h.ip: h for h in hosts}
    assert by_ip["192.168.6.100"].deployed  # matched by host, port ignored
    assert not by_ip["192.168.6.66"].deployed
    assert "#" not in by_ip and "the" not in by_ip  # the comment never became hosts
    assert [h.ip for h in hosts][
        -1
    ] == "camserver7"  # hostnames sort last, no TypeError
    assert "camserver7" in caplog.text  # ...and are flagged: no DB cameras on that name


# --- the generator (deploy/ is not a package: load it by path) ----------------

_GEN = Path(__file__).resolve().parents[1] / "deploy" / "gen_fleet_status.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_fleet_status", _GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_screen_rows_deployed_vs_not(fake_db, tmp_path):
    """Deployed rows carry live PVs + restart; not-deployed rows a label, no restart."""
    gen = _load_generator()
    cfg = _config(
        tmp_path,
        "[Experiment]\nexpt = Undulator\n[pva]\naddr_list = 192.168.6.100 192.168.7.161\n",
    )
    assert gen.main(["--config", str(cfg), "--out-dir", str(tmp_path)]) == 0
    text = (tmp_path / "fleet_status_undulator.bob").read_text()
    assert "3 camera servers in the DB, 2 deployed" in text
    assert "pva://undulator:pvagateway:192_168_6_100:version" in text
    assert "<name>restart_192_168_6_100</name>" in text
    assert "<name>not_deployed_192_168_6_66</name>" in text
    assert "<name>restart_192_168_6_66</name>" not in text
    assert "192_168_6_66:version" not in text
    assert "UC_LoneCam" in text  # the tooltip names the host's cameras


def test_screen_is_well_formed_xml_with_hostile_names(monkeypatch, tmp_path):
    """DB names with XML metacharacters and hostname entries still yield valid, unique widgets."""
    gen = _load_generator()
    hosts = [
        FleetHost(ip="192.168.6.100", cameras=["UC_Cam&A", "UC_Cam<B>", "UC--Dash"]),
        FleetHost(ip="camserver7", cameras=[], deployed=True),
        FleetHost(ip="camserver8", cameras=[], deployed=True),
    ]
    text = gen.render("R&D", hosts)
    root = ET.fromstring(text)  # raises on any escaping or comment mistake
    names = [w.findtext("name") for w in root.iter("widget")]
    assert len(names) == len(set(names)), names  # hostnames no longer collide
    assert "host_camserver7" in names and "host_camserver8" in names
    assert (
        root.find("widget[name='host_192_168_6_100']/tooltip").text
        == "UC_Cam&A, UC_Cam<B>, UC--Dash"
    )
    assert "R&D" in root.find("widget[name='title']/text").text
    assert (
        "UC_Cam" not in text.split("<widget")[0]
    )  # the header comment carries no names


def test_generator_needs_an_experiment(tmp_path, capsys):
    gen = _load_generator()
    assert (
        gen.main(
            ["--config", str(tmp_path / "missing.ini"), "--out-dir", str(tmp_path)]
        )
        == 2
    )
    assert "no experiment" in capsys.readouterr().err


# --- the probe (`geecs-pva-gateway fleet`) ------------------------------------

from geecs_pva_gateway import __main__ as cli  # noqa: E402
from geecs_pva_gateway.fleet import probe_fleet  # noqa: E402


def _hosts():
    return [
        FleetHost(ip="192.168.6.100", cameras=["UC_A", "UC_B"]),
        FleetHost(ip="192.168.7.161", cameras=["UC_C"]),
        FleetHost(ip="192.168.6.66", cameras=["UC_Lone"], deployed=False),
    ]


def _getter(answers: dict[str, object]):
    def get(pv: str) -> object:
        if pv in answers:
            return answers[pv]
        raise TimeoutError(pv)

    return get


def test_probe_lines_and_record_split_findings_from_facts():
    """OK/DOWN/not-deployed lines; the record carries facts as info= and findings as note=."""
    answers = {
        "undulator:pvagateway:192_168_6_100:version": "0.5.0",
        "undulator:pvagateway:192_168_6_100:heartbeat": 42,
    }
    result = probe_fleet("Undulator", _hosts(), getter=_getter(answers))
    lines = result.lines()
    assert (
        lines[0].startswith("  [ OK ] PVA gateway  192.168.6.100")
        and "0.5.0" in lines[0]
    )
    assert (
        lines[1].startswith("  [DOWN] PVA gateway  192.168.7.161")
        and "TimeoutError" in lines[1]
    )
    assert (
        lines[2].startswith("  [ -- ] PVA gateway  192.168.6.66")
        and "not deployed" in lines[2]
    )
    rec = dict(kv.split("=", 1) for kv in result.record().split("\t"))
    assert rec["role"] == "PVA image gateways" and rec["state"] == "ok"
    assert rec["version"] == "0.5.0 ×1"
    fields = result.record().split("\t")
    assert "info=1 of 2 deployed up" in fields and "info=1 not deployed" in fields
    assert "note=1 unreachable: 192.168.7.161" in fields
    assert not any(f.startswith("note=MIXED") for f in fields)


def test_probe_flags_mixed_versions_and_all_down():
    answers = {
        "undulator:pvagateway:192_168_6_100:version": "0.5.0",
        "undulator:pvagateway:192_168_6_100:heartbeat": 1,
        "undulator:pvagateway:192_168_7_161:version": "0.4.4",
        "undulator:pvagateway:192_168_7_161:heartbeat": 2,
    }
    result = probe_fleet("Undulator", _hosts(), getter=_getter(answers))
    assert any("mixed versions" in line for line in result.lines())
    assert "note=MIXED versions" in result.record().split("\t")
    assert "version=0.4.4 ×1, 0.5.0 ×1" in result.record().split("\t")

    result = probe_fleet("Undulator", _hosts(), getter=_getter({}))
    assert "state=down" in result.record().split("\t")


def test_fleet_subcommand_dispatch(monkeypatch, capsys, tmp_path):
    """`geecs-pva-gateway fleet` prints the lines then the record; the serve form is untouched."""
    from geecs_pva_gateway import fleet

    monkeypatch.setattr(
        fleet, "fleet_roster", lambda experiment, config_path=None: _hosts()
    )
    monkeypatch.setattr(
        fleet,
        "_p4p_getter",
        lambda hosts, timeout: _getter(
            {
                "testexp:pvagateway:192_168_6_100:version": "0.5.0",
                "testexp:pvagateway:192_168_6_100:heartbeat": 7,
                "testexp:pvagateway:192_168_7_161:version": "0.5.0",
                "testexp:pvagateway:192_168_7_161:heartbeat": 8,
            }
        ),
    )
    assert cli.main(["fleet", "--experiment", "testexp", "--timeout", "0.1"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-1].startswith("role=PVA image gateways\tstate=ok")
    assert sum(line.startswith("  [ OK ]") for line in out) == 2
    assert any(line.startswith("  [ -- ]") for line in out)

    # No experiment anywhere: exit 2 and still a record line for the table.
    assert cli.main(["fleet", "--config", str(tmp_path / "none.ini")]) == 2
    assert "note=no experiment name" in capsys.readouterr().out

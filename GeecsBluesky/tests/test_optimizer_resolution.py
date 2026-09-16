"""Config roots, rollout availability and client-independent optimizer expansion."""

from types import SimpleNamespace

import pytest
import yaml
from geecs_schemas import OptimizerConfig, Preset
from geecs_bluesky.config_resolver import ConfigsRepoResolver
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.qs_client import QserverConfig, ZmqQueueClient
from geecs_bluesky.qs_client.presets import expand_preset
from geecs_bluesky.qs_client import submit_preflight


@pytest.mark.parametrize("source", ["env", "ini"])
def test_analysis_root_uses_shared_configuration(tmp_path, monkeypatch, source):
    from geecs_data_utils import config_roots

    analysis = tmp_path / "independent-analysis"
    analysis.mkdir()
    ini = tmp_path / "config.ini"
    ini.write_text(f"[Paths]\nscan_analysis_configs_path = {analysis}\n")
    monkeypatch.setattr(config_roots, "_USER_CONFIG_PATH", ini)
    monkeypatch.delenv("SCAN_ANALYSIS_CONFIG_DIR", raising=False)
    if source == "env":
        monkeypatch.setenv("SCAN_ANALYSIS_CONFIG_DIR", str(analysis))
        ini.write_text("")
    monkeypatch.setattr(config_roots.scan_analysis_config, "_base_dir", None)
    config_roots.scan_analysis_config.bootstrap_from_env_or_fallback()
    resolver = ConfigsRepoResolver("Test", tmp_path / "elsewhere/scanners")
    assert resolver.analysis_config_dir == analysis


def _config():
    return OptimizerConfig(
        vocs={
            "variables": {"Motor:Current": [-1, 1]},
            "objectives": {"score": "MINIMIZE"},
        },
        measurements={"score": {"signal": "Meter:Value"}},
        generator={"name": "random"},
        run={"max_iterations": 3, "shots_per_step": 2},
    )


def test_listing_hides_retired_and_invalid_configs(tmp_path):
    folder = tmp_path / "Test/optimizer_configs"
    folder.mkdir(parents=True)
    (folder / "old.yaml").write_text("evaluator: {}\n")
    (folder / "broken.yaml").write_text("schema_version: 1\n")
    (folder / "malformed.yaml").write_text("vocs: [")
    resolver = ConfigsRepoResolver("Test", tmp_path)
    assert resolver.list_optimizer_configs() == []
    (folder / "native.yaml").write_text(
        yaml.safe_dump(_config().model_dump(mode="json"))
    )
    assert resolver.list_optimizer_configs() == ["native"]
    names, unavailable = resolver.optimizer_config_listing()
    assert names == ["native"]
    assert set(unavailable) == {"old", "broken", "malformed"}
    assert "legacy optimizer config is not loadable" in unavailable["old"]
    assert "vocs" in unavailable["broken"]
    assert "expected" in unavailable["malformed"]


def test_expansion_preflight_and_submission_share_optimizer_resolution(monkeypatch):
    preset = Preset(
        name="opt",
        devices=[{"device": "Meter", "essential": False, "save_images": False}],
        plan={"name": "optimize", "kwargs": {"optimizer_config": "test"}},
    )
    resolver = SimpleNamespace(resolve_optimizer_config=lambda _: _config())
    expected = expand_preset(preset, resolver=resolver)
    assert expected.args == [["Meter"]]
    assert expected.kwargs["max_iterations"] == 3
    assert expected.kwargs["shots_per_step"] == 2
    assert "non_essential" not in expected.kwargs
    assert not preset.devices[0].essential  # input document remains a draft
    with pytest.raises(GeecsConfigurationError, match="resolver"):
        expand_preset(preset)
    observed = {}
    client = ZmqQueueClient(
        QserverConfig(
            control_addr="tcp://localhost:60615", info_addr=None, doc_addr=None
        )
    )

    def submit(name, **kwargs):
        observed.update(name=name, **kwargs)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(client, "submit_plan", submit)
    assert client.submit_preset(preset, resolver=resolver).ok
    assert observed["args"] == expected.args and observed["kwargs"] == expected.kwargs
    monkeypatch.setattr(submit_preflight, "_check_worker_ready", lambda *args: None)
    monkeypatch.setattr(submit_preflight, "_trigger_profile_devices", lambda *args: [])
    monkeypatch.setattr(
        submit_preflight,
        "_check_liveness",
        lambda report, devices, experiment: observed.update(devices=devices),
    )
    assert (
        submit_preflight.run_submit_preflight(preset, "Test", resolver=resolver).refusal
        is None
    )
    assert observed["devices"] == ["Meter"]


@pytest.mark.parametrize("failure", ["unconfigured", "missing", "unreadable"])
def test_optimizer_listing_preserves_root_and_io_failures(
    tmp_path, monkeypatch, failure
):
    from pathlib import Path

    resolver = ConfigsRepoResolver("Test", tmp_path)
    if failure == "unconfigured":
        resolver = ConfigsRepoResolver("Test")

        def unconfigured():
            raise RuntimeError("unconfigured")

        monkeypatch.setattr(
            "geecs_bluesky.config_resolver.scanner_configs_base", unconfigured
        )
    elif failure == "unreadable":
        (tmp_path / "Test/optimizer_configs").mkdir(parents=True)

        def unreadable(self):
            raise PermissionError("configs tree inaccessible")

        monkeypatch.setattr(Path, "iterdir", unreadable)
    with pytest.raises((RuntimeError, FileNotFoundError, PermissionError)):
        resolver.optimizer_config_listing()

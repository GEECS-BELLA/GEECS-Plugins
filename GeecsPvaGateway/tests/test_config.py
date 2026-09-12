"""Config tests: DB-scoped camera selection and PV naming (no network)."""

from __future__ import annotations

import pytest

from geecs_pva_gateway.config import CameraSpec, PvaGatewayConfig

#: The DB ``get='yes'`` list per device (GeecsDbScalarPolicy's query).
SUBSCRIBED = {
    "UC_CamA": [
        "acq_timestamp",
        "MaxCounts",
        "exposure",
        "trigger",
        "localsavingpath",
        "image",
        "ghost",
    ],
    "UC_CamB": ["image"],
}

ENDPOINTS = {
    "UC_CamA": ("192.168.6.100", 65186),
    "UC_CamB": ("192.168.6.100", 65199),
    "U_TimingBox": ("192.168.6.100", 64804),
    "UC_OtherHostCam": ("192.168.6.101", 65001),
}

VAR_MAP = {
    "UC_CamA": [
        {"name": "image", "variabletype": "image", "choices": None},
        {"name": "processed image", "variabletype": "image", "choices": None},
        {"name": "exposure", "variabletype": "numeric", "choices": None},
        {"name": "MaxCounts", "variabletype": "numeric", "choices": None},
        {"name": "trigger", "variabletype": "", "choices": "on,off"},
        {"name": "localsavingpath", "variabletype": "string", "choices": None},
        {"name": "acq_timestamp", "variabletype": "numeric", "choices": None},
    ],
    "UC_CamB": [
        # image typed via the choice-descriptor quirk (#512)
        {"name": "image", "variabletype": "choice", "choices": "image"},
    ],
    "U_TimingBox": [
        {"name": "delay", "variabletype": "numeric", "choices": None},
    ],
    "UC_OtherHostCam": [
        {"name": "image", "variabletype": "image", "choices": None},
    ],
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
    monkeypatch.setattr(
        GeecsDb,
        "get_subscribed_variables",
        classmethod(lambda cls, e, **kw: SUBSCRIBED),
    )


def test_host_scoping_selects_image_devices_only(fake_db):
    """Host filter keeps that host's cameras; non-cameras drop out."""
    cfg = PvaGatewayConfig.from_geecs_experiment("Undulator", host="192.168.6.100")
    assert [c.device for c in cfg.cameras] == ["UC_CamA", "UC_CamB"]
    by_dev = {c.device: c for c in cfg.cameras}
    assert by_dev["UC_CamA"].image_variables == ["image", "processed image"]
    assert by_dev["UC_CamB"].image_variables == ["image"]  # choice-descriptor
    assert by_dev["UC_CamA"].port == 65186
    # The per-frame scalar attributes: the subscribed list in DB order,
    # numbers and enums only, minus the timestamp ladder the stamps carry;
    # strings, images and names without a metadata row drop out.
    assert by_dev["UC_CamA"].scalar_variables == ["MaxCounts", "exposure", "trigger"]
    assert by_dev["UC_CamB"].scalar_variables == []


def test_scalar_attributes_empty_when_the_policy_query_fails(
    fake_db, monkeypatch, caplog
):
    """A DB blip on the get='yes' query degrades to frames-and-stamps with a warning."""
    from geecs_core.db.geecs_db import GeecsDb

    def boom(cls, e, **kw):
        raise RuntimeError("no network")

    monkeypatch.setattr(GeecsDb, "get_subscribed_variables", classmethod(boom))
    cfg = PvaGatewayConfig.from_geecs_experiment("Undulator", host="192.168.6.100")
    assert [c.scalar_variables for c in cfg.cameras] == [[], []]
    assert "Could not read get='yes'" in caplog.text


def test_device_subset_and_missing_warning(fake_db, caplog):
    """Explicit device subset filters; unknown names warn, not raise."""
    cfg = PvaGatewayConfig.from_geecs_experiment(
        "Undulator", host="192.168.6.100", devices=["UC_CamB", "UC_Nonexistent"]
    )
    assert [c.device for c in cfg.cameras] == ["UC_CamB"]
    assert "UC_Nonexistent" in caplog.text


def test_other_host_scoping(fake_db):
    cfg = PvaGatewayConfig.from_geecs_experiment("Undulator", host="192.168.6.101")
    assert [c.device for c in cfg.cameras] == ["UC_OtherHostCam"]


def test_pv_names_follow_shared_contract():
    """PV names come from the gateway's pv_naming: lowercase, sanitized."""
    spec = CameraSpec(
        device="UC_Amp2_IR_input",
        host="192.168.6.100",
        port=65186,
        experiment="Undulator",
        image_variables=["image", "processed image"],
    )
    assert spec.pv_name_for("image") == "undulator:uc_amp2_ir_input:image"
    assert (
        spec.pv_name_for("processed image")
        == "undulator:uc_amp2_ir_input:processed_image"
    )

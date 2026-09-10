"""Hermetic tests for the RE Manager startup profile (issue #640).

No lab, no redis, no queueserver process. Most tests here run
``qserver/startup/startup.py`` in-process via ``runpy.run_path`` the same
way the manager's worker would import it, with the experiment-resolution
and Tiled-subscription config chains monkeypatched so nothing depends on
this machine's ``~/.config/geecs_python_api/config.ini``.

The import-order test is the exception: it shells out to
``_qserver_startup_probe.py`` in a fresh interpreter. The module's own
docstring calls the ``geecs_bluesky``-before-``aioca`` ordering
load-bearing, but by the time any in-process test function runs, both are
already cached in this test session's ``sys.modules`` from earlier
collection — an in-process ``runpy.run_path`` here could never actually
observe first-import order, only appear to.
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

STARTUP_PATH = (
    Path(__file__).resolve().parents[1] / "qserver" / "startup" / "startup.py"
)
_PROBE_PATH = Path(__file__).resolve().parent / "_qserver_startup_probe.py"


@pytest.fixture(autouse=True)
def _no_tiled_subscription(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out Tiled subscription — its config chain is not hermetic.

    ``subscribe_tiled`` itself already degrades gracefully off-network (a
    bounded reachability check), but *reading* its config
    (``geecs_data_utils.tiled_catalog.read_tiled_config``) touches the same
    ``config.ini`` this test suite otherwise avoids entirely. Startup
    profile testing is about the profile's own wiring, not Tiled's config
    resolution, so it is stubbed rather than routed through a real or fake
    config file.
    """
    monkeypatch.setattr(
        "geecs_bluesky.tiled_integration.subscribe_tiled", lambda *a, **kw: None
    )
    # No 0MQ publisher in-process: a connected-but-peerless PUB socket makes
    # the zmq context's teardown block (linger) for the next test's whole
    # timeout — the residual one-test stall of #812.
    monkeypatch.setenv("QS_DOC_PUBLISH_ADDR", "OFF")
    # The DB-backed device namespace is exercised by its own test below;
    # every other in-process run skips it (no GEECS DB here).
    monkeypatch.setenv("QS_DEVICE_NAMESPACE", "off")


def test_startup_profile_defines_re_and_plans_headless(tmp_path: Path) -> None:
    """QS_EXPERIMENT resolves the experiment; RE and the stock plans land in the namespace.

    Also asserts the load-bearing import order documented at the top of
    ``qserver/startup/startup.py``: ``geecs_bluesky`` (which sets
    ``EPICS_CA_ADDR_LIST`` from config) must be imported, and that variable
    must be set, before ``aioca`` is first imported. Run as a subprocess so
    both modules start uncached — see the module docstring above.
    """
    # The probe asks the manager's own plan discovery what it sees.
    pytest.importorskip("bluesky_queueserver")
    config_dir = tmp_path / "home" / ".config" / "geecs_python_api"
    config_dir.mkdir(parents=True)
    (config_dir / "config.ini").write_text("[epics]\nca_addr_list = 127.0.0.1\n")

    env = dict(os.environ)
    env["QS_EXPERIMENT"] = "TestExp"
    env["QS_DEVICE_NAMESPACE"] = "off"  # no DB in the hermetic probe
    env["QS_DOC_PUBLISH_ADDR"] = "OFF"
    env["HOME"] = str(tmp_path / "home")
    env.pop("EPICS_CA_ADDR_LIST", None)
    env.pop("EPICS_CA_AUTO_ADDR_LIST", None)

    result = subprocess.run(
        [sys.executable, str(_PROBE_PATH), str(STARTUP_PATH)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PROBE_OK" in result.stdout, result.stdout + result.stderr


def test_startup_profile_fails_loud_without_an_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither QS_EXPERIMENT nor config.ini's [Experiment] expt: fail at import."""
    monkeypatch.delenv("QS_EXPERIMENT", raising=False)

    class _NoExperimentConfig:
        def __init__(self, *args, **kwargs) -> None:
            self.experiment = None

    monkeypatch.setattr(
        "geecs_data_utils.GeecsPathsConfig", _NoExperimentConfig, raising=False
    )

    with pytest.raises(RuntimeError, match="No GEECS experiment configured"):
        runpy.run_path(str(STARTUP_PATH), run_name="__not_main__")


def test_gen_list_of_plans_and_devices_succeeds_on_startup_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The manager's own profile-load check passes against the startup dir.

    Skipped unless the optional ``qserver`` extra (``bluesky-queueserver``)
    is installed — the default CI job (``poetry install --with dev``) does
    not pull it in, only ``poetry install --with dev -E qserver`` does.
    """
    pytest.importorskip("bluesky_queueserver")
    monkeypatch.setenv("QS_EXPERIMENT", "TestExp")

    from bluesky_queueserver.manager.gen_lists import gen_list_of_plans_and_devices

    out_name = "startup_existing_plans_and_devices.yaml"
    gen_list_of_plans_and_devices(
        startup_dir=str(STARTUP_PATH.parent),
        file_dir=str(tmp_path),
        file_name=out_name,
        overwrite=True,
    )

    assert (tmp_path / out_name).exists()


def test_stock_plans_pass_manager_validation_over_namespace_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real ``queue add`` item for a stock plan validates against the profile.

    The manager re-evaluates each plan's signature at submission; the
    stock verbs' annotations are plain builtins, so ``count([cam], 3)``
    and ``scan([cam], U_S1H.current, -1, 1, 5)`` validate with the
    namespace devices as the allowed devices — the queue-item contract
    the plan layer (PR 2) and the clients build on.  Uses the manager's
    own ``_process_plan``/``validate_plan`` pair (the exact code path
    behind ``queue add``); private queueserver API, accepted for a pin
    this specific.
    """
    pytest.importorskip("bluesky_queueserver")
    pytest.importorskip("aioca")
    from bluesky_queueserver.manager.profile_ops import (
        _process_plan,
        existing_plans_and_devices_from_nspace,
        validate_plan,
    )

    from geecs_bluesky.namespace import GeecsNamespace

    monkeypatch.setenv("QS_EXPERIMENT", "TestExp")
    monkeypatch.setenv("QS_DEVICE_NAMESPACE", "db")
    monkeypatch.setattr(
        GeecsNamespace,
        "from_experiment",
        classmethod(lambda cls, exp, **kw: cls(_make_roster())),
    )
    ns = runpy.run_path(str(STARTUP_PATH), run_name="__not_main__")
    plans, devices, *_ = existing_plans_and_devices_from_nspace(nspace=ns)
    assert {"count", "scan", "mv"} <= set(plans)
    assert "UC_TestCam" in devices and "U_S1H" in devices
    for name, args in (
        ("count", [["UC_TestCam"], 3]),
        ("scan", [["UC_TestCam"], "U_S1H.current", -1, 1, 5]),
        ("mv", ["U_S1H.current", 0.0]),
    ):
        processed = _process_plan(ns[name], existing_devices={}, existing_plans={})
        ok, msg = validate_plan(
            {"name": name, "args": args, "item_type": "plan"},
            allowed_plans={name: processed},
            allowed_devices=devices,
        )
        assert ok, (name, msg)


_ROW = {
    "settable": False,
    "variabletype": None,
    "choices": "numeric",
    "tolerance": None,
    "units": "",
    "min": None,
    "max": None,
}


def _make_roster():
    from geecs_bluesky.namespace import DeviceRoster

    return DeviceRoster(
        experiment="TestExp",
        variables={
            "UC_TestCam": [
                {**_ROW, "name": "trigger", "settable": True, "choices": "on,off"},
                {**_ROW, "name": "MeanCounts"},
            ],
            "U_S1H": [
                {**_ROW, "name": "Current", "settable": True, "tolerance": 0.05},
            ],
        },
        types={"UC_TestCam": "Point Grey Camera", "U_S1H": "Magnet PS"},
        subscribed={"UC_TestCam": ["MeanCounts"], "U_S1H": ["Current"]},
    )


def test_startup_exports_the_device_namespace_and_installs_connect_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Devices from the roster land in the namespace/__all__; connect_on_demand is outermost."""
    pytest.importorskip("aioca")  # the roster builds CA devices
    from geecs_bluesky.namespace import GeecsNamespace
    from geecs_bluesky.preprocessors import connect_on_demand

    monkeypatch.setenv("QS_EXPERIMENT", "TestExp")
    monkeypatch.setenv("QS_DEVICE_NAMESPACE", "db")
    monkeypatch.setattr(
        GeecsNamespace,
        "from_experiment",
        classmethod(lambda cls, exp, **kw: cls(_make_roster())),
    )
    ns = runpy.run_path(str(STARTUP_PATH), run_name="__not_main__")
    assert "U_S1H" in ns and "U_S1H" in ns["__all__"]
    assert ns["U_S1H"].current.name == "u_s1h-current"
    funcs = [getattr(p, "func", p) for p in ns["RE"].preprocessors]
    assert funcs[-1] is connect_on_demand and funcs.count(connect_on_demand) == 1

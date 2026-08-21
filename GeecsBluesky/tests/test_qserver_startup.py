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
    ``config.ini`` this test suite otherwise avoids entirely (see every
    other ``GeecsSession(..., tiled=False, mock=True)`` fixture in this
    package). Startup profile testing is about the profile's own wiring,
    not Tiled's config resolution, so it is stubbed rather than routed
    through a real or fake config file.
    """
    monkeypatch.setattr("geecs_bluesky.session.subscribe_tiled", lambda *a, **kw: None)


def test_startup_profile_defines_re_and_plan_headless(tmp_path: Path) -> None:
    """QS_EXPERIMENT resolves the experiment; RE and the plan land in the namespace.

    Also asserts the load-bearing import order documented at the top of
    ``qserver/startup/startup.py``: ``geecs_bluesky`` (which sets
    ``EPICS_CA_ADDR_LIST`` from config) must be imported, and that variable
    must be set, before ``aioca`` is first imported. Run as a subprocess so
    both modules start uncached — see the module docstring above.
    """
    config_dir = tmp_path / "home" / ".config" / "geecs_python_api"
    config_dir.mkdir(parents=True)
    (config_dir / "config.ini").write_text("[epics]\nca_addr_list = 127.0.0.1\n")

    env = dict(os.environ)
    env["QS_EXPERIMENT"] = "TestExp"
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

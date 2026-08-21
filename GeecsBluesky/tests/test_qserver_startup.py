"""Hermetic tests for the RE Manager startup profile (issue #640).

No lab, no redis, no queueserver process: ``runpy.run_path`` executes
``qserver/startup/startup.py`` in-process the same way the manager's worker
would import it, with the experiment-resolution and Tiled-subscription
config chains monkeypatched so nothing here depends on this machine's
``~/.config/geecs_python_api/config.ini``.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from bluesky import RunEngine

STARTUP_PATH = (
    Path(__file__).resolve().parents[1] / "qserver" / "startup" / "startup.py"
)


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


def test_startup_profile_defines_re_and_plan_headless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QS_EXPERIMENT resolves the experiment; RE and the plan land in the namespace."""
    monkeypatch.setenv("QS_EXPERIMENT", "TestExp")

    ns = runpy.run_path(str(STARTUP_PATH), run_name="__not_main__")

    from geecs_bluesky.plans.scan_request_plan import geecs_scan_request_plan

    assert isinstance(ns["RE"], RunEngine)
    # --keep-re requires the manager's RE to be the plan preamble's RE —
    # a second, independently-constructed RunEngine is unsupported.
    assert ns["RE"] is ns["session"].RE
    assert ns["geecs_scan_request_plan"] is geecs_scan_request_plan
    assert ns["__all__"] == ["RE", "geecs_scan_request_plan"]


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

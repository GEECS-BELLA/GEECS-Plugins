"""Integration test for the experiment-defaults path against the real configs.

Unlike ``test_scan_request_hardware.py`` this needs neither hardware nor the
``ca`` extra — only the configs repository checked out — so it exercises the
one thing hermetic tests deliberately do *not*: that a real
``experiment_defaults.yaml`` is found, parsed, and applied to fill a silent
ScanRequest, with the applied default recorded for provenance.

It is ``integration``-marked (deselected in CI, which has no configs repo) and
self-skips when the experiment has no defaults file in this checkout, so it
never fails for a reason unrelated to the code under test.  The dedicated
scope is the point: hermetic tests build their own ``ExperimentDefaults``
inline and must not read the real file, but exactly one test should prove the
real-file wiring — and catch a renamed profile or a broken defaults-application
path.

Run it by hand from the package dir::

    poetry run pytest tests/test_experiment_defaults_integration.py \
        -m integration -s

Point it at another experiment/profile with ``GEECS_HW_EXPERIMENT`` /
``GEECS_HW_EXPECTED_TRIGGER_PROFILE`` if the Undulator defaults change.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.integration
def test_experiment_defaults_fill_minimal_request_from_real_configs() -> None:
    """A silent ScanRequest is filled from the real defaults file, recorded."""
    from geecs_bluesky.scan_request_runner import (
        ConfigsRepoResolver,
        resolve_defaults_for,
    )
    from geecs_schemas import ScanRequest

    experiment = os.environ.get("GEECS_HW_EXPERIMENT", "Undulator")
    expected_profile = os.environ.get("GEECS_HW_EXPECTED_TRIGGER_PROFILE", "HTU-Normal")

    resolver = ConfigsRepoResolver(experiment)
    defaults = resolver.resolve_experiment_defaults()
    if defaults is None:
        pytest.skip(f"no experiment_defaults.yaml for {experiment!r} in this checkout")

    # The file declares the standard trigger profile the whole experiment uses.
    assert defaults.trigger_profile == expected_profile

    # A request that names no trigger profile must inherit it from the defaults,
    # and the applied default must be recorded for run-metadata provenance.
    minimal = ScanRequest.model_validate(
        {
            "mode": "noscan",
            "shots_per_step": 3,
            "acquisition": "strict",
            "save_set": os.environ.get("GEECS_HW_CAMERA", "Amp4In"),
        }
    )
    assert minimal.trigger_profile is None

    filled, applied = resolve_defaults_for(resolver, minimal)
    assert filled.trigger_profile == expected_profile
    assert applied.get("trigger_profile") == expected_profile

    # An explicit trigger profile must win over the default (defaults only fill
    # blanks — never override what a request says).
    explicit = minimal.model_copy(update={"trigger_profile": "HTU-LaserOFF"})
    kept, applied_explicit = resolve_defaults_for(resolver, explicit)
    assert kept.trigger_profile == "HTU-LaserOFF"
    assert "trigger_profile" not in applied_explicit

"""Backend selection for RunControl — the GEECS_USE_BLUESKY env override.

The GUI constructs RunControl without a ``use_bluesky`` argument
(app_controller.reinitialize_run_control), so the env var is the supported
way to switch a GUI session onto the Bluesky backend during the
legacy → Bluesky transition, mirroring ``GEECS_BLUESKY_ACQUISITION_MODE``.
The resolver lives in ``engine`` (PyQt5-free) so this test runs without
the GUI stack.
"""

from __future__ import annotations

import pytest

from geecs_scanner.engine.backend_selection import (
    resolve_use_bluesky as _resolve_use_bluesky,
)


class TestResolveUseBluesky:
    def test_default_is_legacy(self):
        assert _resolve_use_bluesky(None, env={}) is False

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " On "])
    def test_env_truthy_selects_bluesky(self, raw):
        assert _resolve_use_bluesky(None, env={"GEECS_USE_BLUESKY": raw}) is True

    @pytest.mark.parametrize("raw", ["", "0", "false", "no", "OFF"])
    def test_env_falsy_selects_legacy(self, raw):
        assert _resolve_use_bluesky(None, env={"GEECS_USE_BLUESKY": raw}) is False

    def test_explicit_argument_beats_env(self):
        env = {"GEECS_USE_BLUESKY": "1"}
        assert _resolve_use_bluesky(False, env=env) is False
        assert _resolve_use_bluesky(True, env={}) is True

    def test_unrecognised_env_value_raises(self):
        with pytest.raises(ValueError, match="GEECS_USE_BLUESKY"):
            _resolve_use_bluesky(None, env={"GEECS_USE_BLUESKY": "maybe"})

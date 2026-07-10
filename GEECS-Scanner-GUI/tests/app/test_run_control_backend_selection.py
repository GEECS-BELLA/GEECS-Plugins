"""Backend selection is gone: BlueskyScanner is the only backend (G1).

The legacy ScanManager backend was deleted in G1 of the greenfield cutover
(``Planning/cutover_strategy/00_overview.md``).  ``resolve_use_bluesky`` is
retained as an always-True stub so the import surface stays valid; these
tests pin that always-bluesky behavior — argument and env var are ignored.
The resolver lives in ``engine`` (PyQt5-free) so this test runs without
the GUI stack.
"""

from __future__ import annotations

import pytest

from geecs_scanner.engine.backend_selection import (
    resolve_use_bluesky as _resolve_use_bluesky,
)


class TestResolveUseBluesky:
    def test_default_is_bluesky(self):
        assert _resolve_use_bluesky(None, env={}) is True

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " On "])
    def test_env_truthy_still_bluesky(self, raw):
        assert _resolve_use_bluesky(None, env={"GEECS_USE_BLUESKY": raw}) is True

    @pytest.mark.parametrize("raw", ["", "0", "false", "no", "OFF", "maybe"])
    def test_env_cannot_select_legacy(self, raw):
        """The legacy backend no longer exists — any env value is ignored."""
        assert _resolve_use_bluesky(None, env={"GEECS_USE_BLUESKY": raw}) is True

    def test_explicit_argument_is_ignored(self):
        assert _resolve_use_bluesky(False, env={}) is True
        assert _resolve_use_bluesky(True, env={}) is True

    def test_no_arguments(self):
        assert _resolve_use_bluesky() is True

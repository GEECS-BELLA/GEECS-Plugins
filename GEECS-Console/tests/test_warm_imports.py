"""services/warm_imports.py — the #778 startup import warm-up helper."""

from __future__ import annotations

import logging
import sys

from geecs_console.services.warm_imports import WARM_MODULES, warm_imports


class TestWarmImports:
    def test_imports_every_warm_module(self):
        assert warm_imports() == []
        for name in WARM_MODULES:
            assert name in sys.modules, name

    def test_covers_the_threads_lazy_imports(self):
        # Keep-in-sync pin against the daemon threads' lazy import sites:
        # DocumentStreamWorker._run, ZmqQueueClient._manager,
        # ConsoleStreamWorker._run, ops_paths.todays_scan_folder,
        # GatewayTiledDbHealth._tiled_uri.  Rename there → rename here.
        assert {
            "bluesky.callbacks.zmq",
            "bluesky_queueserver_api.zmq",
            "bluesky_queueserver_api.console_monitor",
            "geecs_data_utils.scan_paths",
            "geecs_bluesky.tiled_integration",
        } <= set(WARM_MODULES)

    def test_a_failing_import_is_logged_and_skipped(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger="geecs_console.services.warm_imports"
        ):
            failed = warm_imports(("geecs_console_no_such_module_778", "os"))
        assert failed == ["geecs_console_no_such_module_778"]
        assert "geecs_console_no_such_module_778" in caplog.text

    def test_is_idempotent(self):
        assert warm_imports() == []
        assert warm_imports() == []

    def test_never_pulls_in_the_legacy_api(self):
        warm_imports()
        assert "geecs_python_api" not in sys.modules

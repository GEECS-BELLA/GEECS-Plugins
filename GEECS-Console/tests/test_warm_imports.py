"""services/warm_imports.py — the #778 startup import warm-up helper."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from geecs_console.services.warm_imports import WARM_MODULES, warm_imports

_PKG = Path(__file__).resolve().parents[1] / "geecs_console"

#: The modules whose bodies run on the console's daemon threads and lazily
#: import at function level (the racing sites #778 is about).
_THREAD_BODY_MODULES = (
    _PKG / "app" / "scan_monitor.py",
    _PKG / "services" / "ops_paths.py",
    _PKG / "services" / "health.py",
)

#: Top-level packages with internal import cycles (a package ``__init__``
#: importing its own submodules) that the daemon threads first-import.
_CYCLE_PACKAGES = {"bluesky", "geecs_data_utils"}

_FUNCTION_LEVEL_FROM_IMPORT = re.compile(r"^[ \t]+from ([\w.]+) import ", re.MULTILINE)


class TestWarmImports:
    def test_imports_every_warm_module(self):
        assert warm_imports() == []
        for name in WARM_MODULES:
            assert name in sys.modules, name

    def test_resolves_the_cycle_bearing_packages(self):
        # The outcome that matters, independent of entry names: after the
        # warm-up both cycle-bearing packages — and the exact modules the
        # issue's two errors named — are fully initialised.
        warm_imports()
        for name in (
            "bluesky",
            "bluesky._vendor.super_state_machine.errors",
            "geecs_data_utils",
            "geecs_data_utils.tiled_catalog",
        ):
            assert name in sys.modules, name

    def test_covers_the_threads_lazy_cycle_imports(self):
        # Keep-in-sync pin with teeth: every function-level `from X import`
        # in a thread-body module that enters a cycle-bearing package must
        # be warmed under exactly that name (renaming the import in
        # DocumentStreamWorker._run, or adding a new geecs_data_utils
        # import to a thread body, fails here until WARM_MODULES follows).
        found: set[str] = set()
        for path in _THREAD_BODY_MODULES:
            for module in _FUNCTION_LEVEL_FROM_IMPORT.findall(path.read_text("utf-8")):
                if module.split(".")[0] in _CYCLE_PACKAGES:
                    found.add(module)
        assert found, "grep found no cycle-package imports — the pin is broken"
        missing = found - set(WARM_MODULES)
        assert not missing, f"thread-side lazy imports not warmed: {sorted(missing)}"

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

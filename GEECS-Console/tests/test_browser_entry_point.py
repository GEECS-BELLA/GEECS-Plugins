"""Entry-point tests: stylesheet loader + offscreen construction smoke."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from geecs_console.browser.__main__ import _load_console_stylesheet

_SMOKE_SCRIPT = """
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
QApplication.exec = lambda *a, **k: 0  # construct + show, no event loop
from geecs_console.browser.__main__ import main
raise SystemExit(main([]))
"""


class TestStylesheetLoader:
    def test_console_family_stylesheet_loads_and_resolves_assets(self):
        qss = _load_console_stylesheet()
        assert qss  # the packaged style.qss exists
        assert "@UI_DIR@" not in qss  # asset token resolved
        assert "arrow_down.svg" in qss


class TestEntryPointSmoke:
    def test_main_constructs_offscreen(self, tmp_path):
        """``python -m``-equivalent smoke: real main(), offscreen, hermetic.

        HOME is pointed at a tmp dir so the TiledScanCatalog's config read
        finds nothing (no network probe target) and QSettings stay isolated.
        """
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["HOME"] = str(tmp_path)
        result = subprocess.run(
            [sys.executable, "-c", _SMOKE_SCRIPT],
            env=env,
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr

    def test_module_help_exits_zero(self):
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        result = subprocess.run(
            [sys.executable, "-m", "geecs_console.browser", "--help"],
            env=env,
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert "scan browser" in result.stdout.lower()

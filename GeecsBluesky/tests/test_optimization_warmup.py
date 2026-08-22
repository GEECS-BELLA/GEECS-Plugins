"""The worker startup warm-up: pre-importing the optimization stack off-thread.

PR #644 review row 6. Uses the fake-stack/event-gated-import technique
(originally shared with the console's deleted GUI-process loader tests)
against ``geecs_bluesky.optimization.worker_loader``.

Hermetic — the ``optimize`` extra's dependencies (xopt) are NOT installed
in CI, where the no-op path is exercised against the real ``find_spec``
probe; on dev machines that installed the extra, that real-probe test
skips (the environment, not the code, decides its premise). The warm path
runs against fake ``geecs_bluesky.optimization`` modules planted in
``sys.modules``, and the never-blocks pin against an import stub gated on
an event. Deliberately placed at the top level (not under
``tests/optimization/``): that directory's ``conftest.py`` skips collection
entirely when the ``optimize`` extra is absent, which is exactly the
no-op branch this file must exercise in CI.
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
import types

import pytest

from geecs_bluesky.optimization import worker_loader as worker_loader_module
from geecs_bluesky.optimization.worker_loader import (
    optimization_available,
    warm_up_optimization_stack,
)

_JOIN_TIMEOUT_S = 5.0


def _plant_fake_stack(monkeypatch) -> None:
    """Put importable fakes at the heavy-module paths in ``sys.modules``."""
    for name in (
        "geecs_bluesky.optimization",
        *worker_loader_module._HEAVY_MODULES,
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


@pytest.mark.skipif(
    # optimization_available() is the probe under test itself — agreement
    # by construction, and (unlike a bare dotted find_spec) it guards the
    # ModuleNotFoundError that an absent parent package raises.
    optimization_available(),
    reason="the optimize extra IS installed here — the test's premise "
    "(extra absent, as in CI) does not hold",
)
def test_warm_up_no_ops_without_the_extra(caplog) -> None:
    """Extra absent (real find_spec probe): no thread, nothing logged loudly."""
    with caplog.at_level(logging.INFO, logger=worker_loader_module.__name__):
        assert warm_up_optimization_stack() is None
    assert "preloaded" not in caplog.text


def test_warm_up_imports_the_heavy_modules_and_logs(monkeypatch, caplog) -> None:
    monkeypatch.setattr(worker_loader_module, "optimization_available", lambda: True)
    _plant_fake_stack(monkeypatch)
    imported: list[str] = []
    real_import = importlib.import_module

    def recording_import(name, package=None):
        imported.append(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", recording_import)

    with caplog.at_level(logging.INFO, logger=worker_loader_module.__name__):
        thread = warm_up_optimization_stack()
        assert thread is not None
        assert thread.daemon is True
        thread.join(timeout=_JOIN_TIMEOUT_S)
        assert not thread.is_alive()

    assert imported == list(worker_loader_module._HEAVY_MODULES)
    assert "optimization stack preloaded in" in caplog.text


def test_warm_up_returns_immediately_while_the_import_runs(monkeypatch) -> None:
    """The startup path never blocks: the call returns mid-import."""
    monkeypatch.setattr(worker_loader_module, "optimization_available", lambda: True)
    release = threading.Event()

    def blocking_import(name, package=None):
        release.wait(timeout=_JOIN_TIMEOUT_S)
        return types.ModuleType(name)

    monkeypatch.setattr(importlib, "import_module", blocking_import)

    thread = warm_up_optimization_stack()
    try:
        # Back on the caller immediately — the import is still parked on
        # the event, so a blocking implementation would never get here.
        assert thread is not None
        assert thread.is_alive()
    finally:
        release.set()
        thread.join(timeout=_JOIN_TIMEOUT_S)
    assert not thread.is_alive()


def test_warm_up_failure_is_logged_and_swallowed(monkeypatch, caplog) -> None:
    monkeypatch.setattr(worker_loader_module, "optimization_available", lambda: True)

    def failing_import(name, package=None):
        raise ImportError(f"boom: {name}")

    monkeypatch.setattr(importlib, "import_module", failing_import)

    with caplog.at_level(logging.WARNING, logger=worker_loader_module.__name__):
        thread = warm_up_optimization_stack()
        assert thread is not None
        thread.join(timeout=_JOIN_TIMEOUT_S)
        assert not thread.is_alive()

    assert "optimization stack warm-up failed" in caplog.text
    assert "preloaded" not in caplog.text

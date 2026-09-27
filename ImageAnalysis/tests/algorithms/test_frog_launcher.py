"""The FROG worker command: run directly on Windows, behind a launcher (Wine) elsewhere."""

from __future__ import annotations

import pickle
import subprocess
import types

import numpy as np
import pytest

from image_analysis.algorithms import frog_dll_retrieval as frog
from image_analysis.algorithms.frog_dll_retrieval import FrogDllRetrieval


@pytest.fixture
def paths(tmp_path):
    dll = tmp_path / "FROG.dll"
    python = tmp_path / "python.exe"
    dll.write_bytes(b"")
    python.write_bytes(b"")
    return dll, python


def command_of(retriever: FrogDllRetrieval, monkeypatch) -> list[str]:
    """The argv the retrieval would start, captured instead of run."""
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="stop")

    monkeypatch.setattr(frog.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="stop"):
        retriever.retrieve_pulse(np.ones((9, 4)), delt=1.0, dellam=-0.1, lam0=400.0)
    return seen[0]


def test_without_a_launcher_the_interpreter_runs_directly(paths, monkeypatch):
    dll, python = paths
    cmd = command_of(FrogDllRetrieval(dll, python), monkeypatch)
    assert cmd[0] == str(python)
    assert cmd[1] == str(frog._WORKER_SCRIPT)


def test_a_launcher_prefixes_the_interpreter(paths, monkeypatch):
    dll, python = paths
    retriever = FrogDllRetrieval(
        dll, python, launcher=("env", "WINEDEBUG=-all", "wine")
    )
    cmd = command_of(retriever, monkeypatch)
    assert cmd[:4] == ["env", "WINEDEBUG=-all", "wine", str(python)]
    assert cmd[4] == str(frog._WORKER_SCRIPT)


def test_from_config_splits_the_configured_launcher(paths, monkeypatch):
    dll, python = paths
    config = types.SimpleNamespace(
        frog_dll_path=dll,
        frog_python32_path=python,
        frog_launcher="env WINEDEBUG=-all wine",
    )
    import geecs_data_utils

    monkeypatch.setattr(geecs_data_utils, "GeecsPathsConfig", lambda: config)
    retriever = FrogDllRetrieval.from_config()
    assert retriever.launcher == ("env", "WINEDEBUG=-all", "wine")
    config.frog_launcher = None
    assert FrogDllRetrieval.from_config().launcher == ()


def test_the_retriever_pickles_for_pool_workers(paths):
    dll, python = paths
    copy = pickle.loads(pickle.dumps(FrogDllRetrieval(dll, python, launcher=["wine"])))
    assert (copy.dll_path, copy.python32_path, copy.launcher) == (
        dll,
        python,
        ("wine",),
    )

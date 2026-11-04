"""Pin ``scripts/ci_cpu_torch.py``'s guard: the CPU re-lock may move torch alone.

The GeecsBluesky CI leg re-locks its working copy with torch from the PyTorch
CPU index. The guard is what keeps that leg testing the environment the lab
installs; these tests pin that it accepts exactly the swap and nothing more.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_cpu_torch.py"
spec = importlib.util.spec_from_file_location("ci_cpu_torch", SCRIPT)
assert spec and spec.loader
ci_cpu_torch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_cpu_torch)

CPU = ci_cpu_torch.CPU_INDEX
BEFORE = {
    "torch": {("2.10.0", "pypi")},
    "nvidia-cudnn-cu12": {("9.10.2.21", "pypi")},
    "triton": {("3.6.0", "pypi")},
    "cuda-bindings": {("12.9.4", "pypi")},
    "numpy": {("2.2.6", "pypi")},
}
SWAPPED = {"torch": {("2.10.0+cpu", CPU)}, "numpy": {("2.2.6", "pypi")}}


def test_the_bare_swap_is_accepted():
    assert ci_cpu_torch.unexpected_changes(BEFORE, SWAPPED, "2.10.0") == []


@pytest.mark.parametrize(
    ("after", "complaint"),
    [
        ({**SWAPPED, "numpy": {("2.3.0", "pypi")}}, "numpy"),
        ({**SWAPPED, "scipy": {("1.15.0", "pypi")}}, "scipy"),
        ({k: v for k, v in SWAPPED.items() if k != "numpy"}, "numpy"),
        ({**SWAPPED, "triton": {("3.6.0", "pypi")}}, "triton is still locked"),
        ({**SWAPPED, "torch": {("2.10.0", "pypi")}}, "torch locked as"),
        ({**SWAPPED, "torch": {("2.11.0+cpu", CPU)}}, "torch locked as"),
    ],
    ids=[
        "version-moved",
        "package-added",
        "package-dropped",
        "cuda-kept",
        "torch-not-cpu",
        "torch-version-moved",
    ],
)
def test_any_other_change_is_refused(after, complaint):
    problems = ci_cpu_torch.unexpected_changes(BEFORE, after, "2.10.0")
    assert any(complaint in p for p in problems), problems


def test_a_lock_without_torch_stops_before_relocking(tmp_path, monkeypatch, capsys):
    (tmp_path / "poetry.lock").write_text(
        '[[package]]\nname = "numpy"\nversion = "2.2.6"\n'
    )
    monkeypatch.chdir(tmp_path)
    assert ci_cpu_torch.main() == 1
    assert "expected one locked torch" in capsys.readouterr().out


def test_a_stale_lock_fails_before_relocking(tmp_path, monkeypatch, capsys):
    """The re-lock rewrites the content-hash, so staleness is checked first."""
    (tmp_path / "poetry.lock").write_text(
        '[[package]]\nname = "torch"\nversion = "2.10.0"\n'
    )
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1)

    monkeypatch.setattr(ci_cpu_torch.subprocess, "run", fake_run)
    assert ci_cpu_torch.main() == 1
    assert [c[2:] for c in calls] == [["check", "--lock"]]
    assert "stale" in capsys.readouterr().out

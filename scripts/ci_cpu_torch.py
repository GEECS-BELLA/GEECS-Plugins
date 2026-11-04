#!/usr/bin/env python3
"""Swap a package's locked torch for the CPU-only build, in CI only.

On Linux the PyPI torch wheel requires the nvidia CUDA wheels (cuBLAS,
cuDNN, NCCL, triton, ...): several GB of downloads, all dead weight on a
CPU runner, enough to fill a hosted runner's disk. Run from the package
directory before ``poetry install``, this script re-locks the checkout's
working copy with torch, at the locked version, taken from the PyPI CPU
index instead. The committed lock is never written back: the lab's
installs keep the normal CUDA-capable torch, and developers'
``scripts/check.sh`` runs use whatever their own environment has.

The re-lock is guarded. Poetry keeps every other locked version, and the
script fails unless the result is the committed lock with exactly this
change: torch at ``<version>+cpu`` from the CPU index, and the CUDA
closure (``nvidia-*``, ``triton``, ``cuda-*``) gone. Anything else moving
would mean CI tests an environment the lab does not install.

Usage: ``python ../scripts/ci_cpu_torch.py`` from the package directory.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

CPU_INDEX = "https://download.pytorch.org/whl/cpu"
SOURCE_NAME = "pytorch-cpu"
CUDA_CLOSURE = re.compile(r"^(nvidia-.*|triton|cuda-.*)$")


def locked(lock: Path) -> dict[str, set[tuple[str, str]]]:
    """Map each locked package name to its (version, source url) pairs."""
    packages: dict[str, set[tuple[str, str]]] = {}
    for package in tomllib.loads(lock.read_text())["package"]:
        source = package.get("source", {}).get("url", "pypi")
        packages.setdefault(package["name"], set()).add((package["version"], source))
    return packages


def unexpected_changes(
    before: dict[str, set[tuple[str, str]]],
    after: dict[str, set[tuple[str, str]]],
    torch_version: str,
) -> list[str]:
    """List every difference between the locks other than the CPU swap."""
    problems = []
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name), after.get(name)
        if name == "torch":
            if new != {(f"{torch_version}+cpu", CPU_INDEX)}:
                problems.append(f"torch locked as {new}, not {torch_version}+cpu")
        elif CUDA_CLOSURE.match(name):
            if new is not None:
                problems.append(f"{name} is still locked: {new}")
        elif old != new:
            problems.append(f"{name}: {old} -> {new}")
    return problems


def main() -> int:
    """Re-lock with the CPU torch and verify nothing else moved."""
    lock = Path("poetry.lock")
    before = locked(lock)
    versions = before.get("torch")
    if not versions or len(versions) != 1:
        print(f"ci_cpu_torch: expected one locked torch, found {versions}")
        return 1
    ((torch_version, _),) = versions

    poetry = ["poetry", "--no-interaction"]
    subprocess.run(
        [*poetry, "source", "add", "--priority=explicit", SOURCE_NAME, CPU_INDEX],
        check=True,
    )
    subprocess.run(
        [
            *poetry,
            "add",
            "--lock",
            f"torch=={torch_version}+cpu",
            "--source",
            SOURCE_NAME,
        ],
        check=True,
    )

    problems = unexpected_changes(before, locked(lock), torch_version)
    if problems:
        print("ci_cpu_torch: the CPU re-lock changed more than torch:")
        print("\n".join(f"  {p}" for p in problems))
        return 1
    dropped = sorted(n for n in before if CUDA_CLOSURE.match(n))
    print(f"ci_cpu_torch: torch {torch_version} -> {torch_version}+cpu ({CPU_INDEX});")
    print(f"ci_cpu_torch: dropped {len(dropped)} CUDA packages: {', '.join(dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

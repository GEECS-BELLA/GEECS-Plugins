"""Helpers for capturing post-run analysis provenance."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from geecs_bluesky.analysis.models import CodeVersion, SoftwareEnvironment


def capture_code_version(repo_root: Path | None = None) -> CodeVersion:
    """Capture git repository state for analysis provenance."""
    root = Path(repo_root or Path.cwd())
    return CodeVersion(
        repository=_git_text(root, "config", "--get", "remote.origin.url"),
        commit=_git_text(root, "rev-parse", "HEAD"),
        branch=_git_text(root, "branch", "--show-current"),
        dirty=_git_dirty(root),
    )


def capture_environment(package_names: list[str] | None = None) -> SoftwareEnvironment:
    """Capture Python/platform/package versions for analysis provenance."""
    packages = {}
    for name in package_names or []:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    return SoftwareEnvironment(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        packages=packages,
    )


def _git_text(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _git_dirty(repo_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())

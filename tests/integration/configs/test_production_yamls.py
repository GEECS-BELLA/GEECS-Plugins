"""Validate every production diagnostic YAML loads under the current schema.

A parametrized smoke test that walks the GEECS-Plugins-Configs repo and
runs each ``scan_analysis_configs/analyzers/`` YAML through
``load_diagnostic``. Catches regressions where the schema and the
production YAMLs drift out of sync — for example, when
``CameraConfig.metadata`` was flipped from "silent extra" to "typed
field" and 26 production YAMLs needed the schema update to keep
loading.

``UNCLASSIFIED/`` is deliberately excluded: those YAMLs are
experimental / legacy and not part of the supported production set.
HTT and HTU diagnostics are what scans actually use.

Run with::

    pytest -m integration tests/integration/configs/

Doesn't require the network data drive — just the configs repo
checked out somewhere we can find it (``ScanPaths.paths_config``
first, then a sibling-of-plugins fallback).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Configs-repo discovery — module-level so parametrize can see the YAML list
# at collection time
# ---------------------------------------------------------------------------


def _find_scan_analysis_configs() -> Optional[Path]:
    """Return the ``scan_analysis_configs`` dir if we can find it, else None.

    Tries ScanPaths first (production-style resolution), then a
    sibling-of-plugins fallback for local dev where the configs repo
    is checked out next to the plugins repo.
    """
    # 1. ScanPaths-resolved path (lab machines, networked dev)
    try:
        from geecs_data_utils import ScanPaths

        configured = getattr(ScanPaths.paths_config, "scan_analysis_configs_path", None)
        if configured:
            base = Path(configured)
            if (base / "analyzers").is_dir():
                return base
    except Exception:
        pass

    # 2. Sibling-of-plugins fallback (local dev convenience)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent.parent / "GEECS-Plugins-Configs" / "scan_analysis_configs"
        if (candidate / "analyzers").is_dir():
            return candidate

    return None


def _discover_production_yamls() -> List[Path]:
    """Find every production-tier diagnostic YAML.

    Production tier = anything under ``analyzers/`` other than
    ``UNCLASSIFIED/`` (which is experimental / legacy and not
    consumed by production scans).
    """
    base = _find_scan_analysis_configs()
    if base is None:
        return []

    analyzers_dir = base / "analyzers"
    return sorted(
        path
        for path in (
            list(analyzers_dir.rglob("*.yaml")) + list(analyzers_dir.rglob("*.yml"))
        )
        if "UNCLASSIFIED" not in path.parts
    )


_PRODUCTION_YAMLS = _discover_production_yamls()


def _yaml_id(path: Path) -> str:
    """Compact, readable parametrize ID — ``namespace/stem``."""
    return f"{path.parent.name}/{path.stem}"


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaml_path",
    _PRODUCTION_YAMLS
    or [
        pytest.param(
            None,
            marks=pytest.mark.skip(
                reason=(
                    "GEECS-Plugins-Configs not found via ScanPaths or as a "
                    "sibling of this repo. Check out the configs repo to run "
                    "this test."
                )
            ),
        )
    ],
    ids=[_yaml_id(p) for p in _PRODUCTION_YAMLS] or ["skipped"],
)
def test_production_yaml_loads(yaml_path: Path) -> None:
    """Every HTT / HTU diagnostic YAML loads cleanly under the current schema.

    This pins the contract between the plugins repo's
    ``DiagnosticAnalysisConfig`` schema and the production configs.
    Any schema change that breaks an existing production YAML will
    fail this test with the specific YAML path and pydantic error,
    rather than only being discovered when someone tries to run a
    real scan in the lab.
    """
    from image_analysis.config import load_diagnostic
    from image_analysis.config.diagnostic import DiagnosticAnalysisConfig

    diag = load_diagnostic(yaml_path)
    assert isinstance(diag, DiagnosticAnalysisConfig), (
        f"load_diagnostic returned {type(diag).__name__}, expected "
        f"DiagnosticAnalysisConfig"
    )

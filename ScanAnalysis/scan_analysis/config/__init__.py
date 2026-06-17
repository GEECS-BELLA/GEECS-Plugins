"""Configuration system for scan analysis.

This package provides the unified diagnostic config schema and
loader for scan analyzers. Each diagnostic is one YAML file under
``scan_analysis_configs/analyzers/<namespace>/<id>.yaml`` carrying
both an ``image:`` section (consumed by ImageAnalysis) and a
``scan:`` section (consumed by ScanAnalysis). Diagnostics are
collected into analysis groups under
``scan_analysis_configs/groups/<namespace>/<group>.yaml``, which
LiveWatch and the task queue consume directly.

Quick start
-----------

Load a group, instantiate its analyzers, run::

    >>> from scan_analysis.config import load_analysis_group, create_scan_analyzer
    >>>
    >>> group = load_analysis_group("baseline", config_dir=<scan_analysis_configs>)
    >>> analyzers = [
    ...     create_scan_analyzer(r.diagnostic, id=r.id, priority=r.priority)
    ...     for r in group.analyzers
    ... ]
    >>> for a in analyzers:
    ...     a.run_analysis(scan_tag)

Or build a single diagnostic directly::

    >>> from image_analysis.config import load_diagnostic
    >>> from scan_analysis.config import create_scan_analyzer
    >>> diag = load_diagnostic("UC_VisaEBeam1")
    >>> diag.image.roi.x_max = 200    # optional notebook tweak
    >>> analyzer = create_scan_analyzer(diag)

Environment
-----------

``SCAN_ANALYSIS_CONFIG_DIR`` (env var) or ``scan_analysis_configs_path``
in ``~/.config/geecs_python_api/config.ini`` sets the configs root.
ImageAnalysis derives its own search root as
``<scan_analysis_configs_path>/analyzers`` automatically.
"""

# Unified diagnostic schema — the top-level model lives in ImageAnalysis
# (it owns the image_analyzer + image: shape), re-exported here for
# back-compat with callers used to importing from scan_analysis.config.
from image_analysis.config import (
    DiagnosticAnalysisConfig,
    ImageAnalyzerSpec,
    resolve_image_analyzer_value,
)
from .diagnostic_models import (
    AnalysisGroupConfig,
    AnalyzerRef,
    AutodetectBackgroundSpec,
    BackgroundSource,
    FromCurrentScanSpec,
    ResolvedDiagnosticConfig,
    ScanRuntimeConfig,
)

# Loader + factory for unified diagnostics
from .analysis_group_loader import (
    LoadedAnalysisGroup,
    discover_analyzers,
    discover_groups,
    load_analysis_group,
    resolve_group,
)
from .diagnostic_factory import create_scan_analyzer

__all__ = [
    # Unified diagnostic models
    "DiagnosticAnalysisConfig",
    "ScanRuntimeConfig",
    "ResolvedDiagnosticConfig",
    "AnalyzerRef",
    "AnalysisGroupConfig",
    "BackgroundSource",
    "AutodetectBackgroundSpec",
    "FromCurrentScanSpec",
    # image_analyzer field model + helpers
    "ImageAnalyzerSpec",
    "resolve_image_analyzer_value",
    # Loader
    "load_analysis_group",
    "discover_analyzers",
    "discover_groups",
    "resolve_group",
    "LoadedAnalysisGroup",
    # Factory
    "create_scan_analyzer",
]

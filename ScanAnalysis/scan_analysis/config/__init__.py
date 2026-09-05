"""Configuration system for scan analysis.

This package provides the loader and factory for scan analyzers over
the analysis-config documents defined in ``geecs_schemas.analysis``.
Each diagnostic is one YAML file under
``scan_analysis_configs/analyzers/<namespace>/<id>.yaml`` carrying an
``analyzer:`` section (which analyzer, with its parameters), an
``image:`` section (consumed by ImageAnalysis) and a typed ``scan:``
section (consumed here). Diagnostics are
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

# The documents live in GEECS-Schemas (geecs_schemas.analysis); the
# historical names are kept as aliases in diagnostic_models.
from image_analysis.config import DiagnosticAnalysisConfig
from .diagnostic_models import (
    AnalysisDiagnostic,
    AnalysisGroup,
    AnalysisGroupConfig,
    AnalyzerRef,
    AutodetectBackgroundSpec,
    BackgroundSource,
    FromCurrentScanSpec,
    RendererOptions,
    ResolvedDiagnosticConfig,
    ScanRuntime,
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
    # The documents (geecs_schemas.analysis) + transitional aliases
    "AnalysisDiagnostic",
    "DiagnosticAnalysisConfig",
    "ScanRuntime",
    "ScanRuntimeConfig",
    "RendererOptions",
    "ResolvedDiagnosticConfig",
    "AnalyzerRef",
    "AnalysisGroup",
    "AnalysisGroupConfig",
    "BackgroundSource",
    "AutodetectBackgroundSpec",
    "FromCurrentScanSpec",
    # Loader
    "load_analysis_group",
    "discover_analyzers",
    "discover_groups",
    "resolve_group",
    "LoadedAnalysisGroup",
    # Factory
    "create_scan_analyzer",
]

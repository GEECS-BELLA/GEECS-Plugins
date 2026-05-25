"""Configuration system for scan analysis.

This package provides the unified diagnostic config schema and
loader for scan analyzers. Each diagnostic is one YAML file under
``scan_analysis_configs/analyzers/<namespace>/<id>.yaml`` carrying
both an ``image:`` section (consumed by ImageAnalysis) and a
``scan:`` section (consumed by ScanAnalysis). Diagnostics are
collected into analysis groups under
``scan_analysis_configs/groups/<namespace>/<group>.yaml``, which
LiveWatch and the task queue consume directly.

Scatter scan analyzers stay on a separate config shape because they
do not consume images.

Quick start
-----------

Load a group, instantiate its analyzers, run::

    >>> from scan_analysis.config import load_analysis_group, create_diagnostic_analyzer
    >>>
    >>> group = load_analysis_group("baseline", config_dir=<scan_analysis_configs>)
    >>> analyzers = [create_diagnostic_analyzer(r) for r in group.analyzers]
    >>> for a in analyzers:
    ...     a.run_analysis(scan_tag)

Or build a single diagnostic directly from its YAML path::

    >>> from scan_analysis.config import DiagnosticAnalysisConfig
    >>> import yaml
    >>> with open("analyzers/HTU/GaiaMode.yaml") as f:
    ...     cfg = DiagnosticAnalysisConfig.model_validate(yaml.safe_load(f))

Environment
-----------

``SCAN_ANALYSIS_CONFIG_DIR`` (env var) or ``scan_analysis_configs_path``
in ``~/.config/geecs_python_api/config.ini`` sets the configs root.
ImageAnalysis derives its own search root as
``<scan_analysis_configs_path>/analyzers`` automatically.
"""

# Unified diagnostic schema
from .aliases import (
    ALIAS_REGISTRY,
    ImageAnalyzerSpec,
    ImageKind,
    ScanType,
    resolve_image_analyzer_value,
)
from .diagnostic_models import (
    AnalysisGroupConfig,
    AnalyzerRef,
    BackgroundSource,
    DiagnosticAnalysisConfig,
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
    load_diagnostic,
    resolve_group,
)
from .diagnostic_factory import create_diagnostic_analyzer

# Scatter analyzers (separate path; not unified)
from .analyzer_config_models import PlotParameterConfig, ScatterAnalyzerConfig
from .analyzer_factory import create_analyzer

__all__ = [
    # Unified diagnostic models
    "DiagnosticAnalysisConfig",
    "ScanRuntimeConfig",
    "ResolvedDiagnosticConfig",
    "AnalyzerRef",
    "AnalysisGroupConfig",
    "BackgroundSource",
    "FromCurrentScanSpec",
    # Alias registry
    "ImageAnalyzerSpec",
    "ImageKind",
    "ScanType",
    "ALIAS_REGISTRY",
    "resolve_image_analyzer_value",
    # Loader
    "load_analysis_group",
    "load_diagnostic",
    "discover_analyzers",
    "discover_groups",
    "resolve_group",
    "LoadedAnalysisGroup",
    # Factory
    "create_diagnostic_analyzer",
    # Scatter (legacy non-unified path)
    "ScatterAnalyzerConfig",
    "PlotParameterConfig",
    "create_analyzer",
]

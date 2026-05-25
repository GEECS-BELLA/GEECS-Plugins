"""Public configuration API for ImageAnalysis.

This subpackage owns the unified diagnostic config schema —
:class:`DiagnosticAnalysisConfig` and its constituent models. The
schema lives here (rather than in ScanAnalysis) because ImageAnalysis
is the natural owner of "which ImageAnalyzer class + what image
config" and the dependency direction is one-way (ScanAnalysis depends
on ImageAnalysis, never the reverse).

The ``scan:`` field on a unified diagnostic is weakly typed at this
layer (``Optional[Dict[str, Any]]``) so ImageAnalysis can validate the
whole YAML without needing to import scan-side runtime types.
ScanAnalysis validates the scan dict against its own
:class:`scan_analysis.config.diagnostic_models.ScanRuntimeConfig` at
build time.

Public API
----------
- :class:`DiagnosticAnalysisConfig` — the unified YAML's top-level model
- :class:`ImageAnalyzerSpec` — the resolved ``image_analyzer`` field
- :func:`load_diagnostic` — load + validate a unified YAML by name/path
- :func:`create_image_analyzer` — build an ``ImageAnalyzer`` from a config
- :data:`ALIAS_REGISTRY` — short-name aliases for common analyzers
- :func:`resolve_image_analyzer_value` — string / dict → spec dict
"""

from .aliases import (
    ALIAS_REGISTRY,
    ImageAnalyzerSpec,
    ImageKind,
    ScanType,
    resolve_image_analyzer_value,
)
from .diagnostic_models import DiagnosticAnalysisConfig
from .factory import create_image_analyzer, load_diagnostic

__all__ = [
    "ALIAS_REGISTRY",
    "DiagnosticAnalysisConfig",
    "ImageAnalyzerSpec",
    "ImageKind",
    "ScanType",
    "create_image_analyzer",
    "load_diagnostic",
    "resolve_image_analyzer_value",
]

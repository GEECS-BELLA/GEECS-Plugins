"""The diagnostic document — re-exported from GEECS-Schemas.

:class:`AnalysisDiagnostic` (format v2) replaced ImageAnalysis' own
``DiagnosticAnalysisConfig`` in 2.0; the old name is kept as an alias for
the transition.  The v1 layout (``image_analyzer`` class path,
``image.analysis``, constructor ``kwargs``) is lifted automatically at
validation — see the schema module for the mapping.
"""

from geecs_schemas.analysis import (
    ANALYZER_SPECS,
    AnalysisDiagnostic,
    AnalyzerSpec,
    ImageSection,
)

#: Transitional alias — the pre-2.0 name of the diagnostic model.
DiagnosticAnalysisConfig = AnalysisDiagnostic

__all__ = [
    "ANALYZER_SPECS",
    "AnalysisDiagnostic",
    "AnalyzerSpec",
    "DiagnosticAnalysisConfig",
    "ImageSection",
]

"""Analysis-config schemas: the diagnostic document, the group document, and their parts.

Everything ImageAnalysis and ScanAnalysis load from ``scan_analysis_configs/``
is described here — pydantic only, so editors, the data portal, MCP and
CI can validate a diagnostic without the analysis stack installed.
"""

from geecs_schemas.analysis.analyzers import (
    ANALYZER_SPECS,
    AnalyzerSpec,
    AnalyzerSpecBase,
    ArrayCalibrationSpec,
    BCaveMagOptSpec,
    BCaveMagSpecStitcherSpec,
    BeamAnalyzerSpec,
    CalibrationSpec,
    DnnAxisCalibrationSpec,
    DownrampPhaseSpec,
    FrogRetrievalSpec,
    FrogSpectralPhaseSpec,
    HasoAnalyzerSpec,
    HiResMagCamSpec,
    IctAnalyzerSpec,
    ImageKind,
    LineAnalyzerSpec,
    LineStitcherSpec,
    MagSpecAnalyzerSpec,
    PhaseDownrampSpec,
    PolynomialCalibrationSpec,
    PupilMask,
    StandardAnalyzerSpec,
    TraceAnalyzerSpec,
)
from geecs_schemas.analysis.diagnostic import (
    CURRENT_SCHEMA_VERSION,
    AnalysisDiagnostic,
    ImageSection,
)
from geecs_schemas.analysis.group import AnalysisGroup, AnalyzerRef
from geecs_schemas.analysis.processing_1d import Data1DLoading, Data1DType, Line1DConfig
from geecs_schemas.analysis.processing_2d import CameraConfig
from geecs_schemas.analysis.renderer import RendererOptions
from geecs_schemas.analysis.scan_runtime import (
    AutodetectBackgroundSpec,
    BackgroundSource,
    FromCurrentScanSpec,
    ScanRuntime,
)

__all__ = [
    "ANALYZER_SPECS",
    "CURRENT_SCHEMA_VERSION",
    "AnalysisDiagnostic",
    "AnalysisGroup",
    "AnalyzerRef",
    "AnalyzerSpec",
    "AnalyzerSpecBase",
    "ArrayCalibrationSpec",
    "AutodetectBackgroundSpec",
    "BCaveMagOptSpec",
    "BCaveMagSpecStitcherSpec",
    "BackgroundSource",
    "BeamAnalyzerSpec",
    "CalibrationSpec",
    "CameraConfig",
    "Data1DLoading",
    "Data1DType",
    "DnnAxisCalibrationSpec",
    "DownrampPhaseSpec",
    "FromCurrentScanSpec",
    "FrogRetrievalSpec",
    "FrogSpectralPhaseSpec",
    "HasoAnalyzerSpec",
    "HiResMagCamSpec",
    "IctAnalyzerSpec",
    "ImageKind",
    "ImageSection",
    "Line1DConfig",
    "LineAnalyzerSpec",
    "LineStitcherSpec",
    "MagSpecAnalyzerSpec",
    "PhaseDownrampSpec",
    "PolynomialCalibrationSpec",
    "PupilMask",
    "RendererOptions",
    "ScanRuntime",
    "StandardAnalyzerSpec",
    "TraceAnalyzerSpec",
]

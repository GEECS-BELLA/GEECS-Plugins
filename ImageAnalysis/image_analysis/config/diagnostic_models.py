"""Top-level Pydantic models for the unified diagnostic config schema.

A diagnostic config is one YAML file per device that bundles two
concerns previously split across ``image_analysis_configs/`` and
``scan_analysis_configs/library/analyzers/``:

* ``image:`` — the ImageAnalysis-owned section (camera config or 1D
  line config). Kept as a raw dict here; ImageAnalysis's typed loader
  validates it against :class:`CameraConfig` /
  :class:`Line1DConfig` on the way through.
* ``scan:`` — the ScanAnalysis-owned section (priority, mode, save
  flags, gdoc slot, file tail, renderer kwargs). Kept weakly typed
  here (``Optional[Dict[str, Any]]``) so this model can live in
  ImageAnalysis without importing scan-side runtime types.
  ScanAnalysis validates the dict against its own ``ScanRuntimeConfig``
  at scan-analyzer build time.

A diagnostic also declares its ImageAnalyzer via ``image_analyzer``,
either as a bare class-path string (defaults to camera + array2d) or
as a verbose dict; both forms validate into :class:`ImageAnalyzerSpec`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .aliases import ImageAnalyzerSpec, resolve_image_analyzer_value


class DiagnosticAnalysisConfig(BaseModel):
    """One device's diagnostic config: image and scan sections in one file.

    The on-disk form lives at ``analyzers/<namespace>/<stem>.yaml``;
    the filename stem becomes the analyzer ID used by the task queue
    and referenced from group configs.

    Attributes
    ----------
    name : str
        Logical device name. Used as the default data folder name and
        as the default metric prefix (i.e. injected into ``image.name``
        before ImageAnalysis validates the embedded image section).
    image_analyzer : ImageAnalyzerSpec
        Identifies the ImageAnalyzer class. Accepts a bare class-path
        string (defaults to camera + array2d) or a verbose
        ``{class_path, image_kind, scan_type, kwargs}`` dict for
        analyzers that need different defaults or per-instance kwargs.
    image : dict, optional
        Raw embedded image config. Validated by ImageAnalysis's
        ``load_camera_config`` / ``load_line_config`` after ``name`` is
        injected. Must be ``None`` (or omitted) when the analyzer's
        ``image_kind`` is ``none``.
    scan : dict, optional
        Raw embedded scan-runtime config. Weakly typed at this layer.
        ScanAnalysis validates this against
        ``scan_analysis.config.diagnostic_models.ScanRuntimeConfig``
        when building a scan-analyzer wrapper.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    image_analyzer: ImageAnalyzerSpec
    image: Optional[Dict[str, Any]] = None
    scan: Optional[Dict[str, Any]] = None

    @field_validator("image_analyzer", mode="before")
    @classmethod
    def _normalise_image_analyzer(cls, value: Any) -> Any:
        """Accept alias string, alias-with-kwargs dict, or verbose dict."""
        return resolve_image_analyzer_value(value)

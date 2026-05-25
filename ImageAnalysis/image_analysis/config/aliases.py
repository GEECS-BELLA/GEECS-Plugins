"""Model for the ``image_analyzer`` field on a unified diagnostic.

A diagnostic YAML names its ImageAnalyzer by full class path. Two
surface forms are accepted on the ``image_analyzer`` field:

* **Bare string** — the class path. Defaults the analyzer to a 2D
  camera + ``Array2DScanAnalyzer`` wrapper. Right for the common case
  (``BeamAnalyzer``, ``StandardAnalyzer``, ``GrenouilleAnalyzer``,
  ``MagSpecManualCalibAnalyzer``, …)::

      image_analyzer: image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer

* **Verbose dict** — class path plus explicit ``image_kind`` /
  ``scan_type`` / ``kwargs``. Required for 1D analyzers (line config
  inputs, ``Array1DScanAnalyzer`` wrapper) and for analyzers that
  consume no embedded image config (HASO)::

      image_analyzer:
        class_path: image_analysis.offline_analyzers.line_analyzer.LineAnalyzer
        image_kind: line
        scan_type: array1d
        kwargs:
          some_extra_arg: 42

Both forms normalise to :class:`ImageAnalyzerSpec`.

The alias registry that used to live here is gone — every analyzer's
fully qualified class path is the identifier. Adding a new analyzer
needs no registry update; just write the class path in the YAML.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Mapping

from pydantic import BaseModel, ConfigDict, Field


class ImageKind(str, Enum):
    """How an ImageAnalyzer consumes the embedded ``image:`` config section.

    The loader uses this to decide which ImageAnalysis Pydantic model
    validates the ``image:`` body:

    * ``camera`` — validate as :class:`CameraConfig` (2D).
    * ``line`` — validate as :class:`Line1DConfig` (1D).
    * ``none`` — analyzer has no embedded image config; the YAML must
      omit the ``image:`` section entirely.
    """

    CAMERA = "camera"
    LINE = "line"
    NONE = "none"


class ScanType(str, Enum):
    """Which ScanAnalyzer wrapper class instantiates this analyzer."""

    ARRAY2D = "array2d"
    ARRAY1D = "array1d"


class ImageAnalyzerSpec(BaseModel):
    """Canonical runtime form of an ``image_analyzer`` field.

    Whether the YAML used the bare-string form or the verbose dict,
    the field validator on
    :class:`DiagnosticAnalysisConfig.image_analyzer` normalises to
    this shape before anything else sees it.

    Attributes
    ----------
    class_path : str
        Fully qualified import path of the ImageAnalyzer class
        (``"image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer"``).
    image_kind : ImageKind
        How the analyzer consumes the embedded ``image:`` config.
        Defaults to ``camera`` (the common case); the verbose form
        must override for line/none analyzers.
    scan_type : ScanType
        Which scan-analyzer wrapper hosts this analyzer. Defaults to
        ``array2d``; the verbose form must override for 1D analyzers.
    kwargs : dict
        Per-instance kwargs passed to the analyzer's constructor. The
        loader injects the resolved embedded image config under the
        constructor's expected name (``camera_config_name`` or
        ``line_config_name``) separately; do not put the image config
        here.
    """

    model_config = ConfigDict(extra="forbid")

    class_path: str = Field(min_length=1)
    image_kind: ImageKind = ImageKind.CAMERA
    scan_type: ScanType = ScanType.ARRAY2D
    kwargs: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Surface-form normalisation
# ---------------------------------------------------------------------------


def resolve_image_analyzer_value(value: Any) -> Dict[str, Any]:
    """Normalise an ``image_analyzer`` YAML value to an :class:`ImageAnalyzerSpec` dict.

    Accepts the two surface forms documented at the module level and
    returns a dict ready to be validated by :class:`ImageAnalyzerSpec`.
    Used by the field validator on
    :class:`DiagnosticAnalysisConfig.image_analyzer`.

    Parameters
    ----------
    value : Any
        Whatever the YAML deserialiser produced for the
        ``image_analyzer`` field — a bare class-path string or a dict
        with ``class_path`` (or YAML-friendly ``class``) plus optional
        ``image_kind`` / ``scan_type`` / ``kwargs``.

    Returns
    -------
    dict
        A dict matching :class:`ImageAnalyzerSpec`'s field layout.
        Missing ``image_kind`` / ``scan_type`` fall through to the
        model defaults (``camera`` / ``array2d``).

    Raises
    ------
    ValueError
        If the dict is missing ``class_path`` / ``class``, or if an
        unrecognised input shape is supplied.
    """
    if isinstance(value, str):
        return {"class_path": value}

    if isinstance(value, Mapping):
        data = dict(value)
        if "class" in data:
            # YAML-friendly alias for the model field.
            data["class_path"] = data.pop("class")
        if "class_path" not in data:
            raise ValueError(
                "image_analyzer dict must contain 'class_path' (or "
                "YAML-friendly 'class') naming the analyzer class. "
                f"Got keys: {sorted(data)}"
            )
        return data

    # Already an ImageAnalyzerSpec instance — let Pydantic pass it through.
    if isinstance(value, ImageAnalyzerSpec):
        return value.model_dump()

    raise ValueError(
        f"image_analyzer must be a class-path string or a mapping; got "
        f"{type(value).__name__}"
    )

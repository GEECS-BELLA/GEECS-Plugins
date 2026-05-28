"""Alias registry for ImageAnalyzer classes used by scan analyzers.

A diagnostic YAML names its ImageAnalyzer either by a short alias
(``image_analyzer: beam``) or by a verbose dict that includes the full
class path and the metadata the loader needs (``image_kind``,
``scan_type``). Both surface forms validate into the same
:class:`ImageAnalyzerSpec` so the rest of the loader and the factory
work with one canonical shape.

Aliases live here, in code, intentionally. Each entry records a class
path plus enough metadata that the loader can decide whether to embed
a 2D camera config, a 1D line config, or no embedded config at all,
and whether the surrounding ScanAnalyzer wrapper should be
``Array2DScanAnalyzer`` or ``Array1DScanAnalyzer``.

Three surface forms are accepted on the ``image_analyzer`` field of a
diagnostic config:

* ``image_analyzer: beam`` — alias only. The matching registry entry
  supplies the class path and metadata.
* ``image_analyzer: {alias: line_stitcher, kwargs: {...}}`` — alias
  plus per-instance kwargs, for analyzers like ``LineStitcher`` that
  take constructor arguments beyond the embedded image config.
* ``image_analyzer: {class: full.path.Class, image_kind: ...,
  scan_type: ..., kwargs: {...}}`` — verbose escape hatch for
  analyzers with no registered alias.
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

    Whether the YAML used an alias string, an ``{alias, kwargs}`` dict,
    or the verbose ``{class, image_kind, scan_type, kwargs}`` dict, the
    field validator on :class:`DiagnosticAnalysisConfig.image_analyzer`
    normalises to this shape before anything else sees it.

    Attributes
    ----------
    class_path : str
        Fully qualified import path of the ImageAnalyzer class
        (``"image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer"``).
    image_kind : ImageKind
        How the analyzer consumes the embedded ``image:`` config.
    scan_type : ScanType
        Which scan-analyzer wrapper should host this analyzer.
    kwargs : dict
        Per-instance kwargs passed to the analyzer's constructor. The
        loader injects the resolved embedded image config under the
        constructor's expected name (``camera_config_name`` or
        ``line_config_name``) separately; do not put the image config
        here.
    """

    model_config = ConfigDict(extra="forbid")

    class_path: str = Field(min_length=1)
    image_kind: ImageKind
    scan_type: ScanType
    kwargs: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in alias registry
# ---------------------------------------------------------------------------

ALIAS_REGISTRY: Dict[str, ImageAnalyzerSpec] = {
    "beam": ImageAnalyzerSpec(
        class_path="image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer",
        image_kind=ImageKind.CAMERA,
        scan_type=ScanType.ARRAY2D,
    ),
    "standard_2d": ImageAnalyzerSpec(
        class_path="image_analysis.offline_analyzers.standard_analyzer.StandardAnalyzer",
        image_kind=ImageKind.CAMERA,
        scan_type=ScanType.ARRAY2D,
    ),
    "grenouille": ImageAnalyzerSpec(
        class_path="image_analysis.offline_analyzers.grenouille_analyzer.GrenouilleAnalyzer",
        image_kind=ImageKind.CAMERA,
        scan_type=ScanType.ARRAY2D,
    ),
    "magspec_manual": ImageAnalyzerSpec(
        class_path=(
            "image_analysis.offline_analyzers."
            "magspec_manual_calib_analyzer.MagSpecManualCalibAnalyzer"
        ),
        image_kind=ImageKind.CAMERA,
        scan_type=ScanType.ARRAY2D,
    ),
    "standard_1d": ImageAnalyzerSpec(
        class_path="image_analysis.offline_analyzers.standard_1d_analyzer.Standard1DAnalyzer",
        image_kind=ImageKind.LINE,
        scan_type=ScanType.ARRAY1D,
    ),
    "line": ImageAnalyzerSpec(
        class_path="image_analysis.offline_analyzers.line_analyzer.LineAnalyzer",
        image_kind=ImageKind.LINE,
        scan_type=ScanType.ARRAY1D,
    ),
    "ict_1d": ImageAnalyzerSpec(
        class_path="image_analysis.offline_analyzers.ict_1d_analyzer.ICT1DAnalyzer",
        image_kind=ImageKind.LINE,
        scan_type=ScanType.ARRAY1D,
    ),
    "line_stitcher": ImageAnalyzerSpec(
        class_path="image_analysis.offline_analyzers.line_stitcher.LineStitcher",
        image_kind=ImageKind.LINE,
        scan_type=ScanType.ARRAY1D,
    ),
    "haso": ImageAnalyzerSpec(
        class_path=(
            "image_analysis.offline_analyzers."
            "HASO_himg_has_processor.HASOHimgHasProcessor"
        ),
        image_kind=ImageKind.NONE,
        scan_type=ScanType.ARRAY2D,
    ),
}


# ---------------------------------------------------------------------------
# Surface-form normalisation
# ---------------------------------------------------------------------------


def resolve_image_analyzer_value(value: Any) -> Dict[str, Any]:
    """Normalise an ``image_analyzer`` YAML value to an :class:`ImageAnalyzerSpec` dict.

    Accepts the three surface forms documented at the module level and
    returns a dict ready to be validated by :class:`ImageAnalyzerSpec`.
    Used by the field validator on
    :class:`DiagnosticAnalysisConfig.image_analyzer`.

    Parameters
    ----------
    value : Any
        Whatever the YAML deserialiser produced for the
        ``image_analyzer`` field — a string alias, a dict with
        ``alias`` and optional ``kwargs``, or a verbose dict with
        ``class``/``image_kind``/``scan_type`` and optional ``kwargs``.

    Returns
    -------
    dict
        A dict matching :class:`ImageAnalyzerSpec`'s field layout.

    Raises
    ------
    ValueError
        If the alias is unknown, if the verbose form is missing required
        keys, or if an unrecognised shape is supplied.
    """
    if isinstance(value, str):
        return _expand_alias(value, kwargs={})

    if isinstance(value, Mapping):
        data = dict(value)

        if "alias" in data:
            alias = data.pop("alias")
            kwargs = data.pop("kwargs", {})
            if data:
                raise ValueError(
                    f"image_analyzer alias form accepts only 'alias' and "
                    f"'kwargs'; got extra keys: {sorted(data)}"
                )
            if not isinstance(alias, str):
                raise ValueError(
                    f"image_analyzer.alias must be a string, got {type(alias).__name__}"
                )
            return _expand_alias(alias, kwargs=kwargs)

        if "class" in data or "class_path" in data:
            # Verbose form. Accept either YAML-friendly `class:` or
            # the model's actual field name.
            if "class" in data:
                data["class_path"] = data.pop("class")
            return data

        raise ValueError(
            "image_analyzer dict must contain either 'alias' (for the "
            "registry form) or 'class' (for the verbose form). Got keys: "
            f"{sorted(data)}"
        )

    # Already an ImageAnalyzerSpec instance — let Pydantic pass it through.
    if isinstance(value, ImageAnalyzerSpec):
        return value.model_dump()

    raise ValueError(
        f"image_analyzer must be a string alias or a mapping; got "
        f"{type(value).__name__}"
    )


def _expand_alias(alias: str, *, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve an alias name to a fresh :class:`ImageAnalyzerSpec` dict."""
    if alias not in ALIAS_REGISTRY:
        raise ValueError(
            f"Unknown image_analyzer alias '{alias}'. Known aliases: "
            f"{sorted(ALIAS_REGISTRY)}. For analyzers without a registered "
            f"alias, use the verbose form: image_analyzer: {{class: ..., "
            f"image_kind: ..., scan_type: ...}}."
        )
    spec = ALIAS_REGISTRY[alias]
    return {
        "class_path": spec.class_path,
        "image_kind": spec.image_kind,
        "scan_type": spec.scan_type,
        "kwargs": dict(kwargs),
    }

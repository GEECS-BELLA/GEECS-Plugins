"""Model for the ``image_analyzer`` field on a unified diagnostic.

A diagnostic YAML names its ImageAnalyzer by full class path. Two
surface forms are accepted on the ``image_analyzer`` field:

* **Bare string** — the class path. Right for the common case
  (``BeamAnalyzer``, ``StandardAnalyzer``, ``GrenouilleAnalyzer``,
  ``MagSpecManualCalibAnalyzer``, …)::

      image_analyzer: image_analysis.analyzers.beam_analyzer.BeamAnalyzer

* **Verbose dict** — class path plus extra constructor kwargs.
  Right for analyzers that take per-instance kwargs (HASO's
  ``wavekit_config_file_path``, for example)::

      image_analyzer:
        class_path: image_analysis.analyzers.HASO_himg_has_processor.HASOHimgHasProcessor
        kwargs:
          wavekit_config_file_path: /path/to/wfs.dat
          mask_top: 125

Both forms normalise to :class:`ImageAnalyzerSpec`.

There is no ``image_kind`` / ``scan_type`` field anymore — those moved
onto the ``image:`` section itself as a Pydantic discriminator on
:class:`CameraConfig` / :class:`Line1DConfig`. The ScanAnalyzer
wrapper (Array2D vs Array1D) is picked the same way at scan-build
time, from the type of ``diag.image``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from pydantic import BaseModel, ConfigDict, Field


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
        (``"image_analysis.analyzers.beam_analyzer.BeamAnalyzer"``).
    kwargs : dict
        Per-instance kwargs passed to the analyzer's constructor. The
        loader injects the resolved embedded image config under the
        constructor's expected name (``camera_config`` or
        ``line_config``) separately; do not put the image config here.
    """

    model_config = ConfigDict(extra="forbid")

    class_path: str = Field(min_length=1)
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
        ``kwargs``.

    Returns
    -------
    dict
        A dict matching :class:`ImageAnalyzerSpec`'s field layout.

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

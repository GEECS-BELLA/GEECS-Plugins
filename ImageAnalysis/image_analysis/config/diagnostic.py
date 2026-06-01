"""Diagnostic-level Pydantic models for the unified diagnostic config schema.

A diagnostic config is one YAML file per device that bundles two
concerns previously split across ``image_analysis_configs/`` and
``scan_analysis_configs/library/analyzers/``:

* ``image:`` — the ImageAnalysis-owned section. A typed
  :class:`CameraConfig` (``type: camera``) or :class:`Line1DConfig`
  (``type: line``). Pydantic's discriminated union routes the dict
  to the right model based on the ``type`` field. Attribute access
  on ``diag.image`` works directly
  (``diag.image.background.method = "constant"``).
* ``scan:`` — the ScanAnalysis-owned section (priority, mode, save
  flags, gdoc slot, file tail, renderer kwargs). Kept weakly typed
  here (``Optional[Dict[str, Any]]``) so this model can live in
  ImageAnalysis without importing scan-side runtime types.
  ScanAnalysis validates the dict against its own ``ScanRuntimeConfig``
  at scan-analyzer build time.

A diagnostic also declares its ImageAnalyzer via ``image_analyzer``,
either as a bare class-path string or as a verbose ``{class_path,
kwargs}`` dict — both forms validate into :class:`ImageAnalyzerSpec`.

The 2D-vs-1D dimension lives in one place: the ``type`` field on the
image section. The ScanAnalyzer wrapper (Array2D vs Array1D) is
picked the same way at scan-build time, from the type of
``diag.image``.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, Mapping, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .array1d_processing import Line1DConfig
from .array2d_processing import CameraConfig


# ---------------------------------------------------------------------------
# image_analyzer field model + surface-form normalisation
# ---------------------------------------------------------------------------


class ImageAnalyzerSpec(BaseModel):
    """Canonical runtime form of a diagnostic's ``image_analyzer`` field.

    Whether the YAML used the bare-string form or the verbose dict,
    the field validator on
    :class:`DiagnosticAnalysisConfig.image_analyzer` normalises to
    this shape before anything else sees it.

    Two surface forms are accepted on the ``image_analyzer`` field:

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


def resolve_image_analyzer_value(value: Any) -> Dict[str, Any]:
    """Normalise an ``image_analyzer`` YAML value to an :class:`ImageAnalyzerSpec` dict.

    Accepts the two surface forms documented on :class:`ImageAnalyzerSpec`
    and returns a dict ready to be validated. Used by the field
    validator on :class:`DiagnosticAnalysisConfig.image_analyzer`.

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


# ---------------------------------------------------------------------------
# image: section — discriminated union of camera / line configs
# ---------------------------------------------------------------------------


# Pydantic uses the ``type`` field on each variant
# (``Literal["camera"]`` on CameraConfig, ``Literal["line"]`` on
# Line1DConfig) to dispatch automatically. Unified diagnostic YAMLs
# must declare ``type: camera`` or ``type: line`` explicitly inside
# ``image:`` — no implicit default. A typo'd line YAML missing
# ``data_loading`` will fail union validation rather than silently
# routing to the lenient CameraConfig (which has ``extra="allow"``).
ImageSection = Annotated[
    Union[CameraConfig, Line1DConfig],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Top-level diagnostic model
# ---------------------------------------------------------------------------


class DiagnosticAnalysisConfig(BaseModel):
    """One device's diagnostic config: image and scan sections in one file.

    The on-disk form lives at ``analyzers/<namespace>/<stem>.yaml``;
    the filename stem becomes the analyzer ID used by the task queue
    and referenced from group configs.

    Attributes
    ----------
    name : str
        Input identifier — the device/channel name used for **input
        data discovery** (which folder under ``scans/ScanNNN/`` holds
        this device's raw data). Also the default for ``output_name``
        when no explicit override is supplied. Required.
    image_analyzer : ImageAnalyzerSpec
        Identifies the ImageAnalyzer class. Accepts a bare class-path
        string or a verbose ``{class_path, kwargs}`` dict.
    image : CameraConfig or Line1DConfig, optional
        Typed embedded image config. The ``type`` field on the dict
        (``"camera"`` or ``"line"``) routes Pydantic to the right
        variant. Omitted ``image:`` (``None``) means the analyzer takes
        no embedded image config — used by HASO-style analyzers.
    output_name : str, optional
        Output identifier — the **stem string used for all outputs**
        produced by this analyzer. Drives:

        * the scalar-key prefix applied to every metric written to the
          s-file (e.g. ``UC_TopView_x_fwhm``);
        * the per-analyzer output directory name under
          ``analysis/Scan<NNN>/<output_name>/``;
        * per-file output basenames (e.g. MagSpec's ``-interp/`` and
          ``-interpSpec/`` subdirectories).

        Defaults to ``name`` via :attr:`effective_output_name` — most
        users never set this explicitly. Override only when you want
        to **separate output naming from the input device identifier**
        — e.g. running two ``BeamAnalyzer`` variants over the same
        camera with different ROIs (``output_name: UC_TopView_left``
        / ``UC_TopView_right`` — same input data folder, distinct
        output dirs and s-file columns).
    metric_suffix : str, optional
        Optional suffix appended to every scalar key after the
        analyzer-emitted name. **Scalar-key-only** — does not affect
        directory or file names (unlike ``output_name``). Use for
        post-processed column disambiguation. Empty / ``None`` means
        no suffix.
    scan : dict, optional
        Raw embedded scan-runtime config. Weakly typed at this layer.
        ScanAnalysis validates this against
        ``scan_analysis.config.diagnostic_models.ScanRuntimeConfig``
        when building a scan-analyzer wrapper.

    Notes
    -----
    The output-naming concern lives at this layer rather than inside
    ImageAnalysis (per issue #412): the analyzer emits **bare** scalar
    keys (``"x_fwhm"``, not ``"UC_TopView_x_fwhm"``), and ScanAnalysis
    applies the ``output_name`` prefix and ``metric_suffix`` when
    storing per-shot results. That keeps ImageAnalysis reusable
    standalone (notebook code doesn't need a fake device name to make
    prefixes work) and eliminates the ``StandardAnalyzer.name_suffix``
    mutation pattern.

    The ``name`` vs ``output_name`` split deliberately separates two
    concerns that were previously conflated: input identification
    (where to load from) and output identification (what to label
    things). They default to the same string because the common case
    is "I want my outputs labelled after my device" — but the override
    is the whole reason both fields exist.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    image_analyzer: ImageAnalyzerSpec
    image: Optional[ImageSection] = None
    output_name: Optional[str] = None
    metric_suffix: Optional[str] = None
    scan: Optional[Dict[str, Any]] = None

    @property
    def effective_output_name(self) -> str:
        """Resolved output identifier: ``output_name`` if set, else ``name``.

        This is the string ScanAnalysis uses to prefix scalar keys in
        the s-file AND to label this analyzer's per-scan output
        directory. The two consumers stay in lock-step because they
        read the same field.
        """
        return self.output_name if self.output_name is not None else self.name

    @field_validator("image_analyzer", mode="before")
    @classmethod
    def _normalise_image_analyzer(cls, value: Any) -> Any:
        """Accept bare class-path string or verbose dict."""
        return resolve_image_analyzer_value(value)

    @model_validator(mode="before")
    @classmethod
    def _inject_name_into_image(cls, data: Any) -> Any:
        """Default ``image.name`` to the top-level ``name`` when absent.

        Per-device defaults work the same way they do in standalone
        camera/line YAMLs — the top-level diagnostic name is the
        device identity, so we inject it into the image dict when the
        user didn't bother to repeat themselves.

        Operates only on dict-form input. Typed image instances passed
        in via Mode-1 construction already have their ``name`` set.
        """
        if not isinstance(data, dict):
            return data
        image = data.get("image")
        if not isinstance(image, dict):
            return data
        top_name = data.get("name")
        if not top_name or "name" in image:
            return data
        data = dict(data)
        data["image"] = {**image, "name": top_name}
        return data

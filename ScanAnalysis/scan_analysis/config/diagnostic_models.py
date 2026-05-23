"""Pydantic models for the unified diagnostic config schema.

A diagnostic config is one YAML file per device that bundles two
concerns previously split across ``image_analysis_configs/`` and
``scan_analysis_configs/library/analyzers/``:

* ``image:`` — the ImageAnalysis-owned section (camera config or 1D
  line config, validated by ImageAnalysis). Kept as a raw dict here;
  the ImageAnalysis loader validates it on the way through.
* ``scan:`` — the ScanAnalysis-owned section (priority, mode, save
  flags, gdoc slot, file tail, renderer kwargs).

A diagnostic also declares its ImageAnalyzer via ``image_analyzer``,
either as an alias from :data:`aliases.ALIAS_REGISTRY` or as a verbose
dict; both forms validate into :class:`ImageAnalyzerSpec`.

Group configs collect diagnostic references — by file stem — into a
runnable unit. References can be plain strings or dicts that override
``priority`` or disable the entry via ``enabled: false``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .aliases import ImageAnalyzerSpec, resolve_image_analyzer_value


class ScanRuntimeConfig(BaseModel):
    """Scan-side runtime config for a diagnostic.

    Fields here are consumed by the ScanAnalyzer wrapper
    (``Array2DScanAnalyzer`` or ``Array1DScanAnalyzer``) and the task
    queue. They have nothing to do with how an individual image is
    analyzed; that's the ``image:`` section's job.

    Attributes
    ----------
    priority : int
        Task-queue execution priority. Lower runs first; the task queue
        sorts ascending. Defaults to 100 (background priority).
    mode : {"per_shot", "per_bin"}
        Whether the analyzer runs once per shot (default) or once per
        bin on the bin-averaged image. ``per_bin`` is for analyzers
        whose result on a bin-averaged image is scientifically distinct
        from per-shot results averaged afterward.
    save : bool
        Whether the analyzer writes its per-shot/per-bin outputs (HDF5,
        PNG, etc.) to the analysis tree. Per-shot s-file scalar updates
        are independent of this flag and always happen.
    gdoc_slot : int, optional
        Index 0–3 selecting a cell in the 2×2 table embedded in each
        scan-log entry. When set, the analyzer's last summary figure is
        inserted into that cell. When ``None`` and the task runner has
        gdoc upload enabled, all display files are uploaded as
        hyperlinks instead.
    device : str, optional
        Name of the data subfolder under the scan directory. Defaults
        to the diagnostic's top-level ``name``. Set explicitly only when
        the data folder name differs from the analyzer's metric prefix
        (e.g. post-processed/stitched outputs that live in a sibling
        folder).
    file_tail : str, optional
        Filename suffix used to match this device's data files
        (``".png"``, ``".tdms"``, ``"_postprocessed.tsv"``). When
        ``None``, the ScanAnalyzer wrapper uses its own default.
    renderer_kwargs : dict
        Extra options forwarded to the dimension-specific renderer
        (colormap mode, waterfall sort key, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    priority: int = Field(default=100, ge=0)
    mode: Literal["per_shot", "per_bin"] = "per_shot"
    save: bool = True
    gdoc_slot: Optional[int] = Field(default=None, ge=0, le=3)
    device: Optional[str] = None
    file_tail: Optional[str] = None
    renderer_kwargs: Dict[str, Any] = Field(default_factory=dict)


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
        Identifies the ImageAnalyzer class. Accepts an alias string
        (``"beam"``), an ``{alias, kwargs}`` dict for analyzers needing
        per-instance kwargs (``"line_stitcher"``), or a verbose
        ``{class, image_kind, scan_type, kwargs}`` dict for analyzers
        without a registered alias.
    image : dict, optional
        Raw embedded image config. Validated by ImageAnalysis's
        ``load_camera_config`` / ``load_line_config`` after ``name`` is
        injected. Must be ``None`` (or omitted) when the analyzer's
        ``image_kind`` is ``none``.
    scan : ScanRuntimeConfig
        ScanAnalysis-side runtime config. Defaults are reasonable for
        most analyzers; the YAML can omit the section entirely.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    image_analyzer: ImageAnalyzerSpec
    image: Optional[Dict[str, Any]] = None
    scan: ScanRuntimeConfig = Field(default_factory=ScanRuntimeConfig)

    @field_validator("image_analyzer", mode="before")
    @classmethod
    def _normalise_image_analyzer(cls, value: Any) -> Any:
        """Accept alias string, alias-with-kwargs dict, or verbose dict."""
        return resolve_image_analyzer_value(value)


class AnalyzerRef(BaseModel):
    """A diagnostic reference inside a group, with optional overrides.

    Group configs can list diagnostics either as plain strings (the
    common case) or as dicts that apply per-group overrides. This model
    represents the dict form. The plain-string form is normalised to
    ``AnalyzerRef(ref=<stem>)`` by the group config validator.

    Attributes
    ----------
    ref : str
        Diagnostic ID (filename stem). The group loader resolves this
        against the diagnostics discovered under
        ``analyzers/<namespace>/``.
    enabled : bool
        Set to ``False`` to keep the reference in the group but skip
        the analyzer at run time. Replaces commented-out entries in
        the previous ``groups.yaml`` format.
    priority : int, optional
        Per-group priority override. When ``None``, the diagnostic's
        own ``scan.priority`` applies.
    """

    model_config = ConfigDict(extra="forbid")

    ref: str = Field(min_length=1)
    enabled: bool = True
    priority: Optional[int] = Field(default=None, ge=0)


class AnalysisGroupConfig(BaseModel):
    """A runnable collection of diagnostic references.

    The on-disk form lives at ``groups/<namespace>/<group>.yaml`` and
    is the entry point that LiveWatch (and the offline task queue)
    consume. There is no separate "experiment" wrapper; a group is the
    unit the runner schedules.

    Attributes
    ----------
    name : str
        Display name. Conventionally ``<namespace>_<group>``
        (``HTU_laser_diagnostics``) but free-form.
    description : str, optional
        Free-text description shown in UIs.
    upload_to_scanlog : bool
        Whether the runner should upload this group's outputs to the
        experiment scan-log when the task succeeds.
    analyzers : list of (str or AnalyzerRef)
        Diagnostic references to include. Plain strings are shorthand
        for ``AnalyzerRef(ref=<string>)``. Dict entries can disable a
        reference or override its priority for this group.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: Optional[str] = None
    upload_to_scanlog: bool = True
    analyzers: List[Union[str, AnalyzerRef]] = Field(default_factory=list)

    @field_validator("analyzers", mode="before")
    @classmethod
    def _normalise_analyzer_entries(cls, value: Any) -> Any:
        """Expand plain-string entries to ``{ref: <string>}`` dicts."""
        if not isinstance(value, list):
            return value
        out: List[Any] = []
        for entry in value:
            if isinstance(entry, str):
                out.append({"ref": entry})
            else:
                out.append(entry)
        return out

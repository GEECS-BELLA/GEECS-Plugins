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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .aliases import ImageAnalyzerSpec, resolve_image_analyzer_value


class FromCurrentScanSpec(BaseModel):
    """How to compute a background from the current scan's own images.

    Used inside :class:`BackgroundSource.from_current_scan`. The
    aggregation method is applied across the image stack to produce a
    single background image, which is then cached as ``.npy`` and
    subtracted from each shot via the standard ``FROM_FILE``
    background path.

    Attributes
    ----------
    method : {"median", "percentile"}
        How to collapse the stack. ``median`` is the safe default for
        scans with real shot-to-shot variation (laser jitter, deliberate
        parameter sweep). ``percentile`` is for cases where the
        background sits below the signal at a known fraction of pixels.
    percentile : float, optional
        Required when ``method=percentile``. Value in [0, 100]. Common
        choice: 5 — "use the 5th-percentile value at each pixel."
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["median", "percentile"] = "median"
    percentile: Optional[float] = Field(default=None, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _percentile_required_when_method_is_percentile(self) -> "FromCurrentScanSpec":
        """Pair ``method=percentile`` with an explicit percentile value."""
        if self.method == "percentile" and self.percentile is None:
            raise ValueError(
                "from_current_scan.percentile is required when method='percentile'"
            )
        if self.method == "median" and self.percentile is not None:
            raise ValueError(
                "from_current_scan.percentile must not be set when method='median'"
            )
        return self


class BackgroundSource(BaseModel):
    """ScanAnalysis-side directive for resolving a background source.

    Set on ``scan.background_source`` when the diagnostic needs a
    background that depends on scan context — a different scan's data,
    or an aggregate computed from the current scan's own shots.
    Exactly one source must be specified.

    The resolver runs once per scan-analyzer execution. It produces a
    cached ``.npy`` and mutates the embedded
    ``image.background.file_path`` to point at the cache, then sets
    ``image.background.method`` to ``from_file``. The downstream
    per-shot pipeline sees a static ``from_file`` background and is
    unaware of the source-selection logic.

    Use the existing ``image.background.file_path`` field directly for
    explicit static paths; this directive is only for sources that
    require scan context.

    Attributes
    ----------
    scan_number : int, optional
        Use this previously-acquired scan's images as the background.
        The scan analyzer loads the device's images from that scan,
        averages them, and caches the result in that scan's analysis
        folder. Any scan that references the same ``scan_number``
        reuses the cache. Replaces the legacy
        ``BackgroundConfig.background_scan_number`` field.
    from_current_scan : FromCurrentScanSpec, optional
        Compute a background from the current scan's own images. First
        consumer triggers the compute; subsequent consumers in the
        same scan hit the cache. **Risky when shot-to-shot variation
        is small** (the bg can swallow the signal); not appropriate
        inside an optimizer evaluation loop, where good states have
        nearly identical shots.
    """

    model_config = ConfigDict(extra="forbid")

    scan_number: Optional[int] = Field(default=None, ge=0)
    from_current_scan: Optional[FromCurrentScanSpec] = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "BackgroundSource":
        """Require exactly one of the source variants to be set."""
        sources = [
            ("scan_number", self.scan_number is not None),
            ("from_current_scan", self.from_current_scan is not None),
        ]
        set_sources = [name for name, is_set in sources if is_set]
        if len(set_sources) == 0:
            raise ValueError(
                "background_source must specify exactly one source: "
                "'scan_number' or 'from_current_scan'"
            )
        if len(set_sources) > 1:
            raise ValueError(
                f"background_source must specify exactly one source; got {set_sources}"
            )
        return self


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
    background_source : BackgroundSource, optional
        Scan-context-dependent background source. When set, the scan
        analyzer resolves the source to a cached ``.npy`` path and
        writes it into ``image.background.file_path`` before per-shot
        analysis. Used for cross-scan dark backgrounds
        (``scan_number``) and for dynamic backgrounds computed from
        the current scan's own images (``from_current_scan``).
        Explicit static paths go directly on
        ``image.background.file_path`` and do not need this directive.
    """

    model_config = ConfigDict(extra="forbid")

    priority: int = Field(default=100, ge=0)
    mode: Literal["per_shot", "per_bin"] = "per_shot"
    save: bool = True
    gdoc_slot: Optional[int] = Field(default=None, ge=0, le=3)
    device: Optional[str] = None
    file_tail: Optional[str] = None
    renderer_kwargs: Dict[str, Any] = Field(default_factory=dict)
    background_source: Optional[BackgroundSource] = None


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


class ResolvedDiagnosticConfig(BaseModel):
    """A diagnostic loaded from disk and resolved against a group reference.

    Produced by the analysis-group loader: pairs the on-disk
    :class:`DiagnosticAnalysisConfig` with its filename-derived ID and
    any per-group overrides (``enabled``, effective ``priority``). This
    is what the factory consumes to build a runnable scan analyzer.

    Attributes
    ----------
    id : str
        Filename stem of the diagnostic YAML (without ``.yaml``). Used
        by the task queue for status tracking. Unique within a resolved
        group.
    enabled : bool
        Whether to execute this diagnostic for the group. Group entries
        can set ``enabled: false`` to keep a reference present but skip
        it at run time.
    priority : int
        Effective execution priority — the group's override if one was
        supplied, otherwise the diagnostic's own ``scan.priority``.
        The loader sorts the resolved list ascending by this value.
    diagnostic : DiagnosticAnalysisConfig
        The validated on-disk diagnostic config.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    enabled: bool = True
    priority: int = Field(ge=0)
    diagnostic: DiagnosticAnalysisConfig

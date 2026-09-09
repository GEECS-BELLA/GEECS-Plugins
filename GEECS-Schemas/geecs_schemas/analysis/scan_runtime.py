"""The ``scan:`` section of a diagnostic — how the analyzer runs over a whole scan.

Nothing here changes how one frame is measured; that is the ``image:`` and
``analyzer:`` sections' job.  These settings tell the scan analyzer where
the device's files are, whether to run per shot or per bin, what to save,
which scan-log cell gets the summary figure, and where a scan-dependent
background comes from.

Formerly ScanAnalysis' own ``ScanRuntimeConfig`` (moved here in GEECS-Schemas
0.19.0) so the whole document types in one place.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel
from geecs_schemas.analysis.renderer import RendererOptions


class FromCurrentScanSpec(SchemaModel):
    """Build the background from this scan's own shots, collapsed pixel by pixel.

    'median' is the safe choice when shots genuinely vary (jitter, a
    parameter sweep).  'percentile' suits cases where the background sits
    below the signal at a known fraction of pixels.  Risky when shots are
    nearly identical — the background can swallow the signal — so avoid
    it inside optimizer loops.
    """

    method: Literal["median", "percentile"] = Field(
        "median",
        description="How the shot stack is collapsed: per-pixel median or percentile.",
    )
    percentile: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Percentile (0-100) for method 'percentile'; must be unset for 'median'.",
    )

    @model_validator(mode="after")
    def _percentile_matches_method(self) -> "FromCurrentScanSpec":
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


class AutodetectBackgroundSpec(SchemaModel):
    """Use the averaged-background file another analyzer already wrote for this scan.

    Looks in the day's ``analysis/`` folder for exactly one
    ``ScanNNN<device>_averaged.<ext>`` matching this scan and device.
    """


class BackgroundSource(SchemaModel):
    """Where a scan-dependent background comes from — exactly one of the three.

    For a fixed background file use ``image.background`` directly; this
    directive is for backgrounds that need scan context.  The scan
    analyzer resolves it to a cached ``.npy`` and points the image
    section's background at that file before shots are processed.
    """

    scan_number: Optional[int] = Field(
        None,
        ge=0,
        description="Average this earlier scan's frames of the same device and use that.",
    )
    from_current_scan: Optional[FromCurrentScanSpec] = Field(
        None, description="Collapse this scan's own shots into a background."
    )
    autodetect: Optional[AutodetectBackgroundSpec] = Field(
        None,
        description="Find a precomputed averaged background in the day's analysis folder.",
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "BackgroundSource":
        """Require exactly one of the source variants to be set."""
        chosen = [
            name
            for name, value in (
                ("scan_number", self.scan_number),
                ("from_current_scan", self.from_current_scan),
                ("autodetect", self.autodetect),
            )
            if value is not None
        ]
        if len(chosen) != 1:
            raise ValueError(
                "background_source must specify exactly one source: "
                f"'scan_number', 'from_current_scan', or 'autodetect' (got {chosen})"
            )
        return self


class ScanRuntime(SchemaModel):
    """How the analyzer runs over a scan: order, granularity, what is saved, where files are.

    Every field has a default, so an omitted ``scan:`` section means
    "priority 100, per shot, save outputs, no scan-log cell, files under
    the device's own folder".
    """

    priority: int = Field(
        100,
        ge=0,
        description="Run order within a group: lower runs first. 100 is the background default.",
    )
    mode: Literal["per_shot", "per_bin"] = Field(
        "per_shot",
        description=(
            "'per_shot' analyzes every frame; 'per_bin' averages each bin's "
            "frames first and analyzes once per bin — for metrics that are "
            "not linear in the image."
        ),
    )
    save: bool = Field(
        True,
        description=(
            "Write per-shot / per-bin outputs (HDF5, PNG) into the analysis "
            "tree. S-file scalar columns are written regardless."
        ),
    )
    gdoc_slot: Optional[int] = Field(
        None,
        ge=0,
        le=3,
        description=(
            "Which cell (0-3) of the scan-log entry's 2x2 figure table gets "
            "this analyzer's summary; unset uploads figures as links instead."
        ),
    )
    device: Optional[str] = Field(
        None,
        description=(
            "Data subfolder under the scan when it differs from the diagnostic "
            "name (stitched or post-processed outputs in a sibling folder)."
        ),
    )
    file_tail: Optional[str] = Field(
        None,
        description=(
            "Filename suffix that identifies this device's files ('.png', "
            "'.tdms', '_postprocessed.tsv'); unset uses the analyzer's default."
        ),
    )
    data_format: Optional[Literal["per_shot_files", "device_hdf5"]] = Field(
        None,
        description=(
            "'device_hdf5' reads the capture daemon's per-device frame stack "
            "(falls back to per-shot files when absent). Only for analyzers "
            "that do not derive output names from the shot file path."
        ),
    )
    renderer: RendererOptions = Field(
        default_factory=RendererOptions,
        description="Summary-figure cosmetics; unset fields keep the renderer defaults.",
    )
    background_source: Optional[BackgroundSource] = Field(
        None,
        description=(
            "A scan-dependent background (another scan, this scan's own shots, "
            "or an autodetected averaged file). Fixed files go on image.background."
        ),
    )


__all__ = [
    "AutodetectBackgroundSpec",
    "BackgroundSource",
    "FromCurrentScanSpec",
    "ScanRuntime",
]

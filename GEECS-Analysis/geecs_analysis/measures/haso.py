"""HASO wavefront reconstruction through a host-supplied WaveKit engine.

The reconstruction is Imagine Optic's WaveKit SDK, a licensed 64-bit
Windows library the core may not start (no processes, paths or config
here). The scan host binds an engine under the service name ``haso`` — in
practice ImageAnalysis's ``HasoWaveKit``, which runs the SDK in a Windows
Python, natively or under Wine — and this measure hands it the frame's
pixels with the spec's parameters, then packages what comes back: the
processed (masked, filtered) zonal phase as the frame, the raw phase, the
intensity, the processed slopes and the pupil as extras, and two scalars
taken inside the pupil.

With a ``reference`` the measure reports a *difference*: the reference
frame's slopes (a probe-only scan's mean image, say) are subtracted from
each shot's slopes by the SDK before the mask and the filters, so the
processed phase is what changed between the two — the plasma's imprint on
the probe. The reference is processed by the recipe's steps exactly as
each shot is. The subtraction is linear: the difference of two processed
phases agrees with it to the reconstruction's own precision (2e-3 um on
26_0929 Scan015 against Scan014).

The frame the measure receives is the sensor's raw pixels, after whatever
steps the recipe lists (a background frame, say); it is rounded and
clipped to the sensor's ``uint16`` before the SDK sees it, because the SDK
reads a real ``.himg`` and nothing else (the host rebuilds one from the
pixels and any header of the same sensor). A background subtracted here in
numpy gives the same slopes as the SDK's own image subtraction (verified
2026-09-28, bit for bit).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

from pydantic import Field, field_validator, model_validator

from geecs_analysis.registry import MeasureSpec, SpecModel, measure

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement

#: The binding key of the host's WaveKit engine.
SERVICE = "haso"

#: The per-scan store a scan host writes every shot's wavefront products to.
SHOT_STORE = "wavefront"

#: The scalar keys: phase statistics inside the pupil.
SCALARS = ("phase_rms", "phase_pv")

#: The extras, in the order the store writes them.
EXTRAS = ("raw_phase", "intensity", "slopes_x", "slopes_y", "pupil")

#: The sensor's pixel range: ``.himg`` pixels are 16-bit words.
PIXEL_MAX = 65535


class HasoMask(SpecModel):
    """A rectangular pupil on the slopes grid, as numpy slice bounds.

    Rows ``top:bottom`` and columns ``left:right`` of the sub-aperture grid
    (not camera pixels) stay in the pupil; everything else is masked before
    the filters and the processed phase. The bounds are clipped to the
    grid, so a mask larger than the sensor keeps everything.
    """

    top: int = Field(ge=0, description="First sub-aperture row kept (inclusive).")
    bottom: int = Field(gt=0, description="Row the pupil stops at (exclusive).")
    left: int = Field(ge=0, description="First sub-aperture column kept (inclusive).")
    right: int = Field(gt=0, description="Column the pupil stops at (exclusive).")

    @model_validator(mode="after")
    def _nonempty(self) -> "HasoMask":
        """A pupil with at least one sub-aperture in each direction."""
        if self.bottom <= self.top or self.right <= self.left:
            raise ValueError("HASO mask bounds must satisfy top < bottom, left < right")
        return self


class HasoFilters(SpecModel):
    """Which aberrations the SDK removes from the slopes before the processed phase.

    The defaults are the legacy analyzer's: tilt, curvature and both
    astigmatisms off, every other aberration kept.
    """

    tilt_x: bool = Field(True, description="Remove x tilt.")
    tilt_y: bool = Field(True, description="Remove y tilt.")
    curvature: bool = Field(True, description="Remove curvature (defocus).")
    astigmatism_0: bool = Field(True, description="Remove 0° astigmatism.")
    astigmatism_45: bool = Field(True, description="Remove 45° astigmatism.")
    others: bool = Field(False, description="Remove every other aberration.")

    def flags(self) -> tuple[bool, bool, bool, bool, bool, bool]:
        """The six flags in the SDK's ``apply_filter`` order."""
        return (
            self.tilt_x,
            self.tilt_y,
            self.curvature,
            self.astigmatism_0,
            self.astigmatism_45,
            self.others,
        )


class HasoSpec(MeasureSpec):
    """WaveKit parameters, passed unchanged to the host's engine.

    ``sensor_config`` is a file name, never a path: the host resolves it
    under its own sensor-configuration directory (``[Paths]
    wavekit_configs_path``), so a recipe names the sensor and the facility
    says where the files are. The engine refuses an image whose embedded
    serial does not match the file.

    ``reference`` names a frame input of the recipe (``inputs: {probe:
    {from_scan: {scan: 14, statistic: mean}}}`` with ``reference: probe``):
    its slopes are subtracted from every shot's before the mask and the
    filters, so the processed phase, the slopes and the scalars describe
    the difference; the raw phase and the intensity stay the shot's own.
    Take the reference the same day: the probe drifts 0.03 um RMS from one
    day to the next, about half a plasma imprint.
    """

    kind: Literal["haso"] = "haso"
    sensor_config: str = Field(
        ...,
        min_length=1,
        description=(
            "The sensor's WaveKit configuration file name (its .dat, e.g. "
            "WFS_HASO4_LIFT_680_8244_gain_enabled.dat), found under the "
            "host's wavekit_configs_path."
        ),
    )
    mask: Optional[HasoMask] = Field(
        None,
        description=(
            "Rectangular pupil on the sub-aperture grid; unset keeps the "
            "sensor's own pupil."
        ),
    )
    filters: HasoFilters = Field(
        default_factory=HasoFilters,
        description="Aberrations removed from the slopes before the processed phase.",
    )
    wavelength_nm: float = Field(
        800.0, gt=0, description="Probe wavelength for the LIFT reconstruction, nm."
    )
    start_subpupil: tuple[int, int] = Field(
        (87, 64),
        description=(
            "(x, y) index of the first sub-aperture the slopes computation "
            "starts from, relative to the pupil's top-left corner."
        ),
    )
    reference: Optional[str] = Field(
        None,
        description=(
            "Frame input whose slopes are subtracted from every shot's before "
            "the mask and filters (a same-day probe-only scan's mean image); "
            "unset measures the shot's own wavefront."
        ),
    )
    zonal_prefs: tuple[int, int, float] = Field(
        (100, 500, 1e-6),
        description=(
            "Zonal reconstruction preferences: maximum weak iterations, "
            "maximum iterations, residual variation limit."
        ),
    )

    @field_validator("sensor_config")
    @classmethod
    def _bare_file_name(cls, value: str) -> str:
        """A single path component; the host chooses the directory."""
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("sensor_config must be a file name, not a path")
        return value

    @field_validator("start_subpupil")
    @classmethod
    def _subpupil_in_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Sub-aperture indices are non-negative."""
        if any(v < 0 for v in value):
            raise ValueError("start_subpupil indices must be non-negative")
        return value

    @field_validator("zonal_prefs")
    @classmethod
    def _prefs_positive(cls, value: tuple[int, int, float]) -> tuple[int, int, float]:
        """Iteration counts and the residual limit are positive."""
        if any(v <= 0 for v in value):
            raise ValueError("zonal_prefs values must be positive")
        return value

    def emitted_scalars(self) -> frozenset[str]:
        """Every reconstruction emits the same two pupil statistics."""
        return frozenset(SCALARS)


def sensor_pixels(frame: Frame):
    """The frame's samples as the sensor's ``uint16`` pixels: rounded, clipped.

    A processed frame is float64 (a subtracted background leaves fractions
    and negatives); the SDK reads 16-bit words, so values are rounded to
    the nearest integer and clipped to ``[0, 65535]`` — the SDK's own image
    subtraction clips at zero the same way.
    """
    import numpy as np

    return np.clip(np.rint(frame.data), 0, PIXEL_MAX).astype(np.uint16)


@measure(
    HasoSpec,
    ndim={2},
    service=SERVICE,
    shot_store=SHOT_STORE,
    input_field="reference",
)
def haso(
    frame: Frame, spec: HasoSpec, wavekit: object, reference: Optional[Frame]
) -> Measurement:
    """Reconstruct the wavefront from the frame's pixels with the host's engine.

    ``reference`` is the bound, processed reference frame (``None`` without
    one); its pixels go to the engine as ``reference=`` and the engine
    subtracts their slopes. ``wavekit.compute(pixels, **parameters)``
    returns an object with the
    SDK's outputs (``HasoWaveKitResult``): ``processed_phase``,
    ``raw_phase``, ``intensity``, ``slopes_x``, ``slopes_y`` (float32
    arrays on the sub-aperture grid, NaN outside the pupil) and ``pupil``
    (bool). The scalars are the processed phase's standard deviation
    (``phase_rms``) and peak-to-valley (``phase_pv``) over the finite
    values inside the pupil; NaN when the pupil holds none.
    """
    import numpy as np

    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement

    if frame.data.ndim != 2:
        raise ValueError("HASO reconstruction requires a 2D frame")
    if reference is not None and reference.data.shape != frame.data.shape:
        raise ValueError(
            f"The HASO reference is {reference.data.shape}, the frame "
            f"{frame.data.shape}: both must be the sensor's full image"
        )
    mask = spec.mask
    result = wavekit.compute(
        sensor_pixels(frame),
        reference=None if reference is None else sensor_pixels(reference),
        sensor_config=spec.sensor_config,
        mask=None if mask is None else (mask.top, mask.bottom, mask.left, mask.right),
        filters=spec.filters.flags(),
        wavelength_nm=spec.wavelength_nm,
        start_subpupil=tuple(spec.start_subpupil),
        zonal_prefs=tuple(spec.zonal_prefs),
    )
    processed = np.asarray(result.processed_phase, dtype=np.float64)
    pupil = np.asarray(result.pupil, dtype=bool)
    if pupil.shape != processed.shape:
        raise ValueError("The pupil must have the processed phase's shape")
    inside = processed[pupil & np.isfinite(processed)]
    if inside.size:
        scalars = {
            "phase_rms": float(inside.std()),
            "phase_pv": float(inside.max() - inside.min()),
        }
    else:
        scalars = {"phase_rms": float("nan"), "phase_pv": float("nan")}
    phase = Frame.from_array(processed, shot=frame.shot, unit="um", label="phase")

    def extra(values: object, unit: str, label: str) -> Frame:
        return Frame.from_array(
            np.asarray(values, dtype=np.float64),
            axes=phase.axes,
            shot=frame.shot,
            unit=unit,
            label=label,
        )

    extras = {
        "raw_phase": extra(result.raw_phase, "um", "raw phase"),
        "intensity": extra(result.intensity, "", "intensity"),
        "slopes_x": extra(result.slopes_x, "mrad", "x slopes"),
        "slopes_y": extra(result.slopes_y, "mrad", "y slopes"),
        "pupil": extra(pupil, "", "pupil"),
    }
    return Measurement(scalars=scalars, frame=phase, extras=extras)

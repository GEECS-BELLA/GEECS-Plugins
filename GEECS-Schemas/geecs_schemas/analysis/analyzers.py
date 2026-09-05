"""The ``analyzer:`` section of a diagnostic — which measurement runs, with what settings.

Every analyzer the analysis suite ships has one *spec* model here, chosen
by the ``kind`` field.  The spec carries exactly the analyzer's own
parameters (a FROG retrieval's grid size, a magspec's energy calibration,
HASO's pupil mask); the frame clean-up lives in the ``image:`` section and
the scan-time behaviour in ``scan:``.  Because the set of kinds is closed
and lives here, a diagnostic document validates completely with pydantic
alone — an editor renders a form per kind from the JSON Schema, and a typo
in a parameter name is a load error rather than a silently ignored key.

ImageAnalysis maps each ``kind`` to the class that implements it
(``image_analysis.config.registry``); adding an analyzer means adding a
spec here and one registry line there.  Each spec's ``image_kind`` says
which ``image:`` section it works on — ``"camera"``, ``"line"`` or ``None``
for analyzers that read their own file formats (HASO, phase maps).
"""

from __future__ import annotations

from pathlib import Path
import typing
from typing import Annotated, ClassVar, List, Literal, Optional, Tuple, Union

from pydantic import Field, field_validator, model_validator

from geecs_schemas._base import SchemaModel

ImageKind = Optional[Literal["camera", "line"]]


class AnalyzerSpecBase(SchemaModel):
    """Shared shape of every analyzer spec: the ``kind`` tag plus its image kind."""

    #: Which ``image:`` section this analyzer consumes: "camera", "line", or
    #: None when it loads its own file format and takes no image section.
    image_kind: ClassVar[ImageKind] = "camera"


# ---------------------------------------------------------------------------
# Framework analyzers
# ---------------------------------------------------------------------------


class StandardAnalyzerSpec(AnalyzerSpecBase):
    """Run the camera pipeline and report the processed frame — no extra metrics."""

    image_kind: ClassVar[ImageKind] = "camera"
    kind: Literal["standard"] = Field(
        "standard", description="Processed-frame-only camera analyzer."
    )


class TraceAnalyzerSpec(AnalyzerSpecBase):
    """Run the trace pipeline and report the processed trace, no extra metrics (the 1D peer of ``standard``)."""

    image_kind: ClassVar[ImageKind] = "line"
    kind: Literal["trace"] = Field(
        "trace", description="Processed-trace-only line analyzer."
    )


class LineAnalyzerSpec(AnalyzerSpecBase):
    """Run the trace pipeline and report basic trace statistics (peak, centroid, width, area)."""

    image_kind: ClassVar[ImageKind] = "line"
    kind: Literal["line"] = Field("line", description="Trace statistics analyzer.")


class BeamAnalyzerSpec(AnalyzerSpecBase):
    """Beam profile metrics: centroid, rms size, FWHM, total counts along x, y and the 45° axes."""

    image_kind: ClassVar[ImageKind] = "camera"
    kind: Literal["beam"] = Field("beam", description="Beam profile analyzer.")
    compute_slopes: bool = Field(
        False,
        description=(
            "Also compute beam slope / straightness metrics from line-by-line "
            "fits. Expensive; leave off unless the tilt matters."
        ),
    )
    enabled_stats: Optional[List[str]] = Field(
        None,
        description=(
            "Emit only these statistics (e.g. ['image_total', 'x_CoM', "
            "'y_fwhm']); unset emits all 18. Names are <axis>_<stat>."
        ),
    )


class PolynomialCalibrationSpec(SchemaModel):
    """Pixel-to-energy map as a polynomial in the column index (E = c0 + c1·x + c2·x² + ...)."""

    kind: Literal["polynomial"] = Field(
        "polynomial", description="Polynomial pixel-to-energy calibration."
    )
    coeffs: List[float] = Field(
        ...,
        min_length=1,
        description="Polynomial coefficients, lowest order first, energy in MeV.",
    )


class ArrayCalibrationSpec(SchemaModel):
    """Pixel-to-energy map given explicitly, one energy per image column."""

    kind: Literal["array"] = Field(
        "array", description="Explicit per-column energy axis."
    )
    values: Optional[List[float]] = Field(
        None, description="Inline energy axis, one value per column."
    )
    file: Optional[str] = Field(
        None, description="A saved .npy energy axis (alternative to values)."
    )

    @model_validator(mode="after")
    def _require_source(self) -> "ArrayCalibrationSpec":
        """Require either inline values or a file."""
        if self.values is None and self.file is None:
            raise ValueError("array calibration requires 'values' or 'file'")
        return self


class DnnAxisCalibrationSpec(SchemaModel):
    """The MATLAB-era DNN spectrometer calibration: camera geometry + electron trajectory tables."""

    kind: Literal["dnn_axis"] = Field(
        "dnn_axis", description="DNN camera + trajectory table calibration."
    )
    camera_calibration_file: str = Field(
        ..., description="Tab-delimited camera geometry table (one row per camera)."
    )
    trajectory_calibration_file: str = Field(
        ..., description="Tab-delimited screen-position vs momentum table."
    )
    camera_number: int = Field(
        ..., ge=1, le=4, description="Which camera row of the geometry table to use."
    )
    magnetic_field_t: float = Field(
        1.0,
        description=(
            "Dipole field in tesla the trajectory table is scaled to. A live "
            "teslameter reading in the shot's auxiliary data overrides it."
        ),
    )
    lanex_calibration_file: Optional[str] = Field(
        None,
        description="Lanex counts-to-charge table; when given, spectra are reported in fC.",
    )


CalibrationSpec = Annotated[
    Union[PolynomialCalibrationSpec, ArrayCalibrationSpec, DnnAxisCalibrationSpec],
    Field(discriminator="kind"),
]


class MagSpecAnalyzerSpec(AnalyzerSpecBase):
    """Magnetic spectrometer: beam metrics plus an energy-calibrated, resampled spectrum."""

    image_kind: ClassVar[ImageKind] = "camera"
    kind: Literal["magspec"] = Field(
        "magspec", description="Energy-calibrated magnetic spectrometer analyzer."
    )
    calibration: CalibrationSpec = Field(
        ..., description="How image columns map to energy."
    )
    energy_range: Tuple[float, float] = Field(
        ...,
        description="(min, max) of the uniform energy grid the spectrum is resampled onto, MeV.",
    )
    num_energy_points: int = Field(
        500, ge=2, description="Number of points on the uniform energy grid."
    )

    @field_validator("energy_range")
    @classmethod
    def _check_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """Require max > min."""
        if v[1] <= v[0]:
            raise ValueError("energy_range max must exceed min")
        return v


class FrogRetrievalSpec(AnalyzerSpecBase):
    """Grenouille / FROG pulse retrieval through the vendor DLL (Windows-only at run time)."""

    image_kind: ClassVar[ImageKind] = "camera"
    kind: Literal["frog_retrieval"] = Field(
        "frog_retrieval", description="FROG pulse retrieval via the vendor DLL."
    )
    delt: float = Field(0.85, description="Time-delay step per raw pixel, fs.")
    dellam: float = Field(
        -0.085,
        description="Wavelength step per raw pixel, nm (negative for Grenouille).",
    )
    lam0: float = Field(400.0, description="Centre wavelength of the trace, nm.")
    N: int = Field(512, description="Retrieval grid size: 512, 256, 128 or 64.")
    target_error: float = Field(
        0.005, description="FROG error at which the retrieval stops early."
    )
    max_time_seconds: float = Field(
        5.0, description="Wall-clock cap on one retrieval, seconds."
    )
    max_iterations: int = Field(
        1_000_000_000, description="Iteration cap on one retrieval."
    )
    noise_subtype: int = Field(
        4, description="Vendor NoiseSubtraction SUBTYPE parameter."
    )
    noise_rad: float = Field(1.0, description="Vendor NoiseSubtraction RAD parameter.")

    @field_validator("N")
    @classmethod
    def _grid_size(cls, v: int) -> int:
        """Restrict to the grid sizes the DLL accepts."""
        if v not in (64, 128, 256, 512):
            raise ValueError("N must be one of 64, 128, 256, 512")
        return v


class FrogSpectralPhaseSpec(AnalyzerSpecBase):
    """Fit a polynomial spectral phase to a retrieved FROG spectrum (GDD, TOD, …)."""

    image_kind: ClassVar[ImageKind] = "line"
    kind: Literal["frog_spectral_phase"] = Field(
        "frog_spectral_phase", description="Spectral-phase polynomial fit."
    )
    fit_order: int = Field(3, ge=0, description="Polynomial order of the phase fit.")
    mask_threshold: Optional[float] = Field(
        0.5,
        ge=0,
        description="Fit only where the spectral intensity exceeds this fraction of its peak.",
    )
    min_points: Optional[int] = Field(
        None, ge=1, description="Minimum number of points required for a fit."
    )
    fit_num_points: int = Field(
        300, ge=2, description="Number of points the fitted phase is evaluated on."
    )
    reference_wavelength_nm: float = Field(
        800.0, gt=0, description="Wavelength the phase expansion is taken about, nm."
    )
    sign_reference_order: Optional[int] = Field(
        2,
        ge=0,
        description=(
            "Polynomial order whose sign is forced to sign_reference (2 = the "
            "GDD term, the default); set null to leave the fit's sign as is."
        ),
    )
    sign_reference: float = Field(
        1.0, description="Sign (+1 / -1) imposed on sign_reference_order."
    )
    sign_epsilon: float = Field(
        0.0,
        ge=0,
        description="Dead band around zero within which the sign is not flipped.",
    )


class IctAnalyzerSpec(AnalyzerSpecBase):
    """Integrating current transformer: charge from a scope trace by low-pass filtering and integrating."""

    image_kind: ClassVar[ImageKind] = "line"
    kind: Literal["ict"] = Field("ict", description="ICT charge analyzer.")
    butterworth_order: int = Field(
        1, ge=1, description="Order of the low-pass Butterworth filter."
    )
    butterworth_crit_f: float = Field(
        0.125, gt=0, description="Normalised critical frequency of the low-pass filter."
    )
    calibration_factor: float = Field(
        0.1, description="ICT calibration factor, V·s per C."
    )
    dt: Optional[float] = Field(
        None, description="Sample interval in seconds; unset derives it from the trace."
    )


class LineStitcherSpec(AnalyzerSpecBase):
    """Concatenate this device's trace with its sibling devices' traces into one spectrum."""

    image_kind: ClassVar[ImageKind] = "line"
    kind: Literal["line_stitcher"] = Field(
        "line_stitcher", description="Multi-device trace stitcher."
    )
    sibling_devices: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "The other devices whose traces are appended to this diagnostic's "
            "device. Each must have a folder in the scan."
        ),
    )
    output_label: Optional[str] = Field(
        None,
        description=(
            "Name of the folder (under the scan) and filename label the "
            "stitched traces are written to; defaults to the diagnostic's "
            "output_name. Must differ from the master device's name — the "
            "stitcher refuses to write into the raw data folder."
        ),
    )


class PupilMask(SchemaModel):
    """Rectangular pupil mask on the HASO slopes grid, inclusive bounds; -1 means the far edge."""

    top: int = Field(1, description="Top row of the pupil (inclusive).")
    bottom: int = Field(
        -1, description="Bottom row of the pupil (inclusive); -1 = last row."
    )
    left: int = Field(1, description="Left column of the pupil (inclusive).")
    right: int = Field(
        -1, description="Right column of the pupil (inclusive); -1 = last column."
    )


class HasoAnalyzerSpec(AnalyzerSpecBase):
    """HASO wavefront sensor: slopes to phase and Zernike terms through WaveKit (Windows, licensed)."""

    image_kind: ClassVar[ImageKind] = None
    kind: Literal["haso"] = Field(
        "haso", description="HASO wavefront analyzer via WaveKit."
    )
    wavekit_config_file_path: Path = Field(
        ..., description="The WaveKit sensor configuration (.dat) for this HASO head."
    )
    mask: PupilMask = Field(
        default_factory=PupilMask, description="Pupil mask applied to the slopes."
    )
    background_path: Optional[Path] = Field(
        None, description="A .has slopes file subtracted as background."
    )
    laser_wavelength: float = Field(800.0, gt=0, description="Probe wavelength, nm.")


# ---------------------------------------------------------------------------
# Experiment-specific analyzers (shipped in the suite, used by one facility)
# ---------------------------------------------------------------------------


class DownrampPhaseSpec(AnalyzerSpecBase):
    """HTU downramp phase-map analyzer over the camera pipeline."""

    image_kind: ClassVar[ImageKind] = "camera"
    kind: Literal["downramp_phase"] = Field(
        "downramp_phase", description="HTU downramp phase analyzer."
    )


class HiResMagCamSpec(AnalyzerSpecBase):
    """HTU high-resolution magspec camera: beam metrics plus a bow-tie fit of the trace."""

    image_kind: ClassVar[ImageKind] = "camera"
    kind: Literal["hi_res_mag_cam"] = Field(
        "hi_res_mag_cam", description="HTU HiResMagCam bow-tie analyzer."
    )
    n_beam_size_clearance: int = Field(
        4, ge=0, description="Bow-tie fit: beam-size clearance in pixels."
    )
    min_total_counts: float = Field(
        2500.0, ge=0, description="Bow-tie fit: skip frames with fewer total counts."
    )
    threshold_factor: float = Field(
        10.0, gt=0, description="Bow-tie fit: threshold factor."
    )


class BCaveMagSpecStitcherSpec(AnalyzerSpecBase):
    """HTU BCave magspec camera with a Gaussian-weighted vertical lineout for optimization."""

    image_kind: ClassVar[ImageKind] = "camera"
    kind: Literal["bcave_magspec_stitcher"] = Field(
        "bcave_magspec_stitcher", description="HTU BCave magspec camera analyzer."
    )
    gaussian_sigma: float = Field(
        20.0, gt=0, description="Width of the Gaussian weighting, pixels."
    )
    gaussian_center: float = Field(
        250.0, description="Centre of the Gaussian weighting, pixels."
    )


class BCaveMagOptSpec(AnalyzerSpecBase):
    """HTU BCave stitched-spectrum optimizer metrics over the trace pipeline."""

    image_kind: ClassVar[ImageKind] = "line"
    kind: Literal["bcave_mag_opt"] = Field(
        "bcave_mag_opt", description="HTU BCave stitched-spectrum optimizer analyzer."
    )


class PhaseDownrampSpec(AnalyzerSpecBase):
    """HTU phase-map processor: density from a probe phase map (reads its own TSV/phase files)."""

    image_kind: ClassVar[ImageKind] = None
    kind: Literal["phase_downramp"] = Field(
        "phase_downramp", description="HTU phase-downramp processor."
    )
    pixel_scale: float = Field(
        ..., gt=0, description="Spatial calibration, µm per pixel (vertical)."
    )
    wavelength_nm: float = Field(..., gt=0, description="Probe wavelength, nm.")
    threshold_fraction: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Zero phase values below this fraction of the maximum.",
    )
    roi: Optional[Tuple[int, int, int, int]] = Field(
        None,
        description="Crop as (x_min, x_max, y_min, y_max); negatives count from the end.",
    )
    background_path: Optional[Path] = Field(
        None, description="A background phase map to subtract."
    )


AnalyzerSpec = Annotated[
    Union[
        StandardAnalyzerSpec,
        TraceAnalyzerSpec,
        LineAnalyzerSpec,
        BeamAnalyzerSpec,
        MagSpecAnalyzerSpec,
        FrogRetrievalSpec,
        FrogSpectralPhaseSpec,
        IctAnalyzerSpec,
        LineStitcherSpec,
        HasoAnalyzerSpec,
        DownrampPhaseSpec,
        HiResMagCamSpec,
        BCaveMagSpecStitcherSpec,
        BCaveMagOptSpec,
        PhaseDownrampSpec,
    ],
    Field(discriminator="kind"),
]

#: kind → spec model, for editors, registries and tests. Built from the
#: union so it cannot drift from it.
ANALYZER_SPECS: dict[str, type[AnalyzerSpecBase]] = {
    model.model_fields["kind"].default: model
    for model in typing.get_args(typing.get_args(AnalyzerSpec)[0])
}


__all__ = [
    "ANALYZER_SPECS",
    "AnalyzerSpec",
    "AnalyzerSpecBase",
    "ArrayCalibrationSpec",
    "BCaveMagOptSpec",
    "BCaveMagSpecStitcherSpec",
    "BeamAnalyzerSpec",
    "CalibrationSpec",
    "DnnAxisCalibrationSpec",
    "DownrampPhaseSpec",
    "FrogRetrievalSpec",
    "FrogSpectralPhaseSpec",
    "HasoAnalyzerSpec",
    "HiResMagCamSpec",
    "IctAnalyzerSpec",
    "ImageKind",
    "LineAnalyzerSpec",
    "LineStitcherSpec",
    "MagSpecAnalyzerSpec",
    "PhaseDownrampSpec",
    "PolynomialCalibrationSpec",
    "PupilMask",
    "StandardAnalyzerSpec",
    "TraceAnalyzerSpec",
]

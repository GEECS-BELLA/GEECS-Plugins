"""
Magnetic spectrometer analyzer with energy calibration.

Provides analysis for magnetic spectrometer images with pixel-to-energy
conversion using device-specific calibrations.
"""

import csv
import numpy as np
import logging
from typing import Optional, Tuple, Dict, List, Callable, Annotated, Union, Literal, Any
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from image_analysis.tools.rendering import base_render_image
from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer
from image_analysis.types import ImageAnalyzerResult
from image_analysis.algorithms.axis_interpolation import interpolate_image_axis

logger = logging.getLogger(__name__)


class EnergyCalibration(ABC):
    """Interface for mapping image column index to energy."""

    @abstractmethod
    def build_axis(self, image_width: int) -> np.ndarray:
        """Return energy axis values corresponding to image columns."""
        raise NotImplementedError


@dataclass
class PolynomialCalibration(EnergyCalibration):
    """Polynomial energy calibration: E(pixel) = c0 + c1*x + c2*x^2 + ..."""

    coeffs: List[float]

    def __post_init__(self):
        """Validate polynomial calibration coefficients."""
        if not self.coeffs:
            raise ValueError("PolynomialCalibration requires non-empty coeffs")

    def build_axis(self, image_width: int) -> np.ndarray:
        """Build a pixel-to-energy axis from polynomial coefficients."""
        pixel_axis = np.arange(image_width)
        pixel_to_energy = np.zeros_like(pixel_axis, dtype=float)
        for i, coeff in enumerate(self.coeffs):
            pixel_to_energy += coeff * np.power(pixel_axis, i)
        return pixel_to_energy


@dataclass
class ArrayCalibration(EnergyCalibration):
    """Calibration from a precomputed axis array (inline or .npy file)."""

    values: Optional[np.ndarray] = None
    file: Optional[str] = None

    def __post_init__(self):
        """Validate that inline values or a file path is provided."""
        if self.values is None and self.file is None:
            raise ValueError("ArrayCalibration requires 'values' or 'file'")

    def build_axis(self, image_width: int) -> np.ndarray:
        """Build a pixel-to-energy axis from an array or `.npy` file."""
        if self.file is not None:
            cal_path = Path(self.file)
            if not cal_path.is_absolute():
                if not cal_path.exists():
                    cal_path = Path(__file__).parent.parent / cal_path
            pixel_to_energy = np.load(cal_path)
        else:
            pixel_to_energy = np.asarray(self.values, dtype=float)

        if len(pixel_to_energy) != image_width:
            raise ValueError(
                f"Calibration array length ({len(pixel_to_energy)}) "
                f"doesn't match image width ({image_width})"
            )
        return pixel_to_energy


@dataclass
class CallableCalibration(EnergyCalibration):
    """Calibration built by evaluating a custom function on pixel index."""

    function: Callable[[np.ndarray], np.ndarray]

    def __post_init__(self):
        """Validate callable calibration function."""
        if self.function is None:
            raise ValueError("CallableCalibration requires 'function'")

    def build_axis(self, image_width: int) -> np.ndarray:
        """Build a pixel-to-energy axis by evaluating a callable."""
        pixel_axis = np.arange(image_width)
        pixel_to_energy = np.asarray(self.function(pixel_axis), dtype=float)
        if len(pixel_to_energy) != image_width:
            raise ValueError(
                f"Callable calibration returned length ({len(pixel_to_energy)}) "
                f"but image width is ({image_width})"
            )
        return pixel_to_energy


@dataclass
class DnnAxisCalibration(EnergyCalibration):
    """
    MATLAB-style DNN magspec horizontal-axis calibration inputs.

    This mirrors the `fDnnAxisClbEach.m` x-axis path:
    pixel -> screen position [mm] -> momentum [MeV/c/T] via spline ->
    energy-like axis [MeV] via magnetic field scaling.
    """

    fov_mm: float
    roi_width_px: int
    left_edge_mm: float
    x_start_px: int
    x_end_px: int
    traj_screen_mm: np.ndarray
    traj_momentum_mev_c_per_t: np.ndarray
    magnetic_field_t: float = 1.0

    def __post_init__(self):
        """Validate MATLAB-style DNN calibration inputs."""
        if self.roi_width_px <= 0:
            raise ValueError("roi_width_px must be > 0")
        if self.x_start_px < 1 or self.x_end_px < self.x_start_px:
            raise ValueError(
                "x_start_px/x_end_px must be 1-based inclusive indices with "
                "x_end_px >= x_start_px >= 1"
            )
        if self.x_end_px > self.roi_width_px:
            raise ValueError(
                f"x_end_px ({self.x_end_px}) cannot exceed roi_width_px "
                f"({self.roi_width_px})"
            )

        self.traj_screen_mm = np.asarray(self.traj_screen_mm, dtype=float)
        self.traj_momentum_mev_c_per_t = np.asarray(
            self.traj_momentum_mev_c_per_t, dtype=float
        )

        if self.traj_screen_mm.shape != self.traj_momentum_mev_c_per_t.shape:
            raise ValueError(
                "traj_screen_mm and traj_momentum_mev_c_per_t must have the same shape"
            )
        if self.traj_screen_mm.ndim != 1 or self.traj_screen_mm.size < 2:
            raise ValueError("trajectory arrays must be 1D with at least 2 points")

    @classmethod
    def from_dnn_calibration_files(
        cls,
        camera_calibration_file: str,
        trajectory_calibration_file: str,
        camera_number: int,
        magnetic_field_t: float = 1.0,
    ) -> "DnnAxisCalibration":
        """
        Load DNN calibration constants from MATLAB-era tab-delimited text files.

        Parameters
        ----------
        camera_calibration_file : str
            Path to camera calibration table (e.g., `260224DnnVarianCam.txt`)
        trajectory_calibration_file : str
            Path to trajectory calibration table (e.g., `170925DnnVarianTrj.txt`)
        camera_number : int
            Camera id from `cam number` column. For the standard 4-camera
            magspec: 1-2 use front trajectory, 3-4 use side trajectory.
        magnetic_field_t : float, default=1.0
            Magnetic field scaling used in MATLAB (`fld * xA.mmt`).
        """
        if camera_number not in (1, 2, 3, 4):
            raise ValueError(
                "camera_number must be one of (1, 2, 3, 4) for DNN magspec cameras"
            )

        cam_rows = cls._read_tab_delimited_rows(camera_calibration_file)
        trj_rows = cls._read_tab_delimited_rows(trajectory_calibration_file)

        cam_row = None
        for row in cam_rows:
            if int(float(row["cam number"])) == camera_number:
                cam_row = row
                break
        if cam_row is None:
            raise ValueError(
                f"Camera number {camera_number} not found in '{camera_calibration_file}'"
            )

        split_idx = None
        for i, row in enumerate(trj_rows):
            if float(row["side logic"]) == 1.0:
                split_idx = i
                break
        if split_idx is None:
            raise ValueError(
                "No 'side logic == 1' split found in trajectory calibration file"
            )

        if camera_number in (1, 2):
            trj_subset = trj_rows[:split_idx]
            screen_col = "front screen [m]"
        else:
            trj_subset = trj_rows[split_idx:]
            screen_col = "side screen [m]"

        traj_screen_mm = 1000.0 * np.asarray(
            [float(r[screen_col]) for r in trj_subset], dtype=float
        )
        traj_momentum_mev_c_per_t = np.asarray(
            [float(r["momentum [MeV/c]"]) for r in trj_subset], dtype=float
        )

        return cls(
            fov_mm=float(cam_row["FOV [mm]"]),
            roi_width_px=int(float(cam_row["ROI width"])),
            left_edge_mm=float(cam_row["Left edge [mm]"]),
            x_start_px=int(float(cam_row["X Start"])),
            x_end_px=int(float(cam_row["X End"])),
            traj_screen_mm=traj_screen_mm,
            traj_momentum_mev_c_per_t=traj_momentum_mev_c_per_t,
            magnetic_field_t=magnetic_field_t,
        )

    @staticmethod
    def _read_tab_delimited_rows(file_path: str) -> List[Dict[str, str]]:
        resolved_path = Path(file_path)
        if not resolved_path.is_absolute():
            if not resolved_path.exists():
                resolved_path = Path(__file__).parent.parent / resolved_path
        with resolved_path.open(newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def build_axis(self, image_width: int) -> np.ndarray:
        """Build MATLAB-style DNN energy-like axis for the current image width."""
        expected_width = self.x_end_px - self.x_start_px + 1
        if expected_width != image_width:
            raise ValueError(
                f"dnn_axis calibration ROI width ({expected_width}) "
                f"doesn't match image width ({image_width}). "
                f"Check ROI cropping and X Start/X End settings."
            )

        # MATLAB fDnnAxisClbEach.m:
        #   dx = fov/width
        #   xx = leftPos:-dx:(leftPos-fov+dx)
        #   x_mm = xx(xSt:xEd)
        dx_mm = self.fov_mm / self.roi_width_px
        x_mm_full = self.left_edge_mm - dx_mm * np.arange(
            self.roi_width_px, dtype=float
        )
        x_mm = x_mm_full[self.x_start_px - 1 : self.x_end_px]

        # MATLAB uses interp1(..., 'spline') for momentum on screen axis.
        order = np.argsort(self.traj_screen_mm)
        traj_screen = self.traj_screen_mm[order]
        traj_momentum = self.traj_momentum_mev_c_per_t[order]
        spline = CubicSpline(traj_screen, traj_momentum, extrapolate=False)
        momentum_axis = spline(x_mm)

        if np.any(~np.isfinite(momentum_axis)):
            raise ValueError(
                "dnn_axis interpolation produced non-finite momentum values. "
                "Check screen range overlap between camera geometry and trajectory table."
            )

        # MATLAB writes energy-like axis as fld * normalized momentum.
        return self.magnetic_field_t * momentum_axis


@dataclass
class AxisResampler:
    """Resample image columns onto a uniform energy axis."""

    energy_range: Tuple[float, float]
    num_points: int

    def resample(
        self, image: np.ndarray, pixel_to_energy_axis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample image columns to a uniform energy axis."""
        return interpolate_image_axis(
            image,
            pixel_to_energy_axis,
            axis=1,
            num_points=self.num_points,
            physical_min=self.energy_range[0],
            physical_max=self.energy_range[1],
        )


class ChargeCalibration(ABC):
    """Interface for converting image counts to charge-like units."""

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply charge calibration to an image."""
        raise NotImplementedError


@dataclass
class ScalarChargeCalibration(ChargeCalibration):
    """
    Scalar charge calibration.

    Multiplies the whole image by one factor (e.g., `c2c` in fC/count).
    """

    factor_fc_per_count: float

    def __post_init__(self):
        """Validate scalar charge calibration factor."""
        if not np.isfinite(self.factor_fc_per_count):
            raise ValueError("factor_fc_per_count must be finite")

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply scalar counts-to-charge calibration."""
        return image * self.factor_fc_per_count

    @classmethod
    def from_dnn_calibration_files(
        cls,
        camera_calibration_file: str,
        lanex_calibration_file: str,
        camera_number: int,
    ) -> "ScalarChargeCalibration":
        """
        Build scalar c2c from DNN camera and LANEX calibration text tables.

        This mirrors MATLAB `fDnnLanex -> fLanexClbOutV01 + fLanexClbV02`:
        - select LANEX set from camera table `set` column
        - compute z = (cam_fov - fov_offset) / fov_slope
        - alsR = sense2*z^2 + sense1*z + sense0
        - screen factor: 1.0 for "back", 1.98 otherwise
        - c2c = screen_factor * alsR / 146
        - sensitivity correction: c2c /= camera_sensitivity
        """
        cam_rows = DnnAxisCalibration._read_tab_delimited_rows(camera_calibration_file)
        lanex_rows = DnnAxisCalibration._read_tab_delimited_rows(lanex_calibration_file)

        cam_row = None
        for row in cam_rows:
            if int(float(row["cam number"])) == camera_number:
                cam_row = row
                break
        if cam_row is None:
            raise ValueError(
                f"Camera number {camera_number} not found in '{camera_calibration_file}'"
            )

        set_n = int(float(cam_row["set"]))
        screen = str(cam_row.get("screen", "")).strip().lower()
        cam_fov = float(cam_row["FOV [mm]"])
        sensitivity = float(cam_row.get("sensitivity", 1.0))
        if sensitivity == 0.0:
            raise ValueError("Camera sensitivity cannot be zero for charge calibration")

        # Prefer explicit setting-number matching when available.
        lanex_row = None
        for row in lanex_rows:
            if "setting number" in row and int(float(row["setting number"])) == set_n:
                lanex_row = row
                break
        if lanex_row is None:
            idx = set_n - 1
            if idx < 0 or idx >= len(lanex_rows):
                raise ValueError(
                    f"LANEX set {set_n} not available in '{lanex_calibration_file}'"
                )
            lanex_row = lanex_rows[idx]

        fov_slope = float(lanex_row["FOV slope"])
        fov_offset = float(lanex_row["FOV offset"])
        sense2 = float(lanex_row["sensitivity 2"])
        sense1 = float(lanex_row["sensitivity 1"])
        sense0 = float(lanex_row["sensitivity 0"])

        z = (cam_fov - fov_offset) / fov_slope
        als_ratio = sense2 * z**2 + sense1 * z + sense0
        screen_factor = 1.0 if screen == "back" else 1.98
        c2c = screen_factor * als_ratio / 146.0
        c2c /= sensitivity

        return cls(factor_fc_per_count=c2c)


class PolynomialCalSpec(BaseModel):
    """External config spec for polynomial calibration."""

    kind: Literal["polynomial"]
    coeffs: List[float]

    @field_validator("coeffs")
    def validate_coeffs(cls, v):
        """Validate polynomial coefficients list."""
        if len(v) == 0:
            raise ValueError("coeffs must contain at least one element")
        return v


class ArrayCalSpec(BaseModel):
    """External config spec for array-based calibration."""

    kind: Literal["array"]
    values: Optional[List[float]] = None
    file: Optional[str] = None

    @model_validator(mode="after")
    def validate_values_or_file(self):
        """Validate that at least one array calibration source is provided."""
        if self.values is None and self.file is None:
            raise ValueError("array calibration requires 'values' or 'file'")
        return self


class CallableCalSpec(BaseModel):
    """External config spec for callable calibration (programmatic only)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["callable"]
    function: Callable[[np.ndarray], np.ndarray]


class DnnAxisCalSpec(BaseModel):
    """External config spec for MATLAB-style DNN axis calibration."""

    kind: Literal["dnn_axis"]
    camera_calibration_file: str
    trajectory_calibration_file: str
    camera_number: int = Field(..., ge=1, le=4)
    magnetic_field_t: float = 1.0


CalibrationSpec = Annotated[
    Union[PolynomialCalSpec, ArrayCalSpec, CallableCalSpec, DnnAxisCalSpec],
    Field(discriminator="kind"),
]


class ScalarChargeCalSpec(BaseModel):
    """External config spec for scalar charge calibration."""

    kind: Literal["scalar"]
    factor_fc_per_count: float


class DnnLanexChargeCalSpec(BaseModel):
    """External config spec for DNN LANEX scalar charge calibration."""

    kind: Literal["dnn_lanex"]
    camera_calibration_file: str
    lanex_calibration_file: str
    camera_number: int = Field(..., ge=1, le=4)


ChargeCalibrationSpec = Annotated[
    Union[ScalarChargeCalSpec, DnnLanexChargeCalSpec],
    Field(discriminator="kind"),
]


class MagSpecAnalyzerConfig(BaseModel):
    """
    High-level external configuration for MagSpecManualCalibAnalyzer.

    This is intended to live under camera config `analysis.magspec`.
    Includes both energy-axis calibration and optional charge calibration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mag_field: str = "custom"
    calibration: CalibrationSpec
    charge_calibration: Optional[ChargeCalibrationSpec] = None
    energy_range: Tuple[float, float]
    num_energy_points: int = 500
    flip_horizontal: bool = False
    flip_vertical: bool = False

    @field_validator("energy_range")
    def validate_energy_range(cls, v):
        """Validate energy range bounds."""
        if len(v) != 2:
            raise ValueError("energy_range must be a 2-tuple/list")
        if v[1] <= v[0]:
            raise ValueError("energy_range max must be greater than min")
        return v

    @classmethod
    def from_camera_analysis(
        cls, analysis: Optional[Dict[str, Any]]
    ) -> Optional["MagSpecAnalyzerConfig"]:
        """
        Parse magspec config from camera `analysis` dict.

        Preferred shape:
        analysis:
          magspec:
            ...

        Backward-compatible fallback:
        analysis:
          calibration: ...
          energy_range: ...
          ...
        """
        if not analysis:
            return None

        if isinstance(analysis.get("magspec"), dict):
            return cls.model_validate(analysis["magspec"])

        if "calibration" in analysis and "energy_range" in analysis:
            return cls.model_validate(analysis)

        return None

    @classmethod
    def for_uc_hiresmag(
        cls,
        mag_field: str = "825mT",
        energy_range: Optional[Tuple[float, float]] = None,
        num_energy_points: int = 500,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> "MagSpecAnalyzerConfig":
        """Preset pydantic config for UC_HiResMagCam."""
        calibrations = {
            "800mT": [8.66599527e01, 1.78007126e-02, 1.10546749e-06],
            "825mT": [8.93681013e01, 1.83568540e-02, 1.14012869e-06],
        }
        if mag_field not in calibrations:
            raise ValueError(
                f"Mag field '{mag_field}' not available for UC_HiResMagCam. "
                f"Available: {list(calibrations.keys())}"
            )
        return cls(
            mag_field=mag_field,
            calibration=PolynomialCalSpec(
                kind="polynomial", coeffs=calibrations[mag_field]
            ),
            energy_range=energy_range or (20, 150),
            num_energy_points=num_energy_points,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
        )

    def to_runtime_config(self) -> "MagSpecConfig":
        """Convert pydantic external config to runtime dataclass config."""
        cal_spec = self.calibration

        if isinstance(cal_spec, PolynomialCalSpec):
            calibration = PolynomialCalibration(coeffs=cal_spec.coeffs)
        elif isinstance(cal_spec, ArrayCalSpec):
            values = None if cal_spec.values is None else np.asarray(cal_spec.values)
            calibration = ArrayCalibration(values=values, file=cal_spec.file)
        elif isinstance(cal_spec, CallableCalSpec):
            calibration = CallableCalibration(function=cal_spec.function)
        elif isinstance(cal_spec, DnnAxisCalSpec):
            calibration = DnnAxisCalibration.from_dnn_calibration_files(
                camera_calibration_file=cal_spec.camera_calibration_file,
                trajectory_calibration_file=cal_spec.trajectory_calibration_file,
                camera_number=cal_spec.camera_number,
                magnetic_field_t=cal_spec.magnetic_field_t,
            )
        else:
            raise ValueError(f"Unsupported calibration spec type: {type(cal_spec)}")

        charge_spec = self.charge_calibration
        if charge_spec is None:
            charge_calibration = None
        elif isinstance(charge_spec, ScalarChargeCalSpec):
            charge_calibration = ScalarChargeCalibration(
                factor_fc_per_count=charge_spec.factor_fc_per_count
            )
        elif isinstance(charge_spec, DnnLanexChargeCalSpec):
            charge_calibration = ScalarChargeCalibration.from_dnn_calibration_files(
                camera_calibration_file=charge_spec.camera_calibration_file,
                lanex_calibration_file=charge_spec.lanex_calibration_file,
                camera_number=charge_spec.camera_number,
            )
        else:
            raise ValueError(
                f"Unsupported charge calibration spec type: {type(charge_spec)}"
            )

        return MagSpecConfig(
            mag_field=self.mag_field,
            calibration=calibration,
            charge_calibration=charge_calibration,
            energy_range=self.energy_range,
            num_energy_points=self.num_energy_points,
            flip_horizontal=self.flip_horizontal,
            flip_vertical=self.flip_vertical,
        )


@dataclass
class MagSpecConfig:
    """
    Configuration for magnetic spectrometer analysis.

    Parameters
    ----------
    mag_field : str
        Magnetic field setting identifier (e.g., "825mT")
    calibration : EnergyCalibration
        Pixel-to-energy calibration strategy (always in canonical orientation)
    energy_range : tuple of (float, float)
        Energy range in MeV (min, max)
    charge_calibration : Optional[ChargeCalibration]
        Optional counts-to-charge calibration strategy applied before
        energy-axis resampling.
    num_energy_points : int, default=500
        Number of points in uniform energy grid
    flip_horizontal : bool, default=False
        Flip raw image horizontally to align with canonical calibration.
        Set to True if camera is mounted backwards relative to calibration.
    flip_vertical : bool, default=False
        Flip raw image vertically to align with canonical calibration.
        Set to True if camera is mounted upside-down.

    Notes
    -----
    The calibration defines the canonical orientation (energy increasing
    left-to-right). Flip parameters correct for camera mounting, ensuring
    the raw image aligns with this canonical orientation before interpolation.
    The output is always in canonical orientation.
    """

    mag_field: str
    calibration: EnergyCalibration
    energy_range: Tuple[float, float]
    charge_calibration: Optional[ChargeCalibration] = None
    num_energy_points: int = 500
    flip_horizontal: bool = False
    flip_vertical: bool = False

    @classmethod
    def for_uc_hiresmag(
        cls,
        mag_field: str = "825mT",
        energy_range: Optional[Tuple[float, float]] = None,
        num_energy_points: int = 500,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> "MagSpecConfig":
        """
        Preset configuration for UC_HiResMagCam device.

        Parameters
        ----------
        mag_field : str, default="825mT"
            Magnetic field setting. Options: "800mT", "825mT"
        energy_range : tuple of (float, float), optional
            Energy range in MeV. Defaults to (20, 150)
        num_energy_points : int, default=500
            Number of points in uniform energy grid
        flip_horizontal : bool, default=False
            Whether to flip image left-right
        flip_vertical : bool, default=False
            Whether to flip image up-down

        Returns
        -------
        MagSpecConfig
            Configuration for UC_HiResMagCam

        Raises
        ------
        ValueError
            If mag_field is not recognized
        """
        return MagSpecAnalyzerConfig.for_uc_hiresmag(
            mag_field=mag_field,
            energy_range=energy_range,
            num_energy_points=num_energy_points,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
        ).to_runtime_config()


class MagSpecManualCalibAnalyzer(BeamAnalyzer):
    """
    Analyzer for magnetic spectrometer with device-specific calibrations.

    Supports multiple calibration formats through MagSpecConfig:
    - Polynomial coefficients
    - Pre-computed arrays (from file or inline)
    - Custom calibration functions
    - MATLAB-style DNN camera + trajectory calibration
    """

    def __init__(
        self,
        camera_config_name: str,
        magspec_config: Optional[MagSpecConfig] = None,
    ):
        """
        Initialize magnetic spectrometer analyzer.

        Parameters
        ----------
        camera_config_name : str
            Camera configuration name (loaded from config system)
        magspec_config : MagSpecConfig, optional
            Magnetic spectrometer runtime configuration. If None, the analyzer
            first tries to parse `camera_config.analysis["magspec"]` using
            MagSpecAnalyzerConfig. If not available, it falls back to the
            UC_HiResMagCam preset when camera_name is "UC_HiResMagCam".

        Raises
        ------
        ValueError
            If magspec_config is None and camera is not UC_HiResMagCam
        """
        super().__init__(camera_config_name)

        # Use provided config, or parse from camera_config.analysis, or fallback preset
        if magspec_config is None:
            parsed_cfg = MagSpecAnalyzerConfig.from_camera_analysis(
                self.camera_config.analysis
            )
            if parsed_cfg is not None:
                self.magspec_config = parsed_cfg.to_runtime_config()
                logger.info("Loaded magspec configuration from camera_config.analysis")
            elif self.camera_name == "UC_HiResMagCam":
                self.magspec_config = MagSpecConfig.for_uc_hiresmag()
                logger.info("Using default UC_HiResMagCam configuration")
            else:
                raise ValueError(
                    f"magspec_config required for camera '{self.camera_name}'. "
                    f"Provide MagSpecAnalyzerConfig under analysis.magspec, "
                    f"use MagSpecConfig.for_uc_hiresmag(), or pass a custom magspec_config."
                )
        else:
            self.magspec_config = magspec_config

        logger.info(
            f"Initialized {self.camera_name} with mag_field={self.magspec_config.mag_field}, "
            f"energy_range={self.magspec_config.energy_range} MeV"
        )

    def _build_energy_calibration(self, image: np.ndarray) -> np.ndarray:
        """
        Build pixel-to-energy mapping for current configuration.

        Parameters
        ----------
        image : np.ndarray
            Image to build calibration for (used to get width)

        Returns
        -------
        np.ndarray
            Array mapping pixel index to energy value
        """
        image_width = image.shape[1]
        return self.magspec_config.calibration.build_axis(image_width=image_width)

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[Dict] = None
    ) -> ImageAnalyzerResult:
        """
        Analyze magnetic spectrometer image with energy calibration.

        Parameters
        ----------
        image : np.ndarray
            Image array to analyze
        auxiliary_data : dict, optional
            Additional data (unused)

        Returns
        -------
        ImageAnalyzerResult
            Analysis result with energy-calibrated image and metadata

        Notes
        -----
        The calibration is always defined in canonical orientation (increasing
        left-to-right). The flip parameters correct for camera mounting:
        - flip_horizontal: Camera is mounted backwards relative to calibration
        - flip_vertical: Camera is mounted upside-down

        Flips are applied to the raw image BEFORE interpolation to align it
        with the canonical calibration. The output is always in canonical
        orientation.
        """
        # Standard processing from BeamAnalyzer
        initial_result: ImageAnalyzerResult = super().analyze_image(
            image=image, auxiliary_data=auxiliary_data
        )

        # Get processed image
        final_image = initial_result.processed_image.copy()

        # Apply flips to align image with canonical calibration orientation
        # These correct for camera mounting - NOT post-processing effects
        if self.magspec_config.flip_horizontal:
            final_image = final_image[:, ::-1]
            logger.debug("Applied horizontal flip to align with canonical calibration")

        if self.magspec_config.flip_vertical:
            final_image = final_image[::-1, :]
            logger.debug("Applied vertical flip to align with canonical calibration")

        # Optional charge calibration (counts -> charge-like units)
        if self.magspec_config.charge_calibration is not None:
            final_image = self.magspec_config.charge_calibration.apply(final_image)

        # Stage 1: pixel -> energy calibration
        pixel_to_energy = self.magspec_config.calibration.build_axis(
            image_width=final_image.shape[1]
        )

        # Stage 2: axis resampling to uniform energy grid
        resampler = AxisResampler(
            energy_range=self.magspec_config.energy_range,
            num_points=self.magspec_config.num_energy_points,
        )
        interp_image, energy_axis = resampler.resample(final_image, pixel_to_energy)

        # Compute analysis metrics
        scalars = self._compute_metrics(interp_image, energy_axis)

        # Create result
        result = ImageAnalyzerResult(
            data_type="2d",
            processed_image=interp_image,
            scalars=scalars,
            metadata={
                "energy_units": "MeV",
                "mag_field": self.magspec_config.mag_field,
                "flip_horizontal": self.magspec_config.flip_horizontal,
                "flip_vertical": self.magspec_config.flip_vertical,
                "charge_calibrated": self.magspec_config.charge_calibration is not None,
            },
        )

        result.render_data = {
            "energy_axis": energy_axis,
        }

        return result

    def _compute_metrics(
        self, image: np.ndarray, energy_axis: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute energy spectrum metrics.

        Parameters
        ----------
        image : np.ndarray
            Energy-calibrated image
        energy_axis : np.ndarray
            Energy axis values in MeV

        Returns
        -------
        dict
            Dictionary of scalar metrics
        """
        # Vertical projection (sum over vertical to get energy spectrum)
        energy_spectrum = np.sum(image, axis=0)

        # Find peak energy
        if len(energy_spectrum) > 0 and np.max(energy_spectrum) > 0:
            peak_idx = np.argmax(energy_spectrum)
            peak_energy = energy_axis[peak_idx]
        else:
            peak_energy = np.nan

        # Total charge (integrated signal)
        total_charge = np.sum(image)

        # Mean energy (weighted by intensity)
        if total_charge > 0:
            mean_energy = np.sum(energy_axis * energy_spectrum) / np.sum(
                energy_spectrum
            )
        else:
            mean_energy = np.nan

        metrics = {
            "peak_energy_MeV": peak_energy,
            "mean_energy_MeV": mean_energy,
            "total_charge_au": total_charge,
            "max_intensity": np.max(image),
        }

        return metrics

    @staticmethod
    def render_image(
        result: ImageAnalyzerResult,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "plasma",
        figsize: Tuple[float, float] = (4, 4),
        dpi: int = 150,
        ax: Optional[plt.Axes] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Custom render function for energy-calibrated spectra.

        Parameters
        ----------
        result : ImageAnalyzerResult
            Analysis result containing energy-calibrated image
        vmin : float, optional
            Minimum value for colormap
        vmax : float, optional
            Maximum value for colormap
        cmap : str, default="plasma"
            Colormap name
        figsize : tuple, default=(4, 4)
            Figure size in inches
        dpi : int, default=150
            Figure DPI
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure

        Returns
        -------
        tuple of (Figure, Axes)
            The matplotlib figure and axes objects
        """
        # Base rendering
        fig, ax = base_render_image(
            result=result,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
            ax=ax,
        )

        # Get energy axis from render_data
        energy_axis = result.render_data.get("energy_axis")

        if energy_axis is not None:
            # Update tick labels to show energy values (not pixels)
            num_ticks = 5
            tick_positions = np.linspace(0, len(energy_axis) - 1, num_ticks)
            tick_labels = [f"{energy_axis[int(pos)]:.1f}" for pos in tick_positions]

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel("Energy (MeV)")
            ax.set_ylabel("Vertical Position (pixels)")

        return fig, ax

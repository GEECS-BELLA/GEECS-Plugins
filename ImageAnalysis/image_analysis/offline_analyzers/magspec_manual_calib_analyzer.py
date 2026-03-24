"""
Magnetic spectrometer analyzer with energy calibration.

Provides analysis for magnetic spectrometer images with pixel-to-energy
conversion using device-specific calibrations.

Supported YAML config format (under ``analysis:``):

.. code-block:: yaml

    analysis:
      energy_range: [76.2, 201.3]
      num_energy_points: 1000
      calibration:
        kind: dnn_axis
        camera_calibration_file: "magspec_code/calibrations/260224DnnVarianCam.txt"
        trajectory_calibration_file: "magspec_code/calibrations/170925DnnVarianTrj.txt"
        lanex_calibration_file: "magspec_code/calibrations/250101lanexCalib.txt"
        camera_number: 1
        magnetic_field_t: 0.2005

Polynomial calibration example:

.. code-block:: yaml

    analysis:
      energy_range: [20, 150]
      num_energy_points: 500
      calibration:
        kind: polynomial
        coeffs: [89.368, 0.01836, 1.14e-06]
"""

import csv
import png
import numpy as np
import logging
from typing import Optional, Tuple, Dict, List, Annotated, Union, Literal, Any
from pathlib import Path
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from image_analysis.tools.rendering import base_render_image
from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer
from image_analysis.types import ImageAnalyzerResult
from image_analysis.algorithms.axis_interpolation import interpolate_image_axis

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _read_tab_delimited_rows(file_path: str) -> List[Dict[str, str]]:
    """Read a tab-delimited text file with a header row into a list of dicts."""
    resolved = Path(file_path)
    if not resolved.is_absolute():
        if not resolved.exists():
            resolved = Path(__file__).parent.parent / resolved
    with resolved.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ------------------------------------------------------------------
# Energy calibration strategy
# ------------------------------------------------------------------


class EnergyCalibration(ABC):
    """Interface for mapping image column index to energy."""

    @abstractmethod
    def build_axis(self, image_width: int) -> np.ndarray:
        """Return energy axis values corresponding to image columns."""
        ...

    def get_charge_factor(self) -> Optional[float]:
        """Return counts-to-fC factor if available, else None."""
        return None


class PolynomialCalibration(EnergyCalibration, BaseModel):
    """Polynomial energy calibration: E(pixel) = c0 + c1*x + c2*x^2 + ...

    YAML example::

        calibration:
          kind: polynomial
          coeffs: [89.368, 0.01836, 1.14e-06]
    """

    kind: Literal["polynomial"] = "polynomial"
    coeffs: List[float]

    @field_validator("coeffs")
    def _check_coeffs(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("coeffs must contain at least one element")
        return v

    def build_axis(self, image_width: int) -> np.ndarray:
        """Evaluate polynomial E(pixel) for each column index."""
        pixel = np.arange(image_width, dtype=float)
        return sum(c * pixel**i for i, c in enumerate(self.coeffs))


class ArrayCalibration(EnergyCalibration, BaseModel):
    """Calibration from a precomputed axis array (inline list or ``.npy`` file).

    YAML example::

        calibration:
          kind: array
          file: "path/to/energy_axis.npy"
    """

    kind: Literal["array"] = "array"
    values: Optional[List[float]] = None
    file: Optional[str] = None

    @model_validator(mode="after")
    def _require_source(self):
        if self.values is None and self.file is None:
            raise ValueError("array calibration requires 'values' or 'file'")
        return self

    def build_axis(self, image_width: int) -> np.ndarray:
        """Load axis from file or inline list and validate against image width."""
        if self.file is not None:
            cal_path = Path(self.file)
            if not cal_path.is_absolute() and not cal_path.exists():
                cal_path = Path(__file__).parent.parent / cal_path
            axis = np.load(cal_path)
        else:
            axis = np.asarray(self.values, dtype=float)

        if len(axis) != image_width:
            raise ValueError(
                f"Calibration array length ({len(axis)}) "
                f"doesn't match image width ({image_width})"
            )
        return axis


class DnnAxisCalibration(EnergyCalibration, BaseModel):
    """MATLAB-style DNN magspec calibration (camera + trajectory files).

    On construction the tab-delimited calibration files are read and the
    internal arrays are populated.  If ``lanex_calibration_file`` is
    provided, a counts-to-fC charge factor is also computed.

    YAML example::

        calibration:
          kind: dnn_axis
          camera_calibration_file: "magspec_code/calibrations/260224DnnVarianCam.txt"
          trajectory_calibration_file: "magspec_code/calibrations/170925DnnVarianTrj.txt"
          lanex_calibration_file: "magspec_code/calibrations/250101lanexCalib.txt"
          camera_number: 1
          magnetic_field_t: 0.2005
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["dnn_axis"] = "dnn_axis"
    camera_calibration_file: str
    trajectory_calibration_file: str
    camera_number: int = Field(..., ge=1, le=4)
    magnetic_field_t: float = 1.0
    lanex_calibration_file: Optional[str] = None

    # Populated during model_post_init
    _fov_mm: float = 0.0
    _roi_width_px: int = 0
    _left_edge_mm: float = 0.0
    _x_start_px: int = 0
    _x_end_px: int = 0
    _traj_screen_mm: np.ndarray = np.empty(0)
    _traj_momentum: np.ndarray = np.empty(0)
    _charge_factor: Optional[float] = None

    def model_post_init(self, __context: Any) -> None:
        """Load calibration data from text files after Pydantic validation."""
        cam_rows = _read_tab_delimited_rows(self.camera_calibration_file)
        trj_rows = _read_tab_delimited_rows(self.trajectory_calibration_file)

        # Find camera row
        cam_row = None
        for row in cam_rows:
            if int(float(row["cam number"])) == self.camera_number:
                cam_row = row
                break
        if cam_row is None:
            raise ValueError(
                f"Camera number {self.camera_number} not found in "
                f"'{self.camera_calibration_file}'"
            )

        self._fov_mm = float(cam_row["FOV [mm]"])
        self._roi_width_px = int(float(cam_row["ROI width"]))
        self._left_edge_mm = float(cam_row["Left edge [mm]"])
        self._x_start_px = int(float(cam_row["X Start"]))
        self._x_end_px = int(float(cam_row["X End"]))

        # Validate indices
        if self._roi_width_px <= 0:
            raise ValueError("ROI width must be > 0")
        if self._x_start_px < 1 or self._x_end_px < self._x_start_px:
            raise ValueError("X Start / X End must be 1-based with X End >= X Start")
        if self._x_end_px > self._roi_width_px:
            raise ValueError(
                f"X End ({self._x_end_px}) exceeds ROI width ({self._roi_width_px})"
            )

        # Split trajectory by front/side
        split_idx = None
        for i, row in enumerate(trj_rows):
            if float(row["side logic"]) == 1.0:
                split_idx = i
                break
        if split_idx is None:
            raise ValueError("No 'side logic == 1' split in trajectory file")

        if self.camera_number in (1, 2):
            trj_subset = trj_rows[:split_idx]
            screen_col = "front screen [m]"
        else:
            trj_subset = trj_rows[split_idx:]
            screen_col = "side screen [m]"

        self._traj_screen_mm = 1000.0 * np.array(
            [float(r[screen_col]) for r in trj_subset], dtype=float
        )
        self._traj_momentum = np.array(
            [float(r["momentum [MeV/c]"]) for r in trj_subset], dtype=float
        )

        # Charge calibration from lanex file (if provided)
        if self.lanex_calibration_file is not None:
            self._charge_factor = self._compute_charge_factor(cam_row)

    def _compute_charge_factor(self, cam_row: Dict[str, str]) -> float:
        """Compute counts-to-fC factor from LANEX calibration table.

        Mirrors MATLAB ``fDnnLanex -> fLanexClbOutV01 + fLanexClbV02``.
        """
        lanex_rows = _read_tab_delimited_rows(self.lanex_calibration_file)

        set_n = int(float(cam_row["set"]))
        screen = str(cam_row.get("screen", "")).strip().lower()
        cam_fov = float(cam_row["FOV [mm]"])
        sensitivity = float(cam_row.get("sensitivity", 1.0))
        if sensitivity == 0.0:
            raise ValueError("Camera sensitivity cannot be zero")

        # Find lanex row by setting number
        lanex_row = None
        for row in lanex_rows:
            if "setting number" in row and int(float(row["setting number"])) == set_n:
                lanex_row = row
                break
        if lanex_row is None:
            idx = set_n - 1
            if idx < 0 or idx >= len(lanex_rows):
                raise ValueError(f"LANEX set {set_n} not available")
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
        return c2c

    def build_axis(self, image_width: int) -> np.ndarray:
        """Build MATLAB-style DNN energy axis for the current image width."""
        expected_width = self._x_end_px - self._x_start_px + 1
        if expected_width != image_width:
            raise ValueError(
                f"DNN calibration ROI width ({expected_width}) doesn't match "
                f"image width ({image_width}). Check ROI and X Start/X End."
            )

        dx_mm = self._fov_mm / self._roi_width_px
        x_mm_full = self._left_edge_mm - dx_mm * np.arange(
            self._roi_width_px, dtype=float
        )
        x_mm = x_mm_full[self._x_start_px - 1 : self._x_end_px]

        order = np.argsort(self._traj_screen_mm)
        spline = CubicSpline(
            self._traj_screen_mm[order],
            self._traj_momentum[order],
            extrapolate=False,
        )
        momentum = spline(x_mm)

        if np.any(~np.isfinite(momentum)):
            raise ValueError(
                "DNN interpolation produced non-finite momentum values. "
                "Check screen range overlap between camera and trajectory."
            )

        return self.magnetic_field_t * momentum

    @property
    def expected_image_width(self) -> int:
        """Expected image width based on X Start / X End from camera calibration."""
        return self._x_end_px - self._x_start_px + 1

    def get_charge_factor(self) -> Optional[float]:
        """Return counts-to-fC conversion factor, or None if no lanex file."""
        return self._charge_factor


# Discriminated union of calibration types
CalibrationSpec = Annotated[
    Union[PolynomialCalibration, ArrayCalibration, DnnAxisCalibration],
    Field(discriminator="kind"),
]


# ------------------------------------------------------------------
# Top-level magspec config
# ------------------------------------------------------------------


class MagSpecAnalyzerConfig(BaseModel):
    """Configuration parsed from ``camera_config.analysis``.

    YAML shape::

        analysis:
          energy_range: [76.2, 201.3]
          num_energy_points: 1000
          calibration:
            kind: dnn_axis
            ...
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    calibration: CalibrationSpec
    energy_range: Tuple[float, float]
    num_energy_points: int = 500

    @field_validator("energy_range")
    def _check_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if len(v) != 2:
            raise ValueError("energy_range must be a 2-element tuple")
        if v[1] <= v[0]:
            raise ValueError("energy_range max must exceed min")
        return v


# ------------------------------------------------------------------
# Analyzer
# ------------------------------------------------------------------


class MagSpecManualCalibAnalyzer(BeamAnalyzer):
    """Magnetic spectrometer analyzer with energy calibration.

    Reads configuration from ``camera_config.analysis`` and supports
    multiple calibration formats via the ``kind`` discriminator:

    - ``polynomial`` — polynomial pixel-to-energy mapping
    - ``array`` — precomputed axis from file or inline list
    - ``dnn_axis`` — MATLAB-era DNN camera + trajectory calibration

    Image flips are handled by the camera-level ``transforms`` config,
    not by the magspec config.  Charge calibration (counts → fC) is
    automatically applied when ``lanex_calibration_file`` is present
    in a ``dnn_axis`` calibration block.
    """

    def __init__(self, camera_config_name: str):
        super().__init__(camera_config_name)

        analysis = self.camera_config.analysis
        if not analysis or "calibration" not in analysis:
            raise ValueError(
                f"Camera '{self.camera_name}' requires an 'analysis' section "
                f"with at least 'calibration' and 'energy_range'."
            )

        # Accept both `analysis.magspec.calibration` (legacy) and
        # `analysis.calibration` (preferred flat format).
        raw = (
            analysis.get("magspec", analysis)
            if isinstance(analysis.get("magspec"), dict)
            else analysis
        )

        self.magspec_config = MagSpecAnalyzerConfig.model_validate(raw)

        logger.info(
            "Initialized %s magspec analyzer: energy_range=%s MeV, calibration=%s",
            self.camera_name,
            self.magspec_config.energy_range,
            self.magspec_config.calibration.kind,
        )

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[Dict] = None
    ) -> ImageAnalyzerResult:
        """Analyze magnetic spectrometer image with energy calibration.

        Processing pipeline:
        1. Standard preprocessing (background, ROI, transforms) via BeamAnalyzer
        2. Optional charge calibration (if lanex file was provided)
        3. Pixel-to-energy axis mapping
        4. Resampling onto uniform energy grid
        5. Spectrum metrics (peak energy, mean energy, total charge)
        """
        initial_result: ImageAnalyzerResult = super().analyze_image(
            image=image, auxiliary_data=auxiliary_data
        )
        final_image = initial_result.processed_image.copy()

        cal = self.magspec_config.calibration

        # check if magnetic field is being passed through analyze_image,
        # overriding the config value in the DNN calibration (if present)
        aux_mag_field = (
            auxiliary_data.get("magnetic_field_t", None) if auxiliary_data else None
        )
        if isinstance(cal, DnnAxisCalibration) and aux_mag_field is not None:
            logger.info(
                "Overriding DNN calibration magnetic field with auxiliary value: "
                "config=%s T, auxiliary=%s T",
                cal.magnetic_field_t,
                aux_mag_field,
            )
            cal.magnetic_field_t = aux_mag_field

        # Charge calibration (if available)
        charge_factor = cal.get_charge_factor()
        if charge_factor is not None:
            final_image = final_image * charge_factor

        # Build pixel-to-energy axis
        pixel_to_energy = cal.build_axis(image_width=final_image.shape[1])

        # Resample to uniform energy grid
        interp_image, energy_axis = interpolate_image_axis(
            final_image,
            pixel_to_energy,
            axis=1,
            num_points=self.magspec_config.num_energy_points,
            physical_min=self.magspec_config.energy_range[0],
            physical_max=self.magspec_config.energy_range[1],
        )

        # Compute metrics
        scalars = self._compute_metrics(interp_image, energy_axis)

        result = ImageAnalyzerResult(
            data_type="2d",
            processed_image=interp_image,
            scalars=scalars,
            metadata={
                "energy_units": "MeV",
                "calibration_kind": cal.kind,
                "charge_calibrated": charge_factor is not None,
                "charge_factor_fc_per_count": charge_factor,
            },
        )
        result.render_data = {"energy_axis": energy_axis}

        file_path = auxiliary_data.get("file_path") if auxiliary_data else None
        if file_path is not None:
            try:
                self._save_calibrated_outputs(result, Path(file_path))
            except Exception as e:
                logger.warning(
                    "Failed to save calibrated outputs for %s: %s", file_path, e
                )

        return result

    @staticmethod
    def _compute_metrics(
        image: np.ndarray, energy_axis: np.ndarray
    ) -> Dict[str, float]:
        """Compute energy spectrum metrics from calibrated image."""
        spectrum = np.sum(image, axis=0)
        total = np.sum(image)

        if len(spectrum) > 0 and np.max(spectrum) > 0:
            peak_energy = energy_axis[np.argmax(spectrum)]
        else:
            peak_energy = np.nan

        if total > 0:
            mean_energy = np.sum(energy_axis * spectrum) / np.sum(spectrum)
        else:
            mean_energy = np.nan

        return {
            "peak_energy_MeV": peak_energy,
            "mean_energy_MeV": mean_energy,
            "total_charge_au": total,
            "max_intensity": float(np.max(image)),
        }

    def _save_calibrated_outputs(
        self, result: ImageAnalyzerResult, file_path: Path
    ) -> None:
        """Save calibrated image and spectrum derived from *result*.

        Called automatically from :meth:`analyze_image` when ``file_path`` is
        present in ``auxiliary_data``.  Two sibling directories are created
        next to the camera's raw-data folder (``file_path.parent.parent``):

        ``{camera_name}-interp/``
            16-bit PNG where each pixel value equals the charge-calibrated
            intensity divided by the energy bin width (units: fC / MeV).

        ``{camera_name}-interpSpec/``
            Two-column TSV: ``Energy [MeV]`` and ``Charge Density [pC/MeV]``
            (vertical integral of the calibrated image, converted fC → pC,
            divided by the energy bin width).

        Parameters
        ----------
        result : ImageAnalyzerResult
            Result from :meth:`analyze_image`.
        file_path : Path
            Path to the source image file (as supplied via ``auxiliary_data``).
        """
        image = result.processed_image
        energy_axis = np.asarray(result.render_data["energy_axis"], dtype=float)

        output_dir = file_path.parent.parent
        file_stem = file_path.stem

        # --- interp: 16-bit PNG with units atto_C/MeV per pixel ---
        interp_dir = output_dir / f"{self.camera_name}-interp"
        interp_dir.mkdir(parents=True, exist_ok=True)
        # scale image by 1000 to convert fC/MeV to aC/MeV, then clip and convert to uint16 for PNG
        image_uint16 = np.clip(image * 1000, 0, 65535).astype(np.uint16)
        with open(interp_dir / f"{file_stem}.png", "wb") as f:
            png.Writer(
                width=image_uint16.shape[1],
                height=image_uint16.shape[0],
                bitdepth=16,
                greyscale=True,
            ).write(f, image_uint16)
        logger.info("Saved calibrated image to %s", interp_dir / f"{file_stem}.png")

        # --- interpSpec: energy / charge-density TSV ---
        spec_dir = output_dir / f"{self.camera_name}-interpSpec"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spectrum_pC_per_MeV = np.sum(image, axis=0) / 1000.0
        data = np.column_stack([energy_axis, spectrum_pC_per_MeV])
        np.savetxt(
            str(spec_dir / f"{file_stem}.tsv"),
            data,
            delimiter="\t",
            header="Energy [MeV]\tCharge Density [pC/MeV]",
            comments="",
        )
        logger.info("Saved spectrum to %s", spec_dir / f"{file_stem}.tsv")

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
        """Render energy-calibrated spectrum with energy axis labels."""
        fig, ax = base_render_image(
            result=result,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
            ax=ax,
        )

        energy_axis = result.render_data.get("energy_axis")
        if energy_axis is not None:
            num_ticks = 5
            positions = np.linspace(0, len(energy_axis) - 1, num_ticks)
            labels = [f"{energy_axis[int(p)]:.1f}" for p in positions]
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Energy (MeV)")
            ax.set_ylabel("Vertical Position (pixels)")

        return fig, ax

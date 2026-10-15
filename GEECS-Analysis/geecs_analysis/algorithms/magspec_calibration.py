"""The HTT magnetic spectrometer's pixel-to-energy calibration and resample, bit for bit.

The legacy ``MagSpecManualCalibAnalyzer`` (ImageAnalysis) built these from
three tab-delimited surveys — the camera geometry table, the electron
trajectory table and the lanex charge table — read at construction. Here
they are pure functions over :class:`~geecs_data_utils.tables.Table` values
a host binds. The MATLAB-era conventions are kept exactly: the 1-based
``X Start`` / ``X End`` slice, ``argsort`` plus a non-extrapolating cubic
spline over screen position, the ``1.98 / 146`` lanex factors; the resample
is the legacy row-wise ``np.interp`` of ``counts / dE`` onto a uniform grid
clipped to the calibration's reach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    import numpy as np
    from geecs_data_utils.tables import Table

#: The camera table's row key and the screen each camera images.
CAMERA_NUMBER = "cam number"
FRONT_SCREEN_CAMERAS = (1, 2)


@dataclass(frozen=True)
class CameraGeometry:
    """One camera's row of the geometry table: the fields the axis needs.

    ``x_start_px`` / ``x_end_px`` are 1-based and inclusive, as the table
    writes them.
    """

    fov_mm: float
    roi_width_px: int
    left_edge_mm: float
    x_start_px: int
    x_end_px: int

    @property
    def width(self) -> int:
        """The image width the calibration expects: ``X End - X Start + 1``."""
        return self.x_end_px - self.x_start_px + 1


@dataclass(frozen=True)
class Trajectory:
    """Screen position (mm) against momentum (MeV/c) for one screen, in table order."""

    screen_mm: np.ndarray
    momentum: np.ndarray


def camera_row(table: Table, camera_number: int) -> Mapping[str, str]:
    """The geometry table's row whose ``cam number`` is ``camera_number``."""
    for index in range(len(table)):
        row = table.row(index)
        if int(float(row[CAMERA_NUMBER])) == camera_number:
            return row
    raise ValueError(f"Camera number {camera_number} not found in the camera table")


def camera_geometry(row: Mapping[str, str]) -> CameraGeometry:
    """The row's geometry, with the legacy range checks."""
    geometry = CameraGeometry(
        fov_mm=float(row["FOV [mm]"]),
        roi_width_px=int(float(row["ROI width"])),
        left_edge_mm=float(row["Left edge [mm]"]),
        x_start_px=int(float(row["X Start"])),
        x_end_px=int(float(row["X End"])),
    )
    if geometry.roi_width_px <= 0:
        raise ValueError("ROI width must be > 0")
    if geometry.x_start_px < 1 or geometry.x_end_px < geometry.x_start_px:
        raise ValueError("X Start / X End must be 1-based with X End >= X Start")
    if geometry.x_end_px > geometry.roi_width_px:
        raise ValueError(
            f"X End ({geometry.x_end_px}) exceeds ROI width ({geometry.roi_width_px})"
        )
    return geometry


def trajectory(table: Table, camera_number: int) -> Trajectory:
    """The trajectory rows of the camera's screen: front (cameras 1-2) or side.

    The table lists the front-screen rows first; the first row whose
    ``side logic`` is 1 starts the side-screen rows, as the legacy split did.
    Only the cells the legacy reader converted are converted here: ``side
    logic`` up to that row, then the camera's screen column and the momentum
    over the camera's rows — a survey may leave the rest blank.
    """
    import numpy as np

    logic = table.strings("side logic")
    first = next((i for i, cell in enumerate(logic) if float(cell) == 1.0), None)
    if first is None:
        raise ValueError("No 'side logic == 1' split in the trajectory table")
    front = camera_number in FRONT_SCREEN_CAMERAS
    rows = slice(None, first) if front else slice(first, None)
    column = "front screen [m]" if front else "side screen [m]"
    screen = np.array([float(c) for c in table.strings(column)[rows]], dtype=float)
    momentum = table.strings("momentum [MeV/c]")[rows]
    return Trajectory(
        1000.0 * screen, np.array([float(c) for c in momentum], dtype=float)
    )


def charge_factor(row: Mapping[str, str], lanex: Table) -> float:
    """Counts-to-fC factor from the lanex table (MATLAB ``fDnnLanex``), as legacy.

    The lanex row is the one whose ``setting number`` equals the camera's
    ``set`` when the table has that column, else the ``set``-th row.
    ``screen == "back"`` drops the ``1.98`` front-screen factor; the result
    is divided by the camera's ``sensitivity``.
    """
    set_number = int(float(row["set"]))
    screen = str(row.get("screen", "")).strip().lower()
    fov = float(row["FOV [mm]"])
    sensitivity = float(row.get("sensitivity", 1.0))
    if sensitivity == 0.0:
        raise ValueError("Camera sensitivity cannot be zero")
    lanex_row = None
    if "setting number" in lanex.columns:
        for index in range(len(lanex)):
            candidate = lanex.row(index)
            if int(float(candidate["setting number"])) == set_number:
                lanex_row = candidate
                break
    if lanex_row is None:
        index = set_number - 1
        if index < 0 or index >= len(lanex):
            raise ValueError(f"LANEX set {set_number} not available")
        lanex_row = lanex.row(index)
    z = (fov - float(lanex_row["FOV offset"])) / float(lanex_row["FOV slope"])
    als_ratio = (
        float(lanex_row["sensitivity 2"]) * z**2
        + float(lanex_row["sensitivity 1"]) * z
        + float(lanex_row["sensitivity 0"])
    )
    screen_factor = 1.0 if screen == "back" else 1.98
    c2c = screen_factor * als_ratio / 146.0
    return c2c / sensitivity


def dnn_energy_axis(
    geometry: CameraGeometry,
    path: Trajectory,
    magnetic_field_t: float,
    image_width: int,
) -> np.ndarray:
    """Energy (MeV) per image column: the MATLAB-style DNN axis at ``magnetic_field_t``."""
    import numpy as np
    from scipy.interpolate import CubicSpline

    if geometry.width != image_width:
        raise ValueError(
            f"DNN calibration ROI width ({geometry.width}) doesn't match "
            f"image width ({image_width}). Check ROI and X Start/X End."
        )
    dx_mm = geometry.fov_mm / geometry.roi_width_px
    x_mm_full = geometry.left_edge_mm - dx_mm * np.arange(
        geometry.roi_width_px, dtype=float
    )
    x_mm = x_mm_full[geometry.x_start_px - 1 : geometry.x_end_px]
    order = np.argsort(path.screen_mm)
    spline = CubicSpline(path.screen_mm[order], path.momentum[order], extrapolate=False)
    momentum = spline(x_mm)
    if np.any(~np.isfinite(momentum)):
        raise ValueError(
            "DNN interpolation produced non-finite momentum values. "
            "Check screen range overlap between camera and trajectory."
        )
    return magnetic_field_t * momentum


def polynomial_axis(coefficients: Sequence[float], image_width: int) -> np.ndarray:
    """``E(column) = c0 + c1·x + c2·x² + …`` over column indices, summed as legacy."""
    import numpy as np

    pixel = np.arange(image_width, dtype=float)
    return sum(c * pixel**i for i, c in enumerate(coefficients))


def array_axis(values: Sequence[float], image_width: int) -> np.ndarray:
    """An explicit energy per column; its length must be the image width."""
    import numpy as np

    axis = np.asarray(values, dtype=float)
    if len(axis) != image_width:
        raise ValueError(
            f"Calibration array length ({len(axis)}) "
            f"doesn't match image width ({image_width})"
        )
    return axis


def resample_columns(
    image: np.ndarray,
    pixel_to_physical: np.ndarray,
    num_points: int,
    physical_min: float,
    physical_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample each row of ``image`` onto a uniform grid of the column coordinate.

    The legacy ``interpolate_image_axis`` along axis 1: the requested range
    is clipped to the calibration's reach (silently — the legacy warned
    twice per shot; the numbers are the same), each row is divided by
    ``np.gradient`` of the column coordinate (a density per unit
    coordinate) and ``np.interp``ed onto ``np.linspace(min, max,
    num_points)``. ``np.interp`` assumes an ascending coordinate, as the
    legacy did. Returns the resampled image and the grid.
    """
    import numpy as np

    if len(pixel_to_physical) != image.shape[1]:
        raise ValueError(
            f"pixel_to_physical length ({len(pixel_to_physical)}) must match "
            f"image dimension along axis 1 ({image.shape[1]})"
        )
    calib_min = np.min(pixel_to_physical)
    calib_max = np.max(pixel_to_physical)
    if physical_min < calib_min:
        physical_min = calib_min
    if physical_max > calib_max:
        physical_max = calib_max
    dE = np.gradient(pixel_to_physical)
    uniform_axis = np.linspace(physical_min, physical_max, num_points)
    output = np.zeros((image.shape[0], num_points), dtype=image.dtype)
    for i in range(image.shape[0]):
        output[i] = np.interp(uniform_axis, pixel_to_physical, image[i, :] / dE)
    return output, uniform_axis


def spectrum_metrics(image: np.ndarray, energy_axis: np.ndarray) -> dict[str, float]:
    """The four legacy scalars of the resampled image: peak and mean energy, total, max."""
    import numpy as np

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
        "peak_energy_MeV": float(peak_energy),
        "mean_energy_MeV": float(mean_energy),
        "total_charge_au": float(total),
        "max_intensity": float(np.max(image)),
    }

"""HTT magnetic spectrometer: the image resampled onto an energy axis, with charge scalars.

The legacy ``MagSpecManualCalibAnalyzer``, ported with its numerics
(``algorithms.magspec_calibration``). The calibration is one of three kinds:
``dnn_axis`` — the MATLAB-era survey tables, bound as the recipe's table
inputs, with the dipole field read per shot from the scan's recorded
scalars when ``magnetic_field_column`` names the column — ``polynomial``
(energy as a polynomial in the column index) or ``array`` (one energy per
column, inline). The processed frame becomes the resampled image on a
uniform energy grid; the four scalars and the ``spectrum`` extra are the
legacy analyzer's numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Mapping, Optional, Union

from pydantic import Field, field_validator, model_validator

from geecs_analysis.registry import MeasureSpec, SpecModel, measure

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from geecs_data_utils.tables import Table
    from geecs_analysis.measurement import Measurement

#: The scalars, in emission order.
SCALARS = ("peak_energy_MeV", "mean_energy_MeV", "total_charge_au", "max_intensity")
#: The spec fields that bind the ``dnn_axis`` survey tables, in the order
#: the measure receives them as keyword arguments.
TABLE_FIELDS = ("camera_table", "trajectory_table", "lanex_table")
#: The spec field naming the per-shot dipole-field column.
FIELD_COLUMN = "magnetic_field_column"


class DnnAxisCalibration(SpecModel):
    """The MATLAB-era DNN calibration: camera geometry plus electron trajectory tables.

    The tables are the recipe's ``camera_table`` and ``trajectory_table``
    inputs; a ``lanex_table`` input adds the counts-to-fC charge factor.
    """

    kind: Literal["dnn_axis"] = Field(
        ..., description="DNN camera + trajectory table calibration."
    )
    camera_number: int = Field(
        ...,
        ge=1,
        le=4,
        description=(
            "Which camera row of the geometry table to use; cameras 1-2 image "
            "the front screen, 3-4 the side screen."
        ),
    )
    magnetic_field_t: float = Field(
        1.0,
        description=(
            "Dipole field in tesla the trajectory table is scaled to: the "
            "field for every shot, unless magnetic_field_column gives a shot's own."
        ),
    )


class PolynomialCalibration(SpecModel):
    """Energy as a polynomial in the column index."""

    kind: Literal["polynomial"] = Field(
        ..., description="Polynomial pixel-to-energy mapping."
    )
    coeffs: tuple[float, ...] = Field(
        ...,
        min_length=1,
        description="Polynomial coefficients, lowest order first, energy in MeV per column index.",
    )


class ArrayCalibration(SpecModel):
    """An explicit energy per image column."""

    kind: Literal["array"] = Field(..., description="Explicit per-column energy axis.")
    values: tuple[float, ...] = Field(
        ...,
        min_length=1,
        description="Energy per column in MeV; as many values as the processed frame has columns.",
    )


Calibration = Annotated[
    Union[DnnAxisCalibration, PolynomialCalibration, ArrayCalibration],
    Field(discriminator="kind"),
]


class MagspecSpec(MeasureSpec):
    """Energy-calibrated spectrometer image: columns resampled onto a uniform energy grid.

    A ``dnn_axis`` calibration reads its survey tables through the recipe's
    table inputs, named here by binding key, and takes each shot's dipole
    field from the scan's recorded column ``magnetic_field_column`` names
    (an s-file column such as a teslameter's ``<device> Field``); without
    it, or for a shot that lacks it, ``magnetic_field_t`` applies.
    ``energy_range`` is a request: the grid is clipped to the calibration's
    reach, so a range wider than the screen gives the calibration's own span.
    Fields, defaults and bounds are the v2 ``magspec`` analyzer's.
    """

    kind: Literal["magspec"] = "magspec"
    scalar_docs: ClassVar[Mapping[str, str]] = {
        "peak_energy_MeV": (
            "Energy of the maximum of the spectrum (the resampled image summed "
            "over rows), in MeV; NaN when no sample is positive"
        ),
        "mean_energy_MeV": (
            "Spectrum-weighted mean energy, in MeV; NaN when the image's total "
            "is not positive"
        ),
        "total_charge_au": (
            "Sum of the resampled image (a per-MeV density over the grid's "
            "samples): fC-scaled with a lanex table, counts otherwise"
        ),
        "max_intensity": (
            "Brightest sample of the resampled image, in fC/MeV with a lanex "
            "table, counts/MeV otherwise"
        ),
    }
    calibration: Calibration = Field(
        ...,
        description="How image columns map to energy: dnn_axis (survey tables), polynomial or array.",
    )
    camera_table: Optional[str] = Field(
        None,
        min_length=1,
        description="Binding key of the camera geometry table input (dnn_axis; required for it).",
    )
    trajectory_table: Optional[str] = Field(
        None,
        min_length=1,
        description="Binding key of the trajectory table input (dnn_axis; required for it).",
    )
    lanex_table: Optional[str] = Field(
        None,
        min_length=1,
        description=(
            "Binding key of the lanex charge table input (dnn_axis); when set, "
            "the image is scaled from counts to fC."
        ),
    )
    magnetic_field_column: Optional[str] = Field(
        None,
        min_length=1,
        description=(
            "The scan's recorded column holding each shot's dipole field in "
            "tesla (dnn_axis); magnetic_field_t is the fallback."
        ),
    )
    energy_range: tuple[float, float] = Field(
        ...,
        description="(min, max) in MeV requested for the uniform grid; clipped to the calibration's reach.",
    )
    num_energy_points: int = Field(
        500, ge=2, description="Number of points on the uniform energy grid."
    )

    @field_validator("energy_range")
    @classmethod
    def _check_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        """Require max > min."""
        if value[1] <= value[0]:
            raise ValueError("energy_range max must exceed min")
        return value

    @model_validator(mode="after")
    def _bindings_fit_the_calibration(self) -> "MagspecSpec":
        """dnn_axis needs its two geometry tables; the other kinds bind nothing."""
        slots = (*TABLE_FIELDS, FIELD_COLUMN)
        if self.calibration.kind == "dnn_axis":
            if self.camera_table is None or self.trajectory_table is None:
                raise ValueError(
                    "a dnn_axis calibration needs camera_table and trajectory_table"
                )
            return self
        if any(getattr(self, name) is not None for name in slots):
            raise ValueError(
                f"{self.calibration.kind} calibration binds no tables or field column"
            )
        return self

    def emitted_scalars(self) -> frozenset[str]:
        """The four spectrum scalars."""
        return frozenset(SCALARS)


@measure(MagspecSpec, ndim={2}, tables=TABLE_FIELDS, shot_scalars=(FIELD_COLUMN,))
def magspec(
    frame: Frame,
    spec: MagspecSpec,
    *,
    camera_table: Optional[Table] = None,
    trajectory_table: Optional[Table] = None,
    lanex_table: Optional[Table] = None,
) -> Measurement:
    """Resample the frame's columns onto the energy grid and measure the spectrum.

    With a lanex table the processed frame is scaled to fC before the
    resample, as the legacy analyzer scaled it. The measured frame is the
    resampled image (rows as they were, ``Energy`` in MeV across) in
    ``fC/MeV`` or ``counts/MeV``; the ``spectrum`` extra is its row sum over
    1000 — the legacy ``-interpSpec`` column in pC/MeV — on the same grid.
    A width the calibration does not expect, a camera the table lacks or a
    screen range the trajectory does not cover raise, as before.
    """
    import numpy as np
    from geecs_data_utils.frames import Axis, Frame

    from geecs_analysis.algorithms import magspec_calibration as calibration
    from geecs_analysis.measurement import Measurement

    if frame.data.ndim != 2:
        raise ValueError("magspec requires a 2D frame")
    width = frame.data.shape[1]
    factor = None
    kind = spec.calibration
    if kind.kind == "dnn_axis":
        row = calibration.camera_row(camera_table, kind.camera_number)
        geometry = calibration.camera_geometry(row)
        path = calibration.trajectory(trajectory_table, kind.camera_number)
        if lanex_table is not None:
            factor = calibration.charge_factor(row, lanex_table)
        axis = calibration.dnn_energy_axis(geometry, path, _field(frame, spec), width)
    elif kind.kind == "polynomial":
        axis = calibration.polynomial_axis(kind.coeffs, width)
    else:
        axis = calibration.array_axis(kind.values, width)
    data = frame.data if factor is None else frame.data * factor
    resampled, energy = calibration.resample_columns(
        data, axis, spec.num_energy_points, *spec.energy_range
    )
    energy_axis = Axis(energy, unit="MeV", label="Energy")
    unit = "counts/MeV" if factor is None else "fC/MeV"
    spectrum = np.sum(resampled, axis=0) / 1000.0
    return Measurement(
        scalars=calibration.spectrum_metrics(resampled, energy),
        frame=Frame(
            resampled, (frame.axes[0], energy_axis), frame.shot, unit, frame.label
        ),
        extras={
            "spectrum": Frame(
                spectrum, (energy_axis,), frame.shot, "pC/MeV", "Charge Density"
            )
        },
    )


def _field(frame: Frame, spec: MagspecSpec) -> float:
    """The shot's recorded dipole field when the recipe names its column, else the spec's.

    Read from the frame's shot, never kept: the legacy override wrote the
    live value into the shared calibration, so a later shot without the
    reading used the previous shot's field.
    """
    column = spec.magnetic_field_column
    if column is not None and frame.shot is not None and column in frame.shot.scalars:
        return frame.shot.scalars[column]
    return spec.calibration.magnetic_field_t

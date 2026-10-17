"""Line (1D trace) processing configuration — the ``image:`` section of a 1D diagnostic.

Covers scope traces, spectra, lineouts and other x-vs-y data.  A trace is
loaded per :class:`Data1DLoading`, scaled by the axis scale factors, and
then run through the ``pipeline`` steps in order — ROI (in x-value units,
not indices), interpolation onto a uniform grid, background, filtering,
thresholding.  As with cameras, a step runs only when listed.

Formerly ImageAnalysis' own 1D processing models (moved here in GEECS-Schemas
0.19.0).  The 1D sub-models carry a ``Line`` prefix so they never collide
with the camera models of the same role; consumers import them by these
names.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from geecs_schemas._base import SchemaModel

NumpyDtypeName = Literal[
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]


class Data1DType(str, Enum):
    """The file formats the 1D reader understands."""

    TEK_SCOPE_HDF5 = "tek_scope_hdf5"
    TDMS_SCOPE = "tdms_scope"
    CSV = "csv"
    TSV = "tsv"
    NPY = "npy"
    PVA_STACK = "pva_stack"


class Data1DLoading(SchemaModel):
    """How to read one trace file into an x-vs-y array.

    Mirrors ``geecs_data_utils.io.array1d.Data1DConfig`` field for field —
    the reader is handed ``Data1DConfig.model_validate(loading.model_dump())``
    — so the schema package stays pydantic-only while the reader keeps its
    own model.
    """

    data_type: Data1DType = Field(
        ...,
        description=(
            "File format: 'tek_scope_hdf5' or 'tdms_scope' for scope captures, "
            "'csv' / 'tsv' for delimited text, 'npy' for a saved array, "
            "'pva_stack' for one shot of the per-device capture stack a "
            "Bluesky scan writes (traces and spectra alike; the reader takes "
            "its axis from the stack and hands back the trace at its true "
            "length, never the padded one)."
        ),
    )
    trace_index: int = Field(
        0, ge=0, description="Which trace / channel holds the y values (scope formats)."
    )
    x_trace_index: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Which trace holds the x values; unset derives x from the waveform "
            "properties (scope formats)."
        ),
    )
    delimiter: Optional[str] = Field(
        None,
        description="Column delimiter for csv/tsv; unset uses the format's default.",
    )
    x_column: int = Field(0, ge=0, description="Column index of x (text formats).")
    y_column: int = Field(1, ge=0, description="Column index of y (text formats).")
    auxiliary_columns: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Extra named columns to load alongside y, for analyzers that need "
            "them (name -> column index)."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_model_instances(cls, data: Any) -> Any:
        """Accept any pydantic model with the same fields (GEECS-Data-Utils' ``Data1DConfig``)."""
        if isinstance(data, BaseModel) and not isinstance(data, cls):
            return data.model_dump(mode="json")
        return data

    @model_validator(mode="after")
    def validate_auxiliary_columns(self) -> "Data1DLoading":
        """Validate named auxiliary column definitions (mirrors ``Data1DConfig``)."""
        seen: set[int] = set()
        for name, column in self.auxiliary_columns.items():
            if not name.strip():
                raise ValueError("auxiliary column names must be non-empty")
            if column < 0:
                raise ValueError("auxiliary column indices must be non-negative")
            if column in {self.x_column, self.y_column}:
                raise ValueError(
                    "auxiliary column indices must differ from x_column and y_column"
                )
            if column in seen:
                raise ValueError("auxiliary column indices must be unique")
            seen.add(column)
        return self


__all__ = ["Data1DLoading", "Data1DType"]

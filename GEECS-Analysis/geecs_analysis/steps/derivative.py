"""Derivative of a trace with respect to its own coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class DerivativeSpec(StepSpec):
    """Differentiate a trace along its coordinate axis (``numpy.gradient``).

    Second-order central differences in the interior, one-sided at the ends,
    over the actual coordinates, so nonuniform and descending axes are
    differentiated correctly. The signal unit becomes ``<unit>/<axis unit>``
    when both are set, otherwise it is cleared. Nonfinite samples propagate
    to their neighbours as numpy computes them; they are never replaced.
    """

    step: Literal["derivative"] = "derivative"


@step(DerivativeSpec, ndim={1})
def derivative(frame: Frame, spec: DerivativeSpec) -> Frame:
    """Return d(samples)/d(coordinates) on the same axis, keeping provenance.

    Parameters
    ----------
    frame : Frame
        A 1D trace with at least two samples.
    spec : DerivativeSpec
        The (parameterless) step spec.

    Returns
    -------
    Frame
        The derivative on unchanged coordinates, label and shot; the unit is
        the signal unit per axis unit when both are nonempty, else ``""``.

    Raises
    ------
    ValueError
        If the trace has fewer than two samples.
    """
    from dataclasses import replace

    import numpy as np

    if frame.data.size < 2:
        raise ValueError("A derivative needs a trace of at least two samples")
    axis = frame.axes[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.gradient(frame.data, axis.values)
    unit = f"{frame.unit}/{axis.unit}" if frame.unit and axis.unit else ""
    return replace(frame.replace(data=values), unit=unit)

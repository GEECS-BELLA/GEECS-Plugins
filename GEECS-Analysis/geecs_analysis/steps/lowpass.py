"""Zero-phase Butterworth low-pass filtering on sample indices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class LowpassSpec(StepSpec):
    """Zero-phase Butterworth low-pass filter (forward-backward, SOS form).

    Acts on sample indices like every filtering step: the cutoff is a
    fraction of the Nyquist frequency of the sample spacing, not of a
    physical-axis frequency, and nonuniform coordinates are not resampled.
    The edges are padded as SciPy's ``sosfiltfilt`` pads them by default, so
    a trace needs one sample more than that padding: 10 samples for the
    default order 2 (7 for order 1, 16 for order 4). A shorter trace is
    refused rather than filtered, because the filter's edge transient would
    be all of it. The filter is recursive: a
    nonfinite sample makes the whole filtered trace nonfinite, which stays
    visible to the measure rather than being masked.
    """

    step: Literal["lowpass"] = "lowpass"
    order: int = Field(
        2,
        ge=1,
        description="Butterworth filter order (applied twice: forward and back).",
    )
    critical_frequency: float = Field(
        0.1,
        gt=0,
        lt=1,
        description=(
            "Cutoff as a fraction of the Nyquist frequency of the sample "
            "spacing (scipy's Wn), exclusive of 0 and 1."
        ),
    )


@step(LowpassSpec, ndim={1})
def lowpass(frame: Frame, spec: LowpassSpec) -> Frame:
    """Filter the samples with ``sosfiltfilt``, keeping axes, unit and label.

    A trace no longer than ``sosfiltfilt``'s default edge padding is refused
    with a ValueError naming the minimum length.

    Parameters
    ----------
    frame : Frame
        A 1D trace.
    spec : LowpassSpec
        Filter order and normalized cutoff.

    Returns
    -------
    Frame
        The filtered samples on the unchanged axis, with provenance.
    """
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    sos = butter(spec.order, spec.critical_frequency, btype="low", output="sos")
    # scipy's own default padlen for sosfiltfilt; the trace must exceed it.
    zeros = min(int(np.sum(sos[:, 2] == 0)), int(np.sum(sos[:, 5] == 0)))
    minimum = 3 * (2 * len(sos) + 1 - zeros) + 1
    if frame.data.size < minimum:
        raise ValueError(
            f"An order-{spec.order} lowpass needs a trace of at least {minimum} "
            f"samples (got {frame.data.size})"
        )
    return frame.replace(data=sosfiltfilt(sos, frame.data))

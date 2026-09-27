"""GRENOUILLE/FROG pulse retrieval through a host-supplied retriever.

The retrieval itself is Kane's FROG.dll, a 32-bit Windows program that the
core may not start (no processes, paths or config here). The scan host binds
a retriever under the service name ``frog`` — in practice ImageAnalysis's
``FrogDllRetrieval``, which runs the DLL in a 32-bit Python, natively or
under Wine — and this measure calls it with the processed trace and the
spec's parameters, then packages the result as the legacy
``GrenouilleAnalyzer`` did: five bare scalars, the retrieved trace as the
frame with its two projections, and the temporal/spectral lineouts as
extras (written per shot as ``*_retrieved_lineouts.tsv``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from geecs_analysis.registry import MeasureSpec, measure

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement

#: The binding key of the host's retriever.
SERVICE = "frog"

#: The scalar keys, in the legacy analyzer's order.
SCALARS = (
    "temporal_fwhm",
    "spectral_fwhm",
    "frog_error",
    "frog_iterations",
    "tw_per_joule",
)


class FrogSpec(MeasureSpec):
    """Retrieval parameters, passed unchanged to the host's retriever.

    The defaults are the v2 ``frog_retrieval`` analyzer's, so a converted
    diagnostic writes only what it set.
    """

    kind: Literal["frog"] = "frog"
    delt: float = Field(0.85, description="Time-delay step per raw pixel, fs.")
    dellam: float = Field(
        -0.085,
        description="Wavelength step per pixel, nm (negative per the instrument convention).",
    )
    lam0: float = Field(400.0, description="Centre wavelength of the trace, nm.")
    N: Literal[64, 128, 256, 512] = Field(
        512, description="Retrieval grid size: 512, 256, 128 or 64."
    )
    target_error: float = Field(
        0.005, gt=0, description="Stop once the FROG error falls below this."
    )
    max_time_seconds: float = Field(
        5.0,
        gt=0,
        description=(
            "Stop the retrieval loop after this many seconds. A time cap makes "
            "the result depend on the host's speed and load; prefer max_iterations."
        ),
    )
    max_iterations: int = Field(
        1_000_000_000,
        ge=1,
        description="Stop the retrieval loop after this many iterations.",
    )
    noise_subtype: int = Field(
        4, description="Vendor NoiseSubtraction SUBTYPE parameter."
    )
    noise_rad: float = Field(1.0, description="Vendor NoiseSubtraction RAD parameter.")

    def emitted_scalars(self) -> frozenset[str]:
        """Every retrieval emits the same five scalars."""
        return frozenset(SCALARS)


@measure(FrogSpec, ndim={2}, service=SERVICE, sidecar="retrieved_lineouts")
def frog(frame: Frame, spec: FrogSpec, retriever: object) -> Measurement:
    """Retrieve the pulse from the processed trace with the host's retriever.

    ``retriever.retrieve_pulse(trace, **parameters)`` returns an object with
    the DLL's outputs (``FrogRetrievalResult``): the five scalars, the
    ``retrieved_trace`` and the ``time``/``wavelength`` lineouts.
    """
    import numpy as np

    from geecs_data_utils.frames import Axis, Frame
    from geecs_analysis.measurement import Measurement, Projection

    if frame.data.ndim != 2:
        raise ValueError("FROG retrieval requires a 2D trace")
    result = retriever.retrieve_pulse(
        np.asarray(frame.data, dtype=np.float64),
        delt=spec.delt,
        dellam=spec.dellam,
        lam0=spec.lam0,
        N=spec.N,
        target_error=spec.target_error,
        max_time_seconds=spec.max_time_seconds,
        max_iterations=spec.max_iterations,
        noise_subtype=spec.noise_subtype,
        noise_rad=spec.noise_rad,
    )
    scalars = {
        "temporal_fwhm": result.temporal_fwhm,
        "spectral_fwhm": result.spectral_fwhm,
        "frog_error": result.frog_error,
        "frog_iterations": result.num_iterations,
        "tw_per_joule": result.tw_per_joule,
    }
    retrieved = Frame.from_array(
        np.asarray(result.retrieved_trace, dtype=np.float64), shot=frame.shot
    )
    overlays = (
        Projection(
            "projection_x",
            1,
            Frame.from_array(
                retrieved.data.sum(axis=0), axes=(retrieved.axes[1],), shot=frame.shot
            ),
        ),
        Projection(
            "projection_y",
            0,
            Frame.from_array(
                retrieved.data.sum(axis=1), axes=(retrieved.axes[0],), shot=frame.shot
            ),
        ),
    )
    time = Axis(np.asarray(result.time, dtype=np.float64), label="time", unit="fs")
    wave = Axis(
        np.asarray(result.wavelength, dtype=np.float64), label="wavelength", unit="nm"
    )

    def lineout(values: object, axis: Axis) -> Frame:
        return Frame.from_array(
            np.asarray(values, dtype=np.float64), axes=(axis,), shot=frame.shot
        )

    extras = {
        "temporal_intensity": lineout(result.temporal_intensity, time),
        "temporal_phase": lineout(result.temporal_phase, time),
        "spectral_intensity": lineout(result.spectral_intensity, wave),
        "spectral_phase": lineout(result.spectral_phase, wave),
    }
    return Measurement(
        scalars=scalars, frame=retrieved, overlays=overlays, extras=extras
    )

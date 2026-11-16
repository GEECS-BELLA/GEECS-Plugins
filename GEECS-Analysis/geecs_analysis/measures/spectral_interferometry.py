"""Spectral interferometry: relative delay and differential GDD of two pulses.

A thin measure over ``algorithms.spectral_interferometry``: the trace is one
spectrometer shot (x in nm, counts) holding the interferogram of two
collinear pulses. It emits τ0 at the reference wavelength, ΔGDD, their 1σ
uncertainties, the reference used and a reliability flag; every raised
check is a note. Undetermined values are NaN with a note, never zero (a
merged peak is not evidence of zero delay). See the algorithm module for
the sign convention: τ0 ≥ 0 is the later pulse's delay, ΔGDD = late − early,
and the arms cannot be told apart from one spectrum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Mapping, Optional

from pydantic import Field

from geecs_analysis.registry import MeasureSpec, measure

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame

    from geecs_analysis.measurement import Measurement

#: Axis units accepted as wavelength in nm ("px" is the unset default).
_NM_UNITS = {"", "nm", "px"}


class SpectralInterferometrySpec(MeasureSpec):
    """Relative delay τ0 and differential GDD from one spectral interferogram.

    The trace's x axis is wavelength in nm. Every field is optional: unset
    means "infer from the data". The reference wavelength defaults to the
    centre of the fit band and is always emitted. Sign convention: τ0 ≥ 0
    is the later pulse's delay behind the earlier one, ΔGDD = GDD_late −
    GDD_early; one spectrum cannot say which arm is late.
    """

    kind: Literal["spectral_interferometry"] = "spectral_interferometry"
    scalar_docs: ClassVar[Mapping[str, str]] = {
        "tau0_fs": "Relative group delay of the later pulse at the reference wavelength, fs (≥ 0 by convention)",
        "tau0_err_fs": "1σ uncertainty of tau0 from the noise under the sideband, fs",
        "delta_gdd_fs2": "Differential GDD, later pulse minus earlier, fs²",
        "delta_gdd_err_fs2": "1σ uncertainty of delta_gdd, fs²",
        "reference_wavelength_nm": "Wavelength tau0 is reported at, nm",
        "reliable": "1 when no reliability check was raised, else 0 (the notes name them)",
    }
    resolution_nm: Optional[float] = Field(
        None,
        gt=0,
        description="Spectrometer optical resolution (FWHM), nm; enables the resolution check.",
    )
    fit_range_nm: Optional[tuple[float, float]] = Field(
        None,
        description="Wavelength range the phase is fitted over, nm; default where the fringes are strong.",
    )
    reference_wavelength_nm: Optional[float] = Field(
        None,
        gt=0,
        description="Wavelength tau0 is reported at, nm; default the fit band's centre.",
    )
    sideband_window_fs: Optional[tuple[float, float]] = Field(
        None, description="Sideband window in |delay|, fs; default found automatically."
    )
    spectral_region_nm: Optional[tuple[float, float]] = Field(
        None,
        description="Wavelength region transformed, nm; default where the signal is.",
    )
    sideband: Literal["positive", "negative"] = Field(
        "positive",
        description="Sideband to recover from; negative returns the conjugate (−τ0, −ΔGDD).",
    )

    def emitted_scalars(self) -> frozenset[str]:
        """The fit values, their uncertainties, the reference and the flag."""
        return frozenset(self.scalar_docs)


@measure(SpectralInterferometrySpec, ndim={1})
def spectral_interferometry(
    frame: Frame, spec: SpectralInterferometrySpec
) -> Measurement:
    """Analyse the trace as one interferogram; the processed frame is unchanged.

    Extras, when a phase was recovered: the unwrapped interference phase and
    its fit residual over the fit band's wavelengths (ascending).
    """
    import numpy as np
    from geecs_data_utils.frames import Axis, Frame

    from geecs_analysis.algorithms.spectral_interferometry import (
        SpectralInterferometryConfig,
        analyze_interferogram,
        omega_to_wavelength,
    )
    from geecs_analysis.measurement import Measurement

    if frame.data.ndim != 1:
        raise ValueError("spectral_interferometry requires a trace")
    axis = frame.axes[0]
    if (axis.unit or "").strip().lower() not in _NM_UNITS:
        raise ValueError(
            f"spectral_interferometry needs wavelength in nm, not {axis.unit!r}"
        )
    config = SpectralInterferometryConfig(
        **spec.model_dump(exclude={"kind"}, exclude_none=True)
    )
    analysis = analyze_interferogram(
        axis.values, np.asarray(frame.data, dtype=float), config
    )
    result, traces = analysis.result, analysis.traces

    def value(v: Optional[float]) -> float:
        return float("nan") if v is None else float(v)

    extras: dict[str, Frame] = {}
    if traces.band_phase.size:
        nm = omega_to_wavelength(traces.band_omega)
        order = np.argsort(nm)
        wavelength = Axis(nm[order], unit="nm", label=axis.label or "wavelength")
        for name, samples in (
            ("interference_phase", traces.band_phase),
            ("phase_residual", traces.band_phase - traces.fitted_phase),
        ):
            extras[name] = Frame.from_array(
                samples[order],
                axes=(wavelength,),
                shot=frame.shot,
                unit="rad",
                label=name,
            )
    return Measurement(
        scalars={
            "tau0_fs": value(result.tau0_fs),
            "tau0_err_fs": value(result.tau0_err_fs),
            "delta_gdd_fs2": value(result.delta_gdd_fs2),
            "delta_gdd_err_fs2": value(result.delta_gdd_err_fs2),
            "reference_wavelength_nm": result.reference_wavelength_nm,
            "reliable": float(result.reliable),
        },
        frame=frame,
        notes=tuple(result.messages),
        extras=extras,
    )

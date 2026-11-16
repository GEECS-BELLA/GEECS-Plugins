# Spectral interferometry: relative delay and ΔGDD

Two collinear pulses on one spectrometer give a spectral interferogram

$$S(\omega) = |A_1|^2 + |A_2|^2 + 2|A_1A_2|\cos\Phi(\omega),\qquad
\Phi = \phi_\text{late} - \phi_\text{early}.$$

From **one** such spectrum the analysis recovers the relative group delay
τ0 (fs) at a reference wavelength and the differential group-delay
dispersion ΔGDD (fs²) between the two arms. It is meant for two arms that
each have their own grating compressor, where only the differences between
the arms matter.

## The method

1. Wavelength → ω = 2πc/λ (rad/fs), sorted, resampled onto a uniform ω grid
   at the finest pixel spacing, over the region where the signal stands
   above its background (padded 15 %, Tukey-tapered).
2. FFT to the delay domain: a central peak at zero delay (the two spectra)
   and two conjugate sidebands at ±τ.
3. The sideband is the strongest local maximum that dips below half its
   height on the way to zero delay. It must clear the noise floor
   (10×) and reach 1 % of the central peak; anything less is a sidelobe of
   the central peak. The window runs to where the sideband falls to 5 % of
   its peak, padded by 25 %.
4. Inverse FFT of the windowed sideband gives the complex term
   `A1 A2 e^{iΦ(ω)}`. Its phase is unwrapped over the fit band, which by
   default is where `|A1 A2|` exceeds 15 % of its peak. A user fit range
   is held to where `|A1 A2|` exceeds 5 % of its peak, because below that
   the phase is noise and unwraps badly. The band actually used is
   reported in `fit_range_nm`.
5. Weighted least squares
   `Φ(ω) = Φ0 + τ0 (ω − ω0) + ½ ΔGDD (ω − ω0)²`, with the residuals
   weighted by the fringe magnitude `|A1 A2|`. ω0 defaults to the centre
   of the fit band and is always reported.
6. Uncertainties come from the noise actually under the sideband,
   measured in the delays just beyond it, and propagated through the
   window. The window band-limits that noise, so neighbouring ω samples
   are correlated, and the covariance carries that correlation. On
   synthetic data with white noise, the pulls come out at 1.0–1.2.

The λ → ω Jacobian is not applied: it reshapes the envelope, never the phase.

## Sign convention

!!! warning "One spectrum cannot tell the arms apart"
    `cos Φ` is even, so (τ0, ΔGDD) and (−τ0, −ΔGDD) fit the measured
    spectrum equally well. The analysis takes the positive-delay sideband.
    So **τ0 ≥ 0 is the delay of the later pulse behind the earlier one**
    at the reference wavelength, and **ΔGDD = GDD_late − GDD_early**, with
    GDD = dτ/dω (positive when the blue arrives after the red).

    Which compressor arm is the later one has to come from outside the
    spectrum. For example, move one arm's delay stage and watch which way
    τ0 moves. `sideband="negative"` returns the conjugate (−τ0, −ΔGDD).
    The result carries this text in `sign_convention`.

## Reliability flags

Any raised flag makes `reliable` false and adds one sentence to
`messages`. A value that could not be determined is `null` (NaN in the
measure), **never 0**: a merged peak is not evidence of zero delay.

| Flag | Raised when |
|---|---|
| `sideband_not_found` | The shot holds no spectrum (the smoothed peak stands fewer than 10 pixel-noise units above the median), or no sideband clears the central peak and the noise. No τ0 or ΔGDD is reported. |
| `sideband_overlaps_central` | The central peak does not fall to 5 % of the sideband peak before the sideband, or the window reaches into the central peak. |
| `sideband_broadened` | The sideband is more than 3× its unchirped width (the transform of `|A1A2|`), and its edge reaches below half its delay. Strong ΔGDD smears it across the gap that isolates it. |
| `window_sensitive` | Moving any window edge in or out by 10 % of the width changes τ0 by more than max(1 fs, 3σ), or ΔGDD by more than max(20 fs², 10 %, 3σ). |
| `fringes_undersampled` | Fewer than 3 detector pixels per fringe, using the local delay τ0 + ΔGDD(ω − ω0) and the local pixel spacing. |
| `fringes_unresolved` | The stated optical resolution (Gaussian FWHM δω) leaves less than half the fringe contrast, exp(−(τ δω)² / 16 ln 2) < 0.5. Checked only when `resolution_nm` is given. |
| `narrow_bandwidth` | The fit band spans fewer than 20 detector pixels, or σ(ΔGDD) > max(50 fs², 25 % of \|ΔGDD\|). The band gives too little quadratic leverage for the noise present. |

Every threshold is a field of `SpectralInterferometryConfig`.

## Using it

One file, now (from an environment with ScanAnalysis installed). The
command prints the JSON result and exits 1 when any flag is raised:

```bash
python -m scan_analysis.tools.spectral_interferometry ScanNNN_SPEC-PL1-Thorlabs_001.txt --resolution-nm 2 --png ~/si.png
```

Options: `--fit-range NM NM`, `--reference-nm`, `--window-fs FS FS`
(|delay|), `--region NM NM`, `--sideband negative` and `--json PATH`.
Outputs go only into folders that already exist. Nothing is ever written
next to the data.

From Python:

```python
from scan_analysis.tools.spectral_interferometry import analyze_spectrum_file
from geecs_analysis.render.spectral_interferometry import diagnostic_figure

analysis = analyze_spectrum_file(path)            # or analyze_interferogram(wl_nm, counts)
print(analysis.result.model_dump_json(indent=2))
diagnostic_figure(analysis)                       # spectrum | delay domain | phase + residuals
```

Per shot, in a recipe over a `line` input with x in nm:
`measure: {kind: spectral_interferometry}`. It emits `tau0_fs`,
`tau0_err_fs`, `delta_gdd_fs2`, `delta_gdd_err_fs2`,
`reference_wavelength_nm` and `reliable`. Raised flags become notes.

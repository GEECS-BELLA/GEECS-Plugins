"""Least-squares curve fits of one column against another — GEECSplotter's Fit menu.

:func:`fit_curve` fits ``y(x)`` over the finite pairs with one of
:data:`FitModel` and returns a :class:`CurveFit`: the parameters with
standard errors, derived widths, ``r2``, and enough to evaluate the curve
again (:meth:`CurveFit.evaluate`, :meth:`CurveFit.curve`). The fit is
unweighted; the standard errors come from the covariance scaled by the
residual variance (``chi2 / (n - p)``), so they assume equal, unknown
per-point errors.

The models (``x0`` is ``center``, ``C`` is ``offset``; the ``_no_offset``
kinds fix ``C = 0``):

- ``gaussian``: ``A·exp(-(x - x0)² / (2σ²)) + C``; derived ``fwhm = 2√(2 ln 2)·σ``.
- ``supergaussian``: ``A·exp(-(|x - x0| / w)^p) + C`` with the power ``p``
  fitted (``p = 2`` is a Gaussian with ``w = √2·σ``); derived
  ``fwhm = 2w·(ln 2)^(1/p)``.
- ``polynomial``: ``c0 + c1·u + … + cN·u^N`` with ``u = x - x_shift``
  (linear least squares, :func:`numpy.polyfit`), ``N = order``. The shift
  is zero — plain powers of ``x`` — unless the ``x`` values sit far from
  zero against their spread (a timestamp column: :data:`POLY_OFFSET`),
  where powers of ``x`` itself cannot be solved in floating point; then
  it is the middle of the range, and ``CurveFit.x_shift`` says so.

The nonlinear models are always solved about the middle of the ``x``
range and their ``center`` reported in ``x``: the optimizer's
finite-difference steps scale with the parameters, so a centre near
``1e9`` would otherwise be stepped far wider than the peak.

A fit that cannot be made — too few points, too few distinct ``x``, an
optimizer that does not converge — is a :class:`CurveFit` with ``error``
set and no parameters, never an exception, so a page can show why.
Undefined numbers are NaN and explained in ``notes``.
"""

from __future__ import annotations

import math
import warnings
from typing import Annotated, Callable, Literal, Sequence

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

FitModel = Literal[
    "gaussian",
    "gaussian_no_offset",
    "supergaussian",
    "supergaussian_no_offset",
    "polynomial",
]

#: Gaussian FWHM over sigma, ``2·sqrt(2·ln 2)``.
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

#: The bounds on the SuperGaussian power, so ``|x|^p`` stays finite.
POWER_BOUNDS = (0.1, 100.0)

#: A polynomial is fitted about the middle of its ``x`` range when that
#: middle is more than this many half-ranges from zero.
POLY_OFFSET = 100.0

#: A float that reads JSON ``null`` (how pydantic writes NaN) back as NaN,
#: so a result survives its own JSON round trip.
Number = Annotated[float, BeforeValidator(lambda v: math.nan if v is None else v)]


def gaussian(
    x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float = 0.0
) -> np.ndarray:
    """Return ``amplitude·exp(-(x - center)² / (2·sigma²)) + offset``."""
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2)) + offset


def supergaussian(
    x: np.ndarray,
    amplitude: float,
    center: float,
    width: float,
    power: float,
    offset: float = 0.0,
) -> np.ndarray:
    """Return ``amplitude·exp(-(|x - center| / width)^power) + offset``."""
    return amplitude * np.exp(-((np.abs(x - center) / width) ** power)) + offset


def _gaussian_no_offset(x, amplitude, center, sigma):
    return gaussian(x, amplitude, center, sigma)


def _supergaussian_no_offset(x, amplitude, center, width, power):
    return supergaussian(x, amplitude, center, width, power)


#: Parameter names per nonlinear model, in the function's argument order.
_NONLINEAR: dict[str, tuple[Callable[..., np.ndarray], tuple[str, ...]]] = {
    "gaussian": (gaussian, ("amplitude", "center", "sigma", "offset")),
    "gaussian_no_offset": (_gaussian_no_offset, ("amplitude", "center", "sigma")),
    "supergaussian": (
        supergaussian,
        ("amplitude", "center", "width", "power", "offset"),
    ),
    "supergaussian_no_offset": (
        _supergaussian_no_offset,
        ("amplitude", "center", "width", "power"),
    ),
}


class FitParameter(BaseModel):
    """One fitted (or derived) number and its standard error (NaN when undefined)."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: Number
    stderr: Number


class CurveFit(BaseModel):
    """The result of :func:`fit_curve`: parameters, quality and the fitted range.

    ``parameters`` are the fitted numbers in the model's order (for
    ``polynomial``, ``c0`` … ``cN`` by ascending power of
    ``x - x_shift``); ``derived`` are
    computed from them with first-order error propagation (``fwhm``).
    ``error`` is set, and ``parameters`` empty, when no fit was made.
    """

    model_config = ConfigDict(frozen=True)

    model: FitModel
    order: int | None = Field(None, description="The polynomial order; None otherwise.")
    parameters: list[FitParameter] = Field(default_factory=list)
    derived: list[FitParameter] = Field(default_factory=list)
    points: int = Field(0, description="Finite (x, y) pairs fitted.")
    r2: Number = math.nan
    x_min: Number = math.nan
    x_max: Number = math.nan
    x_shift: float = Field(
        0.0, description="The polynomial's origin: its powers are of x - x_shift."
    )
    error: str | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether a fit was made."""
        return self.error is None

    def value(self, name: str) -> float:
        """Return the fitted or derived number called ``name``."""
        for parameter in (*self.parameters, *self.derived):
            if parameter.name == name:
                return parameter.value
        raise KeyError(name)

    def evaluate(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        """Evaluate the fitted curve at ``x``; refuses a fit that was not made."""
        if not self.ok:
            raise ValueError(f"no fit to evaluate: {self.error}")
        xs = np.asarray(x, dtype=np.float64)
        values = [p.value for p in self.parameters]
        if self.model == "polynomial":
            return np.polynomial.polynomial.polyval(xs - self.x_shift, values)
        function, _ = _NONLINEAR[self.model]
        return function(xs, *values)

    def curve(self, points: int = 400) -> tuple[np.ndarray, np.ndarray]:
        """Return ``points`` evenly spaced ``x`` over the fitted range and the curve there."""
        xs = np.linspace(self.x_min, self.x_max, points)
        return xs, self.evaluate(xs)


def _step_profile(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The median ``y`` at each distinct ``x``, in ``x`` order (one value per scan step)."""
    steps, where = np.unique(x, return_inverse=True)
    return steps, np.array([np.median(y[where == i]) for i in range(steps.size)])


def _initial_guess(model: str, x: np.ndarray, y: np.ndarray) -> list[float]:
    """Guess a peak or a dip from the per-step medians (:func:`_step_profile`).

    The ends of the scan (the outer tenth of the steps, at least one at
    each end) stand for the baseline: a peak if the highest step stands
    further above them than the lowest stands below. The baseline is
    then the opposite extreme (zero for ``_no_offset``); the centre is
    the extreme step, and the width the span of the steps past half
    height, as a Gaussian sigma, or a tenth of the ``x`` range when that
    span is zero. Medians, so a few stray shots do not turn a peak into a dip.
    """
    steps, profile = _step_profile(x, y)
    k = max(1, steps.size // 10)
    ends = float(np.median(np.concatenate([profile[:k], profile[-k:]])))
    peak = (profile.max() - ends) >= (ends - profile.min())
    top = int(np.argmax(profile) if peak else np.argmin(profile))
    no_offset = model.endswith("_no_offset")
    base = 0.0 if no_offset else float(profile.min() if peak else profile.max())
    amplitude = float(profile[top]) - base
    past_half = np.abs(profile - base) >= abs(amplitude) / 2.0
    span = float(np.ptp(steps[past_half])) if past_half.any() else 0.0
    sigma = span / FWHM_PER_SIGMA if span > 0 else float(np.ptp(x)) / 10.0
    guess = [amplitude, float(steps[top])]
    if model.startswith("supergaussian"):
        guess += [math.sqrt(2.0) * sigma, 2.0]
    else:
        guess += [sigma]
    return guess if no_offset else [*guess, base]


def _derived(model: str, values: np.ndarray, cov: np.ndarray) -> list[FitParameter]:
    """The FWHM and its first-order error through the covariance."""
    if model.startswith("gaussian"):
        sigma = values[2]
        return [
            FitParameter(
                name="fwhm",
                value=float(FWHM_PER_SIGMA * sigma),
                stderr=float(FWHM_PER_SIGMA * math.sqrt(cov[2, 2])),
            )
        ]
    width, power = values[2], values[3]
    ln2_root = math.log(2.0) ** (1.0 / power)
    fwhm = 2.0 * width * ln2_root
    gradient = np.array([2.0 * ln2_root, -fwhm * math.log(math.log(2.0)) / power**2])
    block = cov[2:4, 2:4]
    return [
        FitParameter(
            name="fwhm",
            value=float(fwhm),
            stderr=float(math.sqrt(gradient @ block @ gradient)),
        )
    ]


def _r2(y: np.ndarray, fitted: np.ndarray) -> float:
    residual = y - fitted
    sst = float(((y - y.mean()) ** 2).sum()) if np.ptp(y) > 0 else 0.0
    return 1.0 - float(residual @ residual) / sst if sst > 0 else float("nan")


def _stderrs(cov: np.ndarray) -> np.ndarray:
    diagonal = np.diag(cov).astype(np.float64)
    finite = np.isfinite(diagonal) & (diagonal >= 0)
    return np.where(finite, np.sqrt(np.where(finite, diagonal, 0.0)), np.nan)


def _notes(
    model: str, values: np.ndarray, errors: np.ndarray, exact: bool
) -> list[str]:
    """Say why standard errors are missing, and when the power is pinned."""
    notes: list[str] = []
    if model.startswith("supergaussian") and any(
        math.isclose(values[3], bound, rel_tol=1e-3) for bound in POWER_BOUNDS
    ):
        notes.append(f"the power sits on its bound {values[3]:.3g}")
    if exact:
        notes.append(
            f"{len(values)} points for {len(values)} parameters: the curve is"
            " exact and has no standard errors"
        )
    elif not np.isfinite(errors).all():
        notes.append("some standard errors are undefined (a degenerate fit)")
    return notes


def _solve_polynomial(
    xs: np.ndarray, ys: np.ndarray, order: int, exact: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients by ascending power and their covariance (NaN when exact)."""
    if exact:
        values = np.polyfit(xs, ys, order)[::-1]
        return values, np.full((order + 1, order + 1), np.nan)
    coefficients, cov = np.polyfit(xs, ys, order, cov=True)
    return coefficients[::-1], cov[::-1, ::-1]


def _solve_nonlinear(
    model: str, xs: np.ndarray, ys: np.ndarray, max_evaluations: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded least squares from :func:`_initial_guess`; raises when it fails."""
    from scipy.optimize import OptimizeWarning, curve_fit

    function, names = _NONLINEAR[model]
    guess = _initial_guess(model, xs, ys)
    low = [-np.inf] * len(names)
    high = [np.inf] * len(names)
    low[2] = 0.0  # sigma or width
    if model.startswith("supergaussian"):
        low[3], high[3] = POWER_BOUNDS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        return curve_fit(
            function,
            xs,
            ys,
            p0=guess,
            bounds=(low, high),
            x_scale="jac",
            max_nfev=max_evaluations,
        )


class _NoFit(Exception):
    """Why a fit could not be made: :func:`fit_curve` returns it as ``error``."""


def _solve(
    model: str,
    xs: np.ndarray,
    ys: np.ndarray,
    order: int,
    exact: bool,
    max_evaluations: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """The values (in ``x``), their covariance, and the polynomial's ``x_shift``.

    Both kinds are solved about the middle of the ``x`` range (the module
    docstring); a nonlinear centre is moved back to ``x``, a polynomial
    keeps the shift only when it was needed (:data:`POLY_OFFSET`).
    """
    low, high = float(xs.min()), float(xs.max())
    middle, half = (low + high) / 2.0, (high - low) / 2.0
    if model == "polynomial":
        shift = middle if abs(middle) > POLY_OFFSET * half else 0.0
        try:
            values, cov = _solve_polynomial(xs - shift, ys, order, exact)
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise _NoFit(f"the polynomial could not be solved: {exc}") from None
        return values, cov, shift
    if np.ptp(ys) == 0:
        raise _NoFit("the y values do not vary: no peak to fit")
    try:
        values, cov = _solve_nonlinear(model, xs - middle, ys, max_evaluations)
    except (RuntimeError, ValueError) as exc:
        raise _NoFit(f"the fit did not converge: {exc}") from None
    values = values.copy()
    values[1] += middle  # the centre, back in x; the covariance is unchanged
    if exact:
        cov = np.full((len(values), len(values)), np.nan)
    return values, cov, 0.0


def fit_curve(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    model: FitModel,
    *,
    order: int = 2,
    max_evaluations: int = 10_000,
) -> CurveFit:
    """Fit ``y(x)`` with ``model`` over the finite pairs (see the module docstring).

    Parameters
    ----------
    x, y : sequence of float
        Paired values; a pair with a nonfinite member is dropped.
    model : FitModel
        Which curve.
    order : int
        The polynomial order (``polynomial`` only; 0 or more).
    max_evaluations : int
        The nonlinear optimizer's budget of function evaluations.

    Returns
    -------
    CurveFit
        The fit, or one with ``error`` set saying why none was made. A fit
        needs at least as many distinct ``x`` as parameters; with exactly
        that many points the curve is exact and the standard errors NaN.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.shape != ys.shape or xs.ndim != 1:
        raise ValueError(
            f"x and y must be 1-D and paired, got {xs.shape} and {ys.shape}"
        )
    if model == "polynomial" and order < 0:
        raise ValueError(f"the polynomial order must be 0 or more, got {order}")
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    n = int(xs.size)
    if model == "polynomial":
        names = tuple(f"c{k}" for k in range(order + 1))
    else:
        names = _NONLINEAR[model][1]
    head = {
        "model": model,
        "order": order if model == "polynomial" else None,
        "points": n,
    }
    distinct = int(np.unique(xs).size)
    if distinct < len(names):
        return CurveFit(
            **head,
            error=(
                f"{len(names)} parameters need at least {len(names)} distinct x"
                f" values; the data has {distinct}"
            ),
        )
    low, high = float(xs.min()), float(xs.max())
    head |= {"x_min": low, "x_max": high}
    exact = n == len(names)
    try:
        values, cov, shift = _solve(model, xs, ys, order, exact, max_evaluations)
    except _NoFit as why:
        return CurveFit(**head, error=str(why))
    head["x_shift"] = shift
    derived = [] if model == "polynomial" else _derived(model, values, cov)
    errors = _stderrs(cov)
    notes = _notes(model, values, errors, exact)
    parameters = [
        FitParameter(name=name, value=float(v), stderr=float(e))
        for name, v, e in zip(names, values, errors)
    ]
    result = CurveFit(**head, parameters=parameters, derived=derived, notes=notes)
    return result.model_copy(update={"r2": _r2(ys, result.evaluate(xs))})

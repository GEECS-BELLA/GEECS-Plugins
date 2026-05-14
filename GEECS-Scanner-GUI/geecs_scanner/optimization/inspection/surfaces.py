"""2D slice evaluation of GP surrogate posteriors and acquisition functions.

These helpers all produce 2D slices of an N-D model for visualization. The
underlying generator/model is fully N-D — we just project to a chosen pair
of variables (with the rest held at a reference point) so the result can be
rendered as a contour plot.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from xopt.vocs import VOCS


def evaluate_model_on_grid(
    model,
    vocs: VOCS,
    var_names: Tuple[str, str],
    reference_point: Optional[dict] = None,
    n_grid: int = 50,
    objective_name: Optional[str] = None,
):
    """Evaluate posterior mean and σ over a 2D slice of the input space.

    Returns ``(Xg, Yg, mean, sigma)`` where ``Xg, Yg`` are the meshgrids
    over the two slice variables and ``mean, sigma`` are 2D arrays of the
    posterior of the named objective. Non-slice variables are held at
    ``reference_point[name]``; missing entries default to the midpoint of
    the variable's bounds.

    Parameters
    ----------
    model
        Trained GP model (typically a botorch ``ModelListGP``).
    vocs
        VOCS describing the input space and outputs.
    var_names
        ``(var_x, var_y)`` — the two variables to slice over.
    reference_point
        Mapping ``{var_name: value}`` for non-slice variables. Missing
        names fall back to the midpoint of the bounds.
    n_grid
        Number of grid points along each axis.
    objective_name
        Which objective to render. Defaults to ``vocs.objective_names[0]``.
    """
    reference_point = dict(reference_point or {})
    objective_name = objective_name or vocs.objective_names[0]

    vx, vy = var_names
    x = np.linspace(*vocs.variables[vx], n_grid)
    y = np.linspace(*vocs.variables[vy], n_grid)
    Xg, Yg = np.meshgrid(x, y)

    input_names = vocs.variable_names
    rows = []
    for xi, yi in zip(Xg.ravel(), Yg.ravel()):
        pt = dict(reference_point)
        pt[vx] = xi
        pt[vy] = yi
        for name in input_names:
            if name not in pt:
                lo, hi = vocs.variables[name]
                pt[name] = 0.5 * (lo + hi)
        rows.append([pt[name] for name in input_names])

    X_tensor = torch.tensor(rows, dtype=torch.double)
    with torch.no_grad():
        post = model.posterior(X_tensor)
        mean = post.mean.cpu().numpy()
        var = post.variance.cpu().numpy()

    output_names = vocs.objective_names + vocs.constraint_names
    obj_idx = output_names.index(objective_name)
    Z = mean[:, obj_idx].reshape(n_grid, n_grid)
    Sigma = np.sqrt(np.clip(var[:, obj_idx], 0, None)).reshape(n_grid, n_grid)
    return Xg, Yg, Z, Sigma


def acquisition_surface(
    generator,
    vocs: VOCS,
    var_names: Tuple[str, str],
    reference_point: Optional[dict] = None,
    n_grid: int = 50,
):
    """Evaluate the generator's acquisition function over a 2D slice.

    Acquisition reflects the *policy* (EI / UCB / TuRBO-restricted), not
    the model — given the same model and data, the posterior is identical
    across acquisition choices but the acquisition surface differs.

    Per-generator scale varies wildly (EI is improvement-scaled, UCB is in
    objective units, TuRBO masks outside the trust region) so this function
    does not normalize; the caller picks its own colour scale.

    Same slicing semantics as :func:`evaluate_model_on_grid`.
    """
    reference_point = dict(reference_point or {})
    vx, vy = var_names
    x = np.linspace(*vocs.variables[vx], n_grid)
    y = np.linspace(*vocs.variables[vy], n_grid)
    Xg, Yg = np.meshgrid(x, y)

    input_names = vocs.variable_names
    rows = []
    for xi, yi in zip(Xg.ravel(), Yg.ravel()):
        pt = dict(reference_point)
        pt[vx] = xi
        pt[vy] = yi
        for name in input_names:
            if name not in pt:
                lo, hi = vocs.variables[name]
                pt[name] = 0.5 * (lo + hi)
        rows.append([pt[name] for name in input_names])
    X_tensor = torch.tensor(rows, dtype=torch.double)

    # xopt's BayesianGenerator usually exposes get_acquisition(model);
    # some versions only have the underscore form. Try both.
    try:
        acq_fn = generator.get_acquisition(generator.model)
    except AttributeError:
        acq_fn = generator._get_acquisition(generator.model)

    # botorch acquisition convention: input shape (batch, q, d) with q=1
    X_acq = X_tensor.unsqueeze(1)
    with torch.no_grad():
        A = acq_fn(X_acq).cpu().numpy().reshape(n_grid, n_grid)
    return Xg, Yg, A

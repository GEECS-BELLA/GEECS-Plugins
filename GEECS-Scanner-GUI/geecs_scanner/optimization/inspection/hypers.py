"""Read GP hyperparameters out of a fitted xopt generator.

In current xopt versions the model is wrapped as a botorch ``ModelListGP``
(one sub-model per output, even single-objective). These helpers transparently
unwrap that, and convert the standardised-Y noise back into raw-Y units so
the values are comparable to the spread of your actual data.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def gp_hypers(generator) -> Tuple[float, np.ndarray, float]:
    """Return ``(noise_sigma_raw, lengthscales, y_stdv)`` from a fitted GP.

    - ``noise_sigma_raw`` is the GP's learned observation noise σ in the
      original Y units (un-standardised).
    - ``lengthscales`` is a 1D array, one entry per input dimension, in
      normalised input units.
    - ``y_stdv`` is the standardisation factor the model applied to Y;
      noise σ in standardised-Y units is ``noise_sigma_raw / y_stdv``.
    """
    model = generator.model
    sub = model.models[0] if hasattr(model, "models") else model
    noise_var_z = float(sub.likelihood.noise.flatten().detach().cpu().numpy()[0])
    lengthscales = sub.covar_module.lengthscale.flatten().detach().cpu().numpy()
    y_stdv = float(sub.outcome_transform.stdvs.flatten().detach().cpu().numpy()[0])
    noise_sigma_raw = float(np.sqrt(noise_var_z) * y_stdv)
    return noise_sigma_raw, lengthscales, y_stdv


def gp_summary(generator, label: str = "GP") -> None:
    """Pretty-print the GP hyperparameters for a fitted generator.

    Convenience wrapper around :func:`gp_hypers` that prints the noise in
    both standardised- and raw-Y units, the per-dimension lengthscales,
    and the Y standardisation factor.
    """
    model = generator.model
    submodels = getattr(model, "models", None) or [model]
    print(f"{label}:")
    for i, sub in enumerate(submodels):
        tag = f"[out {i}] " if len(submodels) > 1 else ""
        noise_z = sub.likelihood.noise.flatten().detach().cpu().numpy()
        ls = sub.covar_module.lengthscale.flatten().detach().cpu().numpy()
        print(f"  {tag}noise σ² (in standardised-Y units):  {noise_z}")
        print(f"  {tag}lengthscale (normalised inputs):     {ls}")
        if hasattr(sub, "outcome_transform"):
            stdvs = sub.outcome_transform.stdvs.flatten().detach().cpu().numpy()
            print(f"  {tag}Y standardisation stdv:              {stdvs}")
            print(
                f"  {tag}noise σ (raw-Y units):               "
                f"{np.sqrt(noise_z) * stdvs}"
            )

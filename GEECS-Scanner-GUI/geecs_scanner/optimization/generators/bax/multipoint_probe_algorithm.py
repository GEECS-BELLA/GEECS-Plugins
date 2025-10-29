"""Utilities for BAX multipoint probe algorithms.

These helpers mesh over one or more control variables, sweep a measurement
variable along a virtual probe grid, and delegate the virtual objective
calculation to a user-provided callable.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor
from pydantic import BaseModel, Field, model_validator

from botorch.models import ModelListGP
from botorch.models.model import Model
from xopt.generators.bayesian.bax.algorithms import GridOptimize
from xopt.generators.bayesian.bax_generator import BaxGenerator
from xopt.vocs import VOCS


def _ls_slope(x: Tensor, y: Tensor) -> Tensor:
    """Return the least-squares slope of ``y`` with respect to ``x``."""
    p = x.shape[-1]
    ones = torch.ones(p, dtype=x.dtype, device=x.device)
    a = torch.stack((x, ones), dim=-1)
    pinv = torch.linalg.pinv(a.transpose(-1, -2) @ a) @ a.transpose(-1, -2)
    coeffs = torch.einsum("qp,...p->...q", pinv, y)
    return coeffs[..., 0]


def slope_virtual_objective(grid_v: Tensor, samples: Tensor) -> Tensor:
    """Return absolute slope magnitudes per observable."""
    slopes = []
    for obs_idx in range(samples.shape[-1]):
        slopes.append(torch.abs(_ls_slope(grid_v, samples[..., obs_idx])))
    return torch.stack(slopes, dim=-1)


def l2_slope_virtual_objective(grid_v: Tensor, samples: Tensor) -> Tensor:
    """Return the L2 norm of slope magnitudes across observables."""
    slopes = slope_virtual_objective(grid_v, samples)
    norm = torch.linalg.norm(slopes, dim=-1, keepdim=True)
    return norm.expand_as(slopes)


class MultipointProbeConfig(BaseModel):
    """Configuration for multipoint BAX probing."""

    control_names: Sequence[str] = Field(..., description="Variables optimised by BAX")
    measurement_name: str = Field(
        ..., description="Variable swept for the virtual probe"
    )
    observable_names: Sequence[str] = Field(
        default=("x_CoM",),
        description="Observable names consumed by the virtual objective",
    )

    probe_nominal: Optional[float] = Field(
        default=None, description="Nominal measurement value; defaults to mid-range"
    )
    probe_grid_absolute: Optional[Sequence[float]] = Field(
        default=None,
        description="Absolute offsets from nominal (e.g. [-0.2, 0.0, 0.2])",
    )
    probe_grid_fraction: Optional[Sequence[float]] = Field(
        default=None, description="Fractional offsets of the measurement span"
    )

    n_control_mesh: int = Field(
        33, ge=3, description="Mesh points per control dimension"
    )
    mesh_measurement: bool = Field(
        default=False, description="Include measurement axis in the mesh grid"
    )
    n_measurement_mesh: int = Field(
        11, ge=3, description="Mesh points for measurement axis when meshed"
    )

    n_monte_carlo_samples: int = Field(128, ge=16, description="Monte Carlo samples")
    minimize: bool = Field(
        True, description="Minimise (True) or maximise the virtual objective"
    )
    use_low_noise_prior: bool = Field(False, description="Toggle low-noise GP prior")
    algorithm_results_file: str = Field(
        "bax_probe_results", description="Name prefix for saved algorithm results"
    )

    @model_validator(mode="before")
    def _coerce_observable_names(cls, values):
        if isinstance(values, dict):
            data = dict(values)
            if "observable_names" not in data and "observable_name" in data:
                data["observable_names"] = [data.pop("observable_name")]
            elif "observable_names" in data and isinstance(
                data["observable_names"], str
            ):
                data["observable_names"] = [data["observable_names"]]
            values = data
        return values

    @model_validator(mode="after")
    def _validate_probe_grid(
        cls, values: "MultipointProbeConfig"
    ) -> "MultipointProbeConfig":
        if (
            values.probe_grid_absolute is not None
            and values.probe_grid_fraction is not None
        ):
            raise ValueError(
                "Specify either probe_grid_absolute or probe_grid_fraction, not both."
            )
        return values


class MultipointProbeAlgorithm(GridOptimize):
    """Generic multipoint probe algorithm for BAX (virtual objective supplied externally).

    Parameters
    ----------
    vocs : xopt.vocs.VOCS
        Optimisation problem definition.
    cfg : MultipointProbeConfig
        Configuration describing controls, measurement, and probe grid.
    virtual_objective : Callable[[Tensor, Tensor], Tensor]
        Function mapping ``(grid_v, samples)`` to the desired virtual objective.
    """

    observable_names_ordered: List[str] = Field(default_factory=list)

    def __init__(
        self,
        vocs: VOCS,
        cfg: MultipointProbeConfig,
        virtual_objective: Callable[[Tensor, Tensor], Tensor],
    ):
        self._validate_names(vocs, cfg)

        if not callable(virtual_objective):
            raise TypeError("virtual_objective must be callable")

        measurement_bounds = vocs.variables[cfg.measurement_name]
        lo, hi = float(measurement_bounds[0]), float(measurement_bounds[1])
        mid = 0.5 * (lo + hi)
        nominal = cfg.probe_nominal if cfg.probe_nominal is not None else mid

        if cfg.probe_grid_absolute is not None:
            offsets = torch.tensor(list(cfg.probe_grid_absolute), dtype=torch.double)
        elif cfg.probe_grid_fraction is not None:
            span = hi - lo
            offsets = torch.tensor(
                [float(frac) * span for frac in cfg.probe_grid_fraction],
                dtype=torch.double,
            )
        else:
            span = hi - lo
            offsets = torch.tensor([-0.05 * span, 0.0, 0.05 * span], dtype=torch.double)

        probe_grid = torch.clamp(nominal + offsets, min=lo, max=hi).unique(sorted=True)
        if probe_grid.numel() < 2:
            raise ValueError("Probe grid must contain at least two distinct points.")

        super().__init__(n_mesh_points=cfg.n_control_mesh)

        self._virtual_objective: Callable[[Tensor, Tensor], Tensor] = virtual_objective
        self.observable_names_ordered = list(cfg.observable_names)
        self._control_names = list(cfg.control_names)
        self._measurement_name = cfg.measurement_name
        self._probe_grid = probe_grid
        self._vocs_variables = list(vocs.variable_names)
        self._control_indices = [
            self._vocs_variables.index(name) for name in self._control_names
        ]
        self._measurement_index = self._vocs_variables.index(self._measurement_name)

        self._mesh_measurement = cfg.mesh_measurement
        self._n_measurement_mesh = cfg.n_measurement_mesh
        self.minimize = cfg.minimize

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _validate_names(vocs: VOCS, cfg: MultipointProbeConfig) -> None:
        missing_controls = [
            name for name in cfg.control_names if name not in vocs.variable_names
        ]
        if missing_controls:
            raise ValueError(f"Control variables not in VOCS: {missing_controls}")
        if cfg.measurement_name not in vocs.variable_names:
            raise ValueError(
                f"Measurement variable '{cfg.measurement_name}' not in VOCS variables."
            )
        missing_observables = [
            name for name in cfg.observable_names if name not in vocs.observable_names
        ]
        if missing_observables:
            raise ValueError(
                f"Observable names not in VOCS observables: {missing_observables}"
            )

    # ------------------------------------------------------------------- mesh

    def create_mesh(self, bounds: Tensor) -> Tensor:
        """Return a mesh over control variables (optionally measurement).

        Parameters
        ----------
        bounds : torch.Tensor
            Lower/upper bounds with shape ``[2, n_variables]``.

        Returns
        -------
        torch.Tensor
            Mesh points with one column per VOCS variable.
        """
        control_bounds = torch.stack(
            [bounds[:, idx] for idx in self._control_indices], dim=1
        )
        linspaces = [
            torch.linspace(float(lo), float(hi), self.n_mesh_points, dtype=torch.double)
            for lo, hi in control_bounds.T
        ]

        if self._mesh_measurement:
            meas_lo, meas_hi = bounds[:, self._measurement_index]
            linspaces.append(
                torch.linspace(
                    float(meas_lo),
                    float(meas_hi),
                    self._n_measurement_mesh,
                    dtype=torch.double,
                )
            )
            mesh = torch.meshgrid(*linspaces, indexing="ij")
            grid = torch.stack(mesh).flatten(start_dim=1).T

            full = torch.zeros(
                (grid.shape[0], len(self._vocs_variables)), dtype=torch.double
            )
            for axis, name in enumerate(self._control_names):
                full[:, self._vocs_variables.index(name)] = grid[:, axis]
            full[:, self._measurement_index] = grid[:, -1]
        else:
            mesh = torch.meshgrid(*linspaces, indexing="ij")
            grid = torch.stack(mesh).flatten(start_dim=1).T
            full = torch.zeros(
                (grid.shape[0], len(self._vocs_variables)), dtype=torch.double
            )
            for axis, name in enumerate(self._control_names):
                full[:, self._vocs_variables.index(name)] = grid[:, axis]
            meas_lo, meas_hi = bounds[:, self._measurement_index]
            full[:, self._measurement_index] = 0.5 * (meas_lo + meas_hi)

        return full

    # ------------------------------------------------------- virtual objective

    def evaluate_virtual_objective(
        self,
        model: Model,
        x: Tensor,
        bounds: Tensor,
        n_samples: int,
        tkwargs: Optional[Dict] = None,
    ) -> Tensor:
        """Return samples of ``|d observable / d measurement|``.

        Parameters
        ----------
        model : botorch.models.model.Model
            GP model supplied by Xopt.
        x : torch.Tensor
            Mesh points produced by :meth:`create_mesh`.
        bounds : torch.Tensor
            Optimisation bounds (unused, present for API compatibility).
        n_samples : int
            Number of Monte-Carlo samples to draw.
        tkwargs : dict, optional
            Additional tensor keyword arguments (unused).

        Returns
        -------
        torch.Tensor
            Virtual objective samples with shape
            ``[n_samples, n_points, k]`` where ``k`` is determined by the objective.
        """
        if isinstance(model, ModelListGP):
            models = model.models
        else:
            models = [model]

        if len(models) != len(self.observable_names_ordered):
            raise ValueError(
                "Number of GP sub-models does not match observable_names_ordered."
            )

        try:
            param = next(models[0].parameters())
            device, dtype = param.device, param.dtype
        except StopIteration:
            device, dtype = x.device, x.dtype

        grid_v = self._probe_grid.to(device=device, dtype=dtype)
        points = x.to(device=device, dtype=dtype)
        num_probe = grid_v.numel()
        num_points = points.shape[0]

        expanded = points.repeat_interleave(num_probe, dim=0).clone()
        for i in range(num_points):
            start = i * num_probe
            expanded[start : start + num_probe, self._measurement_index] = grid_v

        sample_list = []
        for sub_model in models:
            with torch.no_grad():
                posterior = sub_model.posterior(expanded)
                draws = posterior.rsample(torch.Size([n_samples]))

            draws = draws.squeeze(-1).view(n_samples, num_points, num_probe)
            sample_list.append(draws)

        stacked = torch.stack(
            sample_list, dim=-1
        )  # [n_samples, n_points, probe, n_obs]
        result = self._virtual_objective(grid_v, stacked)
        if result.ndim == 2:
            result = result.unsqueeze(-1)
        return result


class MultipointBAXGenerator(BaxGenerator):
    """BAX generator variant that allows single-objective VOCS."""

    supports_single_objective: bool = True


def make_multipoint_bax_alignment(vocs: VOCS, overrides: Dict) -> BaxGenerator:
    """Factory function for the slope-minimisation variant."""
    cfg = MultipointProbeConfig.model_validate(
        overrides.get("multipoint_bax_alignment", {})
    )
    algorithm = MultipointProbeAlgorithm(vocs, cfg, slope_virtual_objective)
    generator = MultipointBAXGenerator(
        vocs=vocs,
        algorithm=algorithm,
        algorithm_results_file=cfg.algorithm_results_file,
    )
    generator.n_monte_carlo_samples = cfg.n_monte_carlo_samples
    generator.gp_constructor.use_low_noise_prior = cfg.use_low_noise_prior
    return generator


def make_multipoint_bax_alignment_l2(vocs: VOCS, overrides: Dict) -> BaxGenerator:
    """Factory function for the L2 slope minimisation variant."""
    cfg = MultipointProbeConfig.model_validate(
        overrides.get("multipoint_bax_alignment_l2", {})
    )
    algorithm = MultipointProbeAlgorithm(vocs, cfg, l2_slope_virtual_objective)
    generator = MultipointBAXGenerator(
        vocs=vocs,
        algorithm=algorithm,
        algorithm_results_file=cfg.algorithm_results_file,
    )
    generator.n_monte_carlo_samples = cfg.n_monte_carlo_samples
    generator.gp_constructor.use_low_noise_prior = cfg.use_low_noise_prior
    return generator

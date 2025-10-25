"""GEECS Scanner multipoint BAX alignment generator."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
from pydantic import BaseModel, Field

from xopt.vocs import VOCS
from xopt.generators.bayesian.bax_generator import BaxGenerator
from xopt.generators.bayesian.bax.algorithms import GridOptimize
from botorch.models.model import Model

# ---------- helpers ----------------------------------------------------------


def _ls_slope(xp: Tensor, yp: Tensor) -> Tensor:
    """Compute the least-squares slope of ``yp`` with respect to ``xp``.

    Parameters
    ----------
    xp : torch.Tensor
        Probe grid tensor with shape ``[P]``.
    yp : torch.Tensor
        Tensor of values along the probe grid with shape ``[..., P]``.

    Returns
    -------
    torch.Tensor
        Tensor of slope values with shape ``yp[..., 0].shape``.
    """
    P = xp.shape[-1]
    ones = torch.ones(P, dtype=xp.dtype, device=xp.device)
    A = torch.stack([xp, ones], dim=-1)  # [P,2]
    At = A.transpose(-1, -2)  # [2,P]
    pinv = torch.linalg.pinv(At @ A) @ At  # [2,P]
    coeffs = torch.einsum("qp,...p->...q", pinv, yp)  # [...,2]
    return coeffs[..., 0]  # slope


# ---------- config -----------------------------------------------------------


class MultipointBAXAlignmentConfig(BaseModel):
    """Configuration model for the multipoint BAX alignment generator.

    The alignment procedure treats the correctors as control variables ``u``,
    the quadrupole current as the measurement variable ``v``, and an observable
    ``y(u, v)`` (for example the beam centroid ``x_CoM``) returned by the evaluator.

    A virtual objective ``g(u) = | d y(u, v) / d v |`` is constructed by querying
    the Gaussian Process along a virtual probe grid near a nominal value of
    ``v``. The probe grid can be specified explicitly via absolute offsets or as
    fractions of the measurement range; if omitted, a default symmetric
    three-point grid is used.
    """

    # names
    control_names: Sequence[str] = Field(
        ..., description="control variable names (correctors)"
    )
    measurement_name: str = Field(
        ..., description="measurement variable name (quadrupole)"
    )
    observable_name: str = Field(
        "x_CoM", description="observable key in VOCS.observables"
    )

    # virtual probe grid
    probe_nominal: Optional[float] = Field(
        default=None, description="nominal v; if None, use mid-range of VOCS bounds"
    )
    probe_grid_absolute: Optional[Sequence[float]] = Field(
        default=None,
        description="absolute offsets (A) relative to nominal (e.g. [-0.2, 0.0, +0.2])",
    )
    probe_grid_fraction: Optional[Sequence[float]] = Field(
        default=None, description="fraction of v-range (e.g. [-0.05, 0.0, +0.05])"
    )

    # mesh & MC
    n_control_mesh: int = Field(33, ge=3, description="mesh points per control dim")
    n_monte_carlo_samples: int = Field(128, ge=16)
    minimize: bool = True  # BAX will minimize g(u)
    use_low_noise_prior: bool = False

    algorithm_results_file: str = Field("bax_algo_res", description="per shot logs")


# ---------- Algorithm --------------------------------------------------------


class MultipointBAXAlignmentAlgorithm(GridOptimize):
    """Specialized BAX algorithm for multipoint alignment.

    This algorithm customizes the standard BAX grid optimization by meshing only
    over the control variables and evaluating a virtual objective that probes the
    surrogate model along a virtual measurement grid. The absolute slope of the
    observable with respect to the measurement variable becomes the virtual
    objective evaluated by BAX.
    """

    # expose the observable to BAX (must match VOCS output name)
    observable_names_ordered: List[str] = Field(default_factory=list)

    # config & bookkeeping
    control_names: List[str] = Field(default_factory=list)
    measurement_name: str = ""
    observable_index: int = 0  # index of observable in the GP model
    probe_grid_abs: Tensor = Field(
        default_factory=lambda: torch.zeros(0, dtype=torch.double)
    )  # absolute v values used for virtual VO (shape [P])
    all_vars: List[str] = Field(default_factory=list)  # VOCS variable order
    n_control_mesh: int = 0
    minimize: bool = True

    def __init__(self, vocs: VOCS, cfg: MultipointBAXAlignmentConfig):
        # Build probe grid (absolute values) once
        v_lo_raw, v_hi_raw = vocs.variables[cfg.measurement_name]
        v_lo, v_hi = float(v_lo_raw), float(v_hi_raw)
        v_mid = 0.5 * (v_lo + v_hi)
        v_nom = cfg.probe_nominal if cfg.probe_nominal is not None else v_mid

        if cfg.probe_grid_absolute is not None:
            offsets = torch.tensor(list(cfg.probe_grid_absolute), dtype=torch.double)
        elif cfg.probe_grid_fraction is not None:
            span = v_hi - v_lo
            offsets = torch.tensor(
                [float(fr) * span for fr in cfg.probe_grid_fraction], dtype=torch.double
            )
        else:
            # default small symmetric grid ~5% of span
            span = v_hi - v_lo
            offsets = torch.tensor([-0.05 * span, 0.0, 0.05 * span], dtype=torch.double)

        grid = torch.clamp(v_nom + offsets, min=v_lo, max=v_hi)
        grid, _ = torch.sort(torch.unique(grid))
        if grid.numel() < 2:
            raise ValueError("Virtual probe grid must have at least 2 distinct points.")

        all_vars = list(vocs.variable_names)
        control_names = list(cfg.control_names)
        n_control_mesh = int(cfg.n_control_mesh)
        minimize = bool(cfg.minimize)

        # initialize base class first (required for Pydantic models)
        super().__init__(n_mesh_points=n_control_mesh)

        # store fields
        self.observable_names_ordered = [cfg.observable_name]
        self.control_names = control_names
        self.measurement_name = cfg.measurement_name
        self.probe_grid_abs = grid
        self.all_vars = all_vars
        self.n_control_mesh = n_control_mesh
        self.minimize = minimize

        # Compute the observable index in the GP model
        # BAX models observables, so we look up the index in the observable list
        try:
            self.observable_index = list(vocs.observable_names).index(
                cfg.observable_name
            )
        except ValueError:
            raise ValueError(
                f"Observable '{cfg.observable_name}' not found in VOCS.observables: "
                f"{list(vocs.observable_names)}"
            )

    # ---------- mesh over CONTROL dims only, return full-D points ------------
    def create_mesh(self, bounds: Tensor) -> Tensor:
        """Build a Cartesian mesh over the control dimensions only.

        Parameters
        ----------
        bounds : torch.Tensor
            Tensor of shape ``[2, D_total]`` containing lower and upper bounds for
            each VOCS variable.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``[N, D_total]`` containing the mesh points. Control
            variables follow a Cartesian grid, and the measurement variable is set
            to its mid-range placeholder value.
        """
        # indices
        ctrl_idxs = [self.all_vars.index(n) for n in self.control_names]
        meas_idx = self.all_vars.index(self.measurement_name)

        # extract control bounds
        ctrl_bounds = torch.stack([bounds[:, i] for i in ctrl_idxs], dim=1)  # [2, D_u]

        # make control mesh
        dim_u = ctrl_bounds.shape[1]
        linspaces = [
            torch.linspace(
                float(ctrl_bounds[0, d]),
                float(ctrl_bounds[1, d]),
                self.n_mesh_points,
                dtype=torch.double,
            )
            for d in range(dim_u)
        ]
        mesh = torch.meshgrid(*linspaces, indexing="ij")
        U = torch.stack(mesh).flatten(start_dim=1).T  # [N, D_u]

        # pack to full-D points with measurement at nominal mid (not used in VO)
        full = torch.zeros((U.shape[0], len(self.all_vars)), dtype=torch.double)
        for j, name in enumerate(self.control_names):
            full[:, self.all_vars.index(name)] = U[:, j]

        # set measurement to mid-range (placeholder; VO uses its own grid)
        # (we still need a full-D tensor for Xopt interface consistency)
        # choose mid-range to keep inside bounds
        v_lo, v_hi = bounds[:, meas_idx]
        full[:, meas_idx] = 0.5 * (v_lo + v_hi)

        return full  # [N, D_total]

    # ---------- virtual objective: |slope_v y(u,v)| --------------------------
    def evaluate_virtual_objective(
        self,
        model: Model,
        x: Tensor,  # [N, D_total] (controls populated; measurement placeholder)
        bounds: Tensor,  # [2, D_total]
        n_samples: int,
        tkwargs: dict | None = None,
    ) -> Tensor:
        """
        Calculate virtual objective.

        Virtual objective g(u) = | d/dv y(u, v) |, computed by probing the GP along
        the measurement axis at a fixed control u (per row of `x`).

        Returns
        -------
        Tensor
            Shape [n_samples, N, 1] of absolute slope samples.
        """
        # ---- 1) Subset to the single observable GP
        # BAX models observables, so we use the observable_index to access the correct GP
        from botorch.models import ModelListGP

        if isinstance(model, ModelListGP):
            # Multiple observables - get the model for our observable
            if self.observable_index >= len(model.models):
                raise RuntimeError(
                    f"Observable index {self.observable_index} out of range. "
                    f"Model has {len(model.models)} outputs. "
                    f"Observable: {self.observable_names_ordered[0]}"
                )
            single_model = model.models[self.observable_index]
        else:
            # Single observable - the model IS the observable model
            single_model = model

        # ---- 2) Shapes & dtype/device
        N = x.shape[0]
        P = int(self.probe_grid_abs.numel())

        # Use model's dtype/device for consistency with input transforms
        try:
            base = next(single_model.parameters())
            device = base.device
            dtype = base.dtype
        except StopIteration:
            device = x.device
            dtype = x.dtype

        # ---- 3) Build expanded input [N*P, D_total] sweeping the measurement dim over probe grid
        meas_idx = self.all_vars.index(self.measurement_name)

        X_full = (
            x.to(device=device, dtype=dtype).clone().repeat_interleave(P, dim=0)
        )  # [N*P, D_total]
        grid_v = self.probe_grid_abs.to(device=device, dtype=dtype)  # [P]
        for i in range(N):
            s = i * P
            X_full[s : s + P, meas_idx] = grid_v

        # ---- 4) Sample the observable along v and compute least-squares slope wrt v
        with torch.no_grad():
            post = single_model.posterior(X_full)  # posterior over y(u_i, v_p)
            Y = post.rsample(torch.Size([n_samples]))  # [S, N*P, 1] or [S, N*P]
        Y = Y.squeeze(-1).view(n_samples, N, P)  # [S, N, P]

        slopes = _ls_slope(grid_v, Y)  # [S, N]
        virt = torch.abs(slopes).unsqueeze(-1)  # [S, N, 1]
        return virt


# ---------- custom generator wrapper ----------------------------------------


class MultipointBAXGenerator(BaxGenerator):
    """Variant of ``BaxGenerator`` that allows single-objective optimisation."""

    supports_single_objective: bool = True


# ---------- factory ----------------------------------------------------------


def make_bax_multipoint_alignment_generator(
    vocs: VOCS, overrides: Dict
) -> BaxGenerator:
    """Create a BAX generator configured for multipoint alignment.

    Parameters
    ----------
    vocs : xopt.vocs.VOCS
        VOCS specification describing the optimization problem.
    overrides : dict
        Dictionary containing configuration overrides under the key
        ``"multipoint_bax_alignment"``.

    Returns
    -------
    xopt.generators.bayesian.bax_generator.BaxGenerator
        Configured BAX generator instance.

    Raises
    ------
    ValueError
        If any of the specified control names, measurement name, or observable
        name are not present in the VOCS.
    """
    cfg = MultipointBAXAlignmentConfig.model_validate(
        overrides.get("multipoint_bax_alignment", {})
    )

    # sanity
    for n in cfg.control_names:
        if n not in vocs.variable_names:
            raise ValueError(f"control '{n}' not in VOCS.variables")
    if cfg.measurement_name not in vocs.variable_names:
        raise ValueError(f"measurement '{cfg.measurement_name}' not in VOCS.variables")
    if cfg.observable_name not in vocs.observable_names:
        raise ValueError(f"observable '{cfg.observable_name}' not in VOCS.observables")

    algo = MultipointBAXAlignmentAlgorithm(vocs, cfg)
    gen = MultipointBAXGenerator(
        vocs=vocs, algorithm=algo, algorithm_results_file=cfg.algorithm_results_file
    )
    gen.n_monte_carlo_samples = cfg.n_monte_carlo_samples
    gen.gp_constructor.use_low_noise_prior = cfg.use_low_noise_prior

    return gen

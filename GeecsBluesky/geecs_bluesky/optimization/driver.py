"""Xopt ask/tell assembly; acquisition and analysis belong to the native plan."""

from __future__ import annotations

import math
from pathlib import Path
from collections.abc import Mapping, Sequence

import pandas as pd
from gest_api.vocs import VOCS
from xopt import Xopt
from xopt.vocs import random_inputs, select_best, FeasibilityError
from geecs_schemas.optimizer_config import OptimizerGenerator

from .generators.generator_factory import build_generator_from_config
from .inspection.dump_loader import load_xopt_dump, check_vocs_compatible


def _external_evaluation(inputs: dict) -> dict:
    """Reject accidental use of Xopt's evaluator; the plan supplies observations."""
    raise RuntimeError("native optimization evaluates measurements through the plan")


class XoptDriver:
    """Generate candidates, retain failed observations, and train only on valid data."""

    def __init__(
        self, vocs: VOCS, recipe: OptimizerGenerator, seed_dumps: Sequence[Path] = ()
    ) -> None:
        self.vocs = vocs
        generator = build_generator_from_config(
            {**recipe.options, "name": recipe.name}, vocs
        )
        self.xopt = Xopt(
            generator=generator, evaluator={"function": _external_evaluation}
        )
        self._valid_count = 0
        self._outputs = tuple([*vocs.objectives, *vocs.observables, *vocs.constraints])
        for path in seed_dumps:
            source_vocs, data = load_xopt_dump(path)
            check_vocs_compatible(vocs, source_vocs, path)
            for row in data.to_dict("records"):
                if not row.get("xopt_error", False):
                    self.tell(
                        {name: row[name] for name in vocs.variables},
                        {name: row[name] for name in self._outputs},
                    )

    def bind_folder(self, folder: Path) -> None:
        """Root BAX's optional diagnostic dumps inside the already-claimed scan."""
        generator = self.xopt.generator
        value = getattr(generator, "algorithm_results_file", None)
        if value:
            generator.algorithm_results_file = str(folder / Path(value).name)

    def ask(self) -> dict[str, float]:
        """Use random cold-start points until two valid observations exist."""
        candidate = (
            random_inputs(self.vocs, 1)[0]
            if self._valid_count < 2
            else self.xopt.generator.suggest(1)[0]
        )
        proposal = {name: float(candidate[name]) for name in self.vocs.variables}
        for name, value in proposal.items():
            lo, hi = self.vocs.variables[name].domain
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(
                    f"generator proposed {name}={value} outside [{lo}, {hi}]"
                )
        return proposal

    def tell(self, measured: Mapping[str, float], outputs: Mapping[str, float]) -> None:
        """Record an observation; failed measurements never reach the GP training set."""
        row = {**self.vocs.constants, **measured, **outputs}
        valid = all(
            math.isfinite(float(row.get(name, math.nan)))
            for name in [*self.vocs.variables, *self._outputs]
        )
        row["xopt_error"] = not valid
        data = pd.DataFrame([row])
        if valid:
            self.xopt.add_data(data)
            self._valid_count += 1
        else:
            previous = self.xopt.data
            self.xopt.data = (
                data
                if previous is None
                else pd.concat([previous, data], ignore_index=True)
            )

    @property
    def best(self) -> dict[str, float] | None:
        """Best feasible observed variables and outputs, or None for BAX/no feasible data."""
        data = self.xopt.data
        if len(self.vocs.objectives) != 1 or data is None:
            return None
        data = data.loc[~data.xopt_error.astype(bool)]
        for name, variable in self.vocs.variables.items():
            lo, hi = variable.domain
            data = data.loc[data[name].between(lo, hi)]
        if data.empty:
            return None
        try:
            index, _, _ = select_best(self.vocs, data, n=1)
        except (FeasibilityError, RuntimeError, NotImplementedError):
            return None
        return {
            name: float(data.loc[index[0], name])
            for name in [*self.vocs.variables, *self._outputs]
        }

    def dump(self, path: Path) -> None:
        """Write the complete Xopt record into an existing scan folder."""
        if not path.parent.is_dir():
            raise FileNotFoundError(path.parent)
        self.xopt.dump(str(path))

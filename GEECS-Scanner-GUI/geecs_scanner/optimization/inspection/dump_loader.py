"""Parsing and compatibility checking for Xopt dump files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import yaml
from xopt import VOCS

logger = logging.getLogger(__name__)


def load_xopt_dump(path: Path) -> Tuple[VOCS, pd.DataFrame]:
    """Parse an Xopt dump YAML and return its VOCS and evaluated data.

    Parameters
    ----------
    path:
        Path to an ``xopt_dump.yaml`` file written by ``Xopt.dump()``.

    Returns
    -------
    vocs:
        The VOCS reconstructed from the dump's ``vocs`` block.
    df:
        DataFrame of evaluated points.  Rows where ``xopt_error`` is True
        are retained; callers are responsible for any further filtering.

    Raises
    ------
    KeyError
        If the dump file is missing the required ``vocs`` or ``data`` blocks.
    """
    path = Path(path)
    with open(path, "r") as f:
        dump = yaml.safe_load(f)

    if "vocs" not in dump:
        raise KeyError(f"Dump file has no 'vocs' block: {path}")
    if "data" not in dump:
        raise KeyError(f"Dump file has no 'data' block: {path}")

    vocs = VOCS(**dump["vocs"])

    df = pd.DataFrame({k: pd.Series(v) for k, v in dump["data"].items()})
    df.index = df.index.astype(int)
    df = df.sort_index()

    return vocs, df


def check_vocs_compatible(target: VOCS, source: VOCS, source_path: Path) -> None:
    """Assert that *source* VOCS is compatible with *target* for seeding.

    Variable names and objective names/directions must match exactly (hard
    errors).  Differing bounds or constraints produce warnings only, because
    continuation runs often intentionally tighten or expand bounds.

    Parameters
    ----------
    target:
        VOCS of the new optimization run.
    source:
        VOCS reconstructed from a dump file.
    source_path:
        Path of the dump file (used in error / warning messages).

    Raises
    ------
    ValueError
        On variable-name or objective mismatch.
    """
    source_path = Path(source_path)

    # --- variable names (hard) ---
    target_vars = set(target.variables.keys())
    source_vars = set(source.variables.keys())
    if target_vars != source_vars:
        missing = sorted(target_vars - source_vars)
        extra = sorted(source_vars - target_vars)
        parts = []
        if missing:
            parts.append(f"missing from dump: {missing}")
        if extra:
            parts.append(f"extra in dump: {extra}")
        raise ValueError(
            f"VOCS variable mismatch in {source_path.name}: {'; '.join(parts)}"
        )

    # --- objectives (hard) ---
    if target.objectives != source.objectives:
        raise ValueError(
            f"VOCS objective mismatch in {source_path.name}: "
            f"target={target.objectives}, dump={source.objectives}"
        )

    # --- bounds (soft) ---
    for var, target_bounds in target.variables.items():
        source_bounds = source.variables.get(var)
        if source_bounds is None:
            continue
        tlo, thi = target_bounds
        slo, shi = source_bounds
        if slo != tlo or shi != thi:
            logger.warning(
                "Variable '%s' bounds differ — current run: [%s, %s], "
                "dump (%s): [%s, %s]",
                var,
                tlo,
                thi,
                source_path.name,
                slo,
                shi,
            )

    # --- constraints (soft) ---
    target_constraints = set(target.constraints.keys()) if target.constraints else set()
    source_constraints = set(source.constraints.keys()) if source.constraints else set()
    if target_constraints != source_constraints:
        logger.warning(
            "VOCS constraint mismatch in %s — target: %s, dump: %s",
            source_path.name,
            sorted(target_constraints),
            sorted(source_constraints),
        )


def check_cross_dump_consistency(dump_vocs: List[Tuple[Path, VOCS]]) -> None:
    """Warn when two seed dumps have different bounds for the same variable.

    Parameters
    ----------
    dump_vocs:
        List of (path, VOCS) pairs, one per seed file.
    """
    for i, (path_i, vocs_i) in enumerate(dump_vocs):
        for path_j, vocs_j in dump_vocs[i + 1 :]:
            for var in vocs_i.variables:
                if var not in vocs_j.variables:
                    continue
                bounds_i = vocs_i.variables[var]
                bounds_j = vocs_j.variables[var]
                if bounds_i != bounds_j:
                    logger.warning(
                        "Bounds for '%s' differ between seed files: "
                        "%s has %s, %s has %s",
                        var,
                        path_i.name,
                        bounds_i,
                        path_j.name,
                        bounds_j,
                    )

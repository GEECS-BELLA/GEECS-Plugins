"""Utilities for post-hoc inspection and warm-starting of Xopt optimizers.

Two distinct concerns live here:

1. **Dump loading** — :mod:`dump_loader` parses xopt YAML dumps into
   ``(VOCS, DataFrame)`` pairs and checks compatibility between dumps.
   Used by both :class:`BaseOptimizer.seed_from_dumps` (production) and
   the inspection notebooks under ``docs/geecs_scanner/examples/optimization``.

2. **Surrogate analysis** — :mod:`surfaces`, :mod:`candidates`,
   :mod:`slicing`, :mod:`hypers`, and :mod:`column_match` are the
   visualization and introspection helpers used by the inspection
   notebooks. They operate on any fitted ``Xopt`` object regardless of
   how it was built (live run, dump replay, or warm-start from scans).
"""

from geecs_scanner.optimization.inspection.candidates import (
    next_candidate,
    next_candidate_xy,
)
from geecs_scanner.optimization.inspection.column_match import (
    match_vocs_to_sfile_column,
)
from geecs_scanner.optimization.inspection.dump_loader import (
    check_vocs_compatible,
    load_xopt_dump,
)
from geecs_scanner.optimization.inspection.hypers import gp_hypers, gp_summary
from geecs_scanner.optimization.inspection.slicing import (
    best_observed_point,
    pick_top_varied_pair,
    print_slice_summary,
    resolve_slice_and_fixed,
)
from geecs_scanner.optimization.inspection.surfaces import (
    acquisition_surface,
    evaluate_model_on_grid,
)

# `check_cross_dump_consistency` may exist in the dump_loader added by the
# seeding work — re-export it if so.
try:
    from geecs_scanner.optimization.inspection.dump_loader import (
        check_cross_dump_consistency,
    )
except ImportError:
    check_cross_dump_consistency = None  # type: ignore[assignment]

__all__ = [
    # dump loading
    "load_xopt_dump",
    "check_vocs_compatible",
    "check_cross_dump_consistency",
    # surfaces / acquisition
    "evaluate_model_on_grid",
    "acquisition_surface",
    # candidate proposal
    "next_candidate",
    "next_candidate_xy",
    # slice/fix selection
    "pick_top_varied_pair",
    "best_observed_point",
    "resolve_slice_and_fixed",
    "print_slice_summary",
    # GP hyperparameters
    "gp_hypers",
    "gp_summary",
    # column matching
    "match_vocs_to_sfile_column",
]

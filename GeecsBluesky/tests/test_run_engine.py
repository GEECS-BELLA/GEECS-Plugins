"""make_run_engine: the RunEngine with the GEECS pieces installed."""

from __future__ import annotations

import pytest

pytest.importorskip("aioca")

from bluesky.preprocessors import SupplementalData

from geecs_bluesky.preprocessors import connect_on_demand
from geecs_bluesky.run_engine import make_run_engine


def test_connect_on_demand_is_outermost_and_no_baseline_is_installed() -> None:
    """No run-level baseline stream (#1016): the background rides in the rows."""
    RE = make_run_engine(mock=True)
    funcs = [getattr(p, "func", p) for p in RE.preprocessors]
    assert funcs[-1] is connect_on_demand
    assert not any(isinstance(p, SupplementalData) for p in RE.preprocessors)


def test_claim_needs_the_experiment() -> None:
    with pytest.raises(ValueError, match="experiment"):
        make_run_engine(claim=True)

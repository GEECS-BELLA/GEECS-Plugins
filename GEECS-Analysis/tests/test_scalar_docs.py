"""Every scalar a measure can write has a meaning, and the schema carries it.

``scalar_docs`` is what the published recipe reference and the config
editor show for each s-file column; a measure that adds a scalar without
saying what it means fails here rather than landing undocumented.
"""

from __future__ import annotations

import pytest

import geecs_analysis.specs  # noqa: F401 -- registers the builtin steps and measures
from geecs_analysis.recipe import RECIPE_REFERENCE_URL, recipe_schema
from geecs_analysis.registry import (
    definitions,
    measure_definitions,
    summary_definitions,
)

MEASURES = [item.spec for item in measure_definitions()]


def _option_variants(spec):
    """The spec at its defaults, plus once per boolean option switched on.

    ``model_construct`` skips validation so a spec with a required field
    (haso's ``sensor_config``) still lists its keys.
    """
    yield spec.model_construct()
    for name, field in spec.model_fields.items():
        if field.annotation is bool:
            yield spec.model_construct(**{name: True})


@pytest.mark.parametrize("spec", MEASURES, ids=lambda s: s.__name__)
def test_every_emittable_scalar_is_documented(spec):
    emitted = set().union(*(v.emitted_scalars() for v in _option_variants(spec)))
    missing = emitted - set(spec.scalar_docs)
    assert not missing, f"{spec.__name__}.scalar_docs lacks {sorted(missing)}"


@pytest.mark.parametrize("spec", MEASURES, ids=lambda s: s.__name__)
def test_no_documented_scalar_is_unreachable(spec):
    emitted = set().union(*(v.emitted_scalars() for v in _option_variants(spec)))
    stale = set(spec.scalar_docs) - emitted
    assert not stale, f"{spec.__name__}.scalar_docs documents unemitted {sorted(stale)}"


@pytest.mark.parametrize("spec", MEASURES, ids=lambda s: s.__name__)
def test_each_meaning_is_one_nonempty_line(spec):
    for name, text in spec.scalar_docs.items():
        assert text.strip() and "\n" not in text, f"{spec.__name__}: {name}"


def test_schema_carries_scalar_meanings_and_reference_links():
    defs = recipe_schema()["$defs"]
    for item in measure_definitions():
        variant = defs[item.spec.__name__]
        assert variant["x-scalars"] == dict(item.spec.scalar_docs)
        kind = item.spec.model_fields["kind"].default
        assert variant["x-docs"] == f"{RECIPE_REFERENCE_URL}#measure-{kind}"
    for item in definitions():
        name = item.spec.model_fields["step"].default
        assert defs[item.spec.__name__]["x-docs"].endswith(f"#step-{name}")
    for item in summary_definitions():
        kind = item.spec.model_fields["kind"].default
        assert defs[item.spec.__name__]["x-docs"].endswith(f"#summary-{kind}")

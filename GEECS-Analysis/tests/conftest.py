"""Shared fixtures: committed expected values for the numerical-parity tests."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pytest

#: Expected values the parity tests pin against, one ``.npz`` per case under
#: ``data/expected/<test module>/``. They were computed by the legacy
#: ImageAnalysis oracle while it was present (#1039, tag
#: ``legacy-analysis-final``). Regenerate deliberately with
#: ``GEECS_UPDATE_GOLDENS=1``, which stores the core's value.
EXPECTED_ROOT = Path(__file__).parent / "data" / "expected"
UPDATE_GOLDENS = "GEECS_UPDATE_GOLDENS"

#: Cross-platform float allowance, relative to each array's own scale (the
#: values were generated on macOS and are checked on Linux CI); integers and
#: booleans compare exactly. A ×1.0001 perturbation still fails.
RTOL = 1e-6
#: The key a bare (non-dict) value is stored under.
SINGLE = "value"

Value = np.ndarray | float | int | Mapping[str, object]


def _arrays(value: Value) -> dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        return {str(k): np.asarray(v) for k, v in value.items()}
    return {SINGLE: np.asarray(value)}


def assert_matches(actual: Value, expected: Value, label: str = "") -> None:
    """Assert ``actual`` equals ``expected`` key by key within :data:`RTOL`."""
    got, want = _arrays(actual), _arrays(expected)
    assert got.keys() == want.keys(), label
    for key, e in want.items():
        a = got[key]
        assert a.shape == e.shape, f"{label}{key}: shape {a.shape} != {e.shape}"
        if e.dtype.kind in "fc":
            finite = np.abs(e[np.isfinite(e)])
            scale = float(finite.max()) if finite.size else 0.0
            np.testing.assert_allclose(
                a, e, rtol=0, atol=RTOL * scale, equal_nan=True, err_msg=label + key
            )
        else:
            np.testing.assert_array_equal(a, e, err_msg=label + key)


def _stored(array: np.ndarray) -> np.ndarray:
    """Float arrays of more than a few values are kept in single precision.

    Its rounding (6e-8 relative) sits well inside :data:`RTOL`, and it halves
    the committed bytes of the noise-image cases.
    """
    if array.dtype == np.float64 and array.size > 16:
        return array.astype(np.float32)
    return array


@pytest.fixture
def expected_root() -> Path:
    """Where :func:`expected` keeps its files (a test of the fixture overrides it)."""
    return EXPECTED_ROOT


@pytest.fixture
def expected(request, expected_root):
    """Check ``actual`` against the stored value for ``name`` and return it.

    ``expected(name, actual=None, oracle=None)``: with ``GEECS_UPDATE_GOLDENS``
    set, the oracle's value (or ``actual`` when there is none) is stored
    first; a call with neither only reads, and an object array is refused. A dict is compared key by key; its stored values come back as
    arrays (0-d for scalars). Without ``actual`` only the stored value (and
    the oracle against it) is checked, for tests that compare part of it.
    """
    folder = expected_root / Path(request.node.path).stem

    def check(
        name: str,
        actual: Value | None = None,
        oracle: Callable[[], Value] | None = None,
    ) -> dict[str, np.ndarray] | np.ndarray:
        path = folder / (re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + ".npz")
        reference = oracle() if oracle is not None else None
        value = reference if oracle is not None else actual
        # A read-only call (no oracle, no actual) never writes, whatever the env.
        if value is not None and os.environ.get(UPDATE_GOLDENS):
            arrays = {key: _stored(array) for key, array in _arrays(value).items()}
            for key, array in arrays.items():
                if array.dtype == object:
                    raise TypeError(
                        f"{name}: {key!r} is an object array; not storing it"
                    )
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
        with np.load(path) as handle:
            stored = {key: handle[key] for key in handle.files}
        if oracle is not None:
            assert_matches(reference, stored, f"oracle {name}: ")
        if actual is not None:
            assert_matches(actual, stored, f"core {name}: ")
        return stored[SINGLE] if list(stored) == [SINGLE] else stored

    return check

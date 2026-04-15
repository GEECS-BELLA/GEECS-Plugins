"""Shared fixtures for GEECS-Data-Utils ML tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """A small synthetic DataFrame for ML tests."""
    rng = np.random.RandomState(42)
    n = 100
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    x3 = rng.randn(n)
    noise = rng.randn(n) * 0.1
    # target is a linear combination
    y = 3.0 * x1 - 2.0 * x2 + 0.5 * x3 + noise

    return pd.DataFrame({
        "feature_a": x1,
        "feature_b": x2,
        "feature_c": x3,
        "charge": y,
        "timestamp": rng.rand(n) * 1e9,
        "shotnumber": np.arange(n),
    })


@pytest.fixture
def feature_columns():
    """Feature column names matching sample_df."""
    return ["feature_a", "feature_b", "feature_c"]

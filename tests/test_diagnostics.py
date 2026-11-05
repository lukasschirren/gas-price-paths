"""The variogram (diagnostics.py) against closed forms: a random walk grows with the root of the lag, one
mean-reverting factor levels off at its stationary spread times the root of two, missing days are skipped."""
from __future__ import annotations

import numpy as np
import pytest

from gaspaths import variogram


def test_the_variogram_of_a_random_walk_grows_with_the_root_of_the_gap():
    x = np.cumsum(np.random.default_rng(4).standard_normal(200_000))
    assert np.allclose(variogram(x, [1, 4, 16]), [1.0, 2.0, 4.0], rtol=0.02)


def test_one_factor_levels_off_at_its_stationary_spread():
    phi, n = 0.9, 400_000
    rng = np.random.default_rng(5)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    v = 1.0 / (1 - phi ** 2)                                  # the stationary variance
    lags = np.array([1, 5, 200])
    assert np.allclose(variogram(x, lags), np.sqrt(2 * v * (1 - phi ** lags)), rtol=0.02)


def test_missing_days_are_skipped_and_a_lag_without_pairs_is_nan():
    y = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    assert variogram(y, [1])[0] == pytest.approx(1.0)        # the pairs (0,1) and (3,4)
    assert np.isnan(variogram(y, [10])[0])
    with pytest.raises(ValueError):
        variogram(y, [0])

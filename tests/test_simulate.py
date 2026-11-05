"""Seeded simulation: reproducibility, stationary and conditional moments, anchoring, regime targets, common random
numbers, annual means and the streaming path."""
from __future__ import annotations

import numpy as np
import pytest

from gaspaths import (ModelSpec, Params, State, anchor_offsets, annual_means, log_deviation, ou,
                      simulate_annual_means, simulate_factors)

DT = 1.0 / 365.25


def params(regimes=("normal",), seasonal=(0.0, 0.0), theta=None, sigma=None, kappa=(60.0, 2.0)) -> Params:
    spec = ModelSpec(factors=2, harmonics=1, regimes=regimes)
    nr = len(regimes)
    return Params(spec=spec, mu=0.0, kappa=list(kappa), sigma=np.tile([1.0, 0.6], (nr, 1)) if sigma is None else sigma,
                  rho=0.0, seasonal=list(seasonal), beta=np.zeros((nr, 0)), sigma_obs=0.0, theta=theta)


def test_same_seed_same_paths_and_other_seed_differs():
    p = params()
    a = simulate_factors(p, 50, 10, np.random.default_rng(1))
    b = simulate_factors(p, 50, 10, np.random.default_rng(1))
    c = simulate_factors(p, 50, 10, np.random.default_rng(2))
    assert a.shape == (10, 50, 2) and np.array_equal(a, b) and not np.allclose(a, c)


def test_stationary_start_has_the_stationary_variance():
    p = params()
    x = simulate_factors(p, 101, 20_000, np.random.default_rng(3))
    sig = ou.stationary_cov(p.kappa, p.sigma[0], p.rho)
    for day in (0, 100):
        assert np.allclose(np.var(x[:, day, :], axis=0), np.diag(sig), rtol=0.03)


def test_start_away_reverts_at_the_slow_rate():
    p = params()
    start = State(mean=np.array([0.0, 1.0]), cov=np.zeros((2, 2)), t_day=0)
    x = simulate_factors(p, 180, 20_000, np.random.default_rng(4), start=start)
    expect = np.exp(-p.kappa[1] * DT * 180)
    assert abs(x[:, -1, 1].mean() - expect) < 0.02


def test_mean_anchoring_hits_the_level():
    p = params(seasonal=(0.2, 0.1))
    t = np.arange(730)
    years = np.repeat([2027, 2028], 365)
    _, m = simulate_annual_means(p, t, years, 20_000, np.random.default_rng(5), level=np.ones(730), how="mean")
    assert np.allclose(m.mean(axis=0), 1.0, rtol=0.01)


def test_median_anchoring_puts_the_median_on_the_level():
    p = params()
    t = np.arange(200)
    x = simulate_factors(p, 200, 20_000, np.random.default_rng(6))
    a = anchor_offsets(p, t, np.zeros(200, int), how="median")
    price = np.exp(log_deviation(p, t, x) - a)
    assert np.allclose(np.median(price[:, 150]), 1.0, rtol=0.02) and np.allclose(a, 0.0)


def test_regime_target_moves_the_path():
    theta = np.array([[0.0, 0.0], [0.0, 1.0]])
    p = params(regimes=("normal", "crisis"), theta=theta, kappa=(60.0, np.log(2) / 0.1))
    x = simulate_factors(p, 365, 5_000, np.random.default_rng(7), regime=np.ones(365, int))
    assert x[:, -1, 1].mean() > 0.95 and x[:, -1].sum(axis=1).mean() > 0.8


def test_common_random_numbers_across_regimes():
    sig = np.array([[1.0, 0.6], [2.0, 1.2]])
    p = params(regimes=("normal", "stressed"), sigma=sig)
    a = simulate_factors(p, 100, 50, np.random.default_rng(8), regime=np.zeros(100, int))
    b = simulate_factors(p, 100, 50, np.random.default_rng(8), regime=np.ones(100, int))
    assert np.allclose(b, 2.0 * a)


def test_leap_year_annual_mean_uses_366_days():
    years = np.r_[np.full(366, 2028), np.full(365, 2029)]
    values = np.r_[np.ones(366), 3 * np.ones(365)][None, :]
    ys, m = annual_means(values, years)
    assert list(ys) == [2028, 2029] and np.allclose(m, [[1.0, 3.0]])


def test_streaming_annual_means_equal_the_full_array():
    p = params(seasonal=(0.2, 0.0))
    t = 1000 + np.arange(400)
    years = np.r_[np.full(200, 2030), np.full(200, 2031)]
    level = np.linspace(1.0, 2.0, 400)
    x = simulate_factors(p, 400, 30, np.random.default_rng(9))
    a = anchor_offsets(p, t, years, how="mean")
    _, full = annual_means(level * np.exp(log_deviation(p, t, x) - a), years)
    _, stream = simulate_annual_means(p, t, years, 30, np.random.default_rng(9), level=level, how="mean")
    assert np.allclose(full, stream, rtol=1e-12)


def test_mean_anchoring_refuses_a_random_walk():
    p = params(kappa=(60.0, 0.0))
    with pytest.raises(ValueError):
        anchor_offsets(p, np.arange(10), np.zeros(10, int), how="mean")


def test_per_path_regimes_equal_simulating_each_path_alone():
    sig = np.array([[1.0, 0.6], [2.0, 1.5]])
    theta = np.array([[0.0, 0.0], [0.0, 0.8]])
    p = params(regimes=("normal", "crisis"), sigma=sig, theta=theta)
    r = np.zeros((3, 60), int)
    r[1, 20:40] = 1
    r[2, :] = 1
    t = np.arange(60)
    years = np.repeat([2030, 2031], 30)
    _, together = simulate_annual_means(p, t, years, 3, np.random.default_rng(10), regime=r)
    x_all = simulate_factors(p, 60, 3, np.random.default_rng(10), regime=r)
    for i in range(3):
        xi = simulate_factors(p, 60, 3, np.random.default_rng(10), regime=r[i])[i]
        assert np.allclose(x_all[i], xi)
    assert together.shape == (3, 2)


def test_fast_drivers_move_the_fast_factor():
    spec = ModelSpec(factors=2, harmonics=1, drivers=("wind",), drivers_enter="fast")
    p = Params(spec=spec, mu=0.0, kappa=[60.0, 2.0], sigma=[[1e-6, 1e-6]], rho=0.0, seasonal=[0.0, 0.0],
               beta=[[0.1]], sigma_obs=0.0)
    w = np.zeros((40, 1))
    w[5:15, 0] = 1.0                                    # a ten-day spell
    x = simulate_factors(p, 40, 3, np.random.default_rng(0), drivers=w)
    phi = np.exp(-60.0 * DT)
    psi = np.zeros(40)
    for i in range(40):
        psi[i] = (phi * psi[i - 1] if i else 0.0) + 0.1 * w[i, 0]
    assert np.allclose(x[0, :, 0], psi, atol=1e-5) and np.allclose(x[:, :, 1], 0.0, atol=1e-5)
    t = np.arange(40)
    assert np.allclose(log_deviation(p, t, x, drivers=w)[0], x[0].sum(axis=1))   # not counted twice


def test_mean_anchoring_with_fast_drivers_hits_the_level():
    # the offsets carry the drivers' decaying response, so a windy spell still leaves the expected annual mean on L
    spec = ModelSpec(factors=2, harmonics=1, drivers=("wind",), drivers_enter="fast")
    p = Params(spec=spec, mu=0.0, kappa=[60.0, 2.0], sigma=[[1.0, 0.6]], rho=0.0, seasonal=[0.2, 0.1],
               beta=[[0.05]], sigma_obs=0.0)
    t = np.arange(730)
    years = np.repeat([2027, 2028], 365)
    w = np.where((t % 120) < 30, 2.0, -0.5)[:, None]
    _, m = simulate_annual_means(p, t, years, 20_000, np.random.default_rng(7), drivers=w, level=np.ones(730))
    assert np.allclose(m.mean(axis=0), 1.0, rtol=0.01)


def test_fast_drivers_need_one_regime_per_day():
    spec = ModelSpec(factors=2, harmonics=1, regimes=("normal", "stressed"), drivers=("wind",), drivers_enter="fast")
    p = Params(spec=spec, mu=0.0, kappa=[60.0, 2.0], sigma=[[1.0, 0.6], [2.0, 0.6]], rho=0.0, seasonal=[0.0, 0.0],
               beta=[[0.05], [0.1]], sigma_obs=0.0)
    with pytest.raises(ValueError):
        simulate_factors(p, 10, 2, np.random.default_rng(0), regime=np.zeros((2, 10), int), drivers=np.ones((10, 1)))


def test_small_integer_regimes_give_the_same_paths_without_a_copy():
    # a Monte Carlo's per-path regimes (paths x days) are kept as one byte a day; they must give the int64 paths
    p = params(regimes=("normal", "stressed"), sigma=np.array([[1.0, 0.6], [3.0, 0.6]]))
    rng = np.random.default_rng(3)
    r8 = (rng.random((50, 60)) < 0.3).astype(np.int8)
    a = simulate_factors(p, 60, 50, np.random.default_rng(1), regime=r8)
    b = simulate_factors(p, 60, 50, np.random.default_rng(1), regime=r8.astype(np.int64))
    assert np.array_equal(a, b)
    from gaspaths import simulate as S
    assert S._regime(p, r8, 60, 50) is r8 or np.shares_memory(S._regime(p, r8, 60, 50), r8)

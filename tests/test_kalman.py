"""The Kalman filter against exact Gaussian likelihoods, missing days, regimes, drivers and statsmodels."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from gaspaths import ModelSpec, Params, kalman_filter, ou

DT = 1.0 / 365.25


def one_factor(kappa=5.0, sigma=1.0, sigma_obs=0.0, mu=0.3) -> Params:
    spec = ModelSpec(factors=1, harmonics=0)
    return Params(spec=spec, mu=mu, kappa=[kappa], sigma=[[sigma]], rho=0.0, seasonal=[], beta=np.zeros((1, 0)),
                  sigma_obs=sigma_obs)


def two_factor(regimes=("normal",), drivers=(), sigma=None, beta=None, harmonics=0, rho=0.4) -> Params:
    spec = ModelSpec(factors=2, harmonics=harmonics, regimes=regimes, drivers=drivers)
    nr = len(regimes)
    return Params(spec=spec, mu=4.3, kappa=[60.0, 1.5],
                  sigma=np.tile([1.0, 0.6], (nr, 1)) if sigma is None else sigma, rho=rho,
                  seasonal=np.full(2 * harmonics, 0.05), beta=np.zeros((nr, len(drivers))) if beta is None else beta,
                  sigma_obs=0.02)


def simulate_obs(p: Params, t: int, seed: int = 0) -> np.ndarray:
    """A stationary sample of the observation, for tests only (the package's own simulator is tested elsewhere)."""
    rng = np.random.default_rng(seed)
    n = p.spec.factors
    sig = ou.stationary_cov(p.kappa, p.sigma[0], p.rho)
    w, v = np.linalg.eigh(sig)
    x = (v * np.sqrt(np.clip(w, 0, None))) @ rng.standard_normal(n)
    phi, q = ou.transition(p.kappa, p.sigma[0], p.rho, DT)
    wq, vq = np.linalg.eigh(q)
    lq = vq * np.sqrt(np.clip(wq, 0, None))
    out = np.empty(t)
    for i in range(t):
        if i:
            x = phi * x + lq @ rng.standard_normal(n)
        out[i] = p.mu + x.sum() + p.sigma_obs * rng.standard_normal()
    return out + p.seasonal_at(np.arange(t))


def exact_loglik(p: Params, y: np.ndarray) -> float:
    """The Gaussian log-likelihood of a stationary, single-regime, driver-free series, from its full covariance."""
    t = y.size
    phi, _ = ou.transition(p.kappa, p.sigma[0], p.rho, DT)
    sig = ou.stationary_cov(p.kappa, p.sigma[0], p.rho)
    lags = np.abs(np.subtract.outer(np.arange(t), np.arange(t)))
    cov = np.zeros((t, t))
    for i in range(p.spec.factors):
        for j in range(p.spec.factors):
            cov += sig[i, j] * np.where(np.subtract.outer(np.arange(t), np.arange(t)) >= 0, phi[i], phi[j]) ** lags
    cov += p.sigma_obs ** 2 * np.eye(t)
    mean = p.mu + p.seasonal_at(np.arange(t))
    return float(multivariate_normal(mean, cov).logpdf(y))


def test_one_factor_equals_the_exact_ar1_likelihood():
    p = one_factor(sigma_obs=0.0)
    y = simulate_obs(p, 400)
    assert np.isclose(kalman_filter(p, y, np.arange(400)).loglik, exact_loglik(p, y), rtol=0, atol=1e-6)


def test_two_factor_equals_the_exact_gaussian_likelihood():
    p = two_factor(harmonics=2)
    y = simulate_obs(p, 300, seed=3)
    assert np.isclose(kalman_filter(p, y, np.arange(300)).loglik, exact_loglik(p, y), rtol=0, atol=1e-6)


def test_singular_shocks_at_rho_one_still_filter():
    p = two_factor(rho=1.0)
    y = simulate_obs(p, 300, seed=4)
    assert np.isclose(kalman_filter(p, y, np.arange(300)).loglik, exact_loglik(p, y), rtol=0, atol=1e-6)


def test_missing_day_only_predicts():
    p = one_factor(sigma_obs=0.01, mu=0.0)
    y = simulate_obs(p, 50)
    y[10] = np.nan
    f = kalman_filter(p, y, np.arange(50))
    phi = np.exp(-p.kappa * DT)
    assert np.allclose(f.mean[10], phi * f.mean[9]) and np.isnan(f.innovations[10])
    assert kalman_filter(p, np.full(20, np.nan), np.arange(20)).loglik == 0.0


def test_regime_switches_the_shocks():
    sig = np.array([[1.0, 0.6], [3.0, 1.2]])
    p2 = two_factor(regimes=("normal", "stressed"), sigma=sig)
    p_stressed = two_factor(sigma=sig[1:2])
    y = simulate_obs(p_stressed, 200, seed=5)
    a = kalman_filter(p2, y, np.arange(200), regime=np.ones(200, int)).loglik
    assert np.isclose(a, kalman_filter(p_stressed, y, np.arange(200)).loglik, rtol=0, atol=1e-9)
    b = kalman_filter(p2, y, np.arange(200), regime=np.zeros(200, int)).loglik
    assert not np.isclose(a, b)


def test_drivers_enter_the_observation():
    beta = np.array([[0.05, -0.02]])
    p = two_factor(drivers=("wind", "demand"), beta=beta)
    p0 = p.replace(beta=np.zeros((1, 2)))
    rng = np.random.default_rng(6)
    w = rng.standard_normal((200, 2))
    y = simulate_obs(p0, 200, seed=7) + w @ beta[0]
    a = kalman_filter(p, y, np.arange(200), drivers=w).loglik
    assert np.isclose(a, kalman_filter(p0, y - w @ beta[0], np.arange(200), drivers=w).loglik, atol=1e-9)


def test_last_state_is_the_final_filtered_one():
    p = two_factor()
    y = simulate_obs(p, 100)
    f = kalman_filter(p, y, 1000 + np.arange(100))
    assert f.last.t_day == 1099 and np.allclose(f.last.mean, f.mean[-1]) and np.allclose(f.last.cov, f.cov[-1])


def test_rejects_gaps_and_repeats_in_days():
    p = one_factor()
    with pytest.raises(ValueError):
        kalman_filter(p, np.zeros(3), np.array([0, 1, 3]))
    with pytest.raises(ValueError):
        kalman_filter(p, np.zeros(3), np.array([0, 1, 1]))


def test_agrees_with_statsmodels():
    sm = pytest.importorskip("statsmodels.tsa.statespace.mlemodel")
    p = two_factor()
    y = simulate_obs(p, 300, seed=8)
    phi, q = ou.transition(p.kappa, p.sigma[0], p.rho, DT)
    mod = sm.MLEModel(y - p.mu, k_states=2, k_posdef=2)
    mod.ssm["design"] = np.array([[1.0, 1.0]])
    mod.ssm["transition"] = np.diag(phi)
    mod.ssm["selection"] = np.eye(2)
    mod.ssm["state_cov"] = q
    mod.ssm["obs_cov"] = np.array([[p.sigma_obs ** 2]])
    mod.ssm.initialize_known(np.zeros(2), ou.stationary_cov(p.kappa, p.sigma[0], p.rho))
    mod.ssm.tolerance = 0.0          # statsmodels otherwise switches to a steady-state gain once the covariance settles
    assert np.isclose(kalman_filter(p, y, np.arange(300)).loglik, mod.ssm.loglike(), rtol=0, atol=1e-8)


def test_fast_drivers_equal_the_exact_gaussian_with_their_response():
    # drivers in the fast factor: the mean carries their decaying response psi_t = phi psi_{t-1} + beta w_t
    spec = ModelSpec(factors=1, harmonics=0, drivers=("wind",), drivers_enter="fast")
    p = Params(spec=spec, mu=0.2, kappa=[20.0], sigma=[[0.8]], rho=0.0, seasonal=[], beta=[[0.05]], sigma_obs=0.01)
    rng = np.random.default_rng(11)
    t = 300
    w = rng.standard_normal((t, 1))
    phi = np.exp(-p.kappa[0] * DT)
    psi = np.zeros(t)
    for i in range(t):
        psi[i] = (phi * psi[i - 1] if i else 0.0) + 0.05 * w[i, 0]
    p0 = Params(spec=ModelSpec(factors=1, harmonics=0), mu=0.2, kappa=[20.0], sigma=[[0.8]], rho=0.0, seasonal=[],
                beta=np.zeros((1, 0)), sigma_obs=0.01)
    y = simulate_obs(p0, t, seed=12) + psi
    want = exact_loglik(p0, y - psi)
    assert np.isclose(kalman_filter(p, y, np.arange(t), drivers=w).loglik, want, rtol=0, atol=1e-6)


def test_fast_drivers_stay_in_the_filtered_fast_factor():
    spec = ModelSpec(factors=2, harmonics=0, drivers=("wind", "demand"), drivers_enter="fast")
    beta = np.array([[0.04, -0.02]])
    p = Params(spec=spec, mu=4.3, kappa=[60.0, 1.5], sigma=[[1.0, 0.6]], rho=0.0, seasonal=[], beta=beta,
               sigma_obs=0.02)
    p0 = two_factor(rho=0.0)
    rng = np.random.default_rng(13)
    w = rng.standard_normal((250, 2))
    psi = p.driver_effect(np.zeros(250, int), w)
    y = simulate_obs(p0, 250, seed=14) + psi
    f, f0 = kalman_filter(p, y, np.arange(250), drivers=w), kalman_filter(p0, y - psi, np.arange(250))
    assert np.isclose(f.loglik, f0.loglik, rtol=0, atol=1e-9)
    assert np.allclose(f.mean[:, 0], f0.mean[:, 0] + psi) and np.allclose(f.mean[:, 1], f0.mean[:, 1])
    assert np.isclose(f.last.mean[0], f0.last.mean[0] + psi[-1])

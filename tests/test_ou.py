"""The exact transition of the mean-reverting factors against its closed forms and against simulation."""
from __future__ import annotations

import numpy as np
import pytest

from gaspaths import ou

DT = 1.0 / 365.25
K = np.array([50.0, 1.0])
S = np.array([1.0, 0.5])


def test_q_matches_the_closed_form():
    phi, q = ou.transition(K, S, 0.5, DT)
    assert np.allclose(phi, np.exp(-K * DT))
    r = np.array([[1.0, 0.5], [0.5, 1.0]])
    expect = r * np.outer(S, S) * (1 - np.outer(phi, phi)) / (K[:, None] + K[None, :])
    assert np.allclose(q, expect, rtol=1e-12)


def test_zero_speed_is_the_random_walk_limit():
    phi, q = ou.transition(np.array([0.0]), np.array([0.5]), 0.0, DT)
    assert phi[0] == 1.0 and np.isclose(q[0, 0], 0.25 * DT, rtol=1e-12)
    _, q_small = ou.transition(np.array([1e-10]), np.array([0.5]), 0.0, DT)
    assert np.isclose(q_small[0, 0], q[0, 0], rtol=1e-9)


@pytest.mark.parametrize("rho", [0.3, 1.0, -1.0])
def test_unit_factor_reproduces_q_even_at_rho_one(rho):
    l1 = ou.unit_shock_factor(K, rho, DT)
    _, q1 = ou.transition(K, np.ones(2), rho, DT)
    assert np.allclose(l1 @ l1.T, q1, atol=1e-15)
    _, q = ou.transition(K, S, rho, DT)
    ls = np.diag(S) @ l1
    assert np.allclose(ls @ ls.T, q, atol=1e-15)


def test_simulated_moments_match_stationary_and_lags():
    rng = np.random.default_rng(0)
    n, rho = 200_000, 0.5
    sig = ou.stationary_cov(K, S, rho)
    w, v = np.linalg.eigh(sig)
    x = rng.standard_normal((n, 2)) @ (v * np.sqrt(w)).T
    phi, _ = ou.transition(K, S, rho, DT)
    lf = np.diag(S) @ ou.unit_shock_factor(K, rho, DT)
    x0 = x.copy()
    for _ in range(7):
        x = x * phi + rng.standard_normal((n, 2)) @ lf.T
    assert np.allclose(np.var(x, axis=0), np.diag(sig), rtol=0.02)
    lag = np.mean(x0 * x, axis=0)
    assert np.allclose(lag, phi ** 7 * np.diag(sig), rtol=0.03)


def test_ahead_reverts_to_theta():
    theta, m0 = np.array([0.0, 1.0]), np.array([0.5, -0.5])
    m, c = ou.ahead(K, S, 0.5, theta, m0, np.zeros((2, 2)), 30, DT)
    assert np.allclose(m, theta + np.exp(-K * 30 * DT) * (m0 - theta))
    m_far, c_far = ou.ahead(K, S, 0.5, theta, m0, np.zeros((2, 2)), 365 * 200, DT)
    assert np.allclose(m_far, theta, atol=1e-6) and np.allclose(c_far, ou.stationary_cov(K, S, 0.5), rtol=1e-6)


def test_stationary_cov_refuses_a_random_walk():
    with pytest.raises(ValueError):
        ou.stationary_cov(np.array([5.0, 0.0]), S, 0.0)

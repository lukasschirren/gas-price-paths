"""The exact transition of correlated Ornstein-Uhlenbeck factors.

A factor x reverting to its target theta at speed kappa (per year) with shocks of size sigma (per square-root year),
    dx = kappa (theta - x) dt + sigma dW,
steps over an interval dt exactly as
    x' = theta + phi (x - theta) + eta,   phi = exp(-kappa dt),
with the shocks of two factors, correlated by rho in their Brownian motions, jointly normal of covariance
    Q_ij = rho_ij sigma_i sigma_j (1 - exp(-(kappa_i + kappa_j) dt)) / (kappa_i + kappa_j),
whose limit where kappa_i + kappa_j = 0 is rho_ij sigma_i sigma_j dt, the random walk's. The stationary covariance is
rho_ij sigma_i sigma_j / (kappa_i + kappa_j) and exists only when every speed is positive. Because the step is exact,
simulated moments equal the continuous model's at every horizon, and no step size needs choosing.

At rho = +-1 the two shocks are one shock moving both factors at two speeds, and Q is singular. unit_shock_factor
therefore factorises Q by its eigendecomposition rather than by Cholesky, at unit shock sizes, so that the shocks of
any regime are diag(sigma) times one fixed factor: the same standard normal draws then give common random numbers
across regimes whose shock sizes differ (simulate.py).

Reference: Durbin, J. and Koopman, S.J. (2012), Time Series Analysis by State Space Methods, 2nd edn, Oxford
University Press, doi:10.1093/acprof:oso/9780199641178.001.0001 (the state-space form).
"""
from __future__ import annotations

import numpy as np


def correlation(n: int, rho: float) -> np.ndarray:
    """The factors' correlation matrix: [[1]] for one factor, [[1, rho], [rho, 1]] for two."""
    if n == 1:
        return np.ones((1, 1))
    return np.array([[1.0, rho], [rho, 1.0]])


def _integral(s: np.ndarray, dt: float) -> np.ndarray:
    """(1 - exp(-s dt)) / s elementwise, with its limit dt at s = 0 (expm1 keeps small s accurate)."""
    s = np.asarray(s, float)
    out = np.full(s.shape, dt)
    nz = s > 0
    out[nz] = -np.expm1(-s[nz] * dt) / s[nz]
    return out


def transition(kappa, sigma, rho: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """(phi, Q) of one step dt (module docstring); kappa and sigma of shape (n,)."""
    kappa, sigma = np.asarray(kappa, float), np.asarray(sigma, float)
    phi = np.exp(-kappa * dt)
    q = correlation(kappa.size, rho) * np.outer(sigma, sigma) * _integral(kappa[:, None] + kappa[None, :], dt)
    return phi, q


def unit_shock_factor(kappa, rho: float, dt: float) -> np.ndarray:
    """L with L L' = Q at unit shock sizes, by eigendecomposition (it accepts the singular Q of rho = +-1); the
    shocks of shock sizes sigma are diag(sigma) L z for standard normal z."""
    kappa = np.asarray(kappa, float)
    _, q = transition(kappa, np.ones(kappa.size), rho, dt)
    w, v = np.linalg.eigh(q)
    return v * np.sqrt(np.clip(w, 0.0, None))


def stationary_cov(kappa, sigma, rho: float) -> np.ndarray:
    """The stationary covariance (module docstring); a factor with a speed of zero has none."""
    kappa, sigma = np.asarray(kappa, float), np.asarray(sigma, float)
    if np.any(kappa <= 0):
        raise ValueError(f"a random-walk factor (speed 0) has no stationary distribution: {kappa}")
    return correlation(kappa.size, rho) * np.outer(sigma, sigma) / (kappa[:, None] + kappa[None, :])


def ahead(kappa, sigma, rho: float, theta, mean0, cov0, h: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance h steps of dt ahead from (mean0, cov0), in one regime: theta + phi^h (mean0 - theta) and
    Phi^h cov0 Phi^h + Q(h dt)."""
    kappa = np.asarray(kappa, float)
    phi_h, q_h = transition(kappa, sigma, rho, h * dt)
    theta = np.asarray(theta, float)
    mean = theta + phi_h * (np.asarray(mean0, float) - theta)
    cov = np.outer(phi_h, phi_h) * np.asarray(cov0, float) + q_h
    return mean, cov

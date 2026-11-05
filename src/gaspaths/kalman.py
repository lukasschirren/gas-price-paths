"""The Kalman filter of the model and its Gaussian log-likelihood.

STATE SPACE. The state is the vector of factors x_t. Between day t-1 and day t it moves by the exact transition of
ou.py with the shock sizes and targets of day t's regime (the day entered sets the shock, so the first stressed day's
move is already stressed):
    x_t = theta[r_t] + phi (x_{t-1} - theta[r_t]) + eta_t,    eta_t ~ N(0, Q(sigma[r_t]))
and the observation is
    y_t = z_t + sum_j x_t[j] + e_t,    z_t = mu + s(t) + beta[r_t] . w_t,    e_t ~ N(0, sigma_obs^2).
The filter predicts each day and updates on its observation; a missing day (NaN) only predicts, so a gap in the
data widens the state's uncertainty without biasing it. Days must be consecutive integers, since each step is one
day of the model's dt.

DRIVERS IN THE FAST FACTOR. With drivers_enter "fast" the drivers enter the fast factor's transition instead (model.py,
WHERE THE DRIVERS ENTER). Their response psi is fixed by the drivers, so the filter runs on the fast factor less psi,
with z_t = mu + s(t) + psi_t, and adds psi back to the filtered fast factor: the means it reports, and its last state,
are the whole factor, and a simulation started from that state lets the latest days' weather fade at the factor's
speed. The likelihood is that of the model with the drivers in the state, exactly (tests/test_kalman.py).

START. Factors with a positive speed start from their stationary distribution around the first day's targets; a
random-walk factor (speed 0) starts from a diffuse prior of variance DIFFUSE, which adds a constant to the
log-likelihood and leaves the comparison of fits with the same factors unchanged.

SPEED. The fit evaluates the likelihood thousands of times, so the recursion is written out in scalars for one and
two factors (the only cases of v0.1.0) instead of small matrix products; loglik() skips storing the states.

Reference: Durbin, J. and Koopman, S.J. (2012), Time Series Analysis by State Space Methods, 2nd edn, Oxford
University Press, doi:10.1093/acprof:oso/9780199641178.001.0001.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gaspaths import ou
from gaspaths.model import Params

DIFFUSE = 1e6
LOG2PI = math.log(2.0 * math.pi)


@dataclass
class State:
    """The factors' mean and covariance on one day (a filtered state, or a start for simulation)."""
    mean: np.ndarray
    cov: np.ndarray
    t_day: int


@dataclass
class FilterResult:
    """The filter's log-likelihood, its filtered means and covariances by day, and the standardised innovations
    (NaN on missing days)."""
    loglik: float
    mean: np.ndarray
    cov: np.ndarray
    innovations: np.ndarray
    t_days: np.ndarray

    @property
    def last(self) -> State:
        return State(mean=self.mean[-1].copy(), cov=self.cov[-1].copy(), t_day=int(self.t_days[-1]))


def prepare(params: Params, y, t_days, regime=None, drivers=None) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                          np.ndarray]:
    """Checked inputs (y, t_days, regime, z): z_t = mu + s(t) + the drivers' effect, the deterministic part."""
    return _prepare(params, y, t_days, regime, drivers)[:4]


def _prepare(params: Params, y, t_days, regime=None, drivers=None):
    """prepare's four values and the drivers' effect by day (0.0 without drivers)."""
    spec = params.spec
    y = np.asarray(y, float).reshape(-1)
    t = np.asarray(t_days)
    if t.shape != y.shape:
        raise ValueError(f"t_days has shape {t.shape}, y {y.shape}")
    if t.size and not np.issubdtype(t.dtype, np.integer):
        if not np.allclose(t, np.round(t)):
            raise ValueError("t_days must be whole days")
        t = np.round(t).astype(np.int64)
    if t.size > 1 and not np.all(np.diff(t) == 1):
        raise ValueError("t_days must be consecutive days (a missing observation is a NaN in y, not a skipped day)")
    r = np.zeros(y.size, np.int64) if regime is None else np.asarray(regime).astype(np.int64).reshape(-1)
    if r.shape != y.shape or (r.size and (r.min() < 0 or r.max() >= spec.n_regimes)):
        raise ValueError(f"regime must index {spec.regimes} for every day")
    if spec.n_drivers:
        w = np.zeros((y.size, spec.n_drivers)) if drivers is None else np.asarray(drivers, float)
        if w.shape != (y.size, spec.n_drivers):
            raise ValueError(f"drivers need shape {(y.size, spec.n_drivers)}, have {w.shape}")
        if not np.all(np.isfinite(w)):
            raise ValueError("drivers must be finite (set a missing driver to 0, its anomaly's mean)")
        drv = params.driver_effect(r, w)
    else:
        drv = 0.0
    z = params.mu + params.seasonal_at(t) + drv
    return y, t, r, z, drv


def initial_state(params: Params, regime0: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance before the first observation (module docstring, START)."""
    k, s, th = params.kappa, params.sigma[regime0], params.theta[regime0]
    pos = k > 0
    cov = np.zeros((k.size, k.size))
    if pos.any():
        idx = np.flatnonzero(pos)
        cov[np.ix_(idx, idx)] = ou.stationary_cov(k[idx], s[idx], params.rho if idx.size == 2 else 0.0)
    cov[~pos, ~pos] = DIFFUSE
    return th.copy(), cov


def _run(params: Params, y, t, r, z, store: bool) -> FilterResult:
    n = params.spec.factors
    T = y.size
    phi, _ = ou.transition(params.kappa, params.sigma[0], params.rho, params.spec.dt)
    qs = [ou.transition(params.kappa, params.sigma[g], params.rho, params.spec.dt)[1]
          for g in range(params.spec.n_regimes)]
    th = params.theta
    so2 = params.sigma_obs ** 2
    yl, zl, rl = y.tolist(), np.broadcast_to(z, y.shape).tolist(), r.tolist()
    mean = np.empty((T, n)) if store else None
    cov = np.empty((T, n, n)) if store else None
    innov = np.full(T, np.nan) if store else None
    a0, p0 = initial_state(params, rl[0] if T else 0)
    ll = 0.0
    if n == 2:
        f1, f2 = float(phi[0]), float(phi[1])
        q11 = [float(q[0, 0]) for q in qs]
        q12 = [float(q[0, 1]) for q in qs]
        q22 = [float(q[1, 1]) for q in qs]
        t1 = [float(v) for v in th[:, 0]]
        t2 = [float(v) for v in th[:, 1]]
        a1, a2 = float(a0[0]), float(a0[1])
        p11, p12, p22 = float(p0[0, 0]), float(p0[0, 1]), float(p0[1, 1])
        for i in range(T):
            g = rl[i]
            if i:
                a1 = t1[g] + f1 * (a1 - t1[g])
                a2 = t2[g] + f2 * (a2 - t2[g])
                p11 = f1 * f1 * p11 + q11[g]
                p12 = f1 * f2 * p12 + q12[g]
                p22 = f2 * f2 * p22 + q22[g]
            yi = yl[i]
            if yi == yi:
                v = yi - zl[i] - a1 - a2
                h1, h2 = p11 + p12, p12 + p22
                s_ = h1 + h2 + so2
                if s_ <= 1e-300:
                    s_ = 1e-300
                k1, k2 = h1 / s_, h2 / s_
                a1 += k1 * v
                a2 += k2 * v
                p11 -= k1 * h1
                p12 -= k1 * h2
                p22 -= k2 * h2
                ll -= 0.5 * (LOG2PI + math.log(s_) + v * v / s_)
                if store:
                    innov[i] = v / math.sqrt(s_)
            if store:
                mean[i, 0], mean[i, 1] = a1, a2
                cov[i, 0, 0], cov[i, 0, 1], cov[i, 1, 0], cov[i, 1, 1] = p11, p12, p12, p22
    else:
        f1 = float(phi[0])
        q11 = [float(q[0, 0]) for q in qs]
        t1 = [float(v) for v in th[:, 0]]
        a1, p11 = float(a0[0]), float(p0[0, 0])
        for i in range(T):
            g = rl[i]
            if i:
                a1 = t1[g] + f1 * (a1 - t1[g])
                p11 = f1 * f1 * p11 + q11[g]
            yi = yl[i]
            if yi == yi:
                v = yi - zl[i] - a1
                s_ = p11 + so2
                if s_ <= 1e-300:
                    s_ = 1e-300
                k1 = p11 / s_
                a1 += k1 * v
                p11 -= k1 * p11
                ll -= 0.5 * (LOG2PI + math.log(s_) + v * v / s_)
                if store:
                    innov[i] = v / math.sqrt(s_)
            if store:
                mean[i, 0], cov[i, 0, 0] = a1, p11
    return FilterResult(loglik=ll, mean=mean, cov=cov, innovations=innov, t_days=t)


def kalman_filter(params: Params, y, t_days, regime=None, drivers=None) -> FilterResult:
    """Filter a daily log-price series (module docstring): y with NaN where missing, t_days consecutive integer days,
    regime an index into params.spec.regimes per day (default the first), drivers (days, drivers) (default zeros)."""
    y, t, r, z, drv = _prepare(params, y, t_days, regime, drivers)
    res = _run(params, y, t, r, z, store=True)
    if params.spec.drivers_enter == "fast" and params.spec.n_drivers:
        res.mean[:, 0] += drv             # module docstring, DRIVERS IN THE FAST FACTOR
    return res


def loglik(params: Params, y, t_days, regime=None, drivers=None) -> float:
    """The log-likelihood alone, without storing the states (the fit's objective)."""
    return _run(params, *prepare(params, y, t_days, regime, drivers), store=False).loglik

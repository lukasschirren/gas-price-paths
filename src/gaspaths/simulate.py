"""Seeded simulation of the factors, the log price around its level, anchoring to a level curve, and annual means.

PATHS. Each simulated day steps the factors by the exact transition of ou.py with the shock sizes and targets of
that day's regime (the day entered sets the shock, as in kalman.py):
    x_d = theta[r_d] + phi (x_{d-1} - theta[r_d]) + diag(sigma[r_d]) L z_d,    z_d ~ N(0, I),
with L the unit-shock factor of ou.unit_shock_factor. The first simulated day is the day after a given start (a
filtered State, whose mean and covariance draw x_0), or, without a start, a draw from the stationary distribution of
the first day's regime. The standard normals z are drawn day by day in one fixed order from the caller's generator,
so the same generator state gives the same shocks whatever the regimes, drivers or level: paths of different regimes
or weather years differ only through what they are meant to differ in (common random numbers). The regime is one
per day, or one per path and day (shape (n, days)), which lets a Monte Carlo give each path its own regime history,
crises arriving at random for instance. With drivers_enter "fast" each day also adds its impulse beta[r_d] . w_d to
the fast factor (model.py, WHERE THE DRIVERS ENTER), on the first day too, so the simulated fast factor carries the
drivers' response and a start's own response fades at the factor's speed.

THE PRICE. log_deviation is the log price less the level mu, s(t) + beta[r_t] . w_t + sum of the factors, without the
driver term when the drivers enter the fast factor (they are inside the factors then). A level curve L (a price per
day) replaces mu: the price is L exp(log_deviation - a_y), where the offset a_y of each year anchors it. With mean
anchoring
    a_y = V / 2 + ln mean_{t in y} exp( s(t) + D_t ),
D_t the drivers' effect (Params.driver_effect: beta[r_t] . w_t, or its response psi_t from the first day given), V
the stationary variance of the sum of the factors (of variance_regime's shock sizes), so the expected annual mean
price of a stationary path equals the annual mean of L. Median anchoring drops V / 2, which leaves the median path on
the curve. A random-walk factor has no stationary variance, so it admits median anchoring only. With a regime per
path the offsets take the seasonal curve alone, so such runs take no drivers.

STREAMING. A Monte Carlo of 20,000 paths over ten years would need about a gigabyte for its daily values, so
simulate_annual_means steps the same paths and keeps only each year's sum of prices; with the same generator it gives
exactly the annual means of the full array (tests/test_simulate.py).
"""
from __future__ import annotations

import numpy as np

from gaspaths import ou
from gaspaths.kalman import State
from gaspaths.model import Params


def _regime(params: Params, regime, n_days: int, n: int | None = None) -> np.ndarray:
    """The regime of each day, shape (n_days,), or of each path and day, shape (n, n_days); an integer array is kept
    as it is (a Monte Carlo's per-path regimes may take one byte a day), anything else becomes int64."""
    if regime is None:
        return np.zeros(n_days, np.int64)
    r = np.asarray(regime)
    if not np.issubdtype(r.dtype, np.integer):
        r = r.astype(np.int64)
    if r.ndim == 1:
        ok = r.size == n_days
    elif r.ndim == 2:
        ok = r.shape[1] == n_days and (n is None or r.shape[0] == n)
    else:
        ok = False
    if not ok or (r.size and (r.min() < 0 or r.max() >= params.spec.n_regimes)):
        raise ValueError(f"regime must index {params.spec.regimes} for each of the {n_days} days"
                         + (f" (or of {n} paths and the days)" if n is not None else ""))
    return r


def _drivers(params: Params, drivers, r: np.ndarray, n_days: int) -> np.ndarray | None:
    """The checked drivers (days, drivers), or None where the model has none or none are given."""
    if not params.spec.n_drivers or drivers is None:
        return None
    if r.ndim != 1:
        raise ValueError("drivers need one regime per day, not one per path")
    w = np.asarray(drivers, float)
    if w.shape != (n_days, params.spec.n_drivers):
        raise ValueError(f"drivers need shape {(n_days, params.spec.n_drivers)}, have {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("drivers must be finite (set a missing driver to 0, its anomaly's mean)")
    return w


def _impulse(params: Params, r: np.ndarray, w: np.ndarray | None) -> np.ndarray | None:
    """What the drivers add to the fast factor each day: beta[r_d] . w_d with drivers_enter "fast", else None."""
    if w is None or params.spec.drivers_enter != "fast":
        return None
    return params.driver_impulse(r, w)


def _sqrt_psd(cov: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(np.asarray(cov, float))
    return v * np.sqrt(np.clip(w, 0.0, None))


def _steps(params: Params, n_days: int, n: int, rng: np.random.Generator, start: State | None, r: np.ndarray,
           impulse: np.ndarray | None = None):
    """The factors of each simulated day, (n, factors), one day at a time (module docstring, PATHS); impulse is what
    the drivers add to the fast factor each day, if anything."""
    F, dt = params.spec.factors, params.spec.dt
    phi, _ = ou.transition(params.kappa, params.sigma[0], params.rho, dt)
    unit = ou.unit_shock_factor(params.kappa, params.rho, dt)
    per_path = r.ndim == 2
    z = rng.standard_normal((n, F))
    if start is None:
        g0 = r[:, 0] if per_path else np.full(n, r[0])
        roots = {g: _sqrt_psd(ou.stationary_cov(params.kappa, params.sigma[g], params.rho)) for g in np.unique(g0)}
        x = np.empty((n, F))
        for g, root in roots.items():
            m = g0 == g
            x[m] = params.theta[g] + z[m] @ root.T
    else:
        x = np.asarray(start.mean, float) + z @ _sqrt_psd(start.cov).T
    for d in range(n_days):
        if d or start is not None:
            g = r[:, d] if per_path else r[d]
            th = params.theta[g]
            x = th + phi * (x - th) + (rng.standard_normal((n, F)) @ unit.T) * params.sigma[g]
        if impulse is not None:
            x[:, 0] += impulse[d]
        yield x


def simulate_factors(params: Params, n_days: int, n: int, rng: np.random.Generator, start: State | None = None,
                     regime=None, drivers=None) -> np.ndarray:
    """n paths of the factors over n_days days, (n, n_days, factors) (module docstring, PATHS). drivers (n_days,
    drivers) move the fast factor where they enter it, and nothing here where they enter the observation."""
    r = _regime(params, regime, n_days, n)
    imp = _impulse(params, r, _drivers(params, drivers, r, n_days))
    out = np.empty((n, n_days, params.spec.factors))
    for d, x in enumerate(_steps(params, n_days, n, rng, start, r, imp)):
        out[:, d, :] = x
    return out


def _deterministic(params: Params, t_days, regime, drivers, effect: bool = False) -> np.ndarray:
    """The log price less mu and the factors by day, s(t) plus the drivers' term where they enter the observation
    (module docstring, THE PRICE); with effect, s(t) + D_t, the anchoring's curve. One regime per day; a regime per
    path takes the seasonal curve alone."""
    t = np.asarray(t_days, float).reshape(-1)
    det = params.seasonal_at(t)
    r = _regime(params, regime, t.size)
    w = _drivers(params, drivers, r, t.size)
    if w is not None and (effect or params.spec.drivers_enter == "observation"):
        det = det + params.driver_effect(r, w)
    return det


def log_deviation(params: Params, t_days, factors: np.ndarray, regime=None, drivers=None) -> np.ndarray:
    """The log price less mu, (n, days): s(t) + beta[r_t] . w_t + the sum of the factors (n, days, factors)."""
    factors = np.asarray(factors, float)
    det = _deterministic(params, t_days, regime, drivers)
    if factors.shape[1] != det.size:
        raise ValueError(f"factors cover {factors.shape[1]} days, t_days {det.size}")
    return det[None, :] + factors.sum(axis=2)


def anchor_offsets(params: Params, t_days, years, regime=None, drivers=None, how: str = "mean",
                   variance_regime: int = 0) -> np.ndarray:
    """The offset a_y of every day's year (module docstring, THE PRICE); how is "mean" or "median"."""
    if how not in ("mean", "median"):
        raise ValueError(f"how is 'mean' or 'median', not {how!r}")
    det = _deterministic(params, t_days, regime, drivers, effect=True)
    years = np.asarray(years).reshape(-1)
    if years.size != det.size:
        raise ValueError("years needs one entry per day")
    half_v = 0.0
    if how == "mean":
        cov = ou.stationary_cov(params.kappa, params.sigma[variance_regime], params.rho)
        half_v = 0.5 * float(cov.sum())
    out = np.empty(det.size)
    for y in np.unique(years):
        m = years == y
        out[m] = half_v + float(np.log(np.mean(np.exp(det[m]))))
    return out


def annual_means(values: np.ndarray, years) -> tuple[np.ndarray, np.ndarray]:
    """(the years in order, the mean of each path's values over each year), values (n, days)."""
    values = np.atleast_2d(np.asarray(values, float))
    years = np.asarray(years).reshape(-1)
    if years.size != values.shape[1]:
        raise ValueError("years needs one entry per day")
    ys = np.unique(years)
    return ys, np.column_stack([values[:, years == y].mean(axis=1) for y in ys])


def simulate_annual_means(params: Params, t_days, years, n: int, rng: np.random.Generator, start: State | None = None,
                          regime=None, drivers=None, level=None, how: str | None = "mean",
                          variance_regime: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Annual mean prices of n paths, streamed (module docstring, STREAMING): the price is level exp(log_deviation -
    a_y) with a level curve, anchored as how says (None for no offset), and exp(mu + log_deviation) without one.
    regime is one per day or one per path and day."""
    t = np.asarray(t_days).reshape(-1)
    years = np.asarray(years).reshape(-1)
    if years.size != t.size:
        raise ValueError("years needs one entry per day")
    r = _regime(params, regime, t.size, n)
    w = _drivers(params, drivers, r, t.size)
    r_day = None if r.ndim == 2 else r
    det = _deterministic(params, t, r_day, w)
    if level is None:
        base, off = np.full(t.size, np.exp(params.mu)), np.zeros(t.size)
    else:
        base = np.asarray(level, float).reshape(-1)
        if base.size != t.size:
            raise ValueError("level needs one price per day")
        off = (np.zeros(t.size) if how is None
               else anchor_offsets(params, t, years, r_day, w, how=how, variance_regime=variance_regime))
    ys, idx = np.unique(years, return_inverse=True)
    sums = np.zeros((n, ys.size))
    counts = np.bincount(idx, minlength=ys.size).astype(float)
    for d, x in enumerate(_steps(params, t.size, n, rng, start, r, _impulse(params, r, w))):
        sums[:, idx[d]] += base[d] * np.exp(det[d] + x.sum(axis=1) - off[d])
    return ys, sums / counts

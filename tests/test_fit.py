"""Maximum likelihood on synthetic series of known parameters: recovery, fixed parameters, shared shock sizes, the
correlation rule, one factor against least squares."""
from __future__ import annotations

import numpy as np
import pytest

from gaspaths import ModelSpec, Params, fit, ou, simulate_factors

DT = 1.0 / 365.25


def synthetic(p: Params, t: int, regime=None, seed: int = 0) -> np.ndarray:
    """A sample of the observation with regimes (the day entered sets the shocks), for tests only."""
    rng = np.random.default_rng(seed)
    n = p.spec.factors
    r = np.zeros(t, int) if regime is None else regime
    phi, _ = ou.transition(p.kappa, p.sigma[0], p.rho, DT)
    lq = [np.diag(p.sigma[g]) @ ou.unit_shock_factor(p.kappa, p.rho, DT) for g in range(p.spec.n_regimes)]
    sig = ou.stationary_cov(p.kappa, p.sigma[r[0]], p.rho)
    w, v = np.linalg.eigh(sig)
    x = (v * np.sqrt(np.clip(w, 0, None))) @ rng.standard_normal(n)
    out = np.empty(t)
    for i in range(t):
        if i:
            x = phi * x + lq[r[i]] @ rng.standard_normal(n)
        out[i] = x.sum()
    return p.mu + out + p.seasonal_at(np.arange(t)) + p.sigma_obs * rng.standard_normal(t)


def truth(rho=0.5, spec=None) -> Params:
    spec = spec or ModelSpec(factors=2, harmonics=1)
    nr = spec.n_regimes
    sigma = np.array([[1.1, 0.9], [2.5, 0.9]])[:nr]
    return Params(spec=spec, mu=4.3, kappa=[90.0, 1.2], sigma=sigma, rho=rho, seasonal=[0.15, 0.0],
                  beta=np.zeros((nr, spec.n_drivers)), sigma_obs=0.01)


@pytest.fixture(scope="module")
def recovered():
    p = truth()
    y = synthetic(p, 3000, seed=1)
    return p, fit(p.spec, y, np.arange(3000), fixed={"rho": p.rho}, starts=3)


def test_recovers_known_parameters(recovered):
    p, res = recovered
    q = res.params
    assert res.converged
    assert abs(q.kappa[0] / p.kappa[0] - 1) < 0.30
    assert abs(q.sigma[0, 0] / p.sigma[0, 0] - 1) < 0.15
    assert abs(q.sigma[0, 1] / p.sigma[0, 1] - 1) < 0.30
    # a slow factor of half-life 0.6 years over 8 years leaves the seasonal amplitude an error of about 0.065
    se_a1 = res.se["seasonal"][0]
    assert 0.0 < se_a1 < 0.1 and abs(q.seasonal[0] - p.seasonal[0]) < 3 * se_a1
    assert res.n_obs == 3000 and res.n_params == 8


def test_standard_errors_are_finite_where_estimated(recovered):
    _, res = recovered
    assert np.all(np.isfinite(res.se["kappa"])) and np.isfinite(res.se["mu"]) and np.isnan(res.se["rho"])
    assert np.all(np.asarray(res.se["kappa"]) > 0) and np.all(np.isfinite(res.se["sigma"]))


def _ab(q: Params) -> np.ndarray:
    sig = ou.stationary_cov(q.kappa, q.sigma[0], q.rho)
    return np.array([sig[0, 0] + sig[0, 1], sig[1, 1] + sig[0, 1]])


def test_correlation_is_not_identified_from_a_spot_series():
    # the likelihood depends on the speeds, A = Sigma_11 + Sigma_12, B = Sigma_22 + Sigma_12 and the noise alone
    # (fit.py, IDENTIFICATION OF THE CORRELATION): fits at two fixed correlations reach the same optimum
    p = truth(rho=0.5)
    y = synthetic(p, 2000, seed=6)
    a = fit(p.spec, y, np.arange(2000), fixed={"rho": 0.0}, starts=2)
    b = fit(p.spec, y, np.arange(2000), fixed={"rho": 0.6}, starts=2)
    assert abs(a.loglik - b.loglik) < 0.05
    assert np.allclose(a.params.kappa, b.params.kappa, rtol=0.05)
    assert np.allclose(_ab(a.params), _ab(b.params), rtol=0.05)
    assert not np.isclose(a.params.sigma[0, 0], b.params.sigma[0, 0], rtol=0.05)


def test_correlation_defaults_to_zero():
    p = truth()
    y = synthetic(p, 800, seed=7)
    res = fit(p.spec, y, np.arange(800), starts=1)
    assert res.params.rho == 0.0 and np.isnan(res.se["rho"])


def test_fixed_parameters_stay_fixed():
    p = truth()
    y = synthetic(p, 1500, seed=2)
    res = fit(p.spec, y, np.arange(1500), fixed={"kappa": [np.nan, 1.0], "sigma_obs": 0.01}, starts=2)
    assert res.params.kappa[1] == 1.0 and res.params.sigma_obs == 0.01
    assert np.isnan(res.se["kappa"][1]) and res.params.kappa[0] > 1.0


def test_shared_sigma_ties_the_regimes():
    spec = ModelSpec(factors=2, harmonics=1, regimes=("normal", "stressed"), shared_sigma=(1,))
    p = truth(spec=spec)
    r = np.r_[np.zeros(1000, int), np.ones(500, int)]
    y = synthetic(p, 1500, regime=r, seed=4)
    res = fit(spec, y, np.arange(1500), regime=r, starts=2)
    assert res.params.sigma[0, 1] == res.params.sigma[1, 1]
    assert res.params.sigma[1, 0] > res.params.sigma[0, 0]


def test_one_factor_matches_least_squares():
    spec = ModelSpec(factors=1, harmonics=0)
    p = Params(spec=spec, mu=0.0, kappa=[20.0], sigma=[[1.0]], rho=0.0, seasonal=[], beta=np.zeros((1, 0)),
               sigma_obs=0.0)
    y = synthetic(p, 2000, seed=5)
    res = fit(spec, y, np.arange(2000), fixed={"sigma_obs": 0.0}, starts=2)
    a, b = y[:-1] - y.mean(), y[1:] - y.mean()
    kappa_ols = -np.log(np.sum(a * b) / np.sum(a * a)) / DT
    assert abs(res.params.kappa[0] / kappa_ols - 1) < 0.10


def test_fast_drivers_are_recovered_and_beat_same_day_drivers():
    # drivers in the fast factor's state: the fit recovers the daily impulse, and the same-day form fits worse
    spec = ModelSpec(factors=2, harmonics=1, drivers=("wind",), drivers_enter="fast")
    p = Params(spec=spec, mu=4.3, kappa=[60.0, 1.2], sigma=[[1.1, 0.9]], rho=0.0, seasonal=[0.15, 0.0],
               beta=[[0.03]], sigma_obs=0.01)
    rng = np.random.default_rng(21)
    t = 3000
    w = rng.standard_normal((t, 1))
    x = simulate_factors(p, t, 1, rng, drivers=w)[0]
    y = p.mu + p.seasonal_at(np.arange(t)) + x.sum(axis=1) + p.sigma_obs * rng.standard_normal(t)
    fast = fit(spec, y, np.arange(t), drivers=w, starts=3)
    same_day = fit(ModelSpec(factors=2, harmonics=1, drivers=("wind",)), y, np.arange(t), drivers=w, starts=3)
    se = fast.se["beta"][0][0]
    assert fast.converged and 0.0 < se < 0.01 and abs(fast.params.beta[0, 0] - 0.03) < 3 * se
    assert fast.loglik > same_day.loglik + 50


def test_a_warm_start_is_tried_and_kept_when_best(recovered):
    p, cold = recovered
    y = synthetic(p, 3000, seed=1)
    warm = fit(p.spec, y, np.arange(3000), fixed={"rho": p.rho}, starts=1, start=cold.params)
    assert warm.loglik >= cold.loglik - 1e-6                      # it starts where the cold fit ended
    with pytest.raises(ValueError):
        fit(p.spec, y, np.arange(3000), fixed={"rho": p.rho}, starts=1,
            start=cold.params.replace(spec=ModelSpec(factors=2, harmonics=1, drivers=("wind",)), beta=[[0.0]]))

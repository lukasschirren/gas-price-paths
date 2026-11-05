"""The model specification and the parameter set: validation, the plain-dict round trip, the seasonal curve."""
from __future__ import annotations

import json

import numpy as np
import pytest

from gaspaths import ModelSpec, Params

SPEC = ModelSpec(factors=2, harmonics=2, regimes=("normal", "stressed"), drivers=("wind", "demand"))


def params(**kw) -> Params:
    base = dict(spec=SPEC, mu=4.3, kappa=np.array([90.0, 1.2]), sigma=np.array([[1.1, 0.9], [3.0, 1.5]]), rho=0.5,
                seasonal=np.array([0.15, 0.05, -0.02, 0.01]), beta=np.array([[-0.007, 0.018], [-0.071, 0.010]]),
                sigma_obs=0.01, theta=np.zeros((2, 2)))
    base.update(kw)
    return Params(**base)


def test_round_trip_through_a_plain_dict():
    p = params()
    q = Params.from_dict(json.loads(json.dumps(p.to_dict())))
    assert q.spec == p.spec
    for name in ("mu", "kappa", "sigma", "rho", "seasonal", "beta", "sigma_obs", "theta"):
        assert np.allclose(getattr(q, name), getattr(p, name)), name


@pytest.mark.parametrize("bad", [
    dict(kappa=np.array([1.2, 90.0])),                      # speeds must fall from fast to slow
    dict(kappa=np.array([90.0, -1.0])),
    dict(rho=1.2),
    dict(sigma=np.array([1.1, 0.9])),                       # one row per regime
    dict(sigma=np.array([[1.1, -0.9], [3.0, 1.5]])),
    dict(beta=np.zeros((2, 1))),
    dict(seasonal=np.zeros(3)),
    dict(sigma_obs=-0.1),
    dict(theta=np.zeros((1, 2))),
])
def test_validation_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        params(**bad)


def test_spec_validation():
    with pytest.raises(ValueError):
        ModelSpec(factors=3)
    with pytest.raises(ValueError):
        ModelSpec(factors=2, shared_sigma=(2,))
    one = ModelSpec(factors=1, harmonics=0)
    with pytest.raises(ValueError):
        Params(spec=one, mu=0.0, kappa=np.array([2.0]), sigma=np.array([[0.5]]), rho=0.3, seasonal=np.zeros(0),
               beta=np.zeros((1, 0)), sigma_obs=0.0, theta=np.zeros((1, 1)))      # one factor has no correlation


def test_shared_sigma_must_hold():
    spec = ModelSpec(factors=2, regimes=("normal", "stressed"), shared_sigma=(1,))
    with pytest.raises(ValueError):
        params(spec=spec, beta=np.zeros((2, 0)))
    ok = params(spec=spec, beta=np.zeros((2, 0)), sigma=np.array([[1.1, 0.9], [3.0, 0.9]]))
    assert ok.sigma[0, 1] == ok.sigma[1, 1]


def test_seasonal_at_known_values():
    p = params(seasonal=np.array([0.2, 0.0, 0.0, 0.0]))
    s = p.seasonal_at(np.array([0.0, 365.25 / 4, 365.25 / 2]))
    assert np.allclose(s, [0.0, 0.2, 0.0], atol=1e-12)


def test_half_lives_in_days():
    p = params(kappa=np.array([np.log(2) * 365.25 / 3.0, 0.0]))
    hl = p.half_life_days()
    assert np.isclose(hl[0], 3.0) and np.isinf(hl[1])


def test_drivers_enter_round_trips_and_is_checked():
    spec = ModelSpec(factors=2, drivers=("wind",), drivers_enter="fast")
    assert ModelSpec.from_dict(spec.to_dict()) == spec
    assert ModelSpec.from_dict({k: v for k, v in ModelSpec().to_dict().items() if k != "drivers_enter"}) == ModelSpec()
    with pytest.raises(ValueError):
        ModelSpec(drivers_enter="slow")

"""Maximum-likelihood fitting of the model to a daily series.

TRANSFORMS. The optimiser works on unbounded or box-bounded numbers u; each maps to a parameter so that every u gives
a valid model. With two factors, kappa_slow = exp(u_a) and kappa_fast = kappa_slow + exp(u_b), so the factors stay
ordered and never swap labels (a fixed slow speed may be 0, a random walk; a fixed fast speed bounds a free slow one
through a logistic share). Shock sizes and the measurement noise are exp(u); the correlation is 0.999 tanh(u), so a
free fit cannot reach the singular boundary; the level, the seasonal coefficients and the driver weights are u
themselves. A factor named in spec.shared_sigma has one shock size for every regime. Targets theta are not fitted
(zero unless the caller passes them).

FIXED PARAMETERS. fixed maps a parameter name to its value; arrays take NaN (or None) for the entries left free, so
{"kappa": [nan, 1.0]} fixes the slow speed at 1.0 per year and fits the fast one.

ESTIMATION. L-BFGS-B minimises the negative log-likelihood of kalman.loglik from several starts: the caller's start
where given (a Params of the same spec, a related fit's for instance), a grid of speed pairs (fast, slow), each with
the series' variance split between the factors, then random speeds from the caller's generator; the best fit is
kept. Standard errors come from the inverse of a central finite-difference Hessian on the
u scale, mapped to the parameters by the delta method. They are Gaussian-likelihood errors: daily gas price changes
are fat-tailed, so they understate the uncertainty and are labelled as such wherever they are reported.

IDENTIFICATION OF THE CORRELATION (found on 4 Nov 2026, tests/test_fit.py). A spot series cannot identify the
correlation of the two factors' shocks. The Gaussian likelihood of the observed sum depends only on its
autocovariances, which for a lag of h >= 1 days are A phi_1^h + B phi_2^h with A = Sigma_11 + Sigma_12 and
B = Sigma_22 + Sigma_12 (Sigma the factors' stationary covariance), and at lag 0 are A + B + sigma_obs^2. The data
therefore pin five numbers (two speeds, A, B, sigma_obs) while the model has six parameters, so every correlation on
a ridge of shock sizes gives the same likelihood and the same distribution of the price. Fixing the correlation is a
convention that changes how a path splits between the factors and nothing about the path. fit therefore fixes rho
at RHO_DEFAULT (0, independent factors, the usual convention for spot data) unless the caller fixes another value,
or passes {"rho": nan} to estimate it with information a spot series lacks, futures prices for instance. Exploratory
fits of daily GB gas prices that left rho free ran it to +1 because the optimiser drifted along the flat ridge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from gaspaths.kalman import loglik, prepare
from gaspaths.model import ModelSpec, Params

RHO_MAX = 0.999
RHO_DEFAULT = 0.0                # module docstring, IDENTIFICATION OF THE CORRELATION
FAIL = 1e12
GRID2 = ((50.0, 1.0), (100.0, 0.4), (20.0, 2.5), (200.0, 0.15), (10.0, 0.05), (365.0, 4.0))
GRID1 = (5.0, 1.0, 20.0, 0.5, 50.0, 0.2)


@dataclass
class FitResult:
    """A fit: its parameters, their standard errors (keys and shapes of Params.to_dict, NaN where fixed), the
    log-likelihood, the observations and free parameters counted, and the optimiser's verdict."""
    params: Params
    se: dict
    loglik: float
    n_obs: int
    n_params: int
    converged: bool
    message: str

    @property
    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.loglik


def _free_mask(value, shape) -> np.ndarray:
    """Fixed entries from a fixed-value entry (None/NaN free), as an array of the shape with NaN where free."""
    if value is None:
        return np.full(shape, np.nan)
    a = np.array([np.nan if v is None else v for v in np.ravel(np.array(value, dtype=object))], float)
    if a.size == 1 and int(np.prod(shape)) != 1:
        a = np.full(shape, a[0])
    return a.reshape(shape)


class _Packer:
    """The map between the optimiser's vector u and a Params (module docstring, TRANSFORMS)."""

    def __init__(self, spec: ModelSpec, fixed: dict | None, theta: np.ndarray | None):
        unknown = set(fixed or {}) - {"mu", "kappa", "sigma", "rho", "seasonal", "beta", "sigma_obs"}
        if unknown:
            raise ValueError(f"cannot fix {sorted(unknown)}")
        f = dict(fixed or {})
        self.spec = spec
        F, R = spec.factors, spec.n_regimes
        self.theta = np.zeros((R, F)) if theta is None else np.asarray(theta, float)
        self.mu = _free_mask(f.get("mu"), ())
        self.kappa = _free_mask(f.get("kappa"), (F,))
        self.sigma = _free_mask(f.get("sigma"), (R, F))
        self.rho = _free_mask(f.get("rho"), ()) if F == 2 else np.array(0.0)
        self.seasonal = _free_mask(f.get("seasonal"), (2 * spec.harmonics,))
        self.beta = _free_mask(f.get("beta"), (R, spec.n_drivers))
        self.sigma_obs = _free_mask(f.get("sigma_obs"), ())
        for j in spec.shared_sigma:                      # one free value per shared factor, or one fixed value
            col = self.sigma[:, j]
            if np.isfinite(col).any() and not np.isfinite(col).all():
                raise ValueError(f"factor {j} shares its shock size: fix it in every regime or in none")
        self.slots = self._slots()

    def _slots(self) -> list[tuple[str, tuple, tuple[float | None, float | None]]]:
        """(name, index, bounds) of every free number, in the order of u."""
        s, out = self.spec, []
        if np.isnan(self.mu):
            out.append(("mu", (), (None, None)))
        if s.factors == 2:
            if np.isnan(self.kappa[1]):
                out.append(("kappa_slow", (1,), (math.log(1e-3), math.log(1e4))))
            if np.isnan(self.kappa[0]):
                out.append(("kappa_gap", (0,), (math.log(1e-3), math.log(1e5))))
        elif np.isnan(self.kappa[0]):
            out.append(("kappa", (0,), (math.log(1e-3), math.log(1e5))))
        for j in range(s.factors):
            if j in s.shared_sigma:
                if np.isnan(self.sigma[0, j]):
                    out.append(("sigma_shared", (j,), (math.log(1e-4), math.log(1e2))))
            else:
                for g in range(s.n_regimes):
                    if np.isnan(self.sigma[g, j]):
                        out.append(("sigma", (g, j), (math.log(1e-4), math.log(1e2))))
        if s.factors == 2 and np.isnan(self.rho):
            out.append(("rho", (), (-8.0, 8.0)))
        out += [("seasonal", (i,), (None, None)) for i in range(self.seasonal.size) if np.isnan(self.seasonal[i])]
        out += [("beta", (g, k), (None, None)) for g in range(s.n_regimes) for k in range(s.n_drivers)
                if np.isnan(self.beta[g, k])]
        if np.isnan(self.sigma_obs):
            out.append(("sigma_obs", (), (math.log(1e-6), math.log(10.0))))
        return out

    @property
    def size(self) -> int:
        return len(self.slots)

    @property
    def bounds(self) -> list[tuple[float | None, float | None]]:
        return [b for _, _, b in self.slots]

    def unpack(self, u: np.ndarray) -> Params:
        s = self.spec
        mu, kappa, sigma = self.mu.copy(), self.kappa.copy(), self.sigma.copy()
        rho, seasonal, beta, so = self.rho.copy(), self.seasonal.copy(), self.beta.copy(), self.sigma_obs.copy()
        gap = None
        for (name, idx, _), v in zip(self.slots, u):
            if name == "mu":
                mu = np.array(v)
            elif name == "kappa_slow":
                kappa[1] = math.exp(v)
            elif name == "kappa_gap":
                gap = math.exp(v)
            elif name == "kappa":
                kappa[0] = math.exp(v)
            elif name == "sigma_shared":
                sigma[:, idx[0]] = math.exp(v)
            elif name == "sigma":
                sigma[idx] = math.exp(v)
            elif name == "rho":
                rho = np.array(RHO_MAX * math.tanh(v))
            elif name == "seasonal":
                seasonal[idx] = v
            elif name == "beta":
                beta[idx] = v
            elif name == "sigma_obs":
                so = np.array(math.exp(v))
        if s.factors == 2 and gap is not None:
            kappa[0] = kappa[1] + gap
        for j in s.shared_sigma:
            sigma[:, j] = sigma[0, j]
        return Params(spec=s, mu=float(mu), kappa=kappa, sigma=sigma, rho=float(rho), seasonal=seasonal, beta=beta,
                      sigma_obs=float(so), theta=self.theta)

    def pack(self, p: Params) -> np.ndarray:
        u = []
        for name, idx, _ in self.slots:
            if name == "mu":
                u.append(p.mu)
            elif name == "kappa_slow":
                u.append(math.log(max(p.kappa[1], 1e-3)))
            elif name == "kappa_gap":
                u.append(math.log(max(p.kappa[0] - p.kappa[1], 1e-3)))
            elif name == "kappa":
                u.append(math.log(max(p.kappa[0], 1e-3)))
            elif name == "sigma_shared":
                u.append(math.log(p.sigma[0, idx[0]]))
            elif name == "sigma":
                u.append(math.log(p.sigma[idx]))
            elif name == "rho":
                u.append(math.atanh(max(min(p.rho / RHO_MAX, 0.999999), -0.999999)))
            elif name in ("seasonal", "beta"):
                u.append(getattr(p, name)[idx])
            elif name == "sigma_obs":
                u.append(math.log(max(p.sigma_obs, 1e-6)))
        return np.array(u, float)


def _flat(p: Params) -> np.ndarray:
    return np.r_[p.mu, p.kappa, p.sigma.ravel(), p.rho, p.seasonal, p.beta.ravel(), p.sigma_obs]


def _unflat(spec: ModelSpec, v: np.ndarray) -> dict:
    F, R, H, K = spec.factors, spec.n_regimes, 2 * spec.harmonics, spec.n_drivers
    i = 0
    out = {}
    for name, n, shape in (("mu", 1, ()), ("kappa", F, (F,)), ("sigma", R * F, (R, F)), ("rho", 1, ()),
                           ("seasonal", H, (H,)), ("beta", R * K, (R, K)), ("sigma_obs", 1, ())):
        chunk = v[i:i + n]
        out[name] = float(chunk[0]) if shape == () else chunk.reshape(shape).tolist()
        i += n
    return out


def _start(spec: ModelSpec, packer: _Packer, y, t, speeds) -> np.ndarray:
    """A starting vector: the level and seasonality by least squares, the variance split between the factors."""
    ok = np.isfinite(y)
    F, R = spec.factors, spec.n_regimes
    harm = [np.sin(2 * np.pi * (h + 1) * t / 365.25) for h in range(spec.harmonics)]
    harm = [c for h in range(spec.harmonics) for c in (harm[h], np.cos(2 * np.pi * (h + 1) * t / 365.25))]
    X = np.column_stack([np.ones(t.size)] + harm) if harm else np.ones((t.size, 1))
    coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    resid = y[ok] - X[ok] @ coef
    var = max(float(np.var(resid)), 1e-8)
    kap = np.array(speeds, float).reshape(F)
    share = np.array([0.3, 0.7]) if F == 2 else np.array([1.0])
    sig = np.sqrt(2.0 * np.maximum(kap, 1e-3) * share * var)
    p = Params(spec=spec, mu=float(coef[0]) if np.isnan(packer.mu) else float(packer.mu),
               kappa=np.where(np.isnan(packer.kappa), kap, packer.kappa) if F == 1 else _ordered(kap, packer.kappa),
               sigma=np.where(np.isnan(packer.sigma), np.tile(sig, (R, 1)), packer.sigma),
               rho=0.0 if F == 1 or np.isnan(packer.rho) else float(packer.rho),
               seasonal=np.where(np.isnan(packer.seasonal), coef[1:], packer.seasonal),
               beta=np.where(np.isnan(packer.beta), 0.0, packer.beta),
               sigma_obs=0.01 * math.sqrt(var) if np.isnan(packer.sigma_obs) else float(packer.sigma_obs),
               theta=packer.theta)
    return packer.pack(p)


def _ordered(grid: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Starting speeds: the fixed ones as they are, the free ones from the grid, moved only if a fixed speed has
    crossed them (fast above slow)."""
    k = np.where(np.isnan(fixed), grid, fixed)
    if k[0] <= k[1]:
        if np.isnan(fixed[0]):
            k[0] = 5.0 * k[1] + 1.0
        elif np.isnan(fixed[1]):
            k[1] = 0.1 * k[0]
        else:
            raise ValueError(f"fixed speeds must fall from fast to slow: {fixed}")
    return k


def _hessian(f, u: np.ndarray, rel: float = 1e-4) -> np.ndarray:
    n = u.size
    h = rel * np.maximum(1.0, np.abs(u))
    f0 = f(u)
    hess = np.empty((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        hess[i, i] = (f(u + ei) - 2 * f0 + f(u - ei)) / h[i] ** 2
        for j in range(i):
            ej = np.zeros(n)
            ej[j] = h[j]
            hess[i, j] = hess[j, i] = (f(u + ei + ej) - f(u + ei - ej) - f(u - ei + ej) + f(u - ei - ej)) / (
                4 * h[i] * h[j])
    return hess


def _standard_errors(packer: _Packer, f, u: np.ndarray) -> dict:
    spec = packer.spec
    try:
        hess = _hessian(f, u)
        cov_u = np.linalg.pinv(hess)
        base = _flat(packer.unpack(u))
        jac = np.empty((base.size, u.size))
        for i in range(u.size):
            d = 1e-6 * max(1.0, abs(u[i]))
            up, dn = u.copy(), u.copy()
            up[i] += d
            dn[i] -= d
            jac[:, i] = (_flat(packer.unpack(up)) - _flat(packer.unpack(dn))) / (2 * d)
        var = np.einsum("ij,jk,ik->i", jac, cov_u, jac)
        se = np.sqrt(np.where(var > 0, var, np.nan))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        se = np.full(_flat(packer.unpack(u)).size, np.nan)
    fixed_flat = _flat_fixed(packer)
    se = np.where(np.isfinite(fixed_flat), np.nan, se)
    return _unflat(spec, se)


def _flat_fixed(packer: _Packer) -> np.ndarray:
    """The flat vector of fixed values, NaN where estimated (a shared shock size counts as estimated)."""
    return np.r_[packer.mu, packer.kappa, packer.sigma.ravel(), packer.rho if packer.spec.factors == 2 else 0.0,
                 packer.seasonal, packer.beta.ravel(), packer.sigma_obs]


def _optimise(packer: _Packer, f, u0: np.ndarray):
    return optimize.minimize(f, u0, method="L-BFGS-B", bounds=packer.bounds,
                             options=dict(maxiter=600, maxfun=20000, ftol=1e-10, gtol=1e-6))


def fit(spec: ModelSpec, y, t_days, regime=None, drivers=None, fixed: dict | None = None, starts: int = 6,
        rng: np.random.Generator | None = None, theta=None, start: Params | None = None) -> FitResult:
    """Fit the model to a daily log-price series by maximum likelihood (module docstring). y has NaN where missing;
    t_days are consecutive integer days; regime and drivers as kalman.kalman_filter takes them. With two factors the
    correlation is fixed at RHO_DEFAULT unless fixed names it (IDENTIFICATION OF THE CORRELATION). start, a Params of
    the same spec, is tried before the grid (ESTIMATION)."""
    if start is not None and start.spec != spec:
        raise ValueError(f"start has spec {start.spec}, the fit {spec}")
    rng = np.random.default_rng(0) if rng is None else rng
    fixed = dict(fixed or {})
    if spec.factors == 2 and "rho" not in fixed:
        fixed["rho"] = RHO_DEFAULT
    packer = _Packer(spec, fixed, theta)
    probe = packer.unpack(_start(spec, packer, np.asarray(y, float), np.asarray(t_days),
                                 GRID2[0] if spec.factors == 2 else (GRID1[0],)))
    y_, t_, r_, _ = prepare(probe, y, t_days, regime, drivers)

    def f(u):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                v = -loglik(packer.unpack(u), y_, t_, r_, drivers)
        except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            return FAIL
        return v if math.isfinite(v) else FAIL

    grid = GRID2 if spec.factors == 2 else tuple((k,) for k in GRID1)
    speeds = list(grid[:max(1, min(starts, len(grid)))])
    while len(speeds) < starts:
        fast = float(np.exp(rng.uniform(np.log(5.0), np.log(500.0))))
        speeds.append((fast, fast * float(np.exp(rng.uniform(np.log(1e-3), np.log(0.2)))))
                      if spec.factors == 2 else (fast / 10.0,))
    best = None
    inits = ([packer.pack(start)] if start is not None else []) + [_start(spec, packer, y_, t_, sp) for sp in speeds]
    for u0 in inits:
        res = _optimise(packer, f, u0)
        if best is None or res.fun < best.fun:
            best = res
    return FitResult(params=packer.unpack(best.x), se=_standard_errors(packer, f, best.x), loglik=-float(best.fun),
                     n_obs=int(np.isfinite(y_).sum()), n_params=packer.size, converged=bool(best.success),
                     message=str(best.message))

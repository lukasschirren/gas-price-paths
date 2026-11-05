"""The model specification and its parameter set.

THE MODEL (docs/method.md). The log price of day t is

    y_t = mu + s(t) + sum_k beta[r_t, k] w[t, k] + sum_j x[t, j] + e_t,      e_t ~ N(0, sigma_obs^2)
    s(t) = sum_h ( a_h sin(2 pi h t / 365.25) + b_h cos(2 pi h t / 365.25) ),   seasonal = (a_1, b_1, a_2, b_2, ...)

with one or two mean-reverting factors x (ordered from fast to slow), each reverting to its target theta[r_t, j] at
its speed kappa[j] per year with shocks of size sigma[r_t, j] per square-root year, the two shocks correlated by rho.
The regime r_t of each day is given by the caller (dated, not estimated); a regime sets the shock sizes, the driver
weights beta and the targets, while speeds and the correlation are shared by every regime. A speed of zero (allowed
for the slowest factor) makes that factor a random walk, as in Schwartz and Smith (2000). t counts days, so the
seasonal curve keeps its phase across years.

WHERE THE DRIVERS ENTER. With drivers_enter "observation" (the default) the drivers w move the log price of their own
day only, as written above. With drivers_enter "fast" they enter the fast factor's state instead, and the observation
loses its driver term:
    x[t, 0] = theta[r_t, 0] + phi_0 (x[t-1, 0] - theta[r_t, 0]) + sum_k beta[r_t, k] w[t, k] + eta[t, 0],
with phi_0 = exp(-kappa[0] dt). A day's anomaly then moves the fast factor by beta . w, and the move fades at the fast
factor's speed, so an anomaly held for several half-lives moves the log price by beta . w / (1 - phi_0). The response
    psi_t = phi_0 psi_{t-1} + beta[r_t] . w_t,    psi zero before the first day,
is fixed by the drivers, so the fast factor is psi plus a factor without drivers (driver_effect); the filter and the
simulation use that split.

ModelSpec fixes the shape of a model (how many factors, harmonics, regimes and drivers, which factors share one shock
size across regimes, the day step); Params holds one set of values for it and checks them against the shape. Both
round-trip through plain dictionaries of floats and lists, so a caller can keep them in YAML or JSON; the package
itself writes no files.

References: Schwartz, E.S. (1997), The Journal of Finance 52(3), 923-973, doi:10.1111/j.1540-6261.1997.tb02721.x;
Schwartz, E. and Smith, J.E. (2000), Management Science 46(7), 893-911, doi:10.1287/mnsc.46.7.893.12034.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

YEAR_DAYS = 365.25
DRIVERS_ENTER = ("observation", "fast")     # module docstring, WHERE THE DRIVERS ENTER


@dataclass(frozen=True)
class ModelSpec:
    """The shape of a model: factors (1 or 2, fast to slow), seasonal harmonics, regime and driver names, the factors
    whose shock size is one value for every regime, the day step in years, and where the drivers enter (module
    docstring)."""
    factors: int = 2
    harmonics: int = 2
    regimes: tuple[str, ...] = ("normal",)
    drivers: tuple[str, ...] = ()
    shared_sigma: tuple[int, ...] = ()
    dt: float = 1.0 / YEAR_DAYS
    drivers_enter: str = "observation"

    def __post_init__(self):
        if self.factors not in (1, 2):
            raise ValueError(f"factors must be 1 or 2, not {self.factors}")
        if self.harmonics < 0:
            raise ValueError("harmonics must be zero or more")
        if not self.regimes or len(set(self.regimes)) != len(self.regimes):
            raise ValueError(f"regimes must be distinct names, at least one: {self.regimes}")
        if len(set(self.drivers)) != len(self.drivers):
            raise ValueError(f"drivers must be distinct names: {self.drivers}")
        if any(j not in range(self.factors) for j in self.shared_sigma):
            raise ValueError(f"shared_sigma names factors 0..{self.factors - 1}: {self.shared_sigma}")
        if not self.dt > 0:
            raise ValueError("dt must be positive")
        if self.drivers_enter not in DRIVERS_ENTER:
            raise ValueError(f"drivers_enter is one of {DRIVERS_ENTER}, not {self.drivers_enter!r}")
        object.__setattr__(self, "regimes", tuple(self.regimes))
        object.__setattr__(self, "drivers", tuple(self.drivers))
        object.__setattr__(self, "shared_sigma", tuple(int(j) for j in self.shared_sigma))

    @property
    def n_regimes(self) -> int:
        return len(self.regimes)

    @property
    def n_drivers(self) -> int:
        return len(self.drivers)

    def to_dict(self) -> dict:
        return dict(factors=self.factors, harmonics=self.harmonics, regimes=list(self.regimes),
                    drivers=list(self.drivers), shared_sigma=list(self.shared_sigma), dt=float(self.dt),
                    drivers_enter=self.drivers_enter)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        return cls(factors=int(d["factors"]), harmonics=int(d["harmonics"]), regimes=tuple(d["regimes"]),
                   drivers=tuple(d.get("drivers", ())), shared_sigma=tuple(d.get("shared_sigma", ())),
                   dt=float(d.get("dt", 1.0 / YEAR_DAYS)), drivers_enter=str(d.get("drivers_enter", "observation")))


ARRAYS = ("kappa", "sigma", "seasonal", "beta", "theta")
SCALARS = ("mu", "rho", "sigma_obs")


@dataclass
class Params:
    """One set of values for a ModelSpec (module docstring): mu, kappa (factors,), sigma (regimes, factors), rho,
    seasonal (2 x harmonics,), beta (regimes, drivers), sigma_obs, theta (regimes, factors)."""
    spec: ModelSpec
    mu: float
    kappa: np.ndarray
    sigma: np.ndarray
    rho: float
    seasonal: np.ndarray
    beta: np.ndarray
    sigma_obs: float
    theta: np.ndarray = field(default=None)

    def __post_init__(self):
        s = self.spec
        self.mu, self.rho, self.sigma_obs = float(self.mu), float(self.rho), float(self.sigma_obs)
        self.kappa = np.asarray(self.kappa, float).reshape(-1)
        self.sigma = np.asarray(self.sigma, float)
        self.seasonal = np.asarray(self.seasonal, float).reshape(-1)
        self.beta = np.asarray(self.beta, float)
        self.theta = (np.zeros((s.n_regimes, s.factors)) if self.theta is None
                      else np.asarray(self.theta, float))
        if self.kappa.shape != (s.factors,):
            raise ValueError(f"kappa needs {s.factors} speeds, has shape {self.kappa.shape}")
        if np.any(self.kappa < 0) or not np.all(np.isfinite(self.kappa)):
            raise ValueError(f"speeds must be finite and non-negative: {self.kappa}")
        if s.factors == 2 and not self.kappa[0] > self.kappa[1]:
            raise ValueError(f"speeds must fall from the fast factor to the slow one: {self.kappa}")
        if self.sigma.shape != (s.n_regimes, s.factors):
            raise ValueError(f"sigma needs shape {(s.n_regimes, s.factors)}, has {self.sigma.shape}")
        if np.any(self.sigma <= 0) or not np.all(np.isfinite(self.sigma)):
            raise ValueError(f"shock sizes must be finite and positive: {self.sigma}")
        for j in s.shared_sigma:
            if not np.allclose(self.sigma[:, j], self.sigma[0, j], rtol=1e-12, atol=0.0):
                raise ValueError(f"factor {j} shares one shock size across regimes: {self.sigma[:, j]}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must lie in [-1, 1], not {self.rho}")
        if s.factors == 1 and self.rho != 0.0:
            raise ValueError("one factor has no correlation; rho must be 0")
        if self.seasonal.shape != (2 * s.harmonics,):
            raise ValueError(f"seasonal needs {2 * s.harmonics} coefficients, has {self.seasonal.size}")
        if self.beta.shape != (s.n_regimes, s.n_drivers):
            raise ValueError(f"beta needs shape {(s.n_regimes, s.n_drivers)}, has {self.beta.shape}")
        if not self.sigma_obs >= 0:
            raise ValueError(f"sigma_obs must be zero or more, not {self.sigma_obs}")
        if self.theta.shape != (s.n_regimes, s.factors):
            raise ValueError(f"theta needs shape {(s.n_regimes, s.factors)}, has {self.theta.shape}")

    def seasonal_at(self, t_days) -> np.ndarray:
        """s(t) for integer or fractional days t (module docstring)."""
        t = np.asarray(t_days, float)
        out = np.zeros_like(t)
        for h in range(self.spec.harmonics):
            w = 2.0 * np.pi * (h + 1) * t / YEAR_DAYS
            out = out + self.seasonal[2 * h] * np.sin(w) + self.seasonal[2 * h + 1] * np.cos(w)
        return out

    def driver_impulse(self, regime, drivers) -> np.ndarray:
        """beta[r_t] . w_t by day: regime (days,) indexes the regimes, drivers is (days, drivers)."""
        r = np.asarray(regime).astype(np.int64).reshape(-1)
        return np.einsum("tk,tk->t", self.beta[r], np.asarray(drivers, float))

    def driver_effect(self, regime, drivers) -> np.ndarray:
        """The drivers' part of the log price by day (module docstring, WHERE THE DRIVERS ENTER): the impulse
        beta[r_t] . w_t on its own day, or with drivers_enter "fast" its response psi, zero before the first day."""
        u = self.driver_impulse(regime, drivers)
        if self.spec.drivers_enter == "observation":
            return u
        return signal.lfilter([1.0], [1.0, -math.exp(-self.kappa[0] * self.spec.dt)], u)

    def half_life_days(self) -> np.ndarray:
        """Each factor's half-life in days (infinite for a random walk)."""
        with np.errstate(divide="ignore"):
            return np.where(self.kappa > 0, np.log(2.0) / self.kappa * YEAR_DAYS, np.inf)

    def replace(self, **kw) -> "Params":
        """A copy with some values replaced, validated again."""
        d = {k: getattr(self, k) for k in ("spec",) + SCALARS + ARRAYS}
        d.update(kw)
        return Params(**d)

    def to_dict(self) -> dict:
        d = {"spec": self.spec.to_dict()}
        d.update({k: float(getattr(self, k)) for k in SCALARS})
        d.update({k: getattr(self, k).tolist() for k in ARRAYS})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        spec = ModelSpec.from_dict(d["spec"])
        return cls(spec=spec, mu=d["mu"], kappa=np.array(d["kappa"], float),
                   sigma=np.array(d["sigma"], float).reshape(spec.n_regimes, spec.factors), rho=d["rho"],
                   seasonal=np.array(d["seasonal"], float),
                   beta=np.array(d["beta"], float).reshape(spec.n_regimes, spec.n_drivers),
                   sigma_obs=d["sigma_obs"],
                   theta=np.array(d.get("theta", np.zeros((spec.n_regimes, spec.factors))), float)
                   .reshape(spec.n_regimes, spec.factors))

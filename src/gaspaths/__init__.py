"""gas-price-paths: one- and two-factor mean-reverting models of a daily gas price.

The log price of a day is a level, seasonal harmonics, external drivers weighted by the day's regime, one or two
mean-reverting factors and measurement noise (docs/design.md, section 3). The package dates a stressed regime from the
series' volatility, fits the model to a daily series by maximum likelihood through a Kalman filter, compares nested
fits, and simulates seeded daily paths from it, anchored to a level curve if asked. It reads and writes no files;
callers pass arrays and frames.
"""
from gaspaths.diagnostics import variogram
from gaspaths.fit import FitResult, fit, fixed_half_life, likelihood_ratio
from gaspaths.kalman import FilterResult, State, kalman_filter
from gaspaths.model import ModelSpec, Params
from gaspaths.regimes import episodes, rolling_volatility, stressed_days, windows
from gaspaths.simulate import anchor_offsets, annual_means, log_deviation, simulate_annual_means, simulate_factors

__version__ = "0.1.0"
__all__ = ["ModelSpec", "Params", "State", "FilterResult", "FitResult", "kalman_filter", "fit", "fixed_half_life",
           "likelihood_ratio", "rolling_volatility", "episodes", "windows", "stressed_days", "variogram",
           "simulate_factors", "log_deviation", "anchor_offsets", "annual_means", "simulate_annual_means",
           "__version__"]

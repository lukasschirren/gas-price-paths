"""Checking a fit against its series: the variogram.

VARIOGRAM. How far the log price moves over h days, the root mean square of y[t + h] - y[t] over the pairs of days
that both exist (docs/method.md, checking a fit). Its shape tells the factor structure apart. A random walk's grows
with the square root of h; one mean-reverting factor's levels off at its stationary spread times the square root of
two after a few half-lives; a fast and a slow factor give a quick rise over days and a slow one over months to
years. Comparing the series' variogram with that of paths simulated from a fit (simulate_factors and log_deviation,
the drivers and regimes of the record added) shows at which horizons the fitted model moves too little or too much.
gb-power-model's uncertainty figure 2 makes that comparison for one and two factors on GB's System Average Price.

Written first for gb-power-model's figures and moved here before the first release (5 Nov 2026).
"""
from __future__ import annotations

import numpy as np


def variogram(y, lags) -> np.ndarray:
    """sqrt(mean of (y[t + h] - y[t])^2) for each lag h in lags (positive integers, in days), over the pairs where
    both days are finite; NaN for a lag with no such pair. y is a daily series, NaN on missing days."""
    y = np.asarray(y, float)
    out = []
    for h in np.asarray(lags, int):
        if h < 1:
            raise ValueError(f"lags are positive whole days, not {h}")
        d = y[h:] - y[:-h]
        d = d[np.isfinite(d)]
        out.append(np.sqrt(np.mean(d * d)) if d.size else np.nan)
    return np.array(out)

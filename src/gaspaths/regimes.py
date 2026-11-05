"""Dating a stressed regime: the days of a daily log-price series whose volatility is high.

The model takes the regime of every day from the caller, dated and not estimated (model.py). This module dates one
such regime, a stressed market, by a rule on the series itself (docs/method.md, dating a stressed regime). A day is
stressed when the centred WINDOW-day standard deviation of daily log changes exceeds MULT times its median over the
series (rolling_volatility). Runs of stressed days shorter than MIN_LEN days are dropped, then runs less than
MERGE_GAP days apart are merged, the days between them stressed too (episodes), so a calm week inside a crisis does
not split it and a few volatile days do not make one.

The defaults are the settings gb-power-model uses on GB's daily System Average Price of gas, 2018-2025, where they
mark 13 February - 18 March 2018, 13-29 October 2019 and 15 December 2021 - 5 December 2022. Another series may need
others: the multiple sets how far above its usual volatility a market must move, and the two lengths how long a
stressed spell must last and how close two spells must be to count as one. Estimated regime switching (a hidden
Markov chain) is not in the package (README, Roadmap).

Written first in gb-power-model's gas adapter and moved here before the first release (5 Nov 2026), so that a user
of the library can date the regimes it takes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 31        # days of the centred window
MULT = 2.5         # times the median rolling volatility
MIN_LEN = 14       # days: shorter runs are dropped
MERGE_GAP = 45     # days: runs closer than this are merged


def _series(x, dtype=float) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(np.asarray(x, dtype))


def rolling_volatility(logp, window: int = WINDOW) -> pd.Series:
    """The centred window-day standard deviation of daily log changes, where at least window // 2 + 1 changes exist
    in the window; logp a Series (or array) of daily log prices with NaN on missing days."""
    return _series(logp).diff().rolling(window, center=True, min_periods=window // 2 + 1).std()


def episodes(flag, min_len: int = MIN_LEN, merge_gap: int = MERGE_GAP) -> pd.Series:
    """Runs of True shorter than min_len days dropped, then runs less than merge_gap days apart merged (the days
    between them set True); flag a boolean Series (or array) by day, NaN read as False."""
    flag = _series(flag, object)
    v = flag.to_numpy()
    f = np.where(pd.isna(v), False, v).astype(bool)
    runs = []
    i = 0
    while i < f.size:
        if f[i]:
            j = i
            while j + 1 < f.size and f[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] - 1 < merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = np.zeros(f.size, bool)
    for a, b in merged:
        out[a:b + 1] = True
    return pd.Series(out, index=flag.index)


def windows(flag) -> list[tuple]:
    """The (first, last) day of each run of True: as "YYYY-MM-DD" for a date index, the index labels otherwise."""
    flag = _series(flag, bool)
    s = flag.astype(bool).to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, s.astype(int), 0]))
    days = flag.index
    if isinstance(days, pd.DatetimeIndex):
        return [(days[a].strftime("%Y-%m-%d"), days[b - 1].strftime("%Y-%m-%d"))
                for a, b in zip(edges[::2], edges[1::2])]
    return [(days[a], days[b - 1]) for a, b in zip(edges[::2], edges[1::2])]


def stressed_days(logp, mult: float = MULT, window: int = WINDOW, min_len: int = MIN_LEN,
                  merge_gap: int = MERGE_GAP) -> pd.Series:
    """The stressed days of a daily log-price series (module docstring): True where the rolling volatility exceeds
    mult times its median, kept as episodes of at least min_len days, merged across gaps under merge_gap days. The
    result is a boolean Series on logp's index, to pass as fit's regime (stressed 1, normal 0) after astype(int)."""
    vol = rolling_volatility(logp, window)
    return episodes(vol > mult * vol.median(), min_len, merge_gap)

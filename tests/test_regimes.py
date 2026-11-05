"""Dating a stressed regime (regimes.py) on synthetic series: short runs dropped, close runs merged, a known volatile
spell found, arrays and integer indexes accepted."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gaspaths import episodes, rolling_volatility, stressed_days, windows


def _flag() -> pd.Series:
    flag = pd.Series(False, index=pd.date_range("2020-01-01", periods=400, freq="D"))
    flag.iloc[10:16] = True            # 6 days: dropped
    flag.iloc[100:130] = True          # 30 days
    flag.iloc[150:190] = True          # 40 days, 20 days after: merged with the one before
    flag.iloc[300:320] = True          # 20 days, 110 days after: its own episode
    return flag


def test_episodes_drop_short_runs_and_merge_close_ones():
    e = episodes(_flag(), min_len=14, merge_gap=45)
    assert not e.iloc[10:16].any() and e.iloc[100:190].all() and e.iloc[300:320].all()
    assert int(e.sum()) == 90 + 20
    assert windows(e) == [("2020-04-10", "2020-07-08"), ("2020-10-27", "2020-11-15")]


def test_a_gap_of_exactly_merge_gap_days_is_kept():
    f = pd.Series(False, index=range(100))
    f.iloc[0:20] = True
    f.iloc[30:50] = True               # 10 days between the runs
    assert windows(episodes(f, min_len=14, merge_gap=10)) == [(0, 19), (30, 49)]
    assert windows(episodes(f, min_len=14, merge_gap=11)) == [(0, 49)]


def test_episodes_and_windows_take_arrays_and_missing_flags():
    f = np.zeros(60, bool)
    f[5:25] = True
    e = episodes(f, min_len=14, merge_gap=45)
    assert isinstance(e, pd.Series) and int(e.sum()) == 20 and windows(e) == [(5, 24)]
    with_nan = pd.Series([np.nan] * 3 + [True] * 15 + [np.nan] * 3)
    assert windows(episodes(with_nan, min_len=14)) == [(3, 17)]


def test_stressed_days_cover_a_known_episode():
    rng = np.random.default_rng(1)
    sd = np.full(600, 0.02)
    sd[200:260] = 0.12                   # six times the calm volatility for 60 days
    logp = pd.Series(np.cumsum(sd * rng.standard_normal(600)), index=pd.date_range("2019-01-01", periods=600))
    s = stressed_days(logp)
    flagged = np.flatnonzero(s.to_numpy())
    assert s.dtype == bool and s.index.equals(logp.index)
    assert s.iloc[200:260].all() and flagged.min() >= 180 and flagged.max() <= 280
    assert not s.iloc[:150].any() and not s.iloc[320:].any()
    assert stressed_days(logp.to_numpy()).to_numpy().tolist() == s.to_numpy().tolist()


def test_a_calm_series_has_no_stressed_day():
    rng = np.random.default_rng(2)
    logp = np.cumsum(0.03 * rng.standard_normal(1000))
    assert not stressed_days(logp).any()


def test_rolling_volatility_needs_half_the_window():
    logp = pd.Series(np.cumsum(np.ones(40)))                 # every change is 1, so the spread is 0 where defined
    v = rolling_volatility(logp, window=11)
    assert v.notna().sum() > 0 and np.allclose(v.dropna(), 0.0)
    gappy = logp.copy()
    gappy.iloc[::2] = np.nan                                  # no two consecutive days: no change at all
    assert rolling_volatility(gappy, window=11).isna().all()

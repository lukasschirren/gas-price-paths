"""Fit the two-factor model to a daily price series read from a CSV file the user names, and print the fit.

The file needs a date column and a price column (any positive unit, for instance ONS's System Average Price of gas
in pence per kWh). Dates are put on a daily calendar; a missing day becomes a missing observation, which the filter
predicts through. The log price is fitted with two factors and two seasonal harmonics, the correlation of the
factors' shocks fixed at zero (docs/method.md explains why a spot series cannot identify it).

    python examples/fit_daily_series.py --csv sap.csv --date-col date --price-col price
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import gaspaths as gp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True, help="the CSV file")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--price-col", default="price")
    ap.add_argument("--factors", type=int, default=2, choices=(1, 2))
    ap.add_argument("--harmonics", type=int, default=2)
    ap.add_argument("--starts", type=int, default=4)
    a = ap.parse_args(argv)
    d = pd.read_csv(a.csv, parse_dates=[a.date_col])
    s = d.set_index(a.date_col)[a.price_col].astype(float)
    if s.index.duplicated().any():
        raise SystemExit("the file has repeated dates")
    s = s.asfreq("D")                                   # missing days become NaN
    if (s.dropna() <= 0).any():
        raise SystemExit("prices must be positive to take logs")
    y = np.log(s.to_numpy())
    t = (s.index - pd.Timestamp("1970-01-01")).days.to_numpy()
    res = gp.fit(gp.ModelSpec(factors=a.factors, harmonics=a.harmonics), y, t, starts=a.starts)
    p = res.params
    print(f"{s.index[0].date()} to {s.index[-1].date()}: {res.n_obs} observations, log-likelihood {res.loglik:.1f}")
    print("half-lives (days):", ", ".join(f"{h:.1f}" for h in p.half_life_days()))
    print("shock sizes (per sqrt year):", ", ".join(f"{v:.3f}" for v in p.sigma[0]))
    print(f"seasonal: {np.round(p.seasonal, 3).tolist()}, measurement noise {p.sigma_obs:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# gas-price-paths

One- and two-factor mean-reverting models of a daily gas price, fitted by maximum likelihood through a Kalman filter
and used to simulate seeded daily price paths. The one-factor model is the exponential Vasicek process of Schwartz
(1997); the two-factor model has the structure of Schwartz and Smith (2000), a fast factor for short-term deviations
and a slow one for the level, with the slow factor mean-reverting as well. Both take seasonal harmonics, external
drivers such as the weather, and dated regimes, so a stressed market can have larger shocks and a stronger response
to the weather than a calm one.

The package was written for the projection of gb-power-model, a merit-order model of the GB electricity market,
where it gives the Monte Carlo its annual gas prices and the merit order its daily gas shapes. It knows nothing about
GB; it fits any daily price series a caller passes.

Status: v0.1.0, the first release (`CHANGELOG.md`).

## Install

```bash
pip install "gas-price-paths @ git+https://github.com/lukasschirren/gas-price-paths"
# or, from a clone, with the development extras
pip install -e ".[dev]"
```

It needs Python 3.12 or later and numpy, scipy and pandas, nothing else.

## Example

```python
import numpy as np
import gaspaths as gp

spec = gp.ModelSpec(factors=2, harmonics=2)
res = gp.fit(spec, y, t_days)                    # y the daily log price, t_days consecutive integer days
print(res.params.to_dict(), res.loglik)

last = gp.kalman_filter(res.params, y, t_days).last                  # the state on the last observed day
x = gp.simulate_factors(res.params, n_days=3650, n=20_000, rng=np.random.default_rng(1), start=last)
dev = gp.log_deviation(res.params, t_days[-1] + 1 + np.arange(3650), x)   # log price less the level, by path and day
```

With a stressed regime dated from the series' own volatility, and a sensitivity on the slow half-life:

```python
import pandas as pd

logp = pd.Series(y, index=dates)                 # daily log prices on a date index, NaN on missing days
stressed = gp.stressed_days(logp)                # True on the days of stressed episodes
print(gp.windows(stressed))                      # their first and last days
spec = gp.ModelSpec(factors=2, harmonics=2, regimes=("normal", "stressed"))
res = gp.fit(spec, y, t_days, regime=stressed.astype(int).to_numpy())
slow4 = gp.fit(spec, y, t_days, regime=stressed.astype(int).to_numpy(), fixed=gp.fixed_half_life(res.params, 4.0))
```

`examples/fit_daily_series.py` fits a series read from a CSV file the user names.

## The model

The log price of a day is a level, seasonal harmonics, the drivers weighted by the day's regime, the factors and
measurement noise. The drivers may instead enter the fast factor, so that a day's anomaly keeps moving the price
after the day and its effect fades at the factor's speed. Each factor reverts to its target at its own speed and steps from day to day by its exact
transition, so no discretisation error enters. `docs/method.md` sets out the equations, the filter, the estimation
and the anchoring of paths to a level curve; `docs/design.md` records the design decisions.

Regimes are dated, not estimated. `stressed_days` dates a stressed regime by a rule on the series itself: a day is
stressed when the month-long rolling volatility of daily log changes exceeds a multiple of its median, kept in
episodes of a minimum length and merged across short calm gaps. `likelihood_ratio` compares nested fits (two regimes
sharing a shock size, for instance), `fixed_half_life` holds a poorly identified slow speed for a sensitivity, and
`variogram` compares how far a series and simulated paths move over every horizon from a day to years.

## Roadmap

Calibration on futures curves, jumps in the fast factor, estimated regime switching and Student-t shocks are not in
v0.1.0.

## References

Schwartz, E.S. (1997) 'The stochastic behavior of commodity prices: implications for valuation and hedging', *The
Journal of Finance*, 52(3), pp. 923-973. Available at: <https://doi.org/10.1111/j.1540-6261.1997.tb02721.x>.

Schwartz, E. and Smith, J.E. (2000) 'Short-term variations and long-term dynamics in commodity prices', *Management
Science*, 46(7), pp. 893-911. Available at: <https://doi.org/10.1287/mnsc.46.7.893.12034>.

## Licence and citation

BSD-3-Clause (`LICENSE`). To cite the package, use `CITATION.cff`.

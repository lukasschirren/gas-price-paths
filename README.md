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

Status: pre-release (0.1.0.dev0). The interface may change until v0.1.0.

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

`examples/fit_daily_series.py` fits a series read from a CSV file the user names.

## The model

The log price of a day is a level, seasonal harmonics, the drivers weighted by the day's regime, the factors and
measurement noise. The drivers may instead enter the fast factor, so that a day's anomaly keeps moving the price
after the day and its effect fades at the factor's speed. Each factor reverts to its target at its own speed and steps from day to day by its exact
transition, so no discretisation error enters. `docs/method.md` sets out the equations, the filter, the estimation
and the anchoring of paths to a level curve; `docs/design.md` records the design decisions.

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

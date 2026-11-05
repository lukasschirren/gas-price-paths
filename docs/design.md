# gas-price-paths: design note

Design note of 4 November 2026, agreed with the author in conversation that day. It is the founding document of the
repository `gas-price-paths` (import package `gaspaths`) and moves into that repository as `docs/design.md` with its
first commit. The first user of the library is gb-power-model, whose integration has its own note
(`docs/plans/2026-11-04-gas-paths-design.md` in that repository).

## 1. Purpose

The library fits one- and two-factor mean-reverting models to a daily gas price series and simulates daily price
paths from the fitted model. A search on 4 November 2026 found no installable Python package that does this. Schwartz
and Smith's two-factor model exists for MATLAB (File Exchange) and R (`NFCP`, `PDSim`), and `cmdty-storage` simulates
risk-neutral multi-factor prices through .NET. Four small study repositories fit the model to oil futures in
Python, none with a release. None of them handles a gas spot series with seasonality, dated regimes or external drivers
such as the weather.

The one-factor model is the exponential Vasicek process, the mean-reverting process of Vasicek (1977) applied to the
log price, which Schwartz (1997) brought to commodities as his model 1. The two-factor model has the structure of
Schwartz and Smith (2000), a fast factor for short-term deviations and a slow one for the level, here with the slow
factor mean-reverting too. A reversion speed of zero for the slow factor gives their original random walk.

## 2. Decisions

| Decision | Choice |
| --- | --- |
| Name | repository and distribution `gas-price-paths`, import package `gaspaths`; the name says what a user gets, and the candidates checked (`gas-price-paths`, `gaspaths`, `gasfactor`) were free on PyPI on 4 November 2026 |
| Location | `C:\Users\ls2823\OneDrive - Imperial College London\0_PUBLICATION\Repositories\gas-price-paths`; GitHub `lukasschirren/gas-price-paths` |
| Visibility | private until v0.1.0 passes its tests and its README and method document are written, then public |
| Licence | BSD-3-Clause for the code, as for gb-power-data and gb-power-model |
| Dependencies | numpy, scipy and pandas; pytest for the tests; statsmodels for one development check only |
| Data | none tracked; the library reads and writes no files, and callers pass arrays and frames |
| Branches | `main` until v0.1.0; after that, short feature branches merged into tagged releases, never a long-lived branch for one user |
| Versions | semantic versions as git tags, an entry in `CHANGELOG.md` for each, citation metadata in `CITATION.cff` |
| Scope of v0.1.0 | what gb-power-model needs (sections 3-5), with the general pieces it first kept in its own code (the volatility rule for a stressed regime, the likelihood ratio of nested fits, a fixed half-life with the long-run spread kept, the variogram; moved here on 5 Nov 2026), so that a user can date the regimes the model takes and repeat its checks; calibration on futures curves, jumps and estimated regime switching wait on the roadmap (section 8) |

## 3. The model

The log price of day t is the sum of a level, seasonal harmonics, external drivers weighted by the day's regime,
mean-reverting factors (one or two in v0.1.0) and measurement noise.

```text
y_t       = mu + s(t) + sum_k beta[k, r_t] w[k, t] + sum_j x[j, t] + e_t,          e_t ~ N(0, sigma_e^2)
s(t)      = sum_{h=1..H} ( a_h sin(2 pi h t / 365.25) + b_h cos(2 pi h t / 365.25) )
x[j, t]   = theta[j, r_t] + phi_j (x[j, t-1] - theta[j, r_t]) + eta[j, t],          phi_j = exp(-kappa_j dt)
Cov(eta[i, t], eta[j, t]) = rho_ij sigma[i, r_t] sigma[j, r_t] (1 - phi_i phi_j) / (kappa_i + kappa_j)
```

The step dt is one day in years (1/365.25), so a reversion speed kappa is per year and a factor's half-life is
ln 2 / kappa. A speed of zero makes the factor a random walk, whose shock covariance is the limit rho sigma sigma dt.
The caller gives the regime of every day, so regimes are dated and not estimated, and the day entered sets the shock
of the step into it. `regimes.stressed_days` dates a stressed regime by a rule on the series' volatility
(`docs/method.md`, dating a stressed regime). A regime sets the shock sizes sigma
and the driver weights beta, and in simulation it may also move a factor's target theta, which is zero in fitting.
Reversion speeds and correlations are shared by all regimes. The factors are ordered from fast to slow, which fixes
their labels. The drivers enter the observation as written, or, with `drivers_enter="fast"`, the fast factor's
transition, where a day's anomaly adds beta . w to the factor and fades at its speed (`docs/method.md`, where the
drivers enter). In the second form a day's anomaly keeps moving the price after the day, fading at the fast
factor's speed. A spot series cannot identify the correlation of the two factors' shocks (section 5), so the fit fixes it
at zero by default. The filter and the simulation also accept a correlation of plus or minus one, whose
shock covariance is singular.

## 4. Modules

| Module | Holds |
| --- | --- |
| `model.py` | the model specification (factors, harmonics, regimes, drivers, fixed parameters) and the parameter set, read from and written to a plain dictionary |
| `ou.py` | the exact one-day transition per regime, the stationary covariance, and the conditional mean and covariance h days ahead |
| `kalman.py` | the Kalman filter, its Gaussian log-likelihood, the filtered states and covariances, and the state on the last day; a missing day only predicts |
| `fit.py` | maximum likelihood on transformed parameters (logs for speeds and shock sizes, tanh for a correlation, ordered speeds) by L-BFGS-B from several starts, fixed parameters, standard errors,
the correlation fixed at zero by default; the entries that fix a half-life and keep the long-run spread, and the likelihood ratio of nested fits |
| `simulate.py` | seeded paths from a given state or the stationary distribution, with a regime and drivers per day; offsets that anchor paths to a level curve; annual means |
| `regimes.py` | a stressed regime dated by the rolling volatility of daily log changes, its episodes and their windows |
| `diagnostics.py` | the variogram of a series or a simulated path |

The interface sketch below shows the intended calls, whose names the implementation plan may change.

```python
import numpy as np
import gaspaths as gp

spec = gp.ModelSpec(factors=2, harmonics=2, regimes=("normal", "stressed"), drivers=("wind", "demand"))
res = gp.fit(spec, y, days, regime=r, drivers=w, starts=8)          # parameters, standard errors, log-likelihood
last = gp.kalman_filter(res.params, y, days, regime=r, drivers=w).last   # mean and covariance on the last day
x = gp.simulate(res.params, days_ahead, n=20_000, start=last, regime=r_ahead, rng=np.random.default_rng(1))
```

## 5. Numerical choices

Each day steps by the exact transition of the mean-reverting factors, so simulated moments equal those of the
continuous model at every horizon and no step size has to be chosen. Factors with a positive reversion speed start
from their stationary distribution, and a random-walk factor starts from a diffuse prior with a large variance.

A spot series cannot identify the correlation of the two factors' shocks, which a test found on 4 November
2026. The Gaussian likelihood of the series depends only on its autocovariances. For a lag of one day
or more these are A phi_1^h + B phi_2^h, where A and B each combine the shock sizes with the correlation. The data
therefore pin five numbers (two speeds, A, B and the noise) for six parameters. The fit fixes the
correlation at zero unless the caller fixes another value, which changes how a path splits between the
factors and nothing about its distribution. Exploratory fits that left it free ran it to plus one, since the
optimiser drifted along the flat ridge. Standard errors come from the inverse
numerical Hessian of the negative log-likelihood on the transformed scale, mapped back by the delta method. They are
labelled Gaussian-likelihood errors, since daily gas price changes are fat-tailed.

Anchoring scales paths to a level curve L, a price for each day. With mean anchoring the offset of year y is

```text
a_y = V / 2 + ln mean_{t in y} exp( s(t) + sum_k beta[k, r_t] w[k, t] )
```

where V is the stationary variance of the sum of the factors, so the expected annual mean of L exp(y - mu - a_y)
equals the annual mean of L in the stationary state. Median anchoring drops V / 2, which leaves the median path on
the curve and the expected price above it.

The filter loops over days with explicit small-matrix arithmetic in numpy, and simulation vectorises over paths, so
20,000 paths of ten years take seconds.

## 6. Tests and rules

| Rule | Test |
| --- | --- |
| imports are numpy, scipy, pandas and the standard library only, nothing from `meritorder`, `gbpowerdata`, `gbabm`, `spikeabm` or `cmmodel` | `test_imports.py` imports every module in a fresh interpreter |
| no data beyond small synthetic fixtures, no licensed-data markers (`nordpool`, `n2ex`, `epex`, `dayahead`), no tokens, no file over 5 MB | `test_repo_hygiene.py` |
| every draw comes from a generator the caller passes, and nothing seeds a global one | `test_simulate.py` (same seed, same paths) and a scan of the source |
| the daily step | `test_ou.py` compares simulated means, variances and autocovariances with the closed forms, including a zero speed and a correlation of one |
| the filter | `test_kalman.py` checks that one factor gives the exact AR(1) likelihood, and covers missing days, regimes, drivers and a comparison with statsmodels when it is installed |
| fitting | `test_fit.py` recovers known parameters of synthetic two-factor data within stated tolerances, keeps fixed parameters fixed, and shows that fits at two fixed correlations reach the same optimum |
| simulation | `test_simulate.py` checks that anchoring hits its target, that a start away from the target reverts at the slow factor's rate, and that a regime moves the target |
| regimes | `test_regimes.py` drops short runs, merges close ones at the exact gap, finds a known volatile spell and none in a calm series |
| diagnostics | `test_diagnostics.py` checks the variogram against a random walk's square root and one factor's closed form |

## 7. Layout

```text
gas-price-paths/
  README.md  CLAUDE.md  LICENSE  CHANGELOG.md  CITATION.cff  pyproject.toml  environment.yml
  src/gaspaths/  __init__.py  model.py  ou.py  kalman.py  fit.py  simulate.py  regimes.py  diagnostics.py
  tests/         test_imports.py  test_repo_hygiene.py  test_ou.py  test_kalman.py  test_fit.py  test_simulate.py
                 test_regimes.py  test_diagnostics.py
  docs/          design.md (this note)  method.md (the maths, with references)
  examples/      fit_daily_series.py (a daily price series from a CSV the user names)
```

The working rules in `CLAUDE.md` follow those of gb-power-data and gb-power-model. Commits and pushes happen only
when asked, staging names explicit paths, a commit message is one sentence, docstrings are long and cite their
sources, and names describe function without dates, authors or versions.

## 8. Roadmap

Four extensions are left for later versions, each for a user who has the data or the need. They are calibration
on futures curves (the measurement equation of Schwartz and Smith with its risk premia), jumps in the fast factor,
estimated regime switching and Student-t shocks.

## References

Durbin, J. and Koopman, S.J. (2012) *Time Series Analysis by State Space Methods*. 2nd edn. Oxford University Press.
Available at: <https://doi.org/10.1093/acprof:oso/9780199641178.001.0001>.

Schwartz, E.S. (1997) 'The stochastic behavior of commodity prices: implications for valuation and hedging', *The
Journal of Finance*, 52(3), pp. 923-973. Available at: <https://doi.org/10.1111/j.1540-6261.1997.tb02721.x>.

Schwartz, E. and Smith, J.E. (2000) 'Short-term variations and long-term dynamics in commodity prices', *Management
Science*, 46(7), pp. 893-911. Available at: <https://doi.org/10.1287/mnsc.46.7.893.12034>.

Vasicek, O. (1977) 'An equilibrium characterization of the term structure', *Journal of Financial Economics*, 5(2),
pp. 177-188. Available at: <https://doi.org/10.1016/0304-405X(77)90016-2>.

# Changelog

Versions follow semantic versioning and are git tags (`v0.1.0`). Each release lists what changed for a user.

## Unreleased

## 0.1.0 - 2026-11-05

The first release.

### Added

- One- and two-factor mean-reverting models of a daily log price with seasonal harmonics, external drivers and dated
  regimes (`ModelSpec`, `Params`); the drivers enter the observation or the fast factor's transition
  (`ModelSpec.drivers_enter`).
- A stressed regime dated by a rule on the series: the rolling volatility of daily log changes above a multiple of its
  median, kept in episodes of a minimum length and merged across short gaps (`stressed_days`, `rolling_volatility`,
  `episodes`, `windows`).
- The exact one-day transition of the factors, their stationary covariance and moments ahead (`ou`).
- A Kalman filter with its Gaussian log-likelihood, missing days and the state on the last day (`kalman_filter`).
- Maximum-likelihood fitting with fixed parameters, standard errors and a caller's starting point beside the grid
  of starts; with two factors the correlation of their shocks is fixed at zero by default, since a spot series
  cannot identify it (`fit`).
- The entries of `fit`'s fixed that hold a factor's half-life and keep its long-run spread, for a sensitivity on a
  poorly identified slow speed (`fixed_half_life`).
- The likelihood ratio of nested fits with its chi-squared p-value (`likelihood_ratio`).
- Seeded simulation with common random numbers across regimes and drivers, mean and median anchoring to a level
  curve, and annual means, streamed for large Monte Carlos; a regime per path and day may be kept as small
  integers (`simulate_factors`, `log_deviation`, `anchor_offsets`, `annual_means`, `simulate_annual_means`).
- The variogram, how far a series moves over each horizon, to compare a series with simulated paths (`variogram`).

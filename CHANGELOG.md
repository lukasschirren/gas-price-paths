# Changelog

Versions follow semantic versioning and are git tags (`v0.1.0`). Each release lists what changed for a user.

## Unreleased

### Added

- One- and two-factor mean-reverting models of a daily log price with seasonal harmonics, external drivers and dated
  regimes (`ModelSpec`, `Params`); the drivers enter the observation or the fast factor's transition
  (`ModelSpec.drivers_enter`).
- The exact one-day transition of the factors, their stationary covariance and moments ahead (`ou`).
- A Kalman filter with its Gaussian log-likelihood, missing days and the state on the last day (`kalman_filter`).
- Maximum-likelihood fitting with fixed parameters, standard errors and a caller's starting point beside the grid
  of starts; with two factors the correlation of their shocks is fixed at zero by default, since a spot series
  cannot identify it (`fit`).
- Seeded simulation with common random numbers across regimes and drivers, mean and median anchoring to a level
  curve, and annual means, streamed for large Monte Carlos; a regime per path and day may be kept as small
  integers (`simulate_factors`, `log_deviation`, `anchor_offsets`, `annual_means`, `simulate_annual_means`).

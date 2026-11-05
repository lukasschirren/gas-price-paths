# CLAUDE.md

Mean-reverting models of a daily gas price (import package `gaspaths`, project `gas-price-paths`): the model
specification, the exact transition of its factors, the Kalman filter and its likelihood, maximum-likelihood fitting,
and seeded simulation with anchoring. README.md is for users; `docs/design.md` is the founding design note (4 Nov
2026); `docs/method.md` the maths with references; this file covers what you need to work in the code.

## Rules (each has a test in `tests/`)

1. **Dependencies** (`test_imports.py`): numpy, scipy, pandas and the standard library only. Nothing from the research
   or data packages that use the library (`meritorder`, `gbpowerdata`, `gbabm`, `spikeabm`, `cmmodel`), and no code
   reads or points at another repository's paths.
2. **No files** (`test_repo_hygiene.py`): the library reads and writes no files; callers pass arrays and frames. No
   data is tracked beyond small synthetic fixtures in `tests/fixtures/`, nothing named after a licensed market-data
   source (`nordpool`, `n2ex`, `epex`, `dayahead`), no token or credential file, no file over 5 MB.
3. **Randomness** (`test_repo_hygiene.py`, `test_simulate.py`): every draw comes from a `numpy.random.Generator` the
   caller passes; nothing seeds or draws from numpy's global state.
4. **Exactness** (`test_ou.py`): factors step by the exact Ornstein-Uhlenbeck transition, never an Euler step.
5. **Naming** describes function; no dates, author or version marks in module or file names (they go in docstrings
   and `CHANGELOG.md`).

## This repository will be public

Commit or push only when asked, and stage explicit paths, never `git add -A`. Commit messages are one sentence
stating what changed and why, with no attribution trailer. Versions are git tags (`v0.1.0`), each with a
`CHANGELOG.md` entry and the version in `src/gaspaths/__init__.py` and `CITATION.cff`. After v0.1.0, work goes on
short feature branches merged into tagged releases; no branch is kept for one user.

## Environment

- Python 3.12 or later with numpy, scipy and pandas; `pip install -e ".[dev]"` adds pytest and statsmodels (the
  development check of the filter). Without an install, run with `PYTHONPATH=src`.
- `python -m pytest -q` runs every test on synthetic data, without network or data.
- gb-power-model installs this clone editable (`pip install -e`) into its interpreter and pins a release tag; its
  provenance records this clone's commit and whether it has uncommitted changes, so commit before running its results.

## Layout

| Path | Role |
| --- | --- |
| `src/gaspaths/model.py` | `ModelSpec` and `Params`: factors, harmonics, regimes, drivers, fixed parameters; a plain-dict round trip |
| `src/gaspaths/ou.py` | the exact one-day transition, the stationary covariance, moments h days ahead |
| `src/gaspaths/kalman.py` | `kalman_filter`: log-likelihood, filtered states, the state on the last day |
| `src/gaspaths/fit.py` | `fit`: maximum likelihood, standard errors, the correlation fixed at zero (a spot series cannot identify it); `fixed_half_life`, `likelihood_ratio` |
| `src/gaspaths/simulate.py` | seeded factor paths, the log deviation, anchoring offsets, annual means |
| `src/gaspaths/regimes.py` | `stressed_days`: a stressed regime dated by rolling volatility; `episodes`, `windows` |
| `src/gaspaths/diagnostics.py` | `variogram` |
| `tests/` | the rules above and the numerics, synthetic data only |
| `docs/` | `design.md`, `method.md` |
| `examples/` | `fit_daily_series.py` |

## Style

Module docstrings are long and substantive (the method, its sources with DOIs, the assumptions, the units).
Comments explain why. Prose in docs states what the code does; an inference is marked as one.

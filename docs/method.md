# Method

This document sets out the model of `gaspaths`, how it is filtered and fitted, why one parameter is fixed by
convention, and how simulated paths are anchored to a level curve. `design.md` records the design decisions.

## The model

The log price of day t (t counts days, so the seasonal curve keeps its phase across years) is

```text
y_t  = mu + s(t) + beta[r_t] . w_t + x_t[1] + x_t[2] + e_t,          e_t ~ N(0, sigma_obs^2)
s(t) = sum_{h=1..H} ( a_h sin(2 pi h t / 365.25) + b_h cos(2 pi h t / 365.25) )
x_t  = theta[r_t] + Phi (x_{t-1} - theta[r_t]) + eta_t,               eta_t ~ N(0, Q(sigma[r_t]))
```

The factor x[1] is fast and x[2] slow (one factor is the special case without x[2]). Each reverts to its target at a
speed kappa per year, with Phi = diag(exp(-kappa dt)) for a day step dt of 1/365.25 years. The regime r_t is given
for every day, and it sets the shock sizes sigma, the weights beta of the external drivers w and the targets theta,
which are zero in fitting. The day entered sets the shock of the step into it, so the first stressed day's move is
already stressed. A speed of zero for the slow factor makes it a random walk, the long-term factor of Schwartz and
Smith (2000); a positive one makes it mean-reverting, so the one-factor model is the exponential Vasicek process that
Schwartz (1997) brought to commodities.

## Where the drivers enter

The equations above put the drivers in the observation, so a day's anomaly moves that day's price and no other
(`drivers_enter="observation"`, the default). With `drivers_enter="fast"` they enter the fast factor's transition
instead, and the observation loses its driver term:

```text
x_t[1] = theta[r_t, 1] + phi_1 (x_{t-1}[1] - theta[r_t, 1]) + beta[r_t] . w_t + eta_t[1],      phi_1 = exp(-kappa_1 dt)
```

A day's anomaly then moves the fast factor by beta . w, and the move fades at the fast factor's speed. An anomaly
held for several half-lives moves the log price by beta . w / (1 - phi_1), so beta is a daily impulse and the
sustained effect is larger by 1 / (1 - phi_1), about 10 for a half-life of a week. The fast form suits a driver
whose effect builds up, such as storage drawn down through a cold spell, while a price that clears each day's
balance answers the day's anomaly alone. The two forms are not nested, and their likelihoods tell them apart.

The response psi_t = phi_1 psi_{t-1} + beta[r_t] . w_t, zero before the first day, is fixed by the drivers. The fast
factor is therefore psi plus a factor without drivers, and the filter runs on that second factor with psi added to
the deterministic part of the observation. It then adds psi back to the filtered fast factor, so the state it
reports, and the start it hands to a simulation, carry the latest days' weather, which fades in the simulated days
that follow. The likelihood is that of the model with the drivers in the state, exactly.

## Dating a stressed regime

The regimes are dated before the fit, so the likelihood stays that of a linear Gaussian state-space model with known
switches. `stressed_days` dates a stressed regime from the series alone. With v_t the standard deviation of the daily
log changes in a centred window of W days (31 by default, defined where at least W // 2 + 1 changes exist), a day is
stressed when v_t > m median(v), m = 2.5 by default. Runs of stressed days shorter than L = 14 days are dropped, and
runs less than G = 45 days apart are then merged with the days between them. The rolling window spreads a short spike
over about W days, so L sets how long a market must stay volatile to count, and G keeps one crisis with a calm month
inside it as one regime. The defaults date three episodes in GB's System Average Price of 2018-2025, the late-winter
cold spell of 2018, October 2019 and December 2021 to December 2022. A user of another series should check the dated
windows (`windows`) against what is known of its market and vary m, as gb-power-model does at 2 and 3.

## The exact transition

Over one day the factors move exactly, with no discretisation error, and their shocks have the covariance

```text
Q_ij = rho_ij sigma_i sigma_j (1 - exp(-(kappa_i + kappa_j) dt)) / (kappa_i + kappa_j)
```

whose limit where kappa_i + kappa_j = 0 is rho_ij sigma_i sigma_j dt. The stationary covariance is
rho_ij sigma_i sigma_j / (kappa_i + kappa_j), defined when both speeds are positive. Shocks are drawn as
diag(sigma) L z with L L' equal to Q at unit shock sizes, from an eigendecomposition that also serves a singular Q.
The same standard normals z therefore give the shocks of every regime, which keeps simulated paths comparable across
regimes.

## The filter

The Kalman filter predicts the factors each day and updates them on the day's observation, and its prediction errors
give the Gaussian log-likelihood (Durbin and Koopman, 2012). A missing day only predicts. Mean-reverting factors start
from their stationary distribution and a random-walk factor from a diffuse prior of variance 10^6. The recursion is
written out in scalars for one and two factors, so a likelihood on 3,000 days takes about 2 ms.

## Estimation

`fit` maximises the likelihood by L-BFGS-B on transformed parameters. The slow speed is exp(u) and the fast speed the
slow plus exp(u'), so the factors keep their order; shock sizes and the noise are exponentials; the level, the
seasonal coefficients and the driver weights enter as they are. The fit runs from several starting speeds, and from
a caller's starting point where given (a related fit's, for instance), and keeps the best. Standard errors come from the inverse of a finite-difference Hessian, mapped back by the delta method. They
are Gaussian-likelihood errors and understate the uncertainty of fat-tailed daily price changes.

## The correlation is fixed by convention

A spot series cannot identify the correlation rho of the two factors' shocks. The Gaussian likelihood of the series
depends only on its autocovariances, which are

```text
gamma(0) = A + B + sigma_obs^2,     gamma(h) = A exp(-kappa_1 h dt) + B exp(-kappa_2 h dt)   for h >= 1
A = Sigma_11 + Sigma_12,            B = Sigma_22 + Sigma_12
```

with Sigma the stationary covariance of the factors. The data therefore fix five numbers (two speeds, A, B and the
noise) for six parameters, and every rho on a ridge of shock sizes gives the same likelihood and the same distribution
of the price. `fit` fixes rho at zero unless the caller fixes another value. The choice changes how a path splits
between the fast and the slow factor and nothing about the path. Estimating rho needs information a spot series
lacks, such as futures prices, which the package does not use.

## Nested fits and a fixed half-life

`likelihood_ratio(l_full, l_restricted, df)` returns 2 (l_full - l_restricted) and its p-value against chi-squared
with df degrees of freedom, the parameters the restriction removes (Wilks, 1938). The reference holds for a
restriction inside the parameter space, such as two regimes sharing a shock size. For a parameter held on its
boundary, such as a shock size at zero, the chi-squared p-value is conservative (Self and Liang, 1987). For a
restriction that leaves a parameter unidentified, such as a factor removed with its speed, it is only a guide.

A slow speed is often poorly identified by a few years of daily data, so a projection needs a sensitivity on it.
`fixed_half_life(params, h)` returns the entries of `fit`'s fixed that hold the slow factor's speed at ln 2 / h (h in
years) and its shock size in each regime at sigma_r sqrt(2 ln 2 / h) / sqrt(2 kappa), so that its long-run spread
sigma / sqrt(2 kappa) stays at the fit's. The refit then moves only how fast the slow factor reverts, while the
other parameters adjust to the series.

## Anchoring to a level curve

A caller may replace the level mu by a level curve L, a price for each day, so that the price is
L exp(log deviation - a_y). With mean anchoring

```text
a_y = V / 2 + ln mean_{t in y} exp( s(t) + D_t )
```

where D_t is the drivers' effect (beta[r_t] . w_t in the observation, or the response psi_t when the drivers enter
the fast factor) and V is the stationary variance of the sum of the factors. The expected annual mean price of a stationary path
then equals the annual mean of L. Median anchoring drops V / 2 and leaves the median path on the curve, the expected
price above it. A random-walk factor has no stationary variance and admits median anchoring only.

## Checking a fit

`variogram(y, lags)` is the root mean square of y_{t+h} - y_t over the pairs of days that both exist, for each lag
h. A random walk's grows with the square root of h, and one mean-reverting factor's levels off at sqrt(2 V) after a
few half-lives, V its stationary variance. A fast and a slow factor give a quick rise over days and a slow one over
months to years. Comparing a series' variogram with those of paths simulated from fits with one and two factors
shows at which horizons each model moves too little or too much.

## References

Durbin, J. and Koopman, S.J. (2012) *Time Series Analysis by State Space Methods*. 2nd edn. Oxford University Press.
Available at: <https://doi.org/10.1093/acprof:oso/9780199641178.001.0001>.

Self, S.G. and Liang, K.-Y. (1987) 'Asymptotic properties of maximum likelihood estimators and likelihood ratio tests
under nonstandard conditions', *Journal of the American Statistical Association*, 82(398), pp. 605-610.
Available at: <https://doi.org/10.1080/01621459.1987.10478472>.

Schwartz, E.S. (1997) 'The stochastic behavior of commodity prices: implications for valuation and hedging', *The
Journal of Finance*, 52(3), pp. 923-973. Available at: <https://doi.org/10.1111/j.1540-6261.1997.tb02721.x>.

Schwartz, E. and Smith, J.E. (2000) 'Short-term variations and long-term dynamics in commodity prices', *Management
Science*, 46(7), pp. 893-911. Available at: <https://doi.org/10.1287/mnsc.46.7.893.12034>.

Vasicek, O. (1977) 'An equilibrium characterization of the term structure', *Journal of Financial Economics*, 5(2),
pp. 177-188. Available at: <https://doi.org/10.1016/0304-405X(77)90016-2>.

Wilks, S.S. (1938) 'The large-sample distribution of the likelihood ratio for testing composite hypotheses', *The
Annals of Mathematical Statistics*, 9(1), pp. 60-62. Available at: <https://doi.org/10.1214/aoms/1177732360>.

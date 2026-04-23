# The sequential importance resampling particle filter

The particle filter approximates the Bayesian posterior over a hidden
state $x_t$ given a sequence of observations $y_{1:t}$ by a weighted
ensemble of $N$ samples ("particles"):

$$
p(x_t \mid y_{1:t}) \;\approx\; \sum_{m=1}^{N} w_t^{(m)}
\, \delta\!\left(x_t - x_t^{(m)}\right),
$$

where $\delta$ is the Dirac delta and $\sum_m w_t^{(m)} = 1$.

## The recursion

The standard sequential importance resampling (SIR) recursion alternates
two steps.

**Forecast.** Each particle is propagated by the dynamical model:

$$
x_t^{(m)} \;\sim\; p\!\left(x_t \mid x_{t-1}^{(m)}\right), \quad m=1,\dots,N.
$$

**Analysis.** Weights are updated by the observation likelihood:

$$
w_t^{(m)} \;\propto\; w_{t-1}^{(m)} \, p\!\left(y_t \mid x_t^{(m)}\right).
$$

In `pypfda` we always renormalize after the multiplication; the previous
weights are reset to $1/N$ after every resampling, so in practice the
formula reduces to $w_t^{(m)} \propto p(y_t \mid x_t^{(m)})$.

## Numerical implementation

Likelihoods can be vanishingly small in high dimensions, so the
implementation in {mod}`pypfda.weights` works in the log domain and
applies the log-sum-exp trick:

$$
w_t^{(m)} = \frac{\exp\!\left(\ell_t^{(m)} - L_t^\star\right)}
{\sum_{n=1}^{N} \exp\!\left(\ell_t^{(n)} - L_t^\star\right)}, \qquad
L_t^\star = \max_n \ell_t^{(n)},
$$

where $\ell_t^{(m)} = \log p\!\left(y_t \mid x_t^{(m)}\right)$. Subtracting
$L_t^\star$ leaves the normalized weights unchanged but keeps the
exponentials in a representable range.

## Resampling and the diversity–memory trade-off

When a few particles dominate the weight distribution, the
*effective sample size*

$$
N_{\mathrm{eff}} \;=\; \frac{1}{\sum_m \left(w_t^{(m)}\right)^2}
$$

falls and the ensemble must be **resampled** to restore particle
diversity. `pypfda` ships four schemes (`systematic`, `stratified`,
`residual`, `multinomial`); systematic resampling is the default
because it has the lowest variance among unbiased schemes.

Resampling is a double-edged sword: it reinjects diversity
*statistically* (the resampled ensemble has the same expected mean and
covariance as the weighted one) but it *kills* diversity *physically*
because high-weight particles get duplicated. Repeated resampling
without any process noise leads to genealogical collapse: every
particle eventually descends from a single ancestor.

The standard fix is **inflation** (perturb the ensemble after
resampling). For systems with long memory (deep ocean circulation, ice
sheets), inflation undoes the very thing that makes online filtering
attractive: it scrambles the slowly-evolving state that integrates past
observational corrections forward in time. This *diversity–memory
trade-off* is discussed at length in Fallah et al. (2026, *npj Climate
and Atmospheric Science*, in review).

## References

- Doucet, A., de Freitas, N. & Gordon, N. (2001).
  *Sequential Monte Carlo Methods in Practice*. Springer.
- Dubinkina, S., Goosse, H., Sallaz-Damaz, Y., Crespin, E. & Crucifix,
  M. (2011). *Testing a particle filter to reconstruct climate changes
  over the past centuries*. International Journal of Bifurcation and
  Chaos, 21, 3611.
- Gaspari, G. & Cohn, S. E. (1999). *Construction of correlation
  functions in two and three dimensions*. Quarterly Journal of the
  Royal Meteorological Society, 125, 723–757.

> Note: a forthcoming `pypfda.diagnostics.genealogy` module will provide
> tools to track the ancestor of every particle through time, so that
> the trade-off can be quantified rather than merely described.

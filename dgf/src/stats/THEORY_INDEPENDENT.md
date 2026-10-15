# Theory: finite-sample confidence intervals for independent and effective-sample-size evaluations (`stats.independent`)

> These arguments have not been peer-reviewed; see [EVIDENCE.md](EVIDENCE.md)
> for simulation checks.

## Status of claims, lemmas, and propositions

Each result in this document carries one of four statuses:

*   **argued, checked by simulation** (with the corresponding test or benchmark
    in parentheses; see [EVIDENCE.md](EVIDENCE.md));
*   **argued only**;
*   **approximation** (first-order or asymptotic);
*   **cited** (from the literature, with reference).

| Label | One-line statement | Section | Status |
|---|---|---|---|
| Claim 1 | Finite-sample coverage under independent and identically distributed (i.i.d.) sampling (`P1 ∧ P2*(M)`) | §4 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md), `benchmark/coverage_independent_test.py` block A; `independent/interval_test.py`) |
| Corollary 1 | Root-mean-square error (`rmse`) via square-root transform | §4 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md), `benchmark/coverage_independent_test.py` block A, `benchmark/coverage_dependent_test.py`; `independent/interval_test.py`) |
| Corollary 2 | Upper-endpoint clipping for structurally bounded metrics (`accuracy`) | §4 | argued, checked by simulation (`independent/interval_test.py`) |
| Claim 1′ | Finite-sample coverage under dependence with a declared effective sample size (`P1′(n_eff) ∧ P2*_pool(M)`) | §4.1 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md), `benchmark/coverage_independent_test.py` block D, `benchmark/coverage_dependent_test.py`) |
| Claim 1″ | Finite-sample coverage under uniform sampling without replacement on the independent path | §4.1.1 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md), `benchmark/coverage_independent_test.py` block E; `independent/interval_test.py`) |
| Lemma M.1 | Marginal laws of the coupled without-replacement and with-replacement samples | §4.1.1 | argued only |
| Lemma M.2 | Conditional expectations of the sample mean and sample variance under the coupling | §4.1.1 | argued only |
| Claim 3 | Exact Clopper–Pearson binomial interval for `accuracy` on the independent path | §4.2 | cited (Clopper and Pearson, 1934) |
| Claim 3′ | Exact `accuracy` interval under uniform sampling without replacement for $\alpha \in (0, 2 - \sqrt{2}]$ | §4.2 | argued, checked by simulation (`independent/interval_test.py`, `benchmark/hypergeom_cp_test.py`) |
| Corollary 3 | Symmetry between accuracy and error rate on the exact binomial path | §4.2 | argued only |
| Lemma A | One-sided Cantelli bound under `P1 ∧ P2*(M)` | §5.1 | cited (Cantelli's inequality; see References) |
| Lemma B | Truncated empirical-Bernstein lower sweep (`Maurer and Pontil, 2009`) and sketch relaxation | §5.2 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md), `benchmark/coverage_independent_test.py` block A; `independent/interval_test.py`) |
| Lemma C | Data-independent branch selection preserves per-side level $\alpha'$ | §5.3 | argued only |
| Lemma D | Coverage preservation under strictly increasing monotone transforms | §5.4 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md), `benchmark/coverage_independent_test.py` block A; `independent/interval_test.py`) |
| Proposition S | Exact recovery of truncated moments from a top-$K$ sketch when exceedances $\le K$ | §5.5 | argued only |
| Proposition S′ | Chebyshev upper bound on expected exceedances above the truncation threshold | §5.5 | argued only |
| Proposition F | Asymptotic consistency of the empirical relative second-moment diagnostic $\hat M_n$ | §5.6 | argued only |
| Lemma E | One-sided Bernstein upper endpoint for non-negative summands under `P1 ∧ P2*(M)` | §5.7 | cited (Bernstein's inequality; see References) |
| Lemma A′ | One-sided Cantelli bound under dependence and heterogeneity (`P1′(n_eff) ∧ P2*_pool(M)`) | §5.8 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md), `benchmark/coverage_independent_test.py` block D, `benchmark/coverage_dependent_test.py`) |
| Lemma H | Effective sample size $n_{\mathrm{eff}} = n/\kappa$ from the maximum absolute correlation row sum $\kappa$ | §5.8 | argued only |
| Proposition G | Conservative $\kappa_{\max}$ merge rule for mutually uncorrelated shards under `(V_k)` | §5.10 | argued, checked by simulation (`independent/interval_test.py`) |

---

## What this document argues

This document states and argues the coverage bounds of the `stats.independent`
module (`interval.py`, `bound.py`, `sketch.py`, `metrics.py`, and
`jax_accumulator.py`). Given non-negative per-item losses $s_1, \dots, s_n$
(absolute errors, squared errors, or correctness indicators) and a
caller-declared constant $M \ge 1$ that bounds the relative second moment
$\mathbb{E}[s^2]/(\mathbb{E}[s])^2$, the library returns an interval
$[L_n, U_n]$ for the mean loss $\theta$ that misses $\theta$ with probability
at most $\alpha/2$ on each side under the stated premises, for every sample
size at which it returns an interval. The bound is finite-sample rather than
asymptotic. It holds under the stated premises for independent samples
(Claim 1), for correlated samples with a declared effective sample size
(Claim 1′), for samples drawn uniformly without replacement from a finite pool
(Claim 1″), and, without any moment declaration, for classification accuracy
(Claims 3 and 3′). The bound applies to the population mean loss $\theta$ (the
estimand); it makes no statement about the loss of any individual item, the
empirical mean, or the validity of the declared $M$ and effective sample size,
which the caller supplies. The document also lists what the bounds do not cover
(§8) and summarizes the resulting design choices (§9).

## Who this document is for

This document is for machine learning (ML) engineers and statisticians who want
to inspect the arguments behind `stats.independent` or extend the estimator. It
assumes familiarity with basic probability (Chebyshev-type inequalities and
exponential tail bounds). [README.md](README.md) is the user guide,
[THEORY.md](THEORY.md) covers the cluster, temporal, and graph estimators, and
[DESIGN.md](DESIGN.md) summarizes the design rationale.

**Scope.** This document describes the `stats.independent` subpackage. Release
notes are in [CHANGELOG.md](CHANGELOG.md). Section numbers (§0 to §9), result
labels (Claim 1, Lemma E, Proposition G, …), premise names (`P1`, `P2*(M)`, …),
status names, and limitation identifiers (`L1` to `L17`) are referenced by code
docstrings, error messages, and companion documents.

## Notation

The following symbols and abbreviations are used throughout:

| Symbol / term | Definition |
|---|---|
| i.i.d. | Independent and identically distributed. |
| a.s. | Almost surely (with probability 1). |
| MAE, MSE, RMSE | Mean absolute error, mean squared error, and root-mean-square error. |
| R² | Coefficient of determination ($1 - \theta_A/\theta_B$). |
| AP | Average precision. |
| $n$ | Sample size (number of evaluated items). |
| $e_1, \dots, e_n$ | Per-item errors ($e_i = \hat y_i - y_i$, prediction minus label). |
| $s_1, \dots, s_n$ | Non-negative per-item loss summands ($s_i \ge 0$) formed pointwise from $e_i$ (§1). |
| $F$ | The common probability law of the summands under premise `P1`. $\mathbb{E}_F$ and $\Pr_F$ denote expectation and probability under $F$. |
| $\theta$ | Target estimand: $\theta := \mathbb{E}_F[s]$ under i.i.d. sampling (`P1`), or the pooled mean $\theta := \frac{1}{n}\sum_{i=1}^n \mathbb{E}[s_i]$ under dependence (`P1′`). |
| $\bar s$ | Sample mean of the summands, $\bar s := \frac{1}{n}\sum_{i=1}^n s_i$. |
| $\sigma^2$ | Variance $\mathrm{Var}(s)$ under `P1`. |
| $s_{(1)} \ge s_{(2)} \ge \dots \ge s_{(n)}$ | Decreasing order statistics of $(s_1, \dots, s_n)$. |
| $K$ | Number of top order statistics retained in a sketch ($K = 32$ in sketch mode; $K = n$ in data mode). In §4.2 only, $K$ instead denotes the count $\sum_i s_i$ of correct items. |
| $\alpha,\; \alpha'$ | Two-sided failure probability $\alpha \in (0, 1)$ (nominal coverage level $1 - \alpha$) and per-side budget $\alpha' := \alpha/2$. |
| $M$ | Declared relative second-moment bound ($M \ge 1$). |
| $\mathcal P(M)$ | The set of probability laws $F$ on $[0, \infty)$ that satisfy premise `P2*(M)`. |
| $M(F)$ | The relative second moment of a law, $M(F) := \mathbb{E}_F[s^2]/\theta^2$. |
| $\hat M_n$ | Empirical relative second moment, $\hat M_n := n \sum_{i=1}^n s_i^2 / (\sum_{i=1}^n s_i)^2$, used by the refutation check (§5.6). |
| $L_n,\; U_n$ | Lower and upper endpoints of the returned interval. |
| $L$ | A logarithmic level, **defined locally** in each result: $L := \ln(2/\alpha')$ in Lemma B (§5.2) and Proposition S′ (§5.5), where it equals $\ln(4/\alpha)$; $L := \ln(1/\alpha')$ in Lemma E (§5.7). |
| $c$ | A deviation constant, **defined locally**: the relative deviation in Lemmas A, A′, and E (for example $c_A$, $c_E$), or the truncation threshold $c = \gamma\theta$ in Lemma B. |
| $c_A,\; c_E,\; c_{\text{up}}$ | Cantelli constant (Lemma A), one-sided Bernstein constant (Lemma E), and the upper-endpoint constant $c_{\text{up}} := \min(c_A, c_E)$ (§5.7.1). In code: `c_cantelli`, `c_bernstein`. |
| $\gamma,\; \gamma^*$ | Truncation scale of Lemma B and its reference value $\gamma^* := \sqrt{3M(n-1)/(7L)}$ (§5.2). |
| $X_i,\; \bar X(t),\; \hat V(t)$ | Truncated summands $X_i := s_i \wedge \gamma t$, their sample mean, and their unbiased empirical variance (§5.2). Here $a \wedge b := \min(a, b)$ and $a \vee b := \max(a, b)$. |
| $G,\; H,\; G_{\mathrm{lo}},\; H_{\mathrm{lo}},\; \widetilde H_{\mathrm{lo}}$ | The functions whose crossings define the Lemma B endpoints (§5.2). |
| $\text{slope}_\infty$ | Asymptotic slope of $H$ for large $t$, used for routing (§5.2). |
| $n_A,\; n_U,\; n_B$ | Sample-size thresholds (§5.9). |
| `n_eff` ($n_{\text{eff}}$) | Declared effective sample size ($1 \le n_{\text{eff}} \le n$, `effective_n` in code) under dependence (premise `P1′(n_eff)`); `None` on the i.i.d. path (premise `P1`). |
| $\rho_{ij}$ | Linear correlation between $s_i$ and $s_j$. |
| $\kappa$ | Maximum absolute row sum of the correlation matrix, $\kappa := \max_i \sum_j |\rho_{ij}| \ge 1$, giving $n_{\text{eff}} = n/\kappa$ (§5.8, Lemma H). For a shard $k$ of a merge, $\kappa_k := n_k / n_{\text{eff},k}$ is its variance inflation factor (VIF) and $\kappa_{\max} := \max_k \kappa_k$ (§5.10, Proposition G). |
| $\mathrm{Kurt}(\cdot)$ | Centred fourth-moment kurtosis $\mathrm{Kurt}(X) := \mathbb{E}[(X - \mathbb{E}[X])^4] / \mathrm{Var}(X)^2$ (§2 Remark 4, §6). |
| $\gamma_1$ | Skewness of the error, $\gamma_1 := \mathbb{E}[(e-\mu)^3]/\sigma_e^3$ (§2 Remark 4). Distinct from the truncation scale $\gamma$. |
| $W$ | Total correlation mass $W := \sum_{i,j} \rho_{ij}$ of a common-variance sample, for which $n_{\text{eff}} = n^2/W$ (§5.8). Not a merge quantity: Proposition G (§5.10) merges by $\kappa_{\max}$. |
| $\mathrm{CV}$ | Coefficient of variation (standard deviation divided by mean). |
| PSD | Positive semidefinite (of a matrix). |
| SLLN | Strong law of large numbers. |
| MGF | Moment-generating function, $\lambda \mapsto \mathbb{E}[e^{\lambda X}]$. |
| $\Pr_{\mathrm{wor}},\; \mathbb{E}_{\mathrm{wor}}$ / $\mathbb{E}_{\mathrm{iid}}$ | Probability and expectation under sampling without replacement (`wor`) / with replacement (`iid`) from a finite pool (§4.1.1). |
| `P1`, `P1′(n_eff)` | Sample premises (§2): premise `P1` (independence) assumes $s_1, \dots, s_n$ are i.i.d.; premise `P1′(n_eff)` (variance bound for correlated data) assumes $\mathrm{Var}(\bar s) \le (M-1)\theta^2 / n_{\text{eff}}$. |
| `P2*(M)`, `P2*_pool(M)` | Second-moment premises (§2): premise `P2*(M)` (the declaration on $M$) assumes $\mathbb{E}_F[s^2] \le M\theta^2$; premise `P2*_pool(M)` (pooled declaration on $M$) assumes $\frac{1}{n}\sum_i \mathbb{E}[s_i^2] \le M\theta^2$. |
| `UNREFUTED`, `TAIL_UNRESOLVED`, `ASSUMPTION_REQUIRED` | Diagnostic status codes (§3): premise not contradicted (`UNREFUTED`), $\hat M_n > M$ (`TAIL_UNRESOLVED`), or no finite interval derivable at $(n, M, \alpha, n_{\text{eff}})$ (`ASSUMPTION_REQUIRED`). |
| `L1`–`L17` | Scope and limitation identifiers catalogued in §8. |
| `R1` | The design rule that the diagnostic status never narrows an interval (§5.6). |

## Premises at a glance

Each coverage claim rests on a specific combination of premises. §2 states them
formally.

| Premise | One-line meaning | Used by |
|---|---|---|
| `P1` | The summands are i.i.d. draws from one law $F$ on $[0, \infty)$. | Claims 1 and 3; Lemmas A, B, E |
| `P1′(n_eff)` | The summands may be dependent and heterogeneous, with $\mathrm{Var}(\bar s) \le (M-1)\theta^2/n_{\text{eff}}$ for the caller's $n_{\text{eff}}$. | Claim 1′; Lemma A′ |
| `P2*(M)` | $0 < \theta < \infty$ and $\mathbb{E}_F[s^2] \le M\theta^2$: the declared bound on the relative second moment. Pairs with `P1`. | Claims 1 and 1″ |
| `P2*_pool(M)` | The same bound for the pooled moments $\frac1n\sum_i \mathbb{E}[s_i]$ and $\frac1n\sum_i \mathbb{E}[s_i^2]$. Pairs with `P1′`. | Claim 1′; Lemmas A′, H; Proposition G |
| Uniform sampling without replacement | The $n$ items are a uniform random sample without replacement from a finite pool. | Claims 1″ and 3′; Lemmas M.1, M.2 |
| Mutually uncorrelated shards, each satisfying (V$_k$) | Summands in different merged shards are uncorrelated (a caller contract), and each shard's declared $n_{\text{eff},k}$ bounds its own variance inflation (§5.10). | Proposition G |

---

## 0. Why a declared constant is necessary

Two published impossibility results bound what any distribution-free interval
estimator for a mean can achieve:

* **Bahadur and Savage (1956).** Over the class of all distributions with a
  finite mean — even when every moment is finite — there is **no bounded
  confidence interval for the mean with non-vacuous finite-sample coverage.**
* **Devroye, Lerasle, Lugosi, and Oliveira (2016), *Annals of Statistics*
  44(6).** Under finite variance with the variance unknown, no *empirical*
  confidence interval attains sub-Gaussian length uniformly over the class.

Calibrated finite-sample intervals assuming only that the mean and variance
exist are therefore unachievable. Finite-sample inference **requires a declared
constant** that restricts the class of distributions. This module uses a
single, dimensionless constant: the relative second moment `M` (§2).

---

## 1. Per-item statistics and the sketch

Let `e_1, …, e_n` (`e_i = ŷ_i − y_i`, prediction minus label) be per-item
errors. For each supported metric, a **per-item statistic** `s_i ≥ 0` is formed
pointwise (`metrics.extract_summands`, `metrics.extract_r2_summands`, and
`jax_accumulator.summands`):

| Metric | `s_i` | Estimand reported | Default `M` (`metrics.DEFAULT_M`) |
|---|---|---|---|
| Mean absolute error (`mae`) | `\|e_i\|` | `θ` | `4.0` |
| Mean squared error (`mse`) | `e_i²` | `θ` | `16.0` |
| Root-mean-square error (`rmse`) | `e_i²` | `√θ` | `16.0` |
| Classification accuracy (`accuracy`) | `1{correct_i} ∈ {0, 1}` | `θ` | `4.0` (used only when `effective_n` is set) |
| Coefficient of determination (`r2`) | two sequences (`s_a = e_i²`, `s_b = ½(y_{2i} − y_{2i−1})²`), see §6 | `1 − θ_A/θ_B` | `16.0` |
| Average precision (`average_precision`, `ap`) | — | not supported (Limitation L5, §8: not a mean of per-item terms; raises `ValueError`) | — |

Here `θ := E_F[s]` (or `(1/n)Σ_i E[s_i]` under `P1′`), `s̄ := n⁻¹Σs_i`, and the
decreasing order statistics are `s_(1) ≥ s_(2) ≥ …`.

**The sketch** (`sketch.IndependentSketch`) is the fixed-size summary from which intervals
can be computed without retaining all $n$ items:

$$\mathcal S_n \;=\; \bigl(n,\; \Sigma s_i,\; \Sigma s_i^2,\; s_{(1)},\dots,s_{(K)},\; n_{\text{eff}},\; \texttt{schema\_version}\bigr), \qquad K = 32,$$

where `n_eff` (`effective_n`) is `None` under premise `P1` (independence) and a
finite number in `[1, n]` under premise `P1′(n_eff)` (variance bound for
correlated data). Data mode is the special case `K = n`. Sketches merge
associatively and commutatively (`sketch.merge`): the counts and sums add, the
top-`K` of a union is the top-`K` of the merged multiset of the two top-`K`
tuples, and `n_eff` combines by the $\kappa$-maximum rule of §5.10
(Proposition G): $n_{\text{eff}} = n/\max_k(n_k/n_{\text{eff},k})$. The integer
`schema_version` (default `1`) must match across a merge (`ValueError` on
mismatch).

*(The merge is exact for a single-statistic `IndependentSketch`. The `R2Sketch` container
holds two `IndependentSketch` instances (`sketch_a` and `sketch_b`), and its `b`-sequence
is built from index-local pairs within each shard, so merging shards of odd
size drops one unpaired element per odd shard; see §6 and Limitation L2 in
§8.)*

---

## 2. The premises

Two premises concern the sample, and one of them is assumed in each call. Two
premises concern the moment bound on the law, and each pairs with one sample
premise.

$$\textbf{P1: } s_1,\dots,s_n \text{ are i.i.d. draws from a law } F \text{ on } [0,\infty).$$

$$\textbf{P1}'(n_{\text{eff}})\textbf{: } s_1,\dots,s_n \ge 0 \text{ are jointly distributed — neither independent nor identically distributed — with } \quad \mathrm{Var}(\bar s) \le \frac{(M-1)\theta^2}{n_{\text{eff}}}, \quad 1 \le n_{\text{eff}} \le n.$$

$$\textbf{P2*}(M): \quad 0<\theta<\infty \quad\text{and}\quad \mathbb{E}_F[s^2] \;\le\; M\,\theta^2 .$$

$$\textbf{P2*}_{\text{pool}}(M): \quad \theta := \tfrac1n\textstyle\sum_i \mathbb{E}[s_i] \in (0,\infty) \quad\text{and}\quad \tfrac1n\textstyle\sum_i \mathbb{E}[s_i^2] \;\le\; M\,\theta^2 .$$

Here `M ≥ 1` is a dimensionless constant supplied by the caller (or defaulted
from `metrics.DEFAULT_M`), and so is `n_eff` when premise `P1′(n_eff)` is used.
**`P1` pairs with `P2*`; `P1′` pairs with `P2*_pool`.** Under identical
distribution the two forms of the moment premise coincide, so `P2*_pool` is a
weakening and **`P1 + P2*` implies `P1′(n) + P2*_pool`**. That implication is
one-way.

`P1′` does not require identical distribution because correlated evaluation data
(such as nodes in a graph or items across time) generally have heterogeneous
marginal distributions and variances. Lemma H (§5.8) shows that marginal
heterogeneity requires no extra slack. The empirical diagnostic statistic
`M̂_n = n·Σs²/(Σs)²` equals `((1/n)Σs_i²)/((1/n)Σs_i)²`, whose population
counterpart is the pooled ratio, so the check of §5.6 (Proposition F) targets
`P2*_pool` directly.

> **Important.** **`P1′` is weaker and supports fewer bounds.** It drops
> independence entirely and retains only a second-moment bound on `s̄`, which
> suffices for Cantelli's inequality (Lemma A′) and for nothing else in this
> document — neither Bernstein's inequality (Lemma E) nor the Maurer–Pontil
> truncation sweep (Lemma B; see §5.8). The caller selects `P1′` by passing
> `effective_n`; passing it is an **unchecked declaration** (Limitation L12,
> §8).
>
> Assuming `P1′(n)` is **not** equivalent to assuming `P1`, and the outputs
> differ accordingly: `effective_n = n` never gives a smaller upper endpoint
> than `effective_n = None` and refuses whenever `None` refuses; the lower
> endpoints are identical at or below the sweep threshold `n_B` and are not
> ordered sample by sample above it (§5.8).

Four remarks follow.

1.  **`P2*(M)` is a quantitative finite-second-moment condition.** It requires
    `E[e²] < ∞` for `mae` and `E[e⁴] < ∞` for `mse`, `rmse`, and `r2`, and no
    higher moments.
2.  **It is scale-free.** `𝒫(M)` is a cone: `s ∈ 𝒫(M) ⟹ c s ∈ 𝒫(M)` for every
    `c > 0`. No auxiliary scale parameter is declared or estimated.
3.  **It filters rather than excludes.** `⋃_{M≥1} 𝒫(M) = {F : E[s²] < ∞}`.
    Every law `F` with a positive mean and finite second moment belongs to
    `𝒫(M(F))` for `M(F) = E[s²]/θ²`.
4.  **For `mse` and `rmse`, `M` is the *uncentered* standardized fourth moment
    of the error, not its centred kurtosis.** With `s = e²`,
    `M(F) = E[e⁴]/(E[e²])²`. Write `μ := E[e]` and `σ² := Var(e)` for the mean
    and variance of the error, `u := μ/σ` for the **signed** relative bias,
    `z := (e − μ)/σ` for the standardized error, `Kurt(e)` for the centred
    standardized fourth moment (kurtosis) of `e`, and `γ₁` for the skewness of
    `e`. Then

    $$M \;=\; \frac{\mathrm{Kurt}(e) + 4\gamma_1 u + 6u^2 + u^4}{(1+u^2)^2},$$

    which equals `Kurt(e)` if and only if `μ = 0`. **Bias does not always lower
    `M`.** For small `|u|`,

    $$M \;=\; \mathrm{Kurt}(e) + 4\gamma_1 u + \bigl(6-2\,\mathrm{Kurt}(e)\bigr)u^2 + O(u^3),$$

    so whenever `γ₁u > 0` the linear term dominates and `M > Kurt(e)`; and with
    `γ₁ = 0`, `M − Kurt(e) = [2(3−Kurt(e))u² + (1−Kurt(e))u⁴]/(1+u²)²`, which
    is `≤ 0` for every `u` only if `Kurt(e) ≥ 3`. Examples: `Kurt(e)=16, γ₁=3,
    u=0.2` gives `M = 18.6416/1.0816 = 17.24 > 16`; a symmetric two-point error
    with `u = 1` has `Kurt(e) = 1` but `M = 8/4 = 2`. Two general relations
    hold:

    *   `M ≤ Kurt(e)` **if** `Kurt(e) ≥ 3` **and** `γ₁μ ≤ 0` — in particular
        for every symmetric error distribution with `Kurt(e) ≥ 3`, which covers
        Gaussian and symmetric heavy-tailed laws.
    *   `M < Kurt(e) + 2` **unconditionally.** Since the covariance matrix of
        `(z, z²)` is positive semidefinite (PSD), `γ₁² ≤ Kurt(e) − 1`, and
        therefore

        $$\bigl(\mathrm{Kurt}(e)+2\bigr)(1+u^2)^2 - \bigl(\mathrm{Kurt}(e)+4\gamma_1u+6u^2+u^4\bigr)
        \;\ge\; 2 - 4\sqrt{\mathrm{Kurt}(e)-1}\,|u| + 2\bigl(\mathrm{Kurt}(e)-1\bigr)u^2 + \bigl(\mathrm{Kurt}(e)+1\bigr)u^4
        \;=\; 2\bigl(1-\sqrt{\mathrm{Kurt}(e)-1}\,|u|\bigr)^2 + \bigl(\mathrm{Kurt}(e)+1\bigr)u^4 \;>\; 0,$$

        the last inequality strict because `√(Kurt(e)−1)|u| = 1` forces `u ≠ 0`.
        The constant `2` is sharp: at the moment-space boundary
        `γ₁² = Kurt(e) − 1` with `γ₁u = 1`, that is `u = 1/√(Kurt(e)−1)`, the
        gap is `(Kurt(e)+2) − M = (Kurt(e)+1)u⁴/(1+u²)² = (Kurt(e)+1)/Kurt(e)² → 0`.

    Sizing `M` from a centred kurtosis therefore needs **two units of
    headroom**: the default `M = 16` covers every error law with
    `Kurt(e) ≤ 14`. The distinction also matters for `r2`, where
    `M_B = (Kurt(y)+3)/2 ≠ Kurt(y)` (§6).

**Interpretation.** For the two-point law `s ∈ {0, h}` with `P(s = h) = p`,
`M = 1/p` exactly, and Cantelli's inequality holds with equality there.
Declaring `M` therefore amounts to declaring that *the rare event carrying the
mean has probability at least `1/M`* — or equivalently, *at least `n/M` such
events are expected in a sample of size `n`.*

**Derived quantities.** Under `P2*(M)`, `Var(s) = E[s²] − θ² ≤ (M−1)θ²`, and by
Markov's inequality for `s²`, for any `c > 0`, `P(s > c) ≤ E[s²]/c² ≤ Mθ²/c²`.

---

## 3. Statuses

Every call to `interval.confidence_interval` returns a `Result` carrying one of
three `Status` enum values:

*   **`UNREFUTED`**: no empirical evidence against the declaration (`M̂_n ≤ M`,
    or exact `accuracy` on the independent path). A finite interval is returned.
*   **`TAIL_UNRESOLVED`**: the data contradict the moment declaration
    (`M̂_n > M`; for `r2`, `max(M̂_a, M̂_b) > M`). A finite interval is still
    returned, and its coverage bound still holds if the declaration `M` holds
    for the population, but the premise is in doubt.
*   **`ASSUMPTION_REQUIRED`**: no finite interval is derivable at this
    `(n, M, α, n_eff)` or the input sums are non-finite. The returned endpoints
    are `(nan, nan)` and `Result.level` is `None`.

Within Algorithm 1 (`bound.bound`, §7), `ASSUMPTION_REQUIRED` has one
distributional cause, `c_up ≥ 1` (§5.9), plus the degenerate-input guards of
step 1 (`n < 2`, `M < 1`, `K > n`, or any non-finite value in `sum_s`,
`sum_s2`, or `top_k`). Outside Algorithm 1 (`interval.confidence_interval`), it
has three further sources, none of which depends on the data values beyond
input validity:

*   the `r2` path (§6) returns `ASSUMPTION_REQUIRED` when `effective_n` is set
    on either component sketch or as a call argument, when `n_a == 0` or
    `n_b == 0`, or when either of its two `bound.bound` calls returns
    `ASSUMPTION_REQUIRED`;
*   the exact `accuracy` path (§4.2, `_accuracy_exact`) returns
    `ASSUMPTION_REQUIRED` when `n < 1` or when `Σs` or `Σs²` is non-finite (by
    contrast, finite non-binary inputs raise `ValueError`: specifically, raw 1D
    array values not in `{0, 1}`, an `IndependentSketch` with `top_32[0] > 1.0`, or finite
    sketch sums where `Σs` is not an integer in `[0, n]` or `Σs² ≠ Σs`, as well
    as `level < √2 − 1 ≈ 0.4142` on the independent accuracy path);
*   raw data with no items (`n = 0`; for `r2`, `n < 2` so `⌊n/2⌋ = 0` label
    pairs).

> **Important.** **Coverage does not depend on the status.** Claim 1 holds
> unconditionally across `UNREFUTED` and `TAIL_UNRESOLVED` given `P1 ∧ P2*(M)`,
> and Claim 1′ holds unconditionally given `P1′(n_eff) ∧ P2*_pool(M)`. The
> status is a *diagnostic about the premise*, not a condition in the coverage
> argument.
>
> "Unconditional **given** the premise" matters: the premise is a caller
> declaration, and for `n_eff` there is no empirical test in `stats.independent`
> that can refute it (Limitation L12). The status reports on `M` and on nothing
> else.

---

## 4. Main claims

> **Claim 1.** Let `α ∈ (0, 1)`, `M ≥ 1`, and assume **P1** and
> **P2\*(M)**. Let `(L_n, U_n)` be the output of Algorithm 1 (§7) with
> `effective_n = None` whenever it issues an interval (`c_up < 1`). Then
>
> $$\Pr_F\big(\theta < L_n\big) \le \tfrac{\alpha}{2}, \qquad \Pr_F\big(\theta > U_n\big) \le \tfrac{\alpha}{2},$$
>
> hence `Pr_F(θ ∉ [L_n, U_n]) ≤ α`. Consequently, for **any** status rule
> `S_n`,
>
> $$\Pr_F\big(\theta\notin[L_n,U_n] \;\wedge\; S_n = \texttt{UNREFUTED}\big) \;\le\; \alpha .$$

**Argument.** The argument is assembled in §5: the upper endpoint comes from
Lemmas A and E (§5.1, §5.7), the lower endpoint from Lemma A or the Lemma B
lower sweep (§5.1, §5.2), and the data-independent branch selection from
Lemma C (§5.3). Because the issuance condition `c_up < 1` depends only on
`(n, M, α)` and the endpoints `(L_n, U_n)` do not depend on `M̂_n`, the per-side
bounds hold unconditionally over the sample space whenever `c_up < 1`. The final
clause follows from `Pr(A ∧ B) ≤ Pr(A)`. ∎

The final clause bounds the joint event `θ ∉ [L_n, U_n] ∧ S_n = UNREFUTED`, not
the conditional miss rate among `UNREFUTED` runs (`≤ α / Pr_F(S_n = UNREFUTED)`);
see the warning in
[README.md, "Statuses and what the interval promises"](README.md#statuses-and-what-the-interval-promises).

**Corollary 1 (`rmse`).** `Pr(√θ ∉ [√L_n, √U_n]) ≤ α` holds with
equality of events, since `x ↦ √x` is strictly increasing on `[0, ∞)`
(Lemma D, §5.4).

**Corollary 2 (bounded metrics).** If the metric's definition forces
`θ ≤ θ_max` — as `accuracy` does with `θ_max = 1`
(`metrics.get_theta_max("accuracy") == 1.0`) — then
`Pr(θ ∉ [L_n, min(U_n, θ_max)]) ≤ α` as well, because `{θ > min(U_n, θ_max)}`
and `{θ > U_n}` are **identical** events whenever `θ ≤ θ_max`. Clipping to a
definitional upper bound leaves coverage unchanged.

### 4.1 The correlated case

The second claim is for callers who can bound the variance inflation of `s̄`
under dependence via an effective sample size `effective_n = n_eff`. It is
stated separately because it rests on a **different premise (`P1′`) and a
smaller set of lemmas (Lemma A′ alone)**.

> **Claim 1′.** Let `α ∈ (0, 1)`, `M ≥ 1`, and assume
> **P1′(n_eff)** and **P2\*_pool(M)** (§2). Let `(L_n, U_n)` be the output of
> Algorithm 1 run with `effective_n = n_eff` whenever it issues an interval
> (`c_A(n_eff) < 1`). Then the same two per-side bounds hold:
>
> $$\Pr\big(\theta < L_n\big) \le \tfrac{\alpha}{2}, \qquad \Pr\big(\theta > U_n\big) \le \tfrac{\alpha}{2}.$$

**Argument.** Under `effective_n ≠ None`, both endpoints come from Lemma A′
(§5.8) with the single constant `c = c_A(n_eff)`, which is deterministic in
`(n_eff, M, α')`; the upper endpoint is issued only when `c < 1` and otherwise
refused with `ASSUMPTION_REQUIRED`. There is only one branch on each side, so no
branch-selection step is needed. ∎

> **Caution.** **Claim 1′ depends entirely on the caller's `n_eff`, and
> `stats.independent` cannot check it.** Unlike `M`, which has the empirical check
> `M̂_n > M` (§5.6), `n_eff` has no refutation test in `stats.independent` because
> a single sample without cluster or graph structure provides no consistent
> estimator of `Var(s̄)` (Limitation L12; when cluster, temporal, graph, or
> space-time structure is available, use the cluster-based modules
> documented in [THEORY.md](THEORY.md)). An overstated `n_eff` produces an
> interval that is too narrow, without warning. The same caveat applies to
> `sketch.merge` (Limitation L13): the merge rule assumes that shards are
> mutually uncorrelated, which a sketch without item identifiers cannot verify.

### 4.1.1 Sampling without replacement on the independent path (Claim 1″)

When evaluating on a random test subset drawn uniformly without replacement
from a finite pool of size $N$, callers do not need to set `effective_n = n`.
The default independent path (`effective_n = None` — using the smaller of the
Lemma A Cantelli constant and the Lemma E Bernstein constant on the upper side,
and the Lemma B sweep at $\gamma = 0.35\,\gamma^*$ when
$\text{slope}_\infty > 0.10$, or Lemma A′ otherwise, on the lower side)
maintains finite-sample coverage:

> **Claim 1″ (sampling without replacement on the independent path).**
> Let $\{z_1, \dots, z_N\}$ be a finite pool of $N \ge 2$ non-negative numbers
> with pool mean
> $$\theta \;:=\; \frac{1}{N} \sum_{r=1}^N z_r \;>\; 0$$
> and pool relative second moment satisfying **P2\*(M)** under the uniform pool
> law,
> $$\frac{1}{N} \sum_{r=1}^N z_r^2 \;\le\; M \theta^2 \qquad (M \ge 1).$$
> Let $(s_1, \dots, s_n)$ ($2 \le n \le N$) be a uniform random sample drawn
> **without replacement** from $\{z_1, \dots, z_N\}$, and let $(L_n, U_n)$ be
> the output of Algorithm 1 run on the independent path (`effective_n = None`,
> in either data mode $K = n$ or sketch mode $1 \le K < n$) whenever it issues
> an interval. Then for every $\alpha \in (0, 1)$,
> $$\Pr_{\mathrm{wor}}\bigl(\theta < L_n\bigr) \;\le\; \frac{\alpha}{2}, \qquad \Pr_{\mathrm{wor}}\bigl(\theta > U_n\bigr) \;\le\; \frac{\alpha}{2},$$
> and consequently $\Pr_{\mathrm{wor}}(\theta \notin [L_n, U_n]) \le \alpha$.

**Argument.** We construct an explicit joint probability space
$(\Omega, \mathcal{F}, \mathbb{P})$ coupling a without-replacement sample
$X = (X_1, \dots, X_n)$ and an i.i.d. with-replacement pool sample
$\tilde Y = (\tilde Y_1, \dots, \tilde Y_n)$ on which **both**
$\mathbb{E}[\bar{\tilde Y} \mid X] = \bar X$ **and**
$\mathbb{E}[\hat V_n(\tilde Y) \mid X] = (1 - 1/N)\hat V_n(X)$ hold
simultaneously. The argument has four parts: the coupling construction,
Lemma M.1 (the marginal laws), Lemma M.2 (the conditional expectations), and
the transfer of each endpoint bound.

#### Construction of the joint probability space $(\Omega, \mathcal{F}, \mathbb{P})$

Fix any pool values $z_1, \dots, z_N \in \mathbb{R}$ and sample size
$2 \le n \le N$. Let $[N] := \{1, \dots, N\}$ and $[n] := \{1, \dots, n\}$.
Define three **mutually independent** random objects on
$(\Omega, \mathcal{F}, \mathbb{P})$:

1. **Pool permutation $\Pi$:** A uniform random permutation of the entire pool,
   $$\Pi = (\Pi_1, \Pi_2, \dots, \Pi_N) \;\sim\; \mathrm{Uniform}(S_N),$$
   where $S_N$ is the symmetric group of all $N!$ permutations of $[N]$. Define
   the without-replacement sample $X = (X_1, \dots, X_n)$ as the first $n$
   elements selected by $\Pi$:
   $$X_i \;:=\; z_{\Pi_i}, \qquad i = 1, \dots, n.$$
2. **Occupancy / collision template $I$:** An i.i.d. uniform sequence of length
   $n$ drawn from $[N]$ (not $[n]$),
   $$I = (I_1, I_2, \dots, I_n) \;\stackrel{\mathrm{i.i.d.}}{\sim}\; \mathrm{Uniform}([N]), \qquad I \perp\!\!\!\perp \Pi.$$
   From $I$, define deterministically:
   - the number of distinct values
     $D(I) := |\{I_1, \dots, I_n\}| \in \{1, \dots, n\}$ (since $I$ has $n$
     entries, $1 \le D(I) \le n$ always), and
   - the first-appearance rank function
     $R(I) = (R_1, \dots, R_n) \in \{1, \dots, D(I)\}^n$, defined inductively by
     $R_1 := 1$ and, for $a = 2, \dots, n$,
     $$R_a \;:=\; \begin{cases} R_b & \text{if } I_a = I_b \text{ for some } b < a, \\ 1 + \max_{b < a} R_b & \text{otherwise.} \end{cases}$$
   By construction, for all $a, b \in [n]$,
   $$R_a = R_b \;\Longleftrightarrow\; I_a = I_b, \qquad\text{and}\qquad \{R_1, \dots, R_n\} = \{1, \dots, D(I)\} \subseteq [n].$$
3. **Sample-slot permutation $\tau$:** A uniform random permutation of $[n]$,
   $$\tau = (\tau_1, \dots, \tau_n) \;\sim\; \mathrm{Uniform}(S_n), \qquad \tau \perp\!\!\!\perp (\Pi, I).$$

Now define $\tilde Y = (\tilde Y_1, \dots, \tilde Y_n)$ on
$(\Omega, \mathcal{F}, \mathbb{P})$ by
$$\tilde Y_a \;:=\; X_{\tau(R_a)} \;=\; z_{\Pi_{\tau(R_a)}}, \qquad a = 1, \dots, n.$$
Note that $\tilde Y_a$ is well-defined because
$R_a \in \{1, \dots, D(I)\} \subseteq [n]$ and $\tau : [n] \to [n]$ is a
bijection.

#### The marginal law of $\tilde Y$ (Lemma M.1)

> **Lemma M.1.** On $(\Omega, \mathcal{F}, \mathbb{P})$,
> $X = (X_1, \dots, X_n)$ is a uniform sample of size $n$ without replacement
> from $\{z_1, \dots, z_N\}$, and $\tilde Y = (\tilde Y_1, \dots, \tilde Y_n)$
> is an i.i.d. uniform sample of size $n$ **with replacement** from
> $\{z_1, \dots, z_N\}$.

**Argument.** The marginal law of $X$ is immediate from
$\Pi \sim \mathrm{Uniform}(S_N)$. For $\tilde Y$, define the random pool-index
vector $J = (J_1, \dots, J_n) \in [N]^n$ by
$$J_a \;:=\; \Pi_{\tau(R_a)}, \qquad a = 1, \dots, n,$$
so that $\tilde Y_a = z_{J_a}$. It suffices to show that
$J \stackrel{d}{=} I \sim \mathrm{Uniform}([N]^n)$.

1. Extend $\tau \in S_n$ to a permutation $\tilde\tau \in S_N$ by setting
   $\tilde\tau(i) := \tau(i)$ for $1 \le i \le n$ and $\tilde\tau(i) := i$ for
   $n < i \le N$. Since $\Pi \sim \mathrm{Uniform}(S_N)$ is independent of
   $(\tau, I)$, the right-translated permutation
   $\Pi' := \Pi \circ \tilde\tau$ is also distributed as
   $\mathrm{Uniform}(S_N)$ and is **independent of $I$**.
2. For any fixed $d \in \{1, \dots, n\}$, let $[N]_{(d)}$ denote the set of all
   $N! / (N - d)!$ ordered $d$-tuples of distinct elements of $[N]$. Because
   $\Pi' \sim \mathrm{Uniform}(S_N)$ is independent of $I$, conditional on
   $D(I) = d$ and $R(I) = r = (r_1, \dots, r_n)$, the $d$-tuple of distinct
   indices
   $$(U_1, \dots, U_d) \;:=\; \bigl(\Pi'_1, \dots, \Pi'_d\bigr) \;=\; \bigl(\Pi_{\tau(1)}, \dots, \Pi_{\tau(d)}\bigr)$$
   is uniformly distributed over $[N]_{(d)}$.
3. Now consider $I = (I_1, \dots, I_n) \sim \mathrm{Uniform}([N]^n)$. Let
   $(V_1, \dots, V_{D(I)}) \in [N]_{(D(I))}$ denote the $D(I)$ distinct values
   appearing in $I$, listed in order of first appearance; by definition of
   $R(I)$, $I_a = V_{R_a}$ for every $a \in [n]$. Moreover, the map
   $(I_1, \dots, I_n) \mapsto \bigl(D(I),\, R(I),\, (V_1, \dots, V_{D(I)})\bigr)$
   is a bijection between the fiber
   $\{i \in [N]^n : D(i) = d,\, R(i) = r\}$ and $[N]_{(d)}$. Since every
   $i \in [N]^n$ has equal probability $N^{-n}$, conditional on $D(I) = d$ and
   $R(I) = r$, the $d$-tuple $(V_1, \dots, V_d)$ is also uniformly distributed
   over $[N]_{(d)}$.
4. Therefore,
   $\bigl(D(I),\, R(I),\, (U_1, \dots, U_{D(I)})\bigr) \stackrel{d}{=} \bigl(D(I),\, R(I),\, (V_1, \dots, V_{D(I)})\bigr)$.
   Applying the deterministic reconstruction map
   $(d, r, (u_1, \dots, u_d)) \mapsto (u_{r_1}, \dots, u_{r_n})$ yields
   $$J = (U_{R_1}, \dots, U_{R_n}) \;\stackrel{d}{=}\; (V_{R_1}, \dots, V_{R_n}) = I \;\sim\; \mathrm{Uniform}([N]^n).$$

Hence $\tilde Y_a = z_{J_a}$ ($a = 1, \dots, n$) are i.i.d. uniform draws with
replacement from $\{z_1, \dots, z_N\}$. ∎

#### Exact conditional expectations given $X$ (Lemma M.2)

Let $\bar X := \frac{1}{n}\sum_{i=1}^n X_i$,
$\bar{\tilde Y} := \frac{1}{n}\sum_{a=1}^n \tilde Y_a$, and
$$\hat V_n(X) \;:=\; \frac{1}{n(n-1)} \sum_{1 \le i < j \le n} (X_i - X_j)^2, \qquad \hat V_n(\tilde Y) \;:=\; \frac{1}{n(n-1)} \sum_{1 \le a < b \le n} (\tilde Y_a - \tilde Y_b)^2.$$

> **Lemma M.2.** On $(\Omega, \mathcal{F}, \mathbb{P})$, almost
> surely:
> $$\mathbb{E}\bigl[\bar{\tilde Y} \bigm| X\bigr] \;=\; \bar X \qquad\text{and}\qquad \mathbb{E}\bigl[\hat V_n(\tilde Y) \bigm| X\bigr] \;=\; \Bigl(1 - \frac{1}{N}\Bigr) \hat V_n(X).$$

**Argument.** By construction, $X = (z_{\Pi_1}, \dots, z_{\Pi_n})$ is
$\sigma(\Pi)$-measurable, and $(\tau, I)$ is independent of $\Pi$ (and hence
independent of $X$).

1. **Sample mean:** Fix $a \in [n]$. Conditional on $(X, I)$, the rank
   $R_a \in [n]$ is fixed, and since $\tau \sim \mathrm{Uniform}(S_n)$ is
   independent of $(X, I)$, the random index $\tau(R_a)$ is uniformly
   distributed over $[n]$. Thus
   $$\mathbb{E}\bigl[\tilde Y_a \bigm| X, I\bigr] \;=\; \mathbb{E}\bigl[X_{\tau(R_a)} \bigm| X, I\bigr] \;=\; \frac{1}{n} \sum_{i=1}^n X_i \;=\; \bar X.$$
   Taking $\mathbb{E}[\,\cdot \mid X]$ and averaging over $a = 1, \dots, n$
   yields $\mathbb{E}[\bar{\tilde Y} \mid X] = \bar X$.
2. **Sample variance:** Fix any pair of distinct positions
   $1 \le a < b \le n$, and condition first on $(X, I)$:
   - On the event $\{I_a = I_b\}$, we have $R_a = R_b$, hence
     $\tau(R_a) = \tau(R_b)$ and
     $(\tilde Y_a - \tilde Y_b)^2 = (X_{\tau(R_a)} - X_{\tau(R_b)})^2 = 0$.
   - On the event $\{I_a \ne I_b\}$, we have $R_a \ne R_b$ in $[n]$. Because
     $\tau \sim \mathrm{Uniform}(S_n)$ is independent of $(X, I)$, for any two
     distinct indices $r_1 \ne r_2$ in $[n]$ the ordered pair
     $(\tau(r_1), \tau(r_2))$ is uniformly distributed over all $n(n-1)$
     ordered pairs $(i, j)$ of distinct elements of $[n]$. Consequently, on
     $\{I_a \ne I_b\}$,
     $$\mathbb{E}\Bigl[\bigl(\tilde Y_a - \tilde Y_b\bigr)^2 \Bigm| X, I\Bigr] \;=\; \frac{1}{n(n-1)} \sum_{1 \le i \ne j \le n} (X_i - X_j)^2 \;=\; \frac{2}{n(n-1)} \sum_{1 \le i < j \le n} (X_i - X_j)^2 \;=\; 2 \hat V_n(X).$$

   Combining both cases gives the $\sigma(X, I)$-measurable identity
   $$\mathbb{E}\Bigl[\bigl(\tilde Y_a - \tilde Y_b\bigr)^2 \Bigm| X, I\Bigr] \;=\; \mathbf{1}\{I_a \ne I_b\}\, 2 \hat V_n(X).$$
   Now take $\mathbb{E}[\,\cdot \mid X]$. Since
   $I \sim \mathrm{Uniform}([N]^n)$ is independent of $X$,
   $\mathbb{P}(I_a \ne I_b \mid X) = \mathbb{P}(I_a \ne I_b) = 1 - \frac{1}{N}$,
   so
   $$\mathbb{E}\Bigl[\bigl(\tilde Y_a - \tilde Y_b\bigr)^2 \Bigm| X\Bigr] \;=\; \Bigl(1 - \frac{1}{N}\Bigr) 2 \hat V_n(X).$$
   Summing over all $\binom{n}{2} = \frac{n(n-1)}{2}$ pairs $1 \le a < b \le n$
   and dividing by $n(n-1)$ gives
   $$\mathbb{E}\bigl[\hat V_n(\tilde Y) \bigm| X\bigr] \;=\; \frac{1}{n(n-1)} \cdot \frac{n(n-1)}{2} \Bigl(1 - \frac{1}{N}\Bigr) 2 \hat V_n(X) \;=\; \Bigl(1 - \frac{1}{N}\Bigr) \hat V_n(X).$$

∎

#### Completing the argument for Claim 1″

Let $\alpha' := \alpha / 2$, and let
$\sigma_{\mathrm{pool}}^2 := \frac1N\sum_{r=1}^N z_r^2 - \theta^2$ be the pool
variance, so that $\sigma_{\mathrm{pool}}^2 \le (M-1)\theta^2$ by `P2*(M)`. We
check each endpoint of Algorithm 1 on the independent path
(`effective_n = None`), after one auxiliary moment-generating function (MGF)
bound.

0. **A Bernstein bound at the level of moment-generating functions (used in
   steps 1 and 2).** Let $D_1, \dots, D_n$ be i.i.d. with
   $\mathbb{E}[D_a] = 0$, $\mathbb{E}[D_a^2] = v_1$, and $D_a \le b$ for a
   constant $b > 0$. Let $T$ be a random variable with
   $\mathbb{E}[e^{\lambda n T}] \le \mathbb{E}[e^{\lambda \sum_a D_a}]$ for
   every $\lambda \in (0, 3/b)$. Then for every $\ell > 0$,
   $$\Pr\Bigl(T \;\ge\; \sqrt{2 v_1 \ell / n} + \frac{b\,\ell}{3n}\Bigr) \;\le\; e^{-\ell}.$$
   **Argument.**
   1. Put $\varphi(u) := (e^u - 1 - u)/u^2$ ($\varphi(0) := 1/2$). Then
      $\varphi(u) = \int_0^1 (1-s)\,e^{us}\,ds$, so $\varphi$ is
      non-decreasing. Since $\lambda D_a \le \lambda b$,
      $e^{\lambda D_a} = 1 + \lambda D_a + \lambda^2 D_a^2 \varphi(\lambda D_a) \le 1 + \lambda D_a + \lambda^2 D_a^2 \varphi(\lambda b)$.
      Taking expectations and using $1 + x \le e^x$ gives
      $\mathbb{E}[e^{\lambda D_a}] \le \exp(\lambda^2 v_1 \varphi(\lambda b))$.
   2. For $0 \le u < 3$,
      $\varphi(u) = \sum_{k \ge 2} u^{k-2}/k! \le \sum_{k \ge 2} u^{k-2}/(2 \cdot 3^{k-2}) = 1/(2(1 - u/3))$,
      because $k! \ge 2 \cdot 3^{k-2}$. With $V := n v_1$ and $q := b/3$,
      independence and step 1 give
      $\mathbb{E}[e^{\lambda n T}] \le \exp\bigl(V\lambda^2 / (2(1 - q\lambda))\bigr)$
      for $\lambda \in (0, 1/q)$.
   3. Let $\varepsilon := \sqrt{2 v_1 \ell / n} + b\ell/(3n)$, so
      $n\varepsilon = A + q\ell$ with $A := \sqrt{2V\ell}$. By Markov's
      inequality,
      $\Pr(T \ge \varepsilon) \le \exp\bigl(-\lambda n \varepsilon + V\lambda^2/(2(1 - q\lambda))\bigr)$
      for every $\lambda \in (0, 1/q)$. If $V = 0$, the exponent is
      $-\lambda q \ell \to -\ell$ as $\lambda \uparrow 1/q$. If $V > 0$, take
      $\lambda := A/(V + qA) \in (0, 1/q)$; then $1 - q\lambda = V/(V + qA)$
      and, using $A^2/2 = V\ell$,
      $$\lambda n\varepsilon - \frac{V\lambda^2}{2(1 - q\lambda)} \;=\; \frac{A(A + q\ell) - A^2/2}{V + qA} \;=\; \frac{V\ell + qA\ell}{V + qA} \;=\; \ell .$$
   In both cases $\Pr(T \ge \varepsilon) \le e^{-\ell}$. ∎

1. **Upper endpoint (Lemmas A and E):** On the independent path, Algorithm 1
   uses $U_n = \bar s / (1 - c_{\mathrm{up}})$ (when $c_{\mathrm{up}} < 1$,
   refusing with `ASSUMPTION_REQUIRED` when $c_{\mathrm{up}} \ge 1$), where
   $c_{\mathrm{up}} = \min(c_A(n, M, \alpha'),\, c_E(n, M, \alpha'))$ is
   deterministic in $(n, M, \alpha')$.
   - *Lemma A ($c_A$, Cantelli):* $\mathbb{E}_{\mathrm{wor}}[\bar s] = \theta$
     and
     $\mathrm{Var}_{\mathrm{wor}}(\bar s) = \frac{N-n}{N-1}\frac{\sigma_{\mathrm{pool}}^2}{n} \le \frac{(M-1)\theta^2}{n}$.
     Cantelli's inequality gives
     $\Pr_{\mathrm{wor}}(\bar s \le \theta(1 - c_A)) \le \alpha'$.
   - *Lemma E ($c_E$, one-sided Bernstein):* Apply the coupling above to the
     pool $\{z_1, \dots, z_N\}$, letting $X_i = s_i$ and $\tilde Y_a$ (i.i.d.
     with replacement from the pool). For any $\lambda > 0$,
     $u \mapsto \exp(\lambda n (\theta - u))$ is convex on $\mathbb{R}$. By
     Lemma M.2 ($\bar X = \mathbb{E}[\bar{\tilde Y} \mid X]$) and conditional
     Jensen's inequality,
     $$\mathbb{E}_{\mathrm{wor}}\!\left[\exp\bigl(\lambda n (\theta - \bar s)\bigr)\right] \;=\; \mathbb{E}\!\left[\exp\bigl(\lambda n (\theta - \mathbb{E}[\bar{\tilde Y} \mid X])\bigr)\right] \;\le\; \mathbb{E}_{\mathrm{iid}}\!\left[\exp\bigl(\lambda n (\theta - \bar{\tilde Y})\bigr)\right].$$
     Apply step 0 with $D_a := \theta - \tilde Y_a$ (i.i.d., mean $0$,
     $D_a \le \theta$ because $\tilde Y_a \ge 0$,
     $\mathbb{E}[D_a^2] = \sigma_{\mathrm{pool}}^2 \le (M-1)\theta^2$),
     $b := \theta$, $T := \theta - \bar s$, and $\ell := L_E := \ln(1/\alpha')$
     (the $L$ of Lemma E). This gives
     $\Pr_{\mathrm{wor}}\bigl(\theta - \bar s \ge \theta(\sqrt{2(M-1)L_E/n} + L_E/(3n))\bigr) \le \alpha'$,
     since replacing $\sigma_{\mathrm{pool}}^2$ by its upper bound only raises
     the threshold. The positive root of Lemma E's quadratic is
     $c_E = L_E/(3n) + \sqrt{L_E^2/(9n^2) + 2(M-1)L_E/n} \ge \sqrt{2(M-1)L_E/n} + L_E/(3n)$,
     so $\Pr_{\mathrm{wor}}(\bar s \le \theta(1 - c_E)) \le \alpha'$. Both
     constants are deterministic in $(n, M, \alpha')$, so by Lemma C,
     $\Pr_{\mathrm{wor}}(\theta > U_n) \le \alpha'$.

2. **Lower endpoint (Lemma A and the Lemma B lower sweep):** Algorithm 1
   selects deterministically between Cantelli $L_n = \bar s / (1 + c_A)$ and
   the Lemma B lower sweep based on $(n, \alpha, M, K)$ alone (Lemma C).
   Throughout this step $L := \ln(2/\alpha')$, as in Lemma B (§5.2).
   - When Cantelli is selected: with the mean and variance of step 1,
     Cantelli's inequality for the upper tail gives
     $\Pr_{\mathrm{wor}}(\bar s - \theta \ge c_A\theta) \le \frac{(M-1)\theta^2/n}{(M-1)\theta^2/n + c_A^2\theta^2} = \alpha'$
     (the arithmetic of Lemma A), and
     $\{\theta < \bar s/(1 + c_A)\} \subseteq \{\bar s - \theta \ge c_A\theta\}$.
   - When the Lemma B lower sweep is selected: set $c := \gamma\theta > 0$,
     which is deterministic because $\gamma$ depends only on $(n, M, \alpha)$
     and $\theta$ is a functional of the pool. Truncating a without-replacement
     sample, $X_i := s_i \wedge c$, gives a without-replacement sample from the
     truncated pool $\{x_1, \dots, x_N\}$, $x_r := z_r \wedge c \in [0, c]$,
     whose mean is $\mu_X \le \theta$ and whose variance is
     $\sigma_X^2 := \frac1N\sum_r (x_r - \mu_X)^2$. Couple $X$ with an i.i.d.
     sample $\tilde Y$ from the truncated pool by the construction above. Write
     $\bar X := \bar X(\theta)$ and $\hat V := \hat V(\theta) = \hat V_n(X)$.
     - *Miss event.* In data mode ($K = n$),
       $L_n = \inf\{t > 0 : H_{\mathrm{lo}}(t) \ge 0\}$. In sketch mode
       ($1 \le K < n$), $\widetilde H_{\mathrm{lo}}(t) \ge H_{\mathrm{lo}}(t)$
       for all $t > 0$, so
       $L_n^{\mathrm{sketch}} = \inf\{t > 0 : \widetilde H_{\mathrm{lo}}(t) \ge 0\} \le L_n$.
       In both modes $\{H_{\mathrm{lo}}(\theta) \ge 0\} \subseteq \{\theta \ge L_n\}$,
       so the miss event is contained in
       $\{H_{\mathrm{lo}}(\theta) < 0\} = \{\bar X - \theta > \sqrt{2\hat V L/n} + 7cL/(3(n-1))\}$.
       Since $\mu_X \le \theta$, this needs the **upper** tail of $\bar X$: it
       is contained in
       $\{\bar X - \mu_X > \sqrt{2\hat V L/n} + 7cL/(3(n-1))\}$.
     - *Mean concentration, upper tail (budget $\alpha'/2$):* For $\lambda > 0$,
       $u \mapsto \exp(\lambda n (u - \mu_X))$ is convex, so Lemma M.2 and
       conditional Jensen's inequality give
       $\mathbb{E}_{\mathrm{wor}}[\exp(\lambda n (\bar X - \mu_X))] \le \mathbb{E}_{\mathrm{iid}}[\exp(\lambda n (\bar{\tilde Y} - \mu_X))]$.
       Apply step 0 with $D_a := \tilde Y_a - \mu_X$ (i.i.d., mean $0$,
       $\mathbb{E}[D_a^2] = \sigma_X^2$, $D_a \le c - \mu_X \le c$), $b := c$,
       $T := \bar X - \mu_X$, and $\ell := L$:
       $$\Pr_{\mathrm{wor}}\Bigl(\bar X - \mu_X \ge \sqrt{2\sigma_X^2 L/n} + \frac{cL}{3n}\Bigr) \;\le\; e^{-L} \;=\; \frac{\alpha'}{2}.$$
     - *Empirical-variance lower tail (budget $\alpha'/2$):* Define
       $Z(X) := \frac{n}{2c^2}\hat V_n(X)$ and
       $Z(\tilde Y) := \frac{n}{2c^2}\hat V_n(\tilde Y)$. By Lemma M.2,
       $\mathbb{E}[Z(\tilde Y) \mid X] = (1 - 1/N)Z(X) \le Z(X)$. For every
       $\lambda > 0$, $z \mapsto \exp(-\lambda z)$ is non-increasing and convex
       on $[0, \infty)$, so
       $$\mathbb{E}_{\mathrm{wor}}\!\left[\exp\bigl(-\lambda Z(X)\bigr)\right] \;\le\; \mathbb{E}_{\mathrm{iid}}\!\left[\exp\bigl(-\lambda Z(\tilde Y)\bigr)\right].$$
       Maurer and Pontil's (2009) self-bounding MGF inequality for the empirical
       variance of independent `[0, c]` variables therefore transfers to
       $Z(X)$, giving
       $\Pr_{\mathrm{wor}}(\sigma_X - \sqrt{\hat V_n(X)} \ge c\sqrt{2 L / (n-1)}) \le \alpha'/2$.
     - *Combination:* Outside both events, which have total probability at most
       $\alpha'$, $\sigma_X < \sqrt{\hat V} + c\sqrt{2L/(n-1)}$ and therefore
       $$\bar X - \mu_X \;<\; \sqrt{\tfrac{2L}{n}}\,\sigma_X + \frac{cL}{3n} \;<\; \sqrt{\tfrac{2\hat V L}{n}} + \frac{2cL}{\sqrt{n(n-1)}} + \frac{cL}{3n} \;\le\; \sqrt{\tfrac{2\hat V L}{n}} + \frac{7cL}{3(n-1)},$$
       using $1/\sqrt{n(n-1)} \le 1/(n-1)$ and $1/(3n) \le 1/(3(n-1))$. By the
       miss-event step, $\Pr_{\mathrm{wor}}(\theta < L_n) \le \alpha' = \alpha / 2$. ∎

*(Scope note **L16**: Claim 1″ and Lemmas M.1–M.2 establish that the default
independent path (`effective_n = None`) maintains finite-sample coverage under
uniform sampling without replacement from any finite pool satisfying
**P2\*(M)**.)*

### 4.2 Accuracy on the independent path

For `accuracy` the per-item summand is a `{0, 1}` indicator, so under **P1**
the count `K = Σ s_i` follows `Binomial(n, θ)` and no moment declaration `M` is
needed. (In this subsection `K` denotes this count of correct items, not the
sketch size.) On the independent path (`effective_n is None`),
`interval.confidence_interval` bypasses Algorithm 1 and returns the exact
Clopper–Pearson binomial interval (`_accuracy_exact`).

> **Claim 3.** Let `α ∈ (0, 1)`, assume **P1** with `s_i ∈ {0, 1}`,
> and let `K = Σ s_i = k`. Put `α̃ = α − α²/4` (so `1 − α̃ = (1 − α/2)²`). With
> `B(q; a, b)` the `q`-quantile of `Beta(a, b)` (`scipy.special.betaincinv`),
> let
>
> $$L_n = B\big(\tfrac{\tilde\alpha}{2};\, k,\, n-k+1\big)\ (L_n = 0 \text{ if } k = 0), \qquad U_n = B\big(1-\tfrac{\tilde\alpha}{2};\, k+1,\, n-k\big)\ (U_n = 1 \text{ if } k = n).$$
>
> Then for every `θ ∈ [0, 1]` and every `n ≥ 1`,
> `Pr_θ(θ < L_n) ≤ α̃/2 ≤ α/2` and `Pr_θ(θ > U_n) ≤ α̃/2 ≤ α/2`.

**Argument.** This is the Clopper and Pearson (1934) interval evaluated at
level `1 − α̃`. For `k ≥ 1`, `L_n` is the unique `θ` at which
`Pr_θ(K ≥ k) = α̃/2`, using the identity `Pr_θ(K ≥ k) = I_θ(k, n−k+1)` (the
regularized incomplete Beta function). Since `Pr_θ(K ≥ k)` is strictly
increasing in `θ`, `θ < L_n` holds if and only if `Pr_θ(K ≥ k) < α̃/2`, that is,
if and only if the observed count `k` falls in the upper tail of
`Binomial(n, θ)` of total mass below `α̃/2`. The probability of that set of
counts is at most `α̃/2` for every `θ`. The upper endpoint is symmetric. ∎

The interval is evaluated at `α̃` rather than `α` so that Claim 3′ below holds
for without-replacement sampling as well. The width increase is about `α/4` in
relative terms on the tail quantile (about 0.3% of width at `level = 0.95`).

*   **Premises and input checks.** Only **P1** (independent `{0, 1}` indicators
    with a common mean) or uniform sampling without replacement (Claim 3′);
    **P2\*(M)** is not used. The argument `m` is accepted and reported in
    `Result.m_declared` but has no effect on `[L_n, U_n]`, and the status is
    `UNREFUTED` whenever an interval is returned (`n ≥ 1` and finite sums).
    Input validation enforces binary `{0, 1}` values and `level ≥ √2 − 1`:
    - For a 1D raw array, `metrics.extract_summands` raises `ValueError` if any
      finite entry is not `0.0` or `1.0`.
    - For an `IndependentSketch`, `confidence_interval` and `_accuracy_exact` raise
      `ValueError` if `top_32` contains a value exceeding `1.0`, or if finite
      `sum_s` and `sum_s2` fail `float(round(sum_s)) == sum_s`,
      `sum_s2 == sum_s`, or `0 ≤ sum_s ≤ n`.
    - `_accuracy_exact` raises `ValueError` if
      `level < √2 − 1 ≈ 0.41421356` (`α > 2 − √2`).
*   **Asymptotic rate.** Half-width
    `z_{α/2} √(θ(1−θ)/n) (1 + o(1))`, where `z_{α/2}` is the upper `α/2`
    quantile of the standard normal distribution. Every sample size `n ≥ 1`
    returns a finite interval.
*   **Sharpness.** The per-side bound `α̃/2` is sharp over `θ ∈ [0, 1]`: as `θ`
    increases to `L_n(k)` for fixed `k`, the per-side miss probability tends to
    `α̃/2`, so `sup_θ Pr_θ(θ < L_n) = α̃/2` (and similarly for `U_n`). At other
    values of `θ` the miss probability is strictly smaller because `K` is
    discrete; discreteness and the `α/2 − α̃/2 = α²/8` adjustment are the only
    sources of conservatism.
*   **Correlated path.** When `effective_n` is set (`eff_n is not None`), `K`
    is not binomial, and `confidence_interval` routes `accuracy` to Algorithm 1
    under Claim 1′.

> **Claim 3′ (sampling without replacement).** Let
> `α ∈ (0, 2 − √2]` (so `level = 1 − α ≥ √2 − 1 ≈ 0.4142`). Suppose a finite
> pool of `N` items contains `D` correct items, `θ = D/N`, and the `n ≤ N`
> evaluated items are drawn uniformly without replacement, so `K` follows a
> hypergeometric distribution. Then the interval of Claim 3 satisfies
> `Pr_wor(θ < L_n) ≤ α/2` and `Pr_wor(θ > U_n) ≤ α/2`, for every `N ≥ 1`,
> `0 ≤ D ≤ N`, and `1 ≤ n ≤ N`.

**Argument.** We argue the lower side; the upper side follows by applying the
same argument to the incorrect indicators `1 − s_i`, since the Clopper–Pearson
endpoints satisfy `U_n(k) = 1 − L_n(n − k)` for the complementary count. If
`θ = 0` there is nothing to argue. Let `k₀` be the smallest integer `k` with
`θ < L_n(k)`; the miss event is `{K ≥ k₀}`, and by the argument for Claim 3,
`Pr_{Bin(n,θ)}(X ≥ k₀) < α̃/2`, where `X ~ Binomial(n, θ)`.

1.  The hypergeometric law of `K` is the law of a sum of `n` independent
    Bernoulli random variables with means summing to `nθ` (Vatutin and
    Mikhailov, 1982: its probability generating polynomial has only real
    roots). For such Poisson–binomial sums, Hoeffding (1956, Section 4) shows
    that `Pr_wor(K ≥ c) ≤ Pr_{Bin(n,θ)}(X ≥ c)` whenever `c ≥ nθ + 1`.
2.  **Case `nθ > 1`.** Greenberg and Mohri (2014) show that
    `Pr_{Bin(n,θ)}(X ≥ nθ) > 1/4` whenever `θ > 1/n`. For `α ∈ (0, 1)`,
    `α̃ = α − α²/4 ≤ 1/2` holds if and only if `α² − 4α + 2 ≥ 0`, that is,
    `α ≤ 2 − √2`, which gives `α̃/2 ≤ 1/4`. The binomial upper tail at `⌈nθ⌉` is
    therefore strictly above `α̃/2`, which forces `k₀ ≥ ⌈nθ⌉ + 1 ≥ nθ + 1`. Step
    1 then gives `Pr_wor(K ≥ k₀) ≤ Pr_{Bin(n,θ)}(X ≥ k₀) < α̃/2 ≤ α/2`.
3.  **Case `nθ ≤ 1`, `k₀ ≥ 2`.** Then `k₀ ≥ 2 ≥ nθ + 1`, and step 1 applies as
    in case 2.
4.  **Case `nθ ≤ 1`, `k₀ = 1`.** Markov's inequality gives
    `Pr_wor(K ≥ 1) ≤ E[K] = nθ`. Moreover, `θ < L_n(1)` means
    `1 − (1−θ)^n < α̃/2`, and by the alternating-series inequality
    `1 − (1−θ)^n ≥ nθ − (nθ)²/2`. With `x = nθ ∈ (0, 1]`, the quadratic
    `x − x²/2` is strictly increasing on `[0, 1]`, so `x − x²/2 < α̃/2` implies
    `x < 1 − √(1 − α̃) = α/2` by the definition `α̃ = α − α²/4`. Thus
    `Pr_wor(K ≥ 1) ≤ nθ < α/2`. ∎

*   **Why `α̃` is used.** With the unadjusted level `α`, case 4 can exceed
    `α/2`: for example, `N = 100`, `D = 1`, `n = 5`, and `α = 0.0995` give a
    lower-side miss probability of `Pr_wor(K ≥ 1) = 0.05 > α/2 = 0.04975`. Such
    exceedances occur only when `nθ < 1` or `n(1−θ) < 1` in narrow windows of
    `α`, and the quadratic level adjustment `α̃ = α − α²/4` eliminates them for
    all `α ≤ 2 − √2`.
*   **Level guard.** The condition `α ≤ 2 − √2 ≈ 0.5858` (`level ≥ √2 − 1 ≈
    0.4142`) is used in step 2 (`_MIN_ACCURACY_LEVEL = math.sqrt(2.0) - 1.0` in
    `interval.py`).

*(Scope note **L17**: Claim 3′ establishes that the Clopper–Pearson interval at
the adjusted level $\tilde\alpha = \alpha - \alpha^2/4$ maintains finite-sample
coverage under uniform sampling without replacement for all
$\alpha \le 2 - \sqrt{2}$, i.e., `level ≥ √2 − 1 ≈ 0.4142`.)*

**Corollary 3 (error rate).** Binary `{0, 1}` indicators of any
event can be passed with `metric="accuracy"` on the independent path. Because
Claim 3 does not depend on `M`, estimating an error rate of `0.01` has the same
interval width as estimating an accuracy of `0.99`. (On the correlated path
`effective_n ≠ None`, the generic bound of Claim 1′ applies, where `M = 1/p`
for an indicator of mean `p`: an accuracy of `0.99` has `M ≈ 1.01`, whereas an
error rate of `0.01` has `M = 100` and requires a much larger `effective_n`.)

---

## 5. Supporting lemmas and propositions

This section states and argues the lemmas and propositions that Algorithm 1
(§7) combines. Throughout, `α' := α/2` is the per-side failure budget.

### 5.1 Lemma A — Cantelli bound

> **Lemma A.** Under `P1 ∧ P2*(M)`, let
> `c := √((M−1)(1−α')/(nα'))` (`bound.c_cantelli`). Then
> `Pr_F(θ > s̄/(1−c)) ≤ α'` provided `c < 1`, and `Pr_F(θ < s̄/(1+c)) ≤ α'`
> for every `c ≥ 0`. For `c > 0` the same bounds hold with `≥` and `≤` in
> place of `>` and `<`.

**Argument.** Under `P1`, `Var(s̄) = σ²/n` with `σ² = Var(s)`. Cantelli's
one-sided Chebyshev inequality gives, for any `t > 0`,

$$\Pr_F(\bar s - \theta \le -t) \;\le\; \frac{\sigma^2/n}{\sigma^2/n + t^2} \;=\; \frac{\sigma^2}{\sigma^2 + nt^2}.$$

The map `x ↦ x/(x + nt²)` is non-decreasing on `x ≥ 0`, so substituting the
bound `σ² ≤ (M−1)θ²` from `P2*(M)` can only increase the right-hand side. Take
`t = cθ`:

$$\Pr_F\big(\bar s \le \theta(1-c)\big) \;\le\; \frac{(M-1)\theta^2}{(M-1)\theta^2 + nc^2\theta^2} \;=\; \frac{M-1}{(M-1)+nc^2}.$$

By definition `nc² = (M−1)(1−α')/α'`, so the right-hand side equals
`(M−1) / ((M−1)·[1 + (1−α')/α']) = α'`. This needs `t = cθ > 0`, i.e.
`c > 0`. Hence, for `0 < c < 1`, `Pr_F(θ ≥ s̄/(1−c)) = Pr_F(s̄ ≤ θ(1−c)) ≤ α'`.
Applying Cantelli's inequality to `s̄ − θ ≥ cθ` gives, for every `c > 0`,
`Pr_F(θ ≤ s̄/(1+c)) = Pr_F(s̄ − θ ≥ cθ) ≤ α'`. The strict events are subsets of
these. If `c = 0`, then `M = 1`, `Var(s̄) = 0` and `s̄ = θ` almost surely, so
both strict events `θ > s̄` and `θ < s̄` have probability `0`. ∎

**Finiteness threshold.** `c < 1 ⟺ n > n_A(M, α) := (M−1)(1−α/2)/(α/2)`
(`bound.n_A`). At `α = 0.05`, `n_A = 39(M−1)`. On the independent path this is
not the refusal threshold because Lemma E supplies a second upper bound that is
finite at smaller `n` (§5.9).

### 5.2 Lemma B — truncation (the lower-sweep endpoint)

Lemma B is a truncated empirical-Bernstein bound based on Maurer and Pontil
(2009). Algorithm 1 uses its **lower** half, in both data mode ($K = n$) and
sketch mode ($1 \le K < n$, via the piecewise quadratic relaxation described
below under `L11`), whenever `slope_∞ > 0.10` ($n > n_B(M, \alpha)$, §5.3). The
upper endpoint comes from Lemma E (§5.7) instead, which needs no order
statistics. Both the upper and lower truncation relations are stated below
because the reference scale `γ*` and the routing quantity `slope_∞` are defined
from the upper truncation relation.

Fix a positive number `γ > 0` depending only on `(M, n, α)`. Set

$$c := \gamma\theta, \qquad X_i := s_i \wedge c, \qquad L := \ln(2/\alpha').$$

Because `c = γθ` is a deterministic functional of the law `F`, the truncated
variables `X_i` are i.i.d. bounded in `[0, c]` under `P1`.

**Step B1 (tail bias bound for the upper side).** On `{s > c}`, `1 ≤ s/c`, so
`s·1{s > c} ≤ s²/c`. Under `P2*(M)`,

$$0 \;\le\; \theta - \mathbb{E}[X] \;=\; \mathbb{E}[(s-c)\mathbf 1\{s>c\}] \;\le\; \mathbb{E}[s\,\mathbf 1\{s>c\}] \;\le\; \frac{\mathbb{E}[s^2]}{c} \;\le\; \frac{M\theta^2}{\gamma\theta} \;=\; \frac{M\theta}{\gamma}.$$

**Step B2 (bounded empirical-Bernstein concentration).** Since `X_i ∈ [0, c]`
are i.i.d., the empirical Bernstein inequality of Maurer and Pontil (2009),
rescaled from `[0, 1]` to `[0, c]`, gives with probability at least `1 − α'`

$$\mathbb{E}[X] \;\le\; \bar X + \sqrt{\frac{2\hat V L}{n}} + \frac{7cL}{3(n-1)},\qquad \hat V := \tfrac{1}{n(n-1)}\!\!\sum_{i>j}(X_i-X_j)^2 .$$

The mirror-image bound for the upper tail of `X̄ − E[X]` holds with the same
constants: with probability at least `1 − α'`,

$$\bar X - \mathbb{E}[X] \;\le\; \sqrt{\frac{2\hat V L}{n}} + \frac{7cL}{3(n-1)}.$$

**Step B3 (reference upper function).** Combining Steps B1 and B2 gives, with
probability at least `1 − α'`,

$$\theta \;\le\; \bar X(\theta) + \sqrt{\tfrac{2\hat V(\theta) L}{n}} + \tfrac{7\gamma\theta L}{3(n-1)} + \tfrac{M\theta}{\gamma} \;=:\; G(\theta), \qquad H(t) := t - G(t).$$

**Lower side — the bound implemented in `bound._solve_lower_piecewise`.**

> **Lemma B (lower sweep).** Under `P1`, for any deterministic
> truncation scale `γ > 0` and `L = ln(2/α')`, define for `t > 0`
> $$\bar X(t) := \frac{1}{n}\sum_{i=1}^n (s_i \wedge \gamma t), \qquad \hat V(t) := \frac{1}{n(n-1)}\sum_{1 \le j < i \le n} \bigl((s_i \wedge \gamma t) - (s_j \wedge \gamma t)\bigr)^2,$$
> $$G_{\mathrm{lo}}(t) := \bar X(t) - \sqrt{\frac{2\hat V(t) L}{n}} - \frac{7\gamma t L}{3(n-1)}, \qquad H_{\mathrm{lo}}(t) := t - G_{\mathrm{lo}}(t),$$
> and let $L_n := \inf\{t > 0 : H_{\mathrm{lo}}(t) \ge 0\}$ (or, for any
> relaxation $\widetilde H_{\mathrm{lo}}(t) \ge H_{\mathrm{lo}}(t)$ for all
> $t > 0$, $L_n^{\mathrm{sketch}} := \inf\{t > 0 : \widetilde H_{\mathrm{lo}}(t) \ge 0\}$).
> Then $\Pr_F(\theta < L_n) \le \alpha'$ and
> $\Pr_F(\theta < L_n^{\mathrm{sketch}}) \le \alpha'$.

**Argument.** Since $X_i(\theta) = s_i \wedge \gamma\theta \le s_i$ almost
surely, $\mathbb{E}[X(\theta)] \le \theta$ holds unconditionally without using
Step B1 or `P2*(M)`. By the mirror-image empirical-Bernstein bound of Step B2,
with probability at least $1 - \alpha'$,
$$\theta \;\ge\; \mathbb{E}[X(\theta)] \;\ge\; \bar X(\theta) - \sqrt{\frac{2\hat V(\theta)L}{n}} - \frac{7\gamma\theta L}{3(n-1)} \;=\; G_{\mathrm{lo}}(\theta),$$
which is $H_{\mathrm{lo}}(\theta) \ge 0$. By definition of the infimum,
$\{H_{\mathrm{lo}}(\theta) \ge 0\} \subseteq \{\theta \ge L_n\}$, and when
$\widetilde H_{\mathrm{lo}} \ge H_{\mathrm{lo}}$ pointwise,
$L_n^{\mathrm{sketch}} \le L_n$, so
$\{H_{\mathrm{lo}}(\theta) \ge 0\} \subseteq \{\theta \ge L_n^{\mathrm{sketch}}\}$
as well. ∎

**Piece notation (`bound._solve_lower_piecewise`).** The order statistics divide
`t > 0` into pieces: on the piece where `k` summands exceed `γt` (`k`
exceedances), write `S_1 := Σ_{i : s_i ≤ γt} s_i` and
`S_2 := Σ_{i : s_i ≤ γt} s_i²` for the untruncated sums (`suf1[k]` and
`suf2[k]` in `bound.py`). On that piece,
`X̄(t) = (S_1 + kγt)/n`, `V̂(t) = v_0 + v_1 t + v_2 t²` with

$$v_0 = \frac{n S_2 - S_1^2}{n(n-1)}, \qquad v_1 = -\frac{2 k \gamma S_1}{n(n-1)}, \qquad v_2 = \frac{k(n-k)\gamma^2}{n(n-1)},$$

and `H_lo(t) = d₁ t − d₀ + √(C_L V̂(t))` with `C_L := 2L/n`, `d₀ := S_1/n`, and
`d₁ := 1 − kγ/n + 7γL/(3(n−1))`.

**Shape of `V̂` and `H_lo` on exact pieces.**

*   *Monotonicity of `V̂(t)`.* For `s_i ≥ s_j`,
    `X_i(t) − X_j(t) = 0 ∨ ((s_i ∧ γt) − s_j)`, which is non-negative and
    non-decreasing in `t`, so `V̂(t)` is non-decreasing.
*   *Piecewise convexity of `H_lo(t)`.* On an exact piece, `X(t)` is affine in
    `t`, and `√V̂(t) = ‖P X(t)‖₂ / √(n−1)` where
    `P = I − n⁻¹ 1 1ᵀ` is the centring projector. A Euclidean norm composed
    with an affine map is convex, so `H_lo(t)` is convex on each exact piece.

**Upward sweep and per-piece rule (`_solve_piece` in `bound.py`).** Because
`L_n = inf{t > 0 : H_lo(t) ≥ 0}`, `_solve_lower_piecewise` sweeps pieces in
increasing order of `t`:
1. At the left endpoint `t_lo` of a piece `[t_lo, t_hi]`, if `t_lo > 0` and
   `H_lo(t_lo) ≥ −10⁻¹²`, it returns `t_lo` immediately.
2. At `t_lo = 0`, if `d₀ = v₀ = v₁ = 0` (which occurs when all untruncated
   summands on the bottom piece are exact zeros), `H_lo(0) = 0` and the right
   derivative at `0` is `d₁ + √(C_L v₂)`; if this slope is `≥ 0`, `H_lo(t) ≥ 0`
   on the entire bottom piece and the solver returns `0.0`.
3. Otherwise `H_lo(t_lo) < 0`. Setting `d₁ t − d₀ = −√(C_L(v₂ t² + v₁ t + v₀))`
   requires `d₀ − d₁ t ≥ 0` and squares to the quadratic
   `p₂ t² + p₁ t + p₀ = 0` with
   `p₂ = d₁² − C_L v₂`, `p₁ = −2 d₀ d₁ − C_L v₁`, and `p₀ = d₀² − C_L v₀`. The
   solver filters roots `r > 10⁻¹²` lying in `(t_lo, t_hi]` that satisfy
   `d₀ − d₁ r ≥ −10⁻⁹` and `v₂ r² + v₁ r + v₀ ≥ −10⁻⁹`, and returns
   `min(valid_roots)` if non-empty.

**Reverse-cumulative sums (`suf1`, `suf2`).** In `bound._solve_lower_piecewise`,
the untruncated sums `suf1[k]` and `suf2[k]` are accumulated bottom-up from the
smallest retained order statistics rather than by subtracting prefix sums from
`sum_s`, avoiding catastrophic cancellation on bottom pieces when the smallest
summands are zero.

**Sketch-mode lower sweep (`L11` relaxation in `bound._solve_lower_piecewise`
and `bound._truncated_moments`).** In sketch mode ($1 \le K < n$), the sketch
retains $s_{(1)} \ge \dots \ge s_{(K)} =: s_K$ along with $n$, $\sum_{i=1}^n s_i$,
and $\sum_{i=1}^n s_i^2$. The $n_{\mathrm{rest}} := n - K$ unretained summands
$s_{(K+1)}, \dots, s_{(n)}$ all lie in $[0, s_K]$, with clamped residual sums
(`bound._solve_lower_piecewise`):

$$s_{\mathrm{rest}} := \min\!\Bigl(\max\!\bigl(0,\, \textstyle\sum_{i=1}^n s_i - \sum_{j=1}^K s_{(j)}\bigr),\; n_{\mathrm{rest}}\,s_K\Bigr),$$

$$q_{\mathrm{rest}} := \min\!\Bigl(\max\!\Bigl(\max\!\bigl(0,\, \textstyle\sum_{i=1}^n s_i^2 - \sum_{j=1}^K s_{(j)}^2\bigr),\; \frac{s_{\mathrm{rest}}^2}{n_{\mathrm{rest}}}\Bigr),\; s_{\mathrm{rest}}\,s_K\Bigr).$$

*   For $t \ge s_K / \gamma$ (threshold $c = \gamma t \ge s_K$), none of the
    $n_{\mathrm{rest}}$ unretained summands is truncated, so their contribution
    to $S_1$ and $S_2$ is known exactly as $\text{suf1}[K] = s_{\mathrm{rest}}$
    and $\text{suf2}[K] = q_{\mathrm{rest}}$. The pieces $k = K-1, \dots, 0$
    are therefore identical to data mode.
*   For $t \in [0, s_K / \gamma]$ (threshold $c = \gamma t \in [0, s_K]$), all
    $K$ retained summands are truncated to $c$. If $s_{\mathrm{rest}} \le 0$ or
    $q_{\mathrm{rest}} \le 0$, all unretained summands are zero and the single
    bottom piece has $d_0 = v_0 = v_1 = 0$,
    $d_1 = 1 - K\gamma/n + 7\gamma L/(3(n-1))$, and
    $v_2 = K(n - K)\gamma^2 / (n(n-1))$.
*   When $s_{\mathrm{rest}} > 0$ and $q_{\mathrm{rest}} > 0$, let
    $Y_1(c) := \sum_{i > K} (s_{(i)} \wedge c)$ and
    $Y_2(c) := \sum_{i > K} (s_{(i)} \wedge c)^2$. To obtain an upper bound
    $\widetilde H_{\mathrm{lo}}(t) \ge H_{\mathrm{lo}}(t)$, it suffices to
    bound $Y_1(c)$ from **below** and the total truncated variance from
    **above**:
    1. *Lower envelope for $Y_1(c)$:* For every $u \in [0, s_K]$,
       $u \wedge c \ge (c/s_K) u$, giving the chord bound
       $Y_1(c) \ge (s_{\mathrm{rest}}/s_K)c$. Moreover, for $c > 0$,
       $(u - c)^2 \ge 0$ gives $u \wedge c \ge u - u^2/(4c)$, so
       $Y_1(c) \ge s_{\mathrm{rest}} - q_{\mathrm{rest}}/(4c)$, where
       $\sum_{i>K} s_{(i)} = s_{\mathrm{rest}}$ and
       $\sum_{i>K} s_{(i)}^2 = q_{\mathrm{rest}}$. At
       $c_{\mathrm{mid}} := q_{\mathrm{rest}}/(2 s_{\mathrm{rest}})$ this equals
       $s_{\mathrm{rest}}/2$, and since $Y_1$ is concave with $Y_1(0) = 0$,
       $Y_1(c) \ge (c/c_{\mathrm{mid}})\,Y_1(c_{\mathrm{mid}})$ on
       $[0, c_{\mathrm{mid}}]$. Together these give the two-branch lower bound
       $$q_{\mathrm{quad}}(c) \;:=\; \begin{cases} \dfrac{s_{\mathrm{rest}}^2}{q_{\mathrm{rest}}}\,c & \text{if } c \le c_{\mathrm{mid}}, \\[8pt] s_{\mathrm{rest}} - \dfrac{q_{\mathrm{rest}}}{4c} & \text{if } c > c_{\mathrm{mid}}. \end{cases}$$
       Thus $Y_1(c) \ge y_{\mathrm{lo}}(c) := \min\bigl(s_{\mathrm{rest}},\, \max\bigl(\frac{s_{\mathrm{rest}}}{s_K} c,\, q_{\mathrm{quad}}(c)\bigr)\bigr)$.
       Let $c_1 := s_{\mathrm{rest}} / n_{\mathrm{rest}}$ and
       $c_2 := q_{\mathrm{rest}} / s_{\mathrm{rest}}$. By Cauchy–Schwarz
       $c_1 \le c_2$, and $c_{\mathrm{mid}} = c_2/2$ may lie on either side of
       $c_1$ (equal unretained values give $c_{\mathrm{mid}} = c_1/2$).
       Partition $[0, s_K]$ at the sorted knots $\{0, c_1, c_{\mathrm{mid}},
       c_2, s_K\} \cap [0, s_K]$. On each sub-interval $[c_a, c_b]$ between
       consecutive knots, let
       $B_1 := (y_{\mathrm{lo}}(c_b) - y_{\mathrm{lo}}(c_a))/(c_b - c_a) \ge 0$
       and $A_1 := \max(0, y_{\mathrm{lo}}(c_a) - B_1 c_a)$. The function
       $y_{\mathrm{lo}}$ itself need not be concave (the chord and the second
       branch of $q_{\mathrm{quad}}$ can cross with a convex kink), but
       $Y_1(c) = \sum_{i>K} (s_{(i)} \wedge c)$ is concave and lies above
       $y_{\mathrm{lo}}$ at $c_a$ and $c_b$, so it lies above the secant through
       $(c_a, y_{\mathrm{lo}}(c_a))$ and $(c_b, y_{\mathrm{lo}}(c_b))$ on the
       whole sub-interval. The secant intercept
       $y_{\mathrm{lo}}(c_a) - B_1 c_a$ is non-negative because
       $y_{\mathrm{lo}}(c)/c$ is non-increasing on $(0, s_K]$, so the
       $\max(0, \cdot)$ only guards against rounding. Hence the
       total truncated sum $T_1(c) = K c + Y_1(c)$ satisfies
       $$T_1(c) \;\ge\; A_1 + (K + B_1)c \;=\; A_1 + k_{\mathrm{eff}}\,\gamma t, \qquad k_{\mathrm{eff}} := K + B_1.$$
    2. *Upper bound on $Y_2(c)$ and the truncated variance:* Since each of the
       $n_{\mathrm{rest}}$ unretained items satisfies
       $(s_{(i)} \wedge c)^2 \le \min(c^2,\, s_{(i)} c,\, s_{(i)}^2)$, summing
       over $i > K$ gives
       $$Y_2(c) \;\le\; \min\bigl(n_{\mathrm{rest}}\,c^2,\; s_{\mathrm{rest}}\,c,\; q_{\mathrm{rest}}\bigr) \;=\; \begin{cases} n_{\mathrm{rest}}\,c^2 & \text{for } c \in [0, c_1], \\ s_{\mathrm{rest}}\,c & \text{for } c \in [c_1, c_2], \\ q_{\mathrm{rest}} & \text{for } c \in [c_2, s_K]. \end{cases}$$
       Combining $T_2(c) \le K c^2 + \min(n_{\mathrm{rest}} c^2, s_{\mathrm{rest}} c, q_{\mathrm{rest}})$
       with $T_1(c) \ge A_1 + k_{\mathrm{eff}} c$ in
       $\hat V(t) = (n T_2(c) - T_1(c)^2)/(n(n-1))$ yields the quadratic upper
       bound $v_0 + v_1 t + v_2 t^2 \ge \hat V(t)$ on each sub-piece
       $[c_a/\gamma, c_b/\gamma]$ (`bound._solve_lower_piecewise`), and therefore a
       pointwise upper bound $\widetilde H_{\mathrm{lo}}(t) \ge H_{\mathrm{lo}}(t)$.

**Choice of `γ` and the routing threshold `n_B` (`bound._lemma_b_params`,
`bound.n_B`, `bound._use_lower_sweep`).** For `γt > s_(1)` no summand is
truncated, `X̄(t)` and `V̂(t)` are constant, and the reference upper function
`H(t) = t − G(t)` is affine with asymptotic slope

$$\text{slope}_\infty \;=\; 1 - \frac{7\gamma L}{3(n-1)} - \frac{M}{\gamma}.$$

Maximising `slope_∞` over `γ > 0` gives the reference scale
`γ* = √(3M(n−1)/(7L))` with `L = ln(2/α') = ln(4/α)`. The implementation sets
`γ = 0.35 · γ*` (`_GAMMA_SCALE = 0.35`), which reduces `V̂(t)` at the lower-sweep
crossing, and routes to the lower sweep on the independent path
(`effective_n is None`) whenever `K ≥ 1` and `slope_∞ > 0.10`
(`_SLOPE_INF_MIN = 0.10`).

Writing `a := M/γ* = √(7LM/(3(n−1)))`, at `γ = 0.35 · γ*` we have
`slope_∞ = 1 − (0.35 + 1/0.35) a`. Thus `slope_∞ > 0.10` holds if and only if
`n > n_B(M, α)`, where (`bound.n_B`):

$$n_B(M, \alpha) \;:=\; 1 + \Bigl(\frac{0.35 + 1/0.35}{0.90}\Bigr)^2 \frac{7}{3}\, M \ln(4/\alpha) \;\approx\; 1 + 29.63\, M \ln(4/\alpha).$$

Notice that when `slope_∞ > 0.10` and every `s_i > 0`, the slope of
`H_lo(t)` on `[0, s_(n)/γ)` is `1 − γ + 7γL/(3(n−1)) ≤ 1 − 0.35/a + 0.35a < 0`
because `a < 0.90/(0.35 + 1/0.35) ≈ 0.2806` implies `a + 0.35a² < 0.3083 < 0.35`.

### 5.3 Lemma C — data-independent branch selection

> **Lemma C.** If the choice among a family of valid per-side bounds
> is a function of `(n, M, α, K, n_eff)` alone — that is, of quantities fixed
> before the data values are inspected — the selected bound retains the
> per-side failure probability `α'` with no union-bound penalty.

**Argument.** Every bound in the family holds at level `α'` under the stated
premises. Because the selector depends only on non-random parameters fixed
prior to observing `(s_1, …, s_n)`, the selected bound is a single fixed valid
bound for that `(n, M, α, K, n_eff)`. ∎

> **Caution.** **The selector must be data-independent, not the endpoints.**
> Taking `max(L_cantelli, L_sweep)` or `min(U_cantelli, U_bernstein)` over
> *realised* endpoints would be a union of two data-dependent miss events and
> would require splitting `α'` in half. Taking `c_up = min(c_A, c_E)` (§5.7.1)
> and routing via `_use_lower_sweep(n, alpha, M, K, effective_n)` preserve level
> `α'` because `c_A`, `c_E`, and `slope_∞` are functions of
> `(n, M, α, K, effective_n)` alone.

### 5.4 Lemma D — monotone transforms

> **Lemma D.** If `g` is strictly increasing on the range of `θ`,
> then `Pr(g(θ) ∉ [g(L_n), g(U_n)]) = Pr(θ ∉ [L_n, U_n]) ≤ α`.

**Argument.** A strictly increasing map preserves strict inequalities in both
directions, so `{g(θ) < g(L_n)} = {θ < L_n}` and
`{g(θ) > g(U_n)} = {θ > U_n}` are identical events. ∎

Used in `bound.bound` for `rmse = √mse` (Corollary 1). This is an exact
identity of events, not a first-order delta-method approximation.

### 5.5 Propositions S and S′ — sketch computability and exceedance counts

> **Proposition S.** If `#{i : s_i > c} ≤ K`, then the truncated
> sample mean `X̄` and truncated empirical variance `V̂` at threshold `c` are
> exactly recoverable from the sketch `𝒮_n = (n, Σs_i, Σs_i², s_(1..K))`.

**Argument.** Write `cnt := #{j ≤ K : s_(j) > c}`. When `#{i : s_i > c} ≤ K`,
every summand exceeding `c` is among the top `cnt` order statistics stored in
`top_32`. Hence `Σ_i (s_i ∧ c) = (Σ_i s_i − Σ_{j ≤ cnt} s_(j)) + cnt · c` and
`Σ_i (s_i ∧ c)² = (Σ_i s_i² − Σ_{j ≤ cnt} s_(j)²) + cnt · c²`
(`bound._truncated_moments`), from which
`V̂ = (n Σ(s_i ∧ c)² − (Σ(s_i ∧ c))²) / (n(n−1))`. ∎

> **Proposition S′ (expected exceedance count).** Under `P2*(M)`, at
> any truncation scale `γ > 0`, Markov's inequality on `s²` gives
> `Pr_F(s > γθ) ≤ M/γ²`. At the reference scale `γ* = √(3M(n−1)/(7L))` with
> `L = ln(4/α)`, this probability is `7L/(3(n−1))`, so
>
> $$\mathbb{E}_F[\#\text{exceedances}] \;\le\; \frac{7Ln}{3(n-1)} \;\approx\; \tfrac{7}{3}\ln(4/\alpha).$$
>
> At the calibrated scale `γ = 0.35 · γ*` used by Algorithm 1, the bound is
> `1/0.35² ≈ 8.16` times larger:
> `E_F[#exceedances] ≤ 7Ln/(0.35² · 3(n−1)) ≈ 19.05 · ln(4/α)`.

**Argument.** For each item `i`, `Pr_F(s_i > γθ) ≤ E_F[s_i²]/(γθ)² ≤ M/γ²` (§2,
Derived quantities); summing over `i = 1, …, n` multiplies by `n`. Substituting
`γ = γ*` and `γ = 0.35 · γ*` gives the two expressions. ∎

The table below evaluates both upper bounds at standard values of `α` (for
`n/(n−1) ≈ 1`):

| `α` | `L = ln(4/α)` | at `γ*`: `E[#exceed] ≤ 7L/3` | ratio to `K=32` | at `0.35γ*`: `E[#exceed] ≤ 7L/(3·0.35²)` | ratio to `K=32` |
|---|---|---|---|---|---|
| 0.05 | 4.38 | **10.2** | 3.1× | **83.5** | 0.38× |
| 0.01 | 5.99 | **14.0** | 2.3× | **114.1** | 0.28× |
| 0.001 | 8.29 | **19.4** | 1.65× | **158.0** | 0.20× |

**Why the upper endpoint uses Lemma E and the sketch-mode lower sweep uses the
piecewise relaxation.** At `γ = 0.35 · γ*`, the worst-case expected exceedance
count under `P2*(M)` exceeds `K = 32` at every level in the table. Lemma E
avoids exceedance counts on the upper side because it depends only on
`(n, Σs_i)`. On the lower side, whenever `t < s_(K)/γ` in sketch mode
($K < n$), `bound._solve_lower_piecewise` switches to the piecewise quadratic
relaxation $\widetilde H_{\mathrm{lo}}(t) \ge H_{\mathrm{lo}}(t)$ of §5.2,
which requires only $(n, \Sigma s_i, \Sigma s_i^2, s_{(1..K)})$.

### 5.6 Proposition F — the refutation check

For `n ≥ 2` and `Σs_i > 0`, define the empirical relative second moment
`M̂_n := n · Σs_i² / (Σs_i)²`, computable from the sketch, and the status rule

$$S_n := \begin{cases}\texttt{TAIL\_UNRESOLVED} & \hat M_n > M,\\ \texttt{UNREFUTED} & \text{otherwise.}\end{cases}$$

(When `Σs_i = 0`, `bound.bound` returns `UNREFUTED` at step 9 and
`interval.confidence_interval` reports `m_observed = 1.0`.)

> **Proposition F (asymptotic consistency of $\hat M_n$).** Under
> `P1` with `0 < θ < ∞` and `E_F[s²] < ∞`, `M̂_n → M(F)` almost surely as
> `n → ∞`. Hence if `M(F) > M` then `Pr_F(S_n = TAIL_UNRESOLVED) → 1`, and if
> `M(F) < M` then `Pr_F(S_n = UNREFUTED) → 1`.

**Argument.** By Kolmogorov's strong law of large numbers (SLLN),
`(1/n)Σs_i² → E_F[s²]` and `(1/n)Σs_i → θ > 0` almost surely; the continuous
mapping property of `(u, v) ↦ u / v²` on `v > 0` gives
`M̂_n → E_F[s²]/θ² = M(F)` almost surely. ∎

> **Warning.** **The diagnostic check $\hat M_n \le M$ is asymptotic and has no
> finite-sample power bound.** A distribution-free finite-sample test of `M` is
> impossible by the Bahadur–Savage construction (§0): when a rare heavy-tailed
> spike of probability `p ≪ 1/n` is absent from the sample, `M̂_n` is small
> while `M(F)` is large. Because Claim 1 does not condition on `S_n`, this never
> affects the validity of `[L_n, U_n]` when `P2*(M)` holds. The design rule that
> preserves coverage is:
>
> **R1 — the refutation check may only annotate output (`TAIL_UNRESOLVED`). It
> never narrows an interval.**

### 5.7 Lemma E — one-sided Bernstein upper endpoint

> **Lemma E.** Under `P1 ∧ P2*(M)`, let `L := ln(1/α')` and let
> `c_E` be the unique positive root of `n c² − (2/3) L c − 2(M−1) L = 0`
> (`bound.c_bernstein`), namely
>
> $$c_E \;=\; \frac{\tfrac23 L + \sqrt{\bigl(\tfrac23 L\bigr)^2 + 8n(M-1)L}}{2n}.$$
>
> Then `Pr_F(s̄ ≤ θ(1−c_E)) ≤ α'`, and hence `Pr_F(θ ≥ s̄/(1−c_E)) ≤ α'` provided
> `c_E < 1`.

**Argument.** Put `X_i := −(s_i − θ)`. Under `P1`, the variables `X_i` are
independent with `E_F[X_i] = 0`, `X_i ≤ θ` **because `s_i ≥ 0`** (which
supplies the one-sided upper bound that Bernstein's inequality requires), and
`Var_F(X_i) = Var_F(s_i) ≤ (M−1)θ²` by `P2*(M)`. Bernstein's inequality for
independent zero-mean variables bounded above by `b` gives, for every `ε > 0`,

$$\Pr_F\Bigl(\textstyle\sum_{i=1}^n X_i \ge n\varepsilon\Bigr) \;\le\; \exp\!\left(\frac{-n\varepsilon^2}{2\mathrm{Var}_F(X) + \tfrac23 b\varepsilon}\right).$$

Taking `b = θ` and `ε = cθ` yields

$$\Pr_F\bigl(\bar s \le \theta(1-c)\bigr) \;\le\; \exp\!\left(\frac{-nc^2\theta^2}{2(M-1)\theta^2 + \tfrac23\theta\cdot c\theta}\right) \;=\; \exp\!\left(\frac{-nc^2}{2(M-1) + \tfrac23 c}\right).$$

**The unknown mean `θ` cancels from the exponent.** Setting the right-hand side
equal to `α' = e^{−L}` gives the quadratic equation
`n c² − (2/3) L c − 2(M−1) L = 0`, whose unique non-negative root is `c_E`.
Because `c_E` is a **deterministic function of `(n, M, α')` alone**, Lemma C
applies. ∎

**Comparison with the sub-Gaussian proxy.** A naive one-sided bound with
variance proxy `2M` gives `c_subG := √(2ML/n)`. Writing
`q(c) := n c² − (2/3) L c − 2(M−1) L`, we have `q(c) ≥ 0 ⟺ c ≥ c_E` for
`c > 0`, and

$$q(c_{\mathrm{subG}}) \;=\; 2ML - \tfrac23 L\,c_{\mathrm{subG}} - 2(M-1)L \;=\; L\bigl(2 - \tfrac23 c_{\mathrm{subG}}\bigr),$$

which is non-negative whenever `c_subG ≤ 3`. Since every finite upper endpoint
requires `c < 1`, `c_E ≤ c_subG` holds throughout the entire usable regime.

#### 5.7.1 Why the upper endpoint takes `min(c_A, c_E)` — the `M → 1` regime

Lemma E does **not** dominate Lemma A near `M = 1`:

*   `c_A = √((M−1)(1−α')/(nα'))` uses only the centred variance bound
    `Var(s) ≤ (M−1)θ²` and vanishes as `M → 1⁺` (where `Var(s) = 0` and
    `s̄ = θ` almost surely).
*   `c_E` retains the boundedness term `(2/3)L/n > 0` at `M = 1`.

Because both `c_A` and `c_E` depend only on `(n, M, α')`, setting

$$c_{\mathrm{up}} \;:=\; \min(c_A,\, c_E)$$

(`bound.bound`) is a deterministic selection and preserves the per-side
budget `α'` by Lemma C (§5.3).

### 5.8 Lemma A′ — Cantelli bound under dependence

> **Lemma A′.** Let `s_1, …, s_n ≥ 0` be jointly distributed,
> **neither independent nor identically distributed**, satisfying
> **P2\*_pool(M)** with pooled mean `θ = (1/n)Σ_i E[s_i]`. Suppose a declared
> constant `n_eff ≥ 1` satisfies
>
> $$\mathrm{Var}(\bar s) \;\le\; \frac{(M-1)\theta^2}{n_{\text{eff}}}. \tag{P1′}$$
>
> Then with `c := √((M−1)(1−α')/(n_eff · α'))` (`bound.c_cantelli(effective_n, M, alpha_p)`),
> `Pr(θ > s̄/(1−c)) ≤ α'` provided `c < 1`, and `Pr(θ < s̄/(1+c)) ≤ α'` for every
> `c ≥ 0`; for `c > 0` also with `≥` and `≤`.

**Argument.** Identical to the argument for Lemma A (§5.1) with `n` replaced by
`n_eff`. Cantelli's inequality depends only on the mean and variance of the
scalar random variable `s̄`: `E[s̄] = θ` holds by definition of the pooled mean,
and `P1′(n_eff)` directly supplies `Var(s̄) ≤ (M−1)θ²/n_eff` in place of
`σ²/n`. ∎

Under `P1 ∧ P2*(M)`, `P1′(n)` holds with `n_eff = n`, and Lemma A′ reduces to
Lemma A. The implementation therefore uses `s̄/(1 + c_cantelli(ν, M, α'))` for
the Cantelli lower endpoint on both paths, with `ν = n` or `ν = effective_n`.

When all items share a common variance `Var(s)`, `n_eff := Var(s)/Var(s̄)` is
the classical effective sample size. When item means and variances are
heterogeneous across the sample, Lemma H below provides a valid `n_eff`. Any
lower bound on `n_eff` preserves `P1′(n_eff)`.

#### Lemma H — an `n_eff` that survives marginal heterogeneity

> **Lemma H.** Let `s_1, …, s_n ≥ 0` be square-integrable and
> satisfy **P2\*_pool(M)**. Let `R = (ρ_ij)` be their correlation matrix (with
> `ρ_ij := 0` whenever `Var(s_i) = 0` or `Var(s_j) = 0`, and `ρ_ii := 1`) and
> let
>
> $$\kappa \;:=\; \max_{1 \le i \le n} \sum_{j=1}^n |\rho_{ij}| \;\ge\; 1$$
>
> be its maximum absolute row sum. Then **P1′(n_eff)** holds with
> `n_eff = n/κ`.

**Argument.** Write `σ_i = √Var(s_i)` for the standard deviation of `s_i` and
`σ = (σ_1, …, σ_n)ᵀ`, so that `n² Var(s̄) = σᵀ R σ`. Because `R` is real
symmetric, the Rayleigh quotient satisfies `σᵀ R σ ≤ λ_max(R) ‖σ‖₂²`, and by
Gershgorin's circle bound on the eigenvalues of `R`,
`λ_max(R) ≤ ‖R‖_∞ = max_i Σ_j |ρ_ij| = κ`. Therefore
`Var(s̄) ≤ (κ/n) · σ̄²` with `σ̄² := (1/n)Σ_i σ_i²`. Finally,

$$\bar{\sigma^2} \;=\; \tfrac1n\sum_{i=1}^n \mathbb{E}[s_i^2] \;-\; \tfrac1n\sum_{i=1}^n (\mathbb{E}[s_i])^2 \;\le\; M\theta^2 - \theta^2,$$

where the first term is bounded by `P2*_pool(M)` and the second by the
Cauchy–Schwarz inequality `(1/n)Σ_i (E[s_i])² ≥ ((1/n)Σ_i E[s_i])² = θ²`.
Substituting gives `Var(s̄) ≤ κ(M−1)θ²/n`, which is `P1′(n_eff)` at
`n_eff = n/κ`. ∎

Because `ρ_ii = 1`, `κ ≥ 1` and `n_eff = n/κ ≤ n`, matching the validation check
in `sketch.IndependentSketch.__post_init__` and `interval.confidence_interval`
that raises `ValueError` when `effective_n > n` or `effective_n < 1`.

> **Note (`Lemma H` versus the total-correlation `W` route).** When all items
> share a common variance `Var(s)` and a common mean `θ`,
> `n² Var(s̄) = Var(s) · W` with `W := Σ_{i,j} ρ_ij`, giving
> `n_eff = n²/W ≥ n/κ`. However, the `W` route requires equal item variances
> and equal item means; under heterogeneity a small group of high-variance,
> strongly correlated items can dominate `Var(s̄)` while contributing little to
> `W`. Lemma H (`n_eff = n/κ`) requires neither equal variances nor equal
> means, and its intermediate inequality `n Var(s̄) ≤ κ σ̄²` is the form `(V_k)`
> used by `sketch.merge` in Proposition G (§5.10).

> **Warning (why `effective_n` routes exclusively to Lemma A′).**
>
> *   **Lemma E does not hold under `P1′(n_eff)`.** Its argument factorises the
>     MGF `E[exp(λ Σ X_i)] = ∏_i E[exp(λ X_i)]` using independence. A variance
>     bound on `s̄` does not bound higher moments or the MGF of `s̄`.
> *   **Lemma B does not hold under `P1′(n_eff)`.** Maurer and Pontil's (2009)
>     inequality relies on the concentration of the empirical variance `V̂`
>     around the true variance under independence.
> *   **Lemma A′ holds** because Cantelli's inequality depends only on `E[s̄]`
>     and `Var(s̄)`.
>
> Consequently, whenever `effective_n is not None`, Algorithm 1 uses **Lemma A′
> on both endpoints**.

### 5.9 The sample-size thresholds

Three sample-size thresholds (`bound.n_A`, `bound.n_U`, `bound.n_B`) govern
refusal and lower-sweep routing:

*   **`n_A(M, α) = (M−1)(1−α')/α'`** (`bound.n_A`): the threshold at which
    `c_cantelli = 1`. On the correlated path (`effective_n` set), an interval
    is issued if and only if `effective_n > n_A(M, α)`.
*   **`n_U(M, α) = min(n_A(M, α), (2(M−1) + 2/3)·ln(1/α'))`** (`bound.n_U`):
    **the independent-path refusal threshold:** `c_up < 1 ⟺ n > n_U(M, α)`.
*   **`n_B(M, α) = 1 + ((0.35 + 1/0.35)/0.90)²·(7/3)·M·ln(4/α)`** (`bound.n_B`):
    a **width routing threshold**, not a refusal threshold. On the independent
    path with `K ≥ 1`, `slope_∞ > 0.10 ⟺ n > n_B(M, α)`; above `n_B` the
    Lemma B lower sweep is used, and at or below `n_B` the Cantelli lower bound
    `s̄/(1 + c_cantelli)` is used.

The table below gives, for standard values of `α` and `M`, the smallest
integer `n > n_U` on the independent path and the threshold `n_A` itself on
the correlated path (`n_eff` must exceed it strictly):

| `α` | `M` | Independent path: smallest `n > n_U` | Correlated path: `n_A` |
|---:|---:|---:|---:|
| 0.10 | 4 | 20 | 57 |
| 0.10 | 16 | 92 | 285 |
| 0.05 | 4 | 25 | 117 |
| 0.05 | 16 | 114 | 585 |
| 0.01 | 4 | 36 | 597 |
| 0.01 | 16 | 163 | 2985 |
| 0.001 | 4 | 51 | 5997 |
| 0.001 | 16 | 234 | 29985 |

### 5.10 Proposition G — merging correlated sketches

A scalar `effective_n` cannot be summed across shards when shards have unequal
means, variances, or correlation structures. Instead, `sketch.merge` combines
each shard's **variance inflation factor (VIF)**

$$\kappa_k \;:=\; \frac{n_k}{n_{\text{eff},k}} \;\ge\; 1 \qquad (1 \le n_{\text{eff},k} \le n_k).$$

**What each shard's `effective_n` must satisfy.** Write
`σ̄_k² := (1/n_k) Σ_{i∈k} Var(s_i)` for the average item variance in shard `k`.
The merge rule requires each shard's declaration to satisfy the
**variance-inflation bound**

$$n_k\,\mathrm{Var}(\bar s_k) \;\le\; \kappa_k\,\bar\sigma_k^2 . \qquad (\mathrm{V}_k)$$

Inequality `(V_k)` involves neither `M` nor the shard mean `θ_k`. Both standard
ways of obtaining `n_eff,k` satisfy `(V_k)`:

*   **Lemma H (§5.8):** its argument shows `n_k² Var(s̄_k) ≤ κ_k Σ_{i∈k} Var(s_i)`
    before applying `P2*_pool`, which is `(V_k)` with `κ_k` the maximum
    absolute row sum of shard `k`'s correlation matrix.
*   **Common variance within shard `k`:** `n_eff,k = Var(s)/Var(s̄_k)` gives
    `(V_k)` with equality.

> **Proposition G.** Let shards `k = 1, …, m` of sizes `n_k ≥ 1`
> (`n = Σ_k n_k`) be **mutually uncorrelated** — `Cov(s_i, s_j) = 0` whenever
> `i` and `j` lie in different shards — and let each shard satisfy `(V_k)` with
> `κ_k = n_k / n_eff,k`. Let the pooled sample satisfy **P2\*_pool(M)** with
> pooled mean `θ = (1/n)Σ_i E[s_i]`. The shards may have arbitrary, unequal
> means and variances. Then the pooled sample satisfies **P1′(n_eff)** with
>
> $$n_{\text{eff}} \;=\; \frac{n}{\kappa_{\max}}, \qquad \kappa_{\max} \;:=\; \max_{1 \le k \le m} \kappa_k ,$$
>
> and it satisfies `(V)` (`n Var(s̄) ≤ κ_max σ̄²`), so the merged sketch can be
> merged again by the same rule.

**Argument.**

1.  Write `S_k := Σ_{i∈k} s_i = n_k s̄_k`. Because covariances across shards
    vanish, `n² Var(s̄) = Var(Σ_k S_k) = Σ_k Var(S_k) = Σ_k n_k² Var(s̄_k)`.
2.  By `(V_k)`,
    `n_k² Var(s̄_k) ≤ κ_k n_k σ̄_k² = κ_k Σ_{i∈k} Var(s_i) ≤ κ_max Σ_{i∈k} Var(s_i)`.
3.  Summing over `k = 1, …, m` yields
    `n² Var(s̄) ≤ κ_max Σ_{i=1}^n Var(s_i)`, which is `(V)` for the pooled
    sample at `κ = κ_max`.
4.  By `P2*_pool(M)` and Cauchy–Schwarz,
    `Σ_{i=1}^n Var(s_i) = Σ_{i=1}^n E[s_i²] − Σ_{i=1}^n (E[s_i])² ≤ n M θ² − n θ² = n(M−1)θ²`.
5.  Combining steps 3 and 4 gives
    `Var(s̄) ≤ κ_max (M−1)θ² / n = (M−1)θ² / (n / κ_max)`, which is
    `P1′(n_eff)` at `n_eff = n / κ_max`. ∎

> **Caution (why W-space addition `n² / Σ_k n_k κ_k` is unsafe under
> heterogeneity, and why `(V_k)` is required).**
> 1. Adding `W_k := n_k² / n_eff,k = n_k κ_k` across shards would set
>    `n_eff = n² / Σ_k n_k κ_k`, which fails when a small shard has both much
>    larger item variance `σ̄_k²` and much stronger internal correlation `κ_k`
>    than the remaining shards: step 2 cannot weight `Σ_{i∈k} Var(s_i)` by
>    `n_k / n` when `σ̄_k²` varies across shards. Taking `κ_max = max_k κ_k`
>    bounds `Σ_k κ_k Σ_{i∈k} Var(s_i) ≤ κ_max Σ_i Var(s_i)` without assuming
>    equal shard variances or equal shard means.
> 2. Premise `P1′(n_eff,k)` alone (`Var(s̄_k) ≤ (M−1)θ_k² / n_eff,k`) is weaker
>    than `(V_k)` when `M` is much larger than shard `k`'s relative second
>    moment, so each shard's `effective_n` must be derived from Lemma H or a
>    variance ratio satisfying `(V_k)`.
> 3. **Pooled `M` under unequal shard means:** If shard `k` has mean `θ_k` and
>    relative second moment `M_k ≤ M`, then
>    `M_pool = (Σ_k n_k M_k θ_k²) / (n θ²) ≤ M (1 + CV_θ²)`, where `CV_θ²` is
>    the `n_k`-weighted squared coefficient of variation of `{θ_k}`. The
>    declared `M` must bound `M_pool`.
> 4. **Uncorrelatedness across shards is a caller contract (Limitation L13).**
>    If a correlated group is split across two shards with positive
>    cross-covariance `Cov(S_a, S_b) > 0`, merging understates the variance.
>    Furthermore, `sketch.merge` raises `ValueError` if one sketch has
>    `effective_n=None` and the other has `effective_n` set, preventing an
>    unmarked correlated shard from being treated as independent.

---

## 6. Coefficient of determination (`r2`)

For `metric="r2"`, the population target under `P1` is
`R² = 1 − θ_A / θ_B`, where `θ_A := E_F[e²]` and `θ_B := Var_F(y)`.

1. **Numerator sequence (`s_a`):** `s_{a,i} := e_i² = (ŷ_i − y_i)²` ($i = 1, \dots, n$)
   are i.i.d. non-negative summands with mean `θ_A`.
2. **Denominator sequence (`s_b` via disjoint pairing):** Because `Var_F(y) = ½ E_F[(y − y')²]`
   for independent copies `y, y' ~ F_y`, `metrics.extract_r2_summands` (and
   `jax_accumulator.update_r2`) forms `n_b := ⌊n/2⌋` **disjoint** index-adjacent
   pairs and sets
   $$b_i \;:=\; \tfrac12\bigl(y_{2i} - y_{2i-1}\bigr)^2, \qquad i = 1, \dots, \lfloor n/2 \rfloor.$$
   Under `P1`, `b_1, …, b_{⌊n/2⌋}` are i.i.d. non-negative summands with mean
   `θ_B = Var_F(y)`.

**Closed form of `M_B`.** Write `d := y_2 − y_1`, `σ_y² := Var_F(y)`, and
`Kurt(y) := E_F[(y − E_F[y])⁴] / σ_y⁴`. Since `y_1` and `y_2` are independent
and identically distributed, `E_F[d] = 0`, `E_F[d²] = 2σ_y²`, and
`E_F[d⁴] = 2 Kurt(y) σ_y⁴ + 6 σ_y⁴`. Hence `E_F[b] = σ_y² = θ_B`,
`E_F[b²] = ¼ E_F[d⁴] = ½(Kurt(y) + 3)σ_y⁴`, and

$$M_B \;:=\; \frac{\mathbb{E}_F[b^2]}{(\mathbb{E}_F[b])^2} \;=\; \frac{\mathrm{Kurt}(y) + 3}{2}.$$

Gaussian labels (`Kurt(y) = 3`) have `M_B = 3`; symmetric binary labels
(`Kurt(y) = 1`) have `M_B = 2`; and `M_B ≥ 2` always since `Kurt(y) ≥ 1`. The
implementation (`interval._confidence_interval_r2`) applies the caller's single
declared `M` (default `16.0`) to both sequences, which satisfies the moment
premise for `b` whenever `Kurt(y) ≤ 2M − 3` (for default `M = 16`, whenever
`Kurt(y) ≤ 29`; Limitation L3).

**Combination (`interval._confidence_interval_r2`).** Algorithm 1 (`bound.bound`)
is called twice with `metric="mse"` and failure probability `α_call = α/2` (so
each of the four one-sided endpoints carries failure budget `α_call / 2 = α/4`):
once on `(n_a, sum_a, sum2_a, top_a)` yielding `[L_A, U_A]`, and once on
`(n_b, sum_b, sum2_b, top_b)` yielding `[L_B, U_B]`. By the four-way union
bound, with probability at least `1 − α`, `θ_A ∈ [L_A, U_A]` and
`θ_B ∈ [L_B, U_B]`, so

$$R^2 \;\in\; \Bigl[\,1 - \frac{U_A}{L_B},\; 1 - \frac{L_A}{U_B}\,\Bigr] \quad \text{when } L_B > 0 \text{ and } U_A, U_B < \infty,$$

and `[-∞, 1.0]` when `L_B = 0`. The reported `Result.m_observed` is
`max(m_obs_a, m_obs_b)`, and `Result.status` is `TAIL_UNRESOLVED` if either call
returns `TAIL_UNRESOLVED` (and `ASSUMPTION_REQUIRED` if either call returns
`ASSUMPTION_REQUIRED`).

> **Important (`r2` refuses under correlation).** If `effective_n` is set — on
> either component sketch of an `R2Sketch` or `(sk_a, sk_b)` tuple, or as a call
> argument to `confidence_interval` — `confidence_interval` returns
> `Status.ASSUMPTION_REQUIRED` (and `sketch.from_data(..., metric="r2", effective_n=...)`
> raises `ValueError`). In `stats.independent`, pairing index-adjacent items under
> correlation biases `E[b_i] = (1 − ρ_{2i-1,2i}) Var(y)` downward whenever
> adjacent labels are positively correlated, and a single scalar `effective_n`
> cannot repair that bias. (For correlated `R²` with cluster structure, use the
> cluster-centred `R2ClusterShard` / `compute_r2_interval` path documented in
> [THEORY.md](THEORY.md) §8.)

---

## 7. Algorithm 1 and implementation correspondence

### 7.1 Input preparation and poisoning rules

Before Algorithm 1 (`bound.bound`) is called, raw data or sketches are
validated and summarized as follows:

1. **Parameter validation (`interval.confidence_interval`, `metrics.normalize_metric`):**
   - `level` outside `(0, 1)`, `m` non-finite or `< 1.0`, unknown metric names,
     or `metric in ("average_precision", "averageprecision", "ap")` raise
     `ValueError`.
   - Passing `effective_n` when `data_or_sketch` is already an `IndependentSketch` (whether
     `IndependentSketch.effective_n` is `None` or set) raises `ValueError`. On raw data,
     `effective_n` outside `[1.0, n]` or non-finite raises `ValueError`.
2. **Summand extraction (`metrics.extract_summands`, `metrics.extract_r2_summands`):**
   - For `mae`, `mse`, `rmse`, and `r2`, non-finite labels or predictions
     produce `NaN` summands (poisoning `sum_s` and `sum_s2` so step 1 of
     Algorithm 1 returns `ASSUMPTION_REQUIRED`).
   - For `accuracy`:
     - If `data` is a 2-tuple `(labels, predictions)`, `extract_summands`
       computes `(labels == predictions).astype(float64)` and sets entries
       where a floating-point label or prediction is non-finite to `NaN`.
     - If `data` is a 1D sequence of indicator values, `extract_summands`
       checks all finite entries and raises `ValueError` if any finite value is
       not `0.0` or `1.0`; non-finite entries (`NaN`, `inf`) remain non-finite
       and trigger `ASSUMPTION_REQUIRED`.
     - If `data_or_sketch` is an `IndependentSketch` passed with `metric="accuracy"`,
       `confidence_interval` and `_accuracy_exact` raise `ValueError` if
       `top_32` has `top_32[0] > 1.0` or if
       finite `sum_s`, `sum_s2` fail `float(round(sum_s)) == sum_s`,
       `sum_s2 == sum_s`, or `0 ≤ sum_s ≤ n`.
3. **Sketch construction (`sketch.from_array`) and JAX accumulation (`jax_accumulator`):**
   - `sketch.from_array(arr)` sets `sum_s = sum_s2 = nan` whenever any element
     of `arr` is non-finite or `< 0.0`.
   - `jax_accumulator.summands(metric, labels, predictions)` computes
     `(labels == predictions).astype(float32)` for `accuracy` (mapping
     non-finite float `labels` or `predictions` to `NaN`) and `|e|` or `e²` for
     `mae`, `mse`, `rmse` (mapping non-finite residuals to `NaN`).
   - `jax_accumulator.update` and `jax_accumulator.update_r2` accumulate
     running sums in `dtype` (default `jnp.float32`) using Neumaier compensated
     summation (`_neumaier_add`), ignore entries where `mask` is `False`, and
     count masked-in entries that are non-finite or `< 0.0` in `n_invalid`. On
     the host, `jax_accumulator.to_sketch` converts the compensated sums to
     `float64` and poisons the resulting `IndependentSketch` (`sum_s = sum_s2 = nan`) if
     `n_invalid > 0`, if either sum is non-finite, or if `n < 0` (`int32`
     overflow past $2^{31} - 1$). Note that `float32` rounding of individual
     summands prior to accumulation is a numerical approximation outside
     Claim 1 (`jax_accumulator`).

### 7.2 Algorithm 1 (`bound.bound`)

```
input: n, Σs, Σs², top_k = s_(1..K), α, M, metric, effective_n   # data mode is the case K = n
 1  if n < 2, or Σs / Σs² / any s_(j) is NaN or inf, or M < 1, or K > n → return (nan, nan), ASSUMPTION_REQUIRED
 2  α' ← α/2 ;  s̄ ← Σs/n
 3  if effective_n is None:                                      # independent path
 4      ν ← n ;  c_up ← min(c_cantelli(n, M, α'), c_bernstein(n, M, α'))   # Lemmas A and E, selected by Lemma C
 5  else:                                                        # correlated path
 6      ν ← effective_n ;  c_up ← c_cantelli(effective_n, M, α') # Lemma A′ alone; see §5.8
 7  if not (c_up < 1) → return (nan, nan), ASSUMPTION_REQUIRED   # the only distributional refusal
 8  U ← s̄ / (1 − c_up)
 9  if Σs == 0 → return [0, 0], UNREFUTED                        # see note (a)
10  M̂ ← n · Σs² / (Σs)²                                          # raw n, never ν; see note (e)
11  S ← TAIL_UNRESOLVED if M̂ > M else UNREFUTED
12  if _use_lower_sweep(n, α, M, K, effective_n):                # see note (c)
13      L ← max(0, _solve_lower_piecewise(n, sort_desc(top_k), γ, ln(2/α'), Σs, Σs²))   # see note (d)
14  else:
15      L ← s̄ / (1 + c_cantelli(ν, M, α'))
16  if metric == "rmse": return [√L, √U], S                      # Lemma D; see note (f)
17  if θ_max(metric) ≠ None: U ← min(U, θ_max(metric))           # Corollary 2; see note (f)
18  return [L, U], S

_use_lower_sweep(n, α, M, K, eff) :=  (eff is None)  ∧  (K ≥ 1)  ∧  (slope_∞(n, α, M) > 0.10)
```

The following notes justify individual steps:

**(a) The all-zero sample (step 9) is answered only after step 7 has passed.**
When `Σs = 0`, `M̂` is `0/0` (`interval.confidence_interval` reports
`m_observed = 1.0`). Before step 7, returning `[0, 0]` would under-cover: the
two-point law `s ∈ {0, h}` with `Pr(s = h) = 1/M` satisfies **P2\*(M)** with
equality and has `θ = h/M > 0`, yet the entire sample is zero with probability
`(1 − 1/M)^n`. Once step 7 (`c_up < 1`) passes, for any `θ > 0` the event
`{s̄ = 0}` is contained in `{s̄ ≤ θ(1 − c_up)}`, whose probability is already
bounded by `α'` on the upper side; if `θ = 0`, `[0, 0]` covers `θ` directly.
Returning `[0, 0]` at step 9 therefore adds no miss probability on either path.

**(b) The `min` in step 4 is over deterministic constants, not data-dependent
endpoints.** Both `c_cantelli(n, M, α')` and `c_bernstein(n, M, α')` depend only
on `(n, M, α')`, so taking their minimum costs no union-bound penalty (Lemma C,
§5.7.1). Step 6 has no `min` because Lemma E does not hold under dependence
(§5.8).

**(c) The lower-endpoint branch condition `_use_lower_sweep` is deterministic.**
It reads `(n, α, M, K, effective_n)` and never inspects a sample value, so
Lemma C combines the upper bound and the selected lower bound at total failure
probability `α' + α' = α`.

**(d) Step 13 tests each piece's left endpoint before solving for roots.** As
explained in §5.2, a solver that only searched for interior roots would skip a
piece on which `H_lo ≥ 0` throughout and return a root from a higher piece when
exact zeros are present.

**(e) `M̂` uses the raw sample size `n`, never `ν = effective_n`.**
`M̂ = n · Σs² / (Σs)² = ((1/n)Σs_i²) / ((1/n)Σs_i)²` is a ratio of two sample
means, so its numerator and denominator are unbiased for the pooled second
moment and the square of the pooled mean (up to `Var(s̄)`) regardless of
dependence.

**(f) The `θ ≤ 1` clamp (step 17) preserves coverage, and `rmse` returns at
step 16.** On the correlated path, `accuracy` reaches Algorithm 1 and has its
upper endpoint clamped to `metrics.get_theta_max("accuracy") = 1.0`
(Corollary 2).

---

## 8. Scope and known limitations

Each item below carries its stable identifier (`L2`–`L15`) referenced across the
codebase and companion documents (`L1` is in References; `L16` and `L17` are
scope notes in §4.1.1 and §4.2):

*   **L2** (`R2Sketch` shard-boundary pairing). `R2Sketch.merge` is associative
    and commutative, but **not shard-invariant**: because `b_i` is formed by
    pairing consecutive items locally within each shard (or batch in
    `jax_accumulator.update_r2`), each odd-sized shard drops its final unpaired
    label, yielding `Σ_k ⌊n_k/2⌋ ≤ ⌊(Σ_k n_k)/2⌋` pairs. Because dropping the
    last position of an odd shard depends only on shard sizes and not on label
    values, the remaining pairs are still i.i.d. with mean `Var_F(y)` and
    coverage is preserved, at the cost of a slightly smaller `n_b` (and refusal
    if all shards have size `1` so `n_b = 0`).
*   **L3** (`r2` conservatism on the independent path). The independent-path
    `r2` estimator halves the sample size for `θ_B` (`⌊n/2⌋` disjoint pairs),
    splits `α` across four one-sided bounds (`α/4` each), and reuses the single
    declared `M` for both `s_a = e²` and `s_b = b` (§6).
*   **L4** (`accuracy` routing). On the independent path (`effective_n is None`),
    `accuracy` uses the exact Clopper–Pearson binomial interval at level `α̃`
    (§4.2, Claims 3 and 3′). When `effective_n` is set, `accuracy` uses
    Algorithm 1 with Claim 1′.
*   **L5** (`average_precision` is unsupported). Average precision is a
    rank-dependent functional of the entire sample rather than a mean of
    per-item summands. Applying McDiarmid's bounded-differences inequality to
    empirical average precision gives a half-width of `0.59` at `n = 1000` on a
    `[0, 1]` metric (vacuous below `n ≈ 10⁵`), plus a finite-sample bias
    `E[AP_n] − AP_∞`. `metrics.normalize_metric` therefore rejects
    `average_precision` / `ap` with `ValueError`.
*   **L6** (per-metric coverage). Computing intervals for multiple metrics on
    the same evaluation sample provides `1 − α` coverage **marginally per
    metric**; simultaneous coverage across `m` metrics requires passing
    `level = 1 − α/m` (Bonferroni).
*   **L7** (asymptotic refutation check). The diagnostic check `M̂_n > M`
    (Proposition F, §5.6) has no finite-sample power bound against rare unseen
    spikes (§0).
*   **L8** (default `M` values). The default declarations (`4.0` for `mae` and
    `accuracy`; `16.0` for `mse`, `rmse`, and `r2`) are chosen as reference
    constants (covering error kurtosis up to `14` for `mse`/`rmse` by §2
    Remark 4); heavier-tailed domains require a domain-calibrated `M` (see
    `stats.declare` in [README.md](README.md)).
*   **L9** (interval width conservatism). Finite-sample bounds over the full
    moment class `𝒫(M)` are wider than asymptotic normal intervals. Above `n_B`,
    both data mode and sketch mode use the Lemma B lower sweep; at or below
    `n_B`, the lower endpoint falls back to Cantelli.
*   **L10** (extremal two-point law). The two-point law `s ∈ {0, h}` with
    `Pr(s = h) = 1/M` attains `M(F) = M` and makes Cantelli's inequality tight,
    explaining the `1/(1 − c_A)` upper-endpoint divergence as `n → n_A⁺`.
*   **L11** (sketch-mode lower sweep). In sketch mode ($1 \le K < n$),
    `bound._solve_lower_piecewise` evaluates the Lemma B lower sweep whenever
    `slope_∞ > 0.10` ($n > n_B$), using the piecewise quadratic upper bound
    $\widetilde H_{\mathrm{lo}}(t) \ge H_{\mathrm{lo}}(t)$ over
    $t \in [0, s_{(K)}/\gamma]$ (§5.2) with $K = 32$ retained order statistics.
*   **L12** (`effective_n` is an unchecked declaration). In `stats.independent`,
    `effective_n` is supplied by the caller and cannot be tested from a single
    unstructured sample. An overstated `effective_n` causes under-coverage
    without warning. When cluster, temporal, graph, or space-time structure is
    known, prefer the structured estimators ([THEORY.md](THEORY.md)).
*   **L13** (`sketch.merge` requires mutually uncorrelated shards).
    Proposition G (§5.10) requires that items in different shards have zero
    covariance and that each shard's `effective_n` satisfies `(V_k)`. Because an
    `IndependentSketch` stores no item or cluster identifiers, `sketch.merge` cannot detect
    when a correlated group is split across shards.
*   **L14** (correlated-path refusal threshold). When `effective_n` is set,
    Lemma E is unavailable (§5.8), so the refusal threshold is `n_A(M, α)`
    rather than `n_U(M, α)` (§5.9).
*   **L15** (calibrated lower-sweep scale and routing). The Lemma B lower sweep
    uses truncation scale $\gamma = 0.35\,\gamma^*$ with
    $\gamma^* = \sqrt{3M(n-1)/(7\ln(4/\alpha))}$ and routes to the sweep on the
    independent path whenever $\text{slope}_\infty > 0.10$ ($n > n_B(M, \alpha)$,
    §5.2).

---

## 9. Summary of design choices

The implementation of `stats.independent` follows directly from the results above:

1.  **One dimensionless moment declaration `M` per metric** (§0, §2), with
    per-metric defaults `mae: 4.0`, `accuracy: 4.0`, `mse: 16.0`, `rmse: 16.0`,
    and `r2: 16.0` (`metrics.DEFAULT_M`).
2.  **The diagnostic status never narrows an interval** (rule `R1`, §5.6).
    Coverage under the stated premises holds across both `UNREFUTED` and
    `TAIL_UNRESOLVED`.
3.  **Branch selection is data-independent** (Lemma C, §5.3): `c_up` and
    `_use_lower_sweep` depend only on `(n, M, α, K, effective_n)`, and there is
    no data-dependent or width-based refusal (§5.9).
4.  **The upper endpoint uses `(n, Σs_i)` only** (`min(c_A, c_E)` on the
    independent path, `c_A(n_eff)` on the correlated path), so it is `O(1)` and
    identical in data mode and sketch mode (§5.5, §5.7).
5.  **A fixed sketch size `K = 32`** (`sketch.IndependentSketch`), independent of `α`. The
    retained order statistics are read only by the Lemma B lower sweep (§5.2).
6.  **The correlated premise `P1′(n_eff)` and `P2*_pool(M)` are pooled** (§2,
    Lemma H, §5.8), requiring neither identical means nor identical variances
    across items, and routing to Lemma A′ on both endpoints.
7.  **`effective_n = n` is kept distinct from `effective_n = None`**, and
    merging a sketch with `effective_n = None` and a sketch with `effective_n`
    set raises `ValueError`, while two correlated sketches merge via
    `κ_max = max(κ_a, κ_b)` (Proposition G, §5.10).
8.  **`r2` uses disjoint index-local pairs on the independent path and refuses
    with `ASSUMPTION_REQUIRED` under correlation** (§6). `average_precision` is
    rejected with `ValueError` (`L5`, §8).
9.  **`accuracy` uses the exact Clopper–Pearson interval at `α̃ = α − α²/4` for
    `level ≥ √2 − 1` on the independent path** (§4.2, Claims 3 and 3′), and
    clamps the upper endpoint at `1.0` on the correlated path (Corollary 2).

---

## References

*   R. R. Bahadur and L. J. Savage (1956). The nonexistence of certain
    statistical procedures in nonparametric problems. *Annals of Mathematical
    Statistics* 27(4), 1115–1122.
*   C. J. Clopper and E. S. Pearson (1934). The use of confidence or fiducial
    limits illustrated in the case of the binomial. *Biometrika* 26(4),
    404–413.
*   L. Devroye, M. Lerasle, G. Lugosi, and R. I. Oliveira (2016). Sub-Gaussian
    mean estimators. *Annals of Statistics* 44(6), 2695–2725.
*   S. Greenberg and M. Mohri (2014). Tight lower bound on the probability of a
    binomial exceeding its mean. *Statistics & Probability Letters* 86, 91–98.
*   W. Hoeffding (1956). On the distribution of the number of successes in
    independent trials. *Annals of Mathematical Statistics* 27(3), 713–721.
*   A. Maurer and M. Pontil (2009). Empirical Bernstein bounds and
    sample-variance penalization. *Proceedings of the 22nd Annual Conference on
    Learning Theory (COLT 2009)*.
*   V. A. Vatutin and V. G. Mikhailov (1982). Limit theorems for the number of
    empty cells in an equiprobable scheme for group allocation of particles.
    *Theory of Probability and Its Applications* 27(4), 734–743.
*   **(L1)** Cantelli's inequality (one-sided Chebyshev inequality), Bernstein's
    inequality, McDiarmid's bounded-differences inequality, the strong law of
    large numbers, and Gershgorin's circle bound are standard; see any graduate
    probability or matrix-analysis text.

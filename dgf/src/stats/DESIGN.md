# Design rationale for stats: correlated data, cluster totals, and refutation checks

**Scope.** This document explains why the `stats` package is built
the way it is. It covers the modules `independent`, `cluster_bound` and
`cluster_sketch` (cluster totals), `graph`, `temporal`, `refutation`,
`partitioners`, `declare`, `spacetime` and `range_envelope`.

**Audience.** Machine-learning engineers and statisticians who want to know
which alternatives were considered and why they were accepted or rejected. You
should know the independent interval in outline: one declared moment bound `M`, an
interval from Cantelli's inequality, and a status that reports whether the
data contradict `M` ([THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md)).

**Related documents.**

*   [THEORY.md](THEORY.md) states the premises, claims and arguments of the
    cluster interval. Where this document and `THEORY.md` differ, `THEORY.md`
    is normative.
*   [THEORY_SPACETIME.md](THEORY_SPACETIME.md) and
    [THEORY_RANGE_ENVELOPE.md](THEORY_RANGE_ENVELOPE.md) cover the space-time
    and range-envelope modules.
*   [README.md](README.md) is the user guide; [EVIDENCE.md](EVIDENCE.md)
    summarises the simulation evidence.
*   [THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md) holds the independent results this document cites
    (Lemma C, Lemma H, Proposition F, Proposition G, the limitations `L*`).

File paths in code font (for example `benchmark/coverage_dependent.py`) are relative
to the directory of this document.

**Not covered.** The following are outside the current release and are not discussed
here:

*   intervals for ranking metrics (AUC and average precision);
*   rolling-origin evaluation and grouped holdout;
*   a self-normalised replacement for the `κ > 1` graph check.

## Summary of the design

The design rests on six principles. Section 10 describes these design choices.

*   **Pooled premise.** The correlated premise is stated on pooled moments, so
    it does not require identically distributed losses (§1, §2).
*   **One route, D1.** Losses are summed into cluster or window totals, and
    Cantelli's inequality is applied to the totals. Independence-based bounds
    on the totals (route D2) are rejected (§4).
*   **Fixed estimand.** The estimand is the mean loss over all evaluated
    items, never over a topologically selected subset (§5.2).
*   **Declare and refute.** The caller declares the moment bound `M_c` and the
    variance inflation factor `κ`. The interval is certified under these
    declarations, and refutation checks test them on the data.
*   **Measured cluster size.** The best cluster size is measured, not argued
    (§4, §5.4).
*   **Distributable by construction.** Multi-machine execution is a design
    constraint: every estimator works from mergeable per-cluster summaries.

## Notation and terms

The table defines every symbol used below.

| Symbol / term | Definition |
|---|---|
| $n$ | Total number of evaluated items ($n = \sum_{c=1}^G m_c$). |
| $G$ | Number of disjoint clusters or contiguous time windows ($c \in \{1, \dots, G\}$). |
| $m_c$ (or $n_c$), $\bar m$, $m_{\max}$ (or $n_{\max}$) | Item count in cluster $c$, mean cluster size $\bar m := n/G$, and maximum cluster size $m_{\max} := \max_c m_c$. |
| $r,\; \mathrm{CV}_n^2$ | Cluster-size ratio $r := n_{\max}/\bar m$ and squared coefficient of variation of cluster counts $\mathrm{CV}_n^2 := \frac{G\sum_c n_c^2}{n^2} - 1$. |
| $s_i \ge 0,\; S_c,\; \bar S$ | Non-negative per-item loss summand $s_i$ (for example $e_i^2$ for MSE), cluster total $S_c := \sum_{i \in c} s_i$, and sample mean of cluster totals $\bar S := \frac{1}{G}\sum_c S_c$. |
| $\theta,\; \theta_c,\; \bar\theta_c$ | Pooled per-item expected loss $\theta := \frac{1}{n}\sum_i \mathbb{E}[s_i]$, cluster per-item mean $\theta_c := \frac{1}{m_c}\sum_{i \in c}\mathbb{E}[s_i]$, and mean expected cluster total $\bar\theta_c := \bar m \theta$. |
| $\rho_{ij}$, $R$ | Correlation of $s_i$ and $s_j$, and the correlation matrix $R = (\rho_{ij})$. |
| $\alpha,\; \alpha',\; \tilde\alpha$ | Two-sided failure probability $\alpha \in (0, 1)$ (nominal coverage $1 - \alpha$), per-side budget $\alpha' := \alpha/2$, and finite-pool Clopper–Pearson level $\tilde\alpha := \alpha - \alpha^2/4$ (`alpha_tilde` in `independent/interval.py`, Claims 3 and 3′ in [THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md)). |
| $M,\; M_c,\; M_d$ | Item-level relative second moment $M$, cluster-level relative second moment $M_c$, and caller-declared value $M_d$. A relative second moment is $\mathbb{E}[s^2]/(\mathbb{E}s)^2$. |
| $\hat M_n,\; \hat M_G$ | Empirical relative second moments at the item level ($\hat M_n := n\sum s_i^2 / (\sum s_i)^2$) and cluster level ($\hat M_G := G\sum S_c^2 / (\sum S_c)^2$). |
| $n_{\text{eff}}$ (or $\nu$), $W$ | Effective sample size ($1 \le n_{\text{eff}} \le n$, `effective_n` in code) and additive inverse-$n_{\text{eff}}$ correlation mass $W := n^2/n_{\text{eff}}$. |
| $\kappa,\; \kappa_{\text{cluster}},\; \mathrm{VIF}$ | Item-level maximum absolute correlation row sum $\kappa := \max_i \sum_j |\rho_{ij}|$, and cluster-level variance inflation factor $\kappa_{\text{cluster}} = \mathrm{VIF} := \mathrm{Var}(\sum_c S_c) / \sum_c \mathrm{Var}(S_c)$. In this document, "κ" without subscript means $\kappa_{\text{cluster}}$. |
| $\beta_i,\; \beta_{\max}$ | Per-node cross-cluster correlation mass $\beta_i := \sum_{j \notin c(i)} |\rho_{ij}|$ and its worst case $\beta_{\max} := \max_i \beta_i$. |
| $\kappa^*$ | Normal-approximation robustness threshold (§5.4): maximum true $\kappa$ absorbed by Cantelli's tail slack before coverage drops below $1 - \alpha$, when the caller asserts $\kappa = 1$. |
| $n_A(M, \alpha)$ | Cantelli existence threshold $n_A = (M-1)(1-\alpha')/\alpha'$: Cantelli gives a finite interval only when the (effective) sample size exceeds $n_A$. At $\alpha = 0.05$, $n_A = 39(M-1)$. |
| $n_U(M, \alpha)$ | Threshold below which no finite upper endpoint exists: the smaller of $n_A$ and the corresponding Bernstein threshold (`independent.bound.n_U`). |
| $K$ | Number of top order statistics a sketch retains ($K = 32$; as in [THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md)). In §12.2 only, `K` is the model's message-passing hop count (`k_hops`). |
| $P$ | Number of archived models in the archive panel (§6.2). |
| Premise `P1`, Premise `P1′(n_eff)` | Sample premises ([THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md), Premises): premise `P1` (independence) assumes i.i.d. $s_i$; premise `P1′(n_eff)` (variance bound for correlated data) assumes $\mathrm{Var}(\bar s) \le (M-1)\theta^2 / n_{\text{eff}}$. |
| Premise `P2*(M)`, Premise `P2*_pool(M)` | Second-moment premises ([THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md), Premises): premise `P2*(M)` (the declaration on $M$) assumes $\mathbb{E}[s^2] \le M\theta^2$; premise `P2*_pool(M)` (pooled declaration on $M$) assumes $\frac{1}{n}\sum_i \mathbb{E}[s_i^2] \le M\theta^2$. |
| Premise `P1_c(κ)`, Premise `P2*_c(M_c)`, Premise `(P2v)` | `stats` cluster premises ([THEORY.md](THEORY.md), Premises): premise `P1_c(κ)` (between-cluster dependence) assumes $\mathrm{Var}(\sum_c S_c) \le \kappa\sum_c \mathrm{Var}(S_c)$; premise `P2*_c(M_c)` (cluster-level second moment) assumes $\frac{1}{G}\sum_c \mathbb{E}[S_c^2] \le M_c\bar\theta_c^2$; premise `(P2v)` (cluster-level variance bound) assumes $\sum_c \mathrm{Var}(S_c) \le G(M_c - 1)\bar\theta_c^2$. |
| Routes `D0`, `D1`, `D2` | Candidate correlated-inference routes (§4): `D0` is item-level Cantelli with $\nu = n/\kappa$; `D1` is cluster-total Cantelli with $\nu = G/\kappa_{\text{cluster}}$ (**chosen**); `D2` is applying i.i.d. Bernstein/Maurer–Pontil bounds to cluster totals (**rejected**). |
| Estimands `E1`, `E2` | Target estimands: `E1` is the finite pool mean under simple random sampling without replacement; `E2` is the expected loss on the evaluated set (or future stationary draw) under graph/temporal correlation. |
| Statuses | `UNREFUTED` (the data do not contradict the declared `M`), `TAIL_UNRESOLVED` (`M̂ > M`; the interval is still returned), `ASSUMPTION_REQUIRED` (no interval: sample too small for `M`, or invalid input). Refutation checks report `REFUTED`, `NOT_REFUTED` or `UNTESTABLE`. |
| $L,\; \text{cs},\; \text{CP99}$ | Benchmark parameters: temporal correlation length $L$, target cluster size $\text{cs}$, and 99% Clopper–Pearson interval (`CP99`) on empirical coverage across Monte-Carlo replications. |
| `m_true`, `m_clust` | In the synthetic studies of §4 and §5.4: the size of the true equicorrelated dependence blocks, and the size of the clusters the procedure uses. |
| `miss/α` | Empirical miss rate of the interval divided by the nominal α; values at most 1 mean the interval is at least as reliable as stated. |

---

## 0. What this document settles and what it does not

**Settles:**

1.  A correlated premise that requires identically distributed `s_i` **does
    not apply to graph data at all**: node-degree heterogeneity alone destroys
    identical distribution. §1.
2.  A replacement premise that survives non-identical distributions and
    heteroscedasticity, at a quantified price. §2, *Lemma H*.
3.  Why the obvious cluster formula `W = Σ_c m_c²` is wrong, and what the
    missing term is. §3.
4.  That aggregating to cluster totals **does not narrow the Cantelli interval
    at all**: the two routes give algebraically identical widths. The real
    payoff of clustering is refutability. §4.
5.  That the correlation structure is **not estimable from a single model's
    evaluation pass** without structural or temporal assumptions, and that an
    offline archive-model or topological layer is where replicates exist. §6.

**Does not settle in general:** an assumption-free bound on cross-cluster
correlation mass. §5 states what a topological bound looks like and where it
fails. The topological bound under a declared edge-correlation bound
$\Phi$ is Lemma T′ in [THEORY.md](THEORY.md).

---

## 1. Why the correlated premise is pooled

A correlated premise of the form "let `s_1,…,s_n ≥ 0` be **identically
distributed**, not necessarily independent" defines the additive correlation
mass as

$$W \;=\; \frac{n^2\mathrm{Var}(\bar s)}{\mathrm{Var}(s)}$$

with a single `Var(s)` in the denominator, and its merge rule assumes that the
shards share one marginal law.

On a graph, they do not. A degree-3 leaf and a degree-10⁵ hub have different
loss distributions under any model worth evaluating; that is usually the
*point* of the evaluation. There is no common `Var(s)` to divide by, so `W` is
not even well defined, and `n_eff = n²/W` inherits that.

Under that premise the
`effective_n` parameter is licensed only for **exchangeable** correlated data,
such as equicorrelated Gaussian errors. A benchmark that uses only
exchangeable families cannot detect the problem, because every family it runs
satisfies the hypothesis.

The library therefore states the correlated premise on pooled moments:
premise `P1′(n_eff)` drops identical distribution, and premise `P2*_pool(M)`
replaces the shared marginal law. Lemma H (§2) derives `n_eff` under these
premises, and Proposition G ([THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md)) merges sketches
by the conservative $\kappa_{\max}$ rule without dividing by a common `Var(s)`.
The independent interval computation (`bound()`) is the same under both forms.

---

## 2. Lemma H — dropping identical distribution

The normative statement is Lemma H in [THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md). This
section gives the derivation and its consequences for the design.

### Statement

> **Lemma H.** Let `s_1,…,s_n ≥ 0` be square-integrable, neither independent nor
> identically distributed. Define the **pooled** quantities (premise `P2*_pool(M)`):
>
> $$\theta := \tfrac1n\textstyle\sum_i E[s_i] \in (0,\infty), \qquad \text{(P2*-pooled)}\quad \tfrac1n\textstyle\sum_i E[s_i^2] \;\le\; M\theta^2 .$$
>
> Let `R = (ρ_ij)` be the correlation matrix of `(s_1,…,s_n)` and let
>
> $$\kappa \;:=\; \max_i \textstyle\sum_j |\rho_{ij}| \;\ge\; 1$$
>
> be its maximum absolute row sum. Then **premise `P1′(n_eff)` holds with `n_eff = n/κ`**:
>
> $$\mathrm{Var}(\bar s) \;\le\; \frac{(M-1)\theta^2}{n/\kappa}.$$

### Argument sketch

Premises: `P2*_pool(M)` and square integrability. Write `σ_i = sd(s_i)` and
`σ = (σ_1,…,σ_n)`.

1.  `n²Var(s̄) = σᵀRσ ≤ λ_max(R)·‖σ‖²`, by the Rayleigh quotient.
2.  `R` is symmetric, so `λ_max(R) ≤ ‖R‖_∞ = κ`.
3.  Hence `Var(s̄) ≤ (κ/n)·σ̄²` with `σ̄² := (1/n)Σσ_i²`.
4.  Finally

    $$\bar{\sigma^2} = \tfrac1n\sum_i E[s_i^2] - \tfrac1n\sum_i (E s_i)^2
      \;\le\; M\theta^2 - \theta^2$$

    where the first term is bounded by `P2*_pool(M)` and the second uses
    Cauchy–Schwarz, `(1/n)Σa_i² ≥ ((1/n)Σa_i)² = θ²`. ∎

`E[s̄] = θ` holds by construction, so the argument for Lemma A′
([THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md)) goes through verbatim from its second line.
The identical-distribution hypothesis is therefore unnecessary, and the pooled
definitions above replace it at no cost in the i.i.d. case.

### Three consequences

1.  **`M̂` already estimates the pooled quantity.** `M̂_n = n·Σs²/(Σs)²` is
    exactly `((1/n)Σs_i²)/((1/n)Σs_i)²`, whose limit is the pooled ratio.
    Proposition F's refutation test ([THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md)) needs no
    change. That is a strong hint that the pooled form is the right one and
    that the identical-distribution hypothesis was never load-bearing.
2.  **`κ` discards negative correlation.** `W = Σ_ij ρ_ij` is a signed sum, so
    negative dependence *increases* `n_eff` beyond `n`; `κ` uses `|ρ_ij|` and
    cannot. Where both apply, `n²/W ≥ n/κ`, so the `W` route is tighter, and
    they **coincide exactly for balanced block-diagonal structure** (blocks of
    common size `m`: `W = nm`, `κ = m`, both give `n/m`). For unbalanced blocks
    `κ = max_c m_c` is dominated by the worst block while `n²/W = n²/Σ m_c²` is
    not.
3.  **Consequence 2 is an argument for a size-balanced partitioner**,
    independent of the systems argument for balance. Two different
    requirements point the same way.

---

## 3. Why `W = Σ_c m_c²` is wrong

Given a partition `𝒞`, split the correlation mass:

$$W \;=\; \underbrace{\sum_c \sum_{i,j\in c}\rho_{ij}}_{W_{\text{in}}}
      \;+\; \underbrace{\sum_{c\neq c'}\sum_{i\in c,\,j\in c'}\rho_{ij}}_{W_{\text{out}}}.$$

Bounding `ρ_ij ≤ 1` inside blocks gives `W_in ≤ Σ_c m_c²`. But the naive
formula drops `W_out`, which is **positive** whenever correlations across the
cut are positive on net. Once `W_out > Σ_c m_c² − W_in`, `Σ_c m_c²`
**understates** `W`, **overstates** `n_eff = n²/W`, and **under-covers**. The
data cannot show that this does not happen, so no version of this shortcut is
free of a penalty on `α`.

In the `κ` form the same decomposition is

$$\kappa \;\le\; \max_i\Big[\underbrace{\textstyle\sum_{j\in c(i)}|\rho_{ij}|}_{\le\, m_{c(i)}}
   \;+\; \underbrace{\textstyle\sum_{j\notin c(i)}|\rho_{ij}|}_{=:\ \beta_i}\Big]
   \;\le\; m_{\max} + \beta_{\max}.$$

**`β_max`, the worst-case per-node cross-cluster correlation mass, is the
central open quantity.** §5 analyses how to bound or check it.

---

## 4. Three routes to a valid `ν`, and why clustering does not narrow the interval

The table compares the three candidate routes:

| | route | premise needed | what the sketch carries | status |
|---|---|---|---|---|
| **D0** | item-level Cantelli with `ν = n/κ` (Lemma H) | premise `P2*_pool(M)` + a bound on `κ` | sketch 6-tuple + `effective_n` | available; `stats.independent` implements it |
| **D1** | **batch means**: treat cluster totals `S_c = Σ_{i∈c} s_i` as the observations, Cantelli with `ν = G/κ_cluster` | pooled second-moment premise *at cluster level* + a bound on `κ_cluster` | per-cluster accumulators | **CHOSEN** |
| **D2** | full Claim 1 (Lemma E + Lemma B) on the `S_c` | **independence** of cluster totals | same as D1 | **REJECTED** |

D1 is the classical batch-means device from Markov chain Monte Carlo (MCMC)
output analysis. It operates on a precomputed `cluster_id` node feature; the
evaluation accumulates per-cluster totals.

### The width identity: D1 does not narrow the interval

Take balanced blocks of size `m` with within-block correlation `ρ` and no
leakage, so `G = n/m`. Write `c` for the Cantelli relative half-width.

- **D0:** `κ = 1 + (m−1)ρ`, so `c² ∝ (M−1)κ/n = (M−1)(1+(m−1)ρ)/n`.
- **D1:** `Var(S_c) = mσ²(1+(m−1)ρ)` and `E[S_c] = mθ`, so
  `M_c − 1 = (M−1)(1+(m−1)ρ)/m`, and `c² ∝ (M_c−1)/G = (M−1)(1+(m−1)ρ)/n`.

**The widths are algebraically identical,** as they must be: both are the same
variance bound written in different coordinates. Any claim that clustering
buys width is wrong.

### Lemma I — when the declared `M` carries over to clusters

Lemma I below assumes per-cluster homogeneity or equal cluster sizes. When
cluster sizes vary, the library uses the pooled size correction
$M_c^{\text{default}}$ of Lemma I′ ([THEORY.md](THEORY.md)).

**Statement.** Let `S = Σ_{i∈c} s_i` over a cluster of size `m`, with `s_i ≥ 0`,
`μ_i = E[s_i]`, `σ_i² = Var(s_i)`, and let `θ_c = (1/m)Σ_{i∈c} μ_i` be the
cluster's own per-item mean. If `(1/m)Σσ_i² ≤ (M−1)·θ_c²` (implied by the
pooled premise `P2*_pool(M)` restricted to `c`, `(1/m)Σ E[s_i²] ≤ M·θ_c²`,
because `(1/m)Σ μ_i² ≥ θ_c²`), then

```
M_c − 1  :=  Var(S)/(E S)²  ≤  M − 1,   equivalently   E[S²] ≤ M·(E S)².
```

**Argument.** Premise: `(1/m)Σσ_i² ≤ (M−1)·θ_c²`.

1.  By Cauchy–Schwarz on the covariance matrix,
    `Var(S) = Σ_{i,j} Cov(s_i,s_j) ≤ (Σ_i σ_i)²`.
2.  By Cauchy–Schwarz again, `(Σ_i σ_i)² ≤ m·Σ_i σ_i²`.
3.  With `E S = mθ_c`,

    ```
    M_c − 1  ≤  m·Σσ_i² / (mθ_c)²  =  [(1/m)Σσ_i²] / θ_c²  ≤  M − 1.   ∎
    ```

> [!IMPORTANT]
> **The argument uses no assumption on the within-cluster correlation structure.**
> The first inequality is the worst case `ρ_ij = 1` for every pair. So for
> equal-sized clusters:
>
> - **D1 reuses the item-level declared `M` unchanged** (and for general
>   cluster sizes, adjusts it only by the known count statistics
>   $(r, \mathrm{CV}_n)$ via Lemma I′ in [THEORY.md](THEORY.md)), which is
>   automatically conservative.
> - It holds at every granularity simultaneously, so a caller may change the
>   clustering without revisiting `M`.
> - Sanity check in the equicorrelated case:
>   `M_c − 1 = (M−1)(1+(m−1)ρ)/m ≤ M − 1` iff `ρ ≤ 1`. ✓

The bound is loose when within-cluster correlation is weak. At `ρ = 0` the
truth is `M_c − 1 = (M−1)/m` and the lemma only gives `M − 1`, a factor of
`m`. That looseness is harmless for **validity**, but it is not free: the
declared `M_c` sets the Cantelli existence threshold
`n_A = (M_c−1)(1−α′)/α′`, measured against `G` rather than `n`. Reusing
the item-level `M` therefore costs certification at coarse granularities
unless a tighter `M_c` is declared.

### What D1 buys: refutability

**D1 converts an unverifiable assumption into a refutable one.** Under D0 you
must declare `M` *and* `κ`, and only `M` is subject to Proposition F's
refutation test; the within-cluster correlation enters through `κ`, and
nothing in the data can contradict it. Under D1 the within-cluster correlation
is *absorbed into* `M_c`, and `M̂_G` computed from the cluster totals refutes
it exactly as `M̂_n` refutes `M` on independent data.

**Only the cross-cluster term `β` (or the between-cluster VIF `κ_cluster`)
stays outside `M̂_G`.** That is a strictly smaller surface of unrefuted
dependence. The refutation checks in [THEORY.md](THEORY.md) add dedicated
checks for `κ_cluster`.

> [!WARNING]
> **`β` is not merely unverified by `M̂_G`; it is invisible to `M̂_G` by
> construction.**
>
> `M̂_G = G·Σ_c S_c² / (Σ_c S_c)²` converges to `E[S²]/(E S)²`, a **marginal**
> second moment of a single cluster total. Cross-cluster correlation does not
> appear in the marginal law of `S_c`. So no amount of data can move `M̂_G` in
> response to `β`: it is not a hard signal to detect, it is **not a function
> of the quantity `M̂_G` estimates**.

The measurement below confirms this. It uses `m_clust = 5`, `ρ = 0.8`,
`n = 100000` and 200 replications, and varies only the true block size, so
only `β` changes:

| `m_true` | true `κ_cluster` | `M̂` mean | `M̂` sd |
|---|---|---|---|
| 5 (clusters independent) | 1.0 | 2.4289 | 0.0236 |
| 50 | 9.1 | 2.4239 | 0.0755 |
| 250 | 45.0 | 2.4207 | 0.1601 |

The **target is unmoved** across a 45× change in `κ`; only the estimator's
precision degrades. As a side benefit this confirms the analytic
`M_c = 1 + 2(1+(m_c−1)ρ²)/m_c = 2.424` to four figures, including the
easily-missed fact that Gaussian errors correlated at `ρ` give **summands**
correlated at `ρ²`.

When `M̂_G` is used alone, without a separate `κ` check, two consequences
follow:

1.  **`M̂_G` provides no diagnostic for `κ_cluster`.** Without the dedicated
    `κ` refutation checks in [THEORY.md](THEORY.md), only the conservatism of
    the Cantelli bound stands between a wrong `κ` and silent under-coverage.
    §5.4 quantifies that conservatism as `κ*`.
2.  **The interval width cannot warn the user, and actively misleads.** D1's
    width is a function of `(G, M_declared, κ, s̄)` only; it does not respond
    to the true dependence at all. Measured at `m_true=250, ρ=0.8,
    m_clust=5, α=0.05` with `κ = 1`: D1 returns a width of **0.1538 and covers
    82.3%**, while the correct-by-construction D0 row on the *same data*
    returns **1.0673 and covers 100%**. D1 looks 7× narrower precisely because
    `κ = 1` ignores the cross-cluster dependence.

**Empirical check of the within-cluster channel** (5000 replications per cell).
The *within*-cluster
refutation channel (`M̂_G > M_declared` triggering `TAIL_UNRESOLVED`) works
cleanly across granularities, with one edge-case caveat right at the
threshold. At `ρ = 0.8, m_true = 50, α = 0.05`, the certification rate moves
as follows:

- For `m_clust = 5` (`M_c = 2.424`, `M̂_G = 2.423 ± 0.071`): certification
  transitions from **0.0%** (`M_d ≤ 2.00`) → **2.85%** (`M_d = 2.30`, −5.1%) →
  **40.80%** (`M_d = 2.40`, −1.0%) → **85.05%** (`M_d = 2.50`, +3.1%) →
  **100.0%** (`M_d ≥ 3.00`).
- For `m_clust = 25` (`M_c = 2.309`, `M̂_G = 2.307 ± 0.072`): **0.0%**
  (`M_d ≤ 2.00`) → **47.30%** (`M_d = 2.30`, −0.4%) → **90.35%**
  (`M_d = 2.40`) → **99.00%** (`M_d = 2.50`) → **100.0%** (`M_d ≥ 3.00`).
- **Selection-bias caveat at the knife edge.** At `m_clust = 5, M_d = 2.30`
  (under-declared by 5.1%, combined with an uncompensated cross-cluster
  `κ_true = 9.09`), the 2.85% of replications that happen to draw
  `M̂_G ≤ 2.30` are also the replications with slightly downward-biased sample
  variance/mean. They yield **91.23% coverage among the 57 certified draws**
  (vs 95% nominal; the unconditional false-certificate-and-miscoverage rate is
  `0.0285 × 0.0877 = 0.25%`). At `m_clust = 25` (`κ_true = 1.98`), coverage
  among certified draws remains **100.0%** across the entire transition.

### The cost: refutation power

`M_c = 1 + (M−1)(1 + (m−1)ρ)/m → 1 + (M−1)ρ` as `m` grows, so the Cantelli
existence threshold `n_A(M_c, α)`, which scales with `M_c − 1`, shrinks by the
factor `ρ`. (Route D1 always passes an effective sample size, so only Cantelli
applies and `n_A`, not `n_U`, is the relevant threshold.) With weak
within-cluster correlation, for example `M_c = 1.02` at `α = 0.05`,
`n_A = 0.78 < 1` and the distributional refusal never
fires. That sounds like a win and is partly a trap: **`M̂` has far less power
with `G` observations than with `n`.** Proposition F's refutation test stands
between a declared `M` and silent under-coverage, and D1 weakens it by a
factor of `m` in sample size while shrinking the effect it must detect. Larger
clusters absorb more dependence into the refutable term *and* make the
refutation weaker.

### Measuring the best cluster size

The optimum `m` is measured, not argued, in a synthetic equicorrelated-block
sweep.

The sweep does not perturb a declared `M_c` by a fixed factor. A mis-declared
`M_c` is exactly the thing `M̂_G` *can* refute, so perturbing it would measure
the strength of a check that already exists. The quantity with no `M̂_G` check
behind it is the cross-cluster correlation, and the clustering granularity
controls it.

The sweep therefore holds the **true** dependence fixed (equicorrelated blocks
of size `m_true ∈ {10, 50}` with `ρ ∈ {0.3, 0.8}`) and varies the **declared
granularity** `m_clust ∈ {1, 5, 10, 25, 50, 100, 250, 500}` and the
**alignment** of the cluster grid to the blocks, while always asserting
`κ_cluster = 1`. That assertion is:

- **true** when `m_clust ≥ m_true` and the grid is aligned;
- **false by a factor of about `m_true/m_clust`** when `m_clust < m_true`,
  because each block is then split across several mutually correlated
  clusters;
- **false even at `m_clust = m_true`** when the grid is offset by half a
  block, because every cluster then straddles two blocks.

The realised coverage in the false cells *is* the empirical price of `β`. The
opposing cost, `G = n/m_clust` falling toward the Cantelli threshold, weakening
`M̂_G` and eventually refusing, shows up in the same table as the
certification rate and width. `m_clust = 1` is the D0 reference row and runs
with the analytically exact `effective_n`, so it is correct by construction
and the cluster rows have a sound reference.

The sweep declares `M = 4` against a true `M = 3` (standard normal errors,
`mse`). That declaration is in class, and it stays in class at every
granularity by the `M_c ≤ M_pool` result above. `M = 16` is not used for this
sweep because its existence threshold `n_A = 585` refuses every cell with
`m_clust ≥ 50`, which would measure the threshold rather than `β`.

### D2 is rejected

Aggregation to cluster totals does **not** re-enable Bernstein (Lemma E) or
Maurer–Pontil (Lemma B). Both need factorisation of the moment-generating
function (MGF), i.e. **independence**, and cluster totals with a non-zero edge
cut are at best *uncorrelated*. D2 therefore requires a strictly stronger
premise than D1, one that no topological bound on `β` can ever deliver: `β` is
a second-moment quantity, and independence is not a second-moment property.
**The library does not offer D2. Where D1 cannot certify, the library refuses.**

---

## 5. Where `κ` comes from: topology-only bounds on `β`

### 5.1 The decay assumption

A natural route is a spatial-mixing assumption transplanted to graphs. Let
`d_G(i, j)` be the shortest-path distance between nodes `i` and `j`, and
assume

$$|\rho_{ij}| \;\le\; \phi\big(d_G(i,j)\big), \qquad \phi \text{ non-increasing}.$$

Then with `N_i(r) = |{j : d_G(i,j) = r}|` the sphere sizes,

$$\kappa \;\le\; \max_i \sum_{r\ge 0} N_i(r)\,\phi(r),$$

which is **purely topological given `φ`**. This bound needs no clusters at
all: clustering is a computational and diagnostic device, not a mathematical
necessity.

With exponential decay `φ(r) = φ₀λ^r` and polynomial ball growth
`N_i(r) ≤ C r^{D−1}` (doubling dimension `D`), the sum converges to a constant
and `n_eff ≥ n/κ` with `κ = O(1)`. On a mesh, a road network, or a
spatially-embedded graph this works.

### 5.2 Where it fails

On a small-world or power-law graph, `N_i(2)` can already be a constant
fraction of `n`. Then `κ = Θ(n)`, `n_eff = O(1)`, and the bound is
**vacuous**, correctly so: a loss field with slowly decaying correlation on a
small-world graph genuinely contains `O(1)` independent observations' worth of
information. Refusing is the mathematically sound answer when the
information content is O(1), though it means unconstrained social or citation
graphs will frequently refuse.

> [!IMPORTANT]
> **The estimand is fixed to "mean loss over all nodes."** The consumer is a
> user evaluating their own metric on their own graph. They want a robust
> interval or a straight answer that their `α` is unaffordable, not a silently
> redefined estimand. This **rules out the first two escapes below as
> defaults** and makes refusal the primary answer on badly mixing graphs.

Three escapes were considered. Only the third survives the decision above,
and it is the weakest.

- **`r`-separated evaluation sets: rejected.** Evaluate on an independent set
  of `G^r` (the graph whose edges join nodes at distance at most `r`). This
  kills `κ` by construction. **But it changes the estimand**: an
  `r`-separated set is biased toward the low-degree periphery. That is bias,
  not variance; no `n_eff` can repair it, exactly the failure mode documented
  for correlated R² in [THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md).
- **Boundary peeling: rejected as a default.** Evaluate only on nodes at
  distance `≥ b` from their cluster boundary; then
  `β_i ≤ Σ_{r≥b} N_i(r)φ(r)`. The same bias objection applies, more mildly,
  and `b` trades `n` against `κ`. It may return **only** as an explicit,
  opt-in, differently named estimand ("interior mean loss"), never as an
  internal optimisation.
- **Weighted (Horvitz–Thompson) correction: survives, but weakly.** Sample an
  `r`-separated set at random, then reweight by inclusion probability to
  restore the all-node estimand. This is unbiased. The cost is **not**
  heteroscedasticity, which Lemma H already tolerates; it is the
  **unequal-weight design effect**. For the weighted mean the bound becomes
  `Var ≤ κ·Σᵢwᵢ²σᵢ²/(Σᵢwᵢ)²`, so the `n` in Lemma H is replaced by the Kish
  effective size `(Σwᵢ)²/Σwᵢ²`, which on a degree-skewed graph can be a small
  fraction of the sample. Whether that trade beats plain refusal is an
  empirical question. The library does not implement it.

> [!IMPORTANT]
> **The refusal is the product, and it needs two diagnoses, not one.** Since
> refusal is the main escape, the message must distinguish two situations
> that call for opposite actions:
>
> | diagnosis | condition | honest message |
> |---|---|---|
> | **RESOLVABLE** | `n_eff < ν_required(M, α)` but `n_eff` is still growing in `n` | "collect more nodes", and the library can say **how many**, by inverting the Cantelli existence condition `ν = G/κ > n_A(M,α)` to `G ≥ ⌈κ·n_A(M,α)⌉ + 1` clusters, that is about `m̄·G` items |
> | **SATURATED** | `n_eff` is bounded above by the graph's mixing regardless of `n` (in the limit, `κ = Θ(n)`) | "this graph cannot support this `α`. More nodes will not help. Loosen `α`, or change the evaluation unit." |
>
> Conflating them would send a user off to label a million more nodes for
> zero gain. The required-`n` figure in the RESOLVABLE row is cheap (one
> inversion of an existing threshold) and is the most useful thing the
> library can hand a user who is stuck. It is produced **even when the
> interval is refused**, so the refusal path carries a payload, not just a
> status enum.

The payload is `cluster_bound.DiagnosticPayload`, with
`Diagnosis.RESOLVABLE_SAMPLE_SIZE` and `Diagnosis.SATURATED_GRAPH_MIXING`.
`κ = Θ(n)` is an asymptotic description; the code uses a finite-sample rule
(`cluster_bound.is_mixing_saturated`). It declares saturation when a
caller-supplied `max_achievable_effective_n` is at most `n_A`, or when `G ≥ 4`
and `κ ≥ G/2`. When saturated, the required-size fields are left empty. Otherwise
the inversion uses `n_A(M_d, α)`, or `n_A(max(M_d, M̂_G), α)` when the status is
`TAIL_UNRESOLVED`.

### 5.3 Effective resistance instead of geodesic distance

`d_G` is a poor proxy on graphs with hubs: a single hub makes every node
distance-2 from every other. Effective resistance `R_eff(i,j)` is the natural
alternative. It is a metric, it is computable at scale by
Johnson–Lindenstrauss sketching, and for a Gaussian Markov random field with
precision `L + τI` (graph Laplacian `L`, ridge `τ`) it relates directly to
covariance. Comparing `φ(R_eff)` with `φ(d_G)` on the archive panel of §6.2
is the natural test before committing to a geodesic decay functional. The
library does not ship a decay functional. For edge-local or general pairwise
bounds it ships the spectral bound $\kappa_T = 1 + \lambda_{\max}(\Phi^{\text{cut}})$
of Lemma T′ ([THEORY.md](THEORY.md); `graph.topological_kappa`).

---

### 5.4 `κ*` — how wrong the `κ` assertion may be before it costs coverage

§5.1–5.3 ask how to *bound* `β`. This subsection asks the complementary and
much more tractable question: **when `β` is not bounded a priori, how much
error does the procedure absorb?** The answer is a lot, and it complements the
sensitivity bound of Claim 5 ([THEORY.md](THEORY.md)).

#### Definition

The definition assumes that the caller asserts `κ = 1`, so the interval is
computed with `ν = G`; this is the setting of the sweep below. Cantelli then
gives a relative half-width `c = √((M_d−1)(1−α′)/(G α′))`, while the
sampling standard deviation of the cluster mean is `√((M_c−1)·κ/G)`. Write
`z_{1−α/2}` for the standard normal quantile. Coverage survives while the
former exceeds `z_{1−α/2}` times the latter, i.e. while

```
κ  <  κ*  :=  (M_d − 1)·(1−α′)/α′  /  ( (M_c − 1)·z²_{1−α/2} )
```

`G` cancels. **The robustness margin does not improve with more data**: it is
a property of `(M_d, M_c, α)` alone. More nodes buy a narrower interval, not a
safer one.

> [!CAUTION]
> `κ*` is a **normal approximation**, not a mathematical claim. It assumes the cluster
> totals are near-Gaussian, which needs large `G` and light-tailed summands,
> and it says nothing in the regime where Cantelli is near-vertical. It does
> not justify weakening, removing or gating any branch that issues an interval;
> it is purely a diagnostic for reading experiments and for advising on cluster
> size.

#### Measured (132 cells, 2000 reps, n=100000)

Source: [EVIDENCE.md §9](EVIDENCE.md#9-cluster-granularity-sweep). In every
certified cell below, all repetitions were `UNREFUTED`, so "coverage" here is
`1 − miss rate` over all repetitions.

The table holds `κ_true = 45.04` fixed (`m_true=250, ρ=0.8, m_clust=5`, with
declared `M_d = 4` and true `M_c = 2.424`) and varies only `α`:

| α | `κ*` | `κ_true/κ*` | coverage | nominal | verdict |
|---|---|---|---|---|---|
| 0.10 | 14.8 | 3.04 | **65.75%** | 90% | broken |
| 0.05 | 21.4 | 2.11 | **82.30%** | 95% | broken |
| 0.01 | 63.2 | 0.71 | 99.70% | 99% | holds |

Across all 108 certified cells, **no cell with `κ_true < κ*` under-covered**,
on either a point estimate or a 99% Clopper–Pearson upper bound. `κ*`
concerns the cluster route, which asserts `kappa_cluster = 1`; the 18
`m_clust = 1` cells use the item-level route with the exact `effective_n`,
assert nothing about `κ`, and cover fully even where `κ_true > κ*`.

> [!IMPORTANT]
> **Tightening `α` makes the procedure *more* robust to a wrong `κ`, not
> less.** The same `κ = 45` that destroys coverage at `α = 0.10` is survived
> at `α = 0.01`. The definition shows why: Cantelli's penalty grows like
> `1/α′`, while the quantile a correct interval needs grows only like
> `√log(1/α)`.
>
> This inverts the natural worry. A demanding `α = 0.001` requirement is the
> **safe** end for graph data; the exposure is at `α = 0.1`. It also means an
> `α` chosen for its statistical meaning silently also chooses a
> `β`-tolerance, and the two have no reason to agree.

#### The actionable rule

With `κ_true ≈ m_true/m_clust` and `M_c − 1` roughly flat in `m_clust` below
`m_true`, the safety condition `κ_true < κ*` reduces to a statement about
granularity:

```
m_clust  >  m_true / κ*        (at α = 0.05: κ* ≈ 10 when M_d = M_c; ≈ 21 in the sweep above)
```

**Clusters need only be about 1/20th the size of the true dependence groups
in this sweep (about 1/10th when `M_d = M_c`), not larger than them.** The
grid confirms it: `m_true/m_clust = 10`
(`m_true=250, m_clust=25`) holds at 96.65% against a 90% nominal, while
`m_true/m_clust = 50` breaks to 65.75%.

This is considerably more forgiving than the analysis of §4 suggests, and it
is a practical guideline: when relying on Cantelli's slack alone, a caller
needs an **order-of-magnitude** upper bound on the diameter of the true
dependence groups.

#### Calibration and remaining caveats

- **`κ*` is calibrated to within about 20%** (5000 reps per cell). Filling the
  gap between ratios `0.71` and `2.11` (`m_clust=5, ρ=0.8, n=100000`), the
  empirical crossing sits right at **`κ_true / κ* ≈ 1.0`**:
  - At `α = 0.05` (`κ* = 21.4`): `ratio = 0.85` holds at **97.00%**
    (`CP99 = [96.32%, 97.59%]`); `ratio = 1.06` (`m_true = 125`) crosses
    nominal with point coverage **94.36%** (`CP99 = [93.47%, 95.17%]`);
    `ratio = 1.35` (`m_true = 160`) breaks unambiguously at **91.06%**
    (`CP99 = [89.97%, 92.07%]`).
  - At `α = 0.10` (`κ* = 14.8`): `ratio = 0.98` (`m_true = 80`) sits right on
    the 90% nominal boundary at **89.96%** (`CP99 = [88.82%, 91.03%]`);
    `ratio = 1.22` (`m_true = 100`) breaks at **87.14%**
    (`CP99 = [85.88%, 88.33%]`).
  - Combining across `α`, the point-estimate crossing is bracketed in
    **`(0.85, 0.98)`**, and the statistically established break
    (`CP99_hi < 1 − α`) lies in **`(1.06, 1.22)`**. The normal-approximation
    formula `κ*` predicts the transition accurately, though it remains an
    analytical diagnostic rather than a finite-sample claim.
- **A cell can pass on coverage while being useless.** `m_true=250, ρ=0.8,
  m_clust=1, α=0.01` reports 100% coverage with a median width of **45.9** on
  an estimand of 1.0; its `n_eff = 623.6` sits 4% above `n_A = 597`, on the
  near-vertical part of the Cantelli curve. Any summary that scores cells on
  coverage alone records that as a success.

---

## 6. The estimation problem: one draw of the field

### 6.1 `κ` cannot be estimated from one model without structural assumptions

An evaluation pass gives **one realisation** of the random vector
`s ∈ ℝⁿ`. An `n×n` correlation matrix cannot be estimated from one draw
without a structural assumption that manufactures replicates:
homogeneity/stationarity over the graph, which lets you pool over node pairs
at equal distance. That is the empirical-variogram or Moran's-I approach. It
buys estimability at the price of an assumption (graph stationarity) that is
as strong as the thing it estimates, and that degree heterogeneity badly
violates.

### 6.2 The archive panel is where replicates exist across models

An offline, model-agnostic archive layer provides replicates
across models: run `P` archived models over the same graph to obtain `P`
draws `s^{(k)}` (`k = 1, …, P`) of the loss field. Per-cluster totals `S_c^{(k)}` give a
`G × P` panel from which `Corr(S_c, S_c')`, hence `β`, hence `κ_cluster`, is
estimable across `k` without a graph-stationarity assumption. Cross-model
correlation differs from label-noise correlation under estimand `E2`, so the
panel is not a source of `κ` for `E2`.

The panel's role is therefore limited. Its intended deliverable is a single
number `κ̂_cluster(𝒞)` per partition, computed once offline and stored
alongside the `cluster_id` node feature. The library does not provide it directly;
the panel estimator exists only as an offline simulation instrument.

### 6.3 Validity for a *new* model is an unverifiable assertion, and that is acceptable

A `κ̂` estimated on a model family `ℳ` bounds a new model's `κ` only if the
new model's loss field lies in the same correlation class. That is
unverifiable. But it is the **same category** of risk as Limitation L12
(declared `n_eff`) and Limitation L13 (merge shard-uncorrelatedness) in
[THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md), so it introduces no new kind of risk, only a
cluster-level instance, tracked as **Limitation V1** in
[THEORY.md](THEORY.md).

**The library refutes rather than verifies**, matching the `UNREFUTED`
semantics already in the independent estimator. From the single new draw it computes a
graph-autocorrelation statistic (on the cluster-total residuals, or the lag-1
autocorrelation on time windows; see the refutation checks in
[THEORY.md](THEORY.md)) and compares it against its predicted null under `κ`.
This can only reject, never confirm, which is precisely the existing contract
for `M`. A declared `κ` with a live refutation test is strictly better than a
declared `κ` without one, and it is cheap.

### 6.4 Estimating `κ` from the losses changes the α-budget

This is the trap. Any `κ̂` computed from the same `s` that produces `s̄` makes
`c_cantelli` **random**. Lemma C ([THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md)) licenses
free branch selection only for selectors that are deterministic in
`(n, M, α, K, n_eff)`. A data-dependent `κ̂` voids it, and the interval then
costs a union split `α = (α − 2α_κ) + 2α_κ` plus an actual high-probability
upper bound on `κ̂`, which §6.1 says one draw cannot provide. **Any `κ` that
enters the certificate must come from topology, declarations or archive data
only.** This is not a stylistic preference; it is what keeps Lemma C.

---

## 7. Partitioning at multi-machine scale

The requirements below follow from the analysis above. They are listed in
priority order:

1.  **Minimise cross-cluster correlation mass `β`**, proxied by conductance or
    normalised cut, *not* modularity. Modularity optimises a null-model
    contrast that has nothing to do with `β` and is known to have a resolution
    limit; Leiden is a fine diagnostic and the wrong objective.
2.  **Size-balanced.** By consequence 2 of §2, unbalanced blocks make
    `κ = max_c m_c` dominated by the worst cluster. Balance is also a systems
    requirement. A hard balance constraint is preferable to a penalty.
3.  **Independent of the losses `s`.** This is the real condition, *not*
    determinism. A randomised partitioner is fine provided its randomness is
    independent of `s`, because Lemma C needs the selector to be
    non-data-dependent, not fixed. This admits streaming and sampling-based
    partitioners.
4.  **Computed once, offline, and materialised as a node feature**, which
    reinforces requirement 3.

The candidates considered were:

*   ParHIP / ParMETIS: balanced k-way min-cut; the right objective, with a
    known scaling ceiling.
*   Balanced label propagation in the style of Ugander and Backstrom: scales
    to social-graph size, lower quality, explicitly balance-constrained.
*   Fennel-style streaming: cheapest, one pass, lowest quality.
*   Spectral coarsening: best quality, worst scaling.

The partitioner is a capacity-constrained coarsening designed for
distributability; spectral methods serve only as a
quality reference.

---

## 8. Sketch and API implications

- **Per-cluster accumulators.** The independent sketch 6-tuple is per-dataset; D1 needs
  per-cluster accumulators. The cluster sketch is `cluster_sketch.ClusterSketch`.
- **Minimal D1 sketch:** `G`, `Σ_c S_c`, `Σ_c S_c²`, plus the largest cluster
  totals (the independent sketch's top-`K` order statistics with `K = 32`; field
  `top_32`) for the cluster-level `M̂`. `ClusterSketch` also stores `n_max` and
  `Σ_c n_c²` for the default `M_c` (Lemma I′), the declared `κ_cluster`,
  and 64-bit cluster-id fingerprints. This is structurally the *same* sketch
  as the independent sketch with the summand redefined, so the independent-path interval
  computation is reused unchanged (`ClusterSketch.to_independent_sketch` converts to
  `IndependentSketch` of `stats.independent`), and the new work is entirely in the
  aggregation layer. `stats.independent` implements the independent algorithm inside
  `stats`.
- **JAX/TPU:** cluster totals are a `segment_sum` over a static `cluster_id`
  feature. They fuse into the evaluation step and merge across shards. This is
  the cheapest part of the whole design.
- **Merge:** simple summation over disjoint cluster sets replaces Proposition
  G's `W`-addition, *provided shards do not split a cluster*. The same
  Limitation L13 caveat ([THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md)) applies, with
  a checkable proxy: cluster ids are in the sketch's domain. A shard that
  may split clusters is carried as `PartialClusterShard`, whose `merge` sums
  partial totals exactly before squaring; `ClusterSketch.merge` raises
  `ValueError` when two sketches share a cluster-id fingerprint instead of
  trusting the caller.

---

## 9. Correlated R², briefly

Pairing index-adjacent items inside one dependence group gives
`E[b] = ½E[(y−y′)²] = (1−ρ)Var(y) < θ_B`, a **bias**, so no `n_eff` repairs it
([THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md)). This is why `cluster_bound.cluster_interval`
rejects `metric="r2"`. Under a partition, Claim 7 ([THEORY.md](THEORY.md))
solves correlated R² **without pairing**. It bounds the denominator via
Cantelli on the unobservable centred label squares
$Q_c = \sum_{i \in c} (y_i - \bar\mu)^2$, combined with a Chebyshev bound on
$(\bar y - \bar\mu)^2$. The temporal and graph entry points implement it
(`metric="r2"`, `cluster_bound.compute_r2_interval`).

---

## 10. Design choices

This section summarizes the principal design choices implemented in the codebase.

1.  **Pooled moment premise (`P2*_pool`).** The item-level relative second-moment
    bound $M$ is defined over pooled item summands ($\frac{1}{n}\sum_i \mathbb{E}[s_i^2] \le M\theta^2$)
    rather than assuming an identical marginal distribution for every item. Graph nodes
    and temporal processes exhibit natural degree and variance heterogeneity (such as
    high-degree hubs versus leaf nodes), which violates identical-distribution assumptions.
    Pooling allows finite-sample Cantelli guarantees to hold across heterogeneous items
    without requiring exchangeability.
2.  **Cluster totals under Cantelli (Route D1).** Dependent observations are grouped
    into clusters (contiguous time windows, graph partition components, or space-time
    blocks), and Cantelli's inequality is applied directly to the sum of cluster totals.
    This absorbs arbitrary, unmodeled correlation within each cluster without requiring
    within-cluster covariance structure, while cross-cluster dependence is bounded by a
    single variance inflation factor $\kappa$.
3.  **Fixed full-sample estimand.** The estimand $\theta$ is defined as the mean
    expected loss over the entire evaluated dataset, avoiding boundary peeling or
    subsampling to $r$-separated independent subsets. Practical model evaluation requires
    confidence intervals for the actual evaluation test set. Restricting evaluation to
    peripheral or pruned subgraphs alters the target population and introduces selection bias.

4.  **Declared assumptions with active refutation.** Callers declare bounds on relative
    moments ($M$ or $M_c$) and dependence ($\kappa$), and the library certifies coverage
    under these declarations while running empirical refutation checks (such as lag-1
    autocorrelation and edge correlation checks). Correlation cannot be consistently
    estimated from a single evaluation draw without unverifiable structural assumptions.
    The declare-and-refute contract provides conservative guarantees while actively
    testing the data for evidence contradicting the declared bounds.
5.  **Cluster sketch representation and mergeability.** Datasets are summarized into
    per-cluster accumulators (`ClusterSketch`, `PartialClusterShard`) capturing cluster
    counts, linear sums, sum-of-squares, and top order statistics. This enables distributed
    computation and exact associative, commutative merging across shards without streaming
    raw loss arrays into a single machine.

6.  **Automatic cluster-level moment scaling (Lemma I′).** When callers supply only an
    item-level moment bound $M$, the cluster-level moment bound $M_c$ is derived
    automatically from $M$ and observed cluster size statistics ($r = n_{\max}/\bar m$,
    $\mathrm{CV}_n^2$). This avoids requiring callers to calibrate a separate cluster-level
    parameter, while accounting for the maximum possible variance concentration under
    unequal cluster sizes.
7.  **Balanced size-constrained graph partitioning.** The graph partitioner
    (`partitioners.partition_graph`) uses capacity-constrained coarsening to enforce balance
    constraints: no cluster exceeds $\lceil 1.5 t \rceil$ and at most one cluster falls below
    $\lceil t/2 \rceil$. Strict balance bounds the size ratio $r$ and cluster coefficient
    of variation, preventing giant clusters from inflating $M_c$ and eroding the effective
    sample size.

8.  **Topological spectral bounds for graph $\kappa$.** When edge-level correlation bounds
    $\phi$ are known, `graph.topological_kappa` computes a certified $\kappa$ from the
    spectral radius of the cross-cluster cut adjacency matrix. This provides a mathematically
    grounded upper bound on $\kappa$ under pointwise correlation dominance, eliminating
    arbitrary declarations on graph topologies.
9.  **Projected edge statistic for graph refutation.** `refutation.check_uncorrelated_edges`
    tests $\kappa = 1$ on graphs using a self-normalized quadratic form that explicitly
    projects out the estimated mean $\hat\theta$. Without the projection, centering at the
    sample mean biases the statistic low and overstates its variance on dense or hub-heavy
    cluster graphs, so the test almost never refutes. Subtracting the projected null mean
    $\hat B$ and normalizing by the projected null variance $V'$ restores asymptotic
    $\mathcal{N}(0, 1)$ calibration when no small set of eigenvalues dominates (de Jong,
    1987). The size is asymptotic, not exact; §8 of the README lists the remaining hub-graph
    limitation.
10. **Quantile aggregation for time buckets (Lemma W).** `temporal.temporal_interval_from_buckets`
    aggregates pre-binned time buckets into contiguous windows using cumulative count quantiles.
    Pre-aggregated series have heterogeneous bucket counts; quantile grouping guarantees
    non-empty windows and bounds the count deviation $|n_c - \bar m| < b_{\max}$, so the
    temporal interval can be computed without raw loss records.

11. **Shift-invariant pre-centering for R² cluster shards.** `cluster_sketch.R2ClusterShard`
    pre-centers labels and predictions at a reference scalar before computing cluster totals,
    adjusting references upon shard merging. Raw sums of squares suffer severe catastrophic
    floating-point cancellation when the label mean is large relative to its standard deviation;
    pre-centering preserves precision while maintaining exact shift-invariance.
12. **Design-only space-time partition selection.** `spacetime.choose_partition` selects the
    candidate partition that maximizes size-adjusted effective sample size $N_{\text{eff}}$
    using only graph topology, time indices, and declarations. Evaluating candidates without
    observing test losses ensures that partition selection is strictly independent of
    evaluation data, preventing post-selection bias from voiding coverage.

13. **Matrix-free Kronecker operator for range envelopes.** `range_envelope.range_kappa`
    uses a Kronecker operator combining spatial BFS balls and 1D temporal cumulative sums to
    compute variance inflation bounds. Spatio-temporal evaluation over hundreds of nodes and
    thousands of time steps generates millions of item-level pairs; the matrix-free Kronecker
    method computes certified bounds in seconds without forming dense cut matrices.
14. **Two-sided drift diagnostic for temporal and space-time series.** The library
    splits the windows into early and late halves and computes a level-$(1 - \alpha/2)$
    interval on each. The full-sample interval covers the in-sample pooled mean
    $\theta = \frac{1}{n}\sum_i \mathbb{E}[s_i]$ without stationarity, but reading it as the
    expected loss on future data requires stationarity. Disjoint half-intervals refute equal
    half-means at level $\alpha$ (union bound) and warn of drift; overlap does not certify
    stationarity.

## 11. Summary of coverage properties and operational risks

κ is a declaration on both paths (or is derived from a declared $\Phi$ via
Lemma T′ in [THEORY.md](THEORY.md)), and both paths check it (refutation
checks and Limitation V1 in [THEORY.md](THEORY.md)).

What the library certifies is conditional and stated: Claim 4 under declared
`(M_c, κ)`, and Claim 7 for R². Claim 5 prices a wrong declaration. The
checks refute large errors on the temporal path, except in the blind corner
(small `G`, long memory), and a wrong κ = 1 on graphs; the κ > 1 graph check
is weak on dense cluster graphs. In the blind corner, protection comes from
the `M_c` check and from refusal, as verified in
[EVIDENCE.md](EVIDENCE.md).

Key operational risks:

- **Dense cluster graphs need large κ.** Small per-edge correlation across many
  cut edges still produces a large variance inflation factor. An honest κ can
  then push `G/κ` near or below the existence threshold, so wide intervals or
  refusal are the likely outcome on dense graphs; larger clusters help only
  where dependence is local.
- **Refusals on short series.** With the size-corrected default `M_c`,
  refusal is the common outcome at `n` in the low thousands. This is
  honest, and a smaller declared `M_c` is admissible when justified because
  it is refutable via `M̂_G`.
- **The joint rate** `Pr(miss ∧ not refuted)` is measured in
  [EVIDENCE.md](EVIDENCE.md), not argued mathematically.
- **The drift warning** detects shifts of about twice the interval width, and
  on short series it is usually `UNTESTABLE`.

## 12. Space-time and range-envelope modules

The space-time and range-envelope modules reuse route D1 unchanged; they add
ways to choose the partition and to obtain κ from declarations. Their theory
is in [THEORY_SPACETIME.md](THEORY_SPACETIME.md) and
[THEORY_RANGE_ENVELOPE.md](THEORY_RANGE_ENVELOPE.md); this section records
only how they fit the design above.

### 12.1 `stats.spacetime`: graphs observed over time

*   **Items and clusters.** An item is a (node, time-step) pair
    (`spacetime.spacetime_items`, `SpaceTimeIndex`). Candidate clusters are
    space-time product blocks: spatial communities from the partitioner
    crossed with time windows, plus community-only and
    window-only candidates (`candidate_partitions`). `spacetime_edges` builds
    the item-level strong-product graph on observed items, with a declared
    correlation bound per edge (same-time spatial, same-node temporal and
    cross-lagged edges); it feeds the topological κ and the cluster
    adjacency.
*   **Same interval, same checks.** `spacetime_interval` calls
    `graph.graph_interval` on the chosen partition, so the certificate is
    Claim 4 and the κ checks are the edge and batch refutation checks (§10). Two diagnostics are
    added: marginal checks on window totals against `κ_t` and on
    per-window-centred community totals against `κ_s` (separable route,
    product partitions only), and a window-split drift warning (Proposition
    D′ in [THEORY_SPACETIME.md](THEORY_SPACETIME.md)).
*   **One declaration from choice to interval.** The declared item-level
    bound `M` (`m_item`) is passed to `choose_partition` and stored in the
    returned `Choice`. `spacetime_interval` takes no cluster-level `M_c`
    override: the cluster-level bound is always the size-corrected default of
    Lemma I′, computed from `m_item` and the chosen partition's counts.
    This keeps the bound that scored the partition and the bound that
    certifies the interval the same.
*   **Guards in `spacetime_interval`.** It raises `ValueError` when:
    *   the data length differs from the index's item count;
    *   the metric is R² and `kappa_labels` (the declared VIF of the label
        totals) is not passed, because the space-time κ bounds only the
        losses;
    *   an `m_item` passed to the interval differs from `Choice.m_item`;
    *   a `kappa` override is smaller than the chosen partition's κ
        (overrides may only be more conservative).
*   **Partition choice is design-only.** `choose_partition` picks the
    candidate that maximises the size-adjusted effective sample size
    `N_eff = G / (κ · r_eff)`, where `r_eff = (M_c^default − 1)/(m_item − 1)`
    for `m_item > 1` and `r_eff = n_max/m̄` for `m_item = 1`. Ties break on a
    fixed order (higher `N_eff`, then `G_t ≥ 2`, then fewer clusters, then
    name). The choice depends only on the design and the declarations, never
    on losses (Proposition S5), so it is a data-independent selector in the
    sense of Lemma C and §6.4.
*   **Four κ routes.** The caller must state whether the observed items form
    a complete node × time grid (`is_complete_grid`, a required argument;
    `SpaceTimeIndex.is_complete_grid` holds the value). The route then
    follows from the κ function and the grid:
    *   `"range"`: a range-envelope κ function (§12.2) is used as given, on
        any grid (Claim RE1 in
        [THEORY_RANGE_ENVELOPE.md](THEORY_RANGE_ENVELOPE.md)).
    *   `"separable"`: on a complete grid, a separable declaration gives
        `κ = κ_s · κ_t` for product partitions, `κ_s` for community-only
        partitions (`G_t = 1`) and `κ_t` for window-only partitions
        (`G_s = 1`) (Claim S1).
    *   `"topological"`: a topological declaration gives κ from the
        cut-matrix eigenvalue bound on any grid (Claim S2B).
    *   `"fallback_topological"`: on an incomplete grid a separable
        declaration is not argued mathematically, so `choose_partition` requires a
        `fallback_kappa_fn` and raises `ValueError` without one.

### 12.2 `stats.range_envelope`: κ from model and data ranges

*   **Declaration.** `RangeDeclaration` takes the model's message-passing hops
    `K` (`k_hops`), lookback `L`, horizon `H`, the data's noise ranges `R_s`
    and `R_t`, and an out-of-range correlation budget `γ` (`gamma`, required,
    no default). In this subsection `K`, `L` and `H` are model parameters: `K`
    is not the sketch's top-`K` count (§8), and `L` is not the correlation
    length of the benchmarks.
*   **Bounding box.** Beyond the budget `γ`, two losses can be correlated only
    within `2K + R_s` hops and `L + H + R_t` steps (Lemma RE1 in
    [THEORY_RANGE_ENVELOPE.md](THEORY_RANGE_ENVELOPE.md)).
*   **κ bound.** `range_kappa` returns `κ ≤ 1 + λ_max(A_cut) + γ` (Claim RE1
    in [THEORY_RANGE_ENVELOPE.md](THEORY_RANGE_ENVELOPE.md)), where `A_cut`
    joins in-range item pairs in different clusters. `method="kronecker"`
    (default) uses a matrix-free operator for product partitions;
    `method="direct"` materialises item-level range edges (at most
    `max_edges = 20,000,000`) and uses `graph.topological_kappa`.
*   **Why this fits the design.** κ comes from declarations and topology
    only, never from the evaluated losses, which keeps Lemma C (§6.4).
    `range_candidates` and `range_kappa_fn` connect the declaration to
    `spacetime.choose_partition`.

## References

The real-graph topology benchmarks in [EVIDENCE.md](EVIDENCE.md) use these
datasets:

- **GraphLand** (`hm-prices`, `avazu-ctr`, `city-roads-L`).
  Gleb Bazhenov, Oleg Platonov, and Liudmila Prokhorenkova. *GraphLand:
  Evaluating graph machine learning models on diverse industrial data.* In The
  Thirty-Ninth Annual Conference on Neural Information Processing Systems
  (Datasets and Benchmarks Track), 2025.
- **Open Graph Benchmark** (`ogbn-arxiv`). Weihua Hu, Matthias
  Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and
  Jure Leskovec. *Open Graph Benchmark: Datasets for Machine Learning on
  Graphs.* In Advances in Neural Information Processing Systems 33 (NeurIPS),
  2020.

Methods named in the text:

- Batch means for MCMC output analysis (§4).
- Ugander and Backstrom, balanced label propagation (§7).
- Meyerhenke, Sanders and Schulz, "Partitioning Complex Networks via
  Size-constrained Clustering", SEA 2014 (background of the partitioner;
  `partitioners.py` docstring).
- de Jong (1987), central limit theorem for quadratic forms (the edge test
  described in §10; `refutation.py` docstring).

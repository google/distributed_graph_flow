# Theory: confidence intervals for correlated losses from cluster and window totals

> These arguments have not been peer-reviewed; see [EVIDENCE.md](./EVIDENCE.md)
> for simulation checks.

## Status of claims, lemmas, and propositions

Each result in this document carries one of four statuses:

*   **argued, checked by simulation** (with the corresponding test or benchmark
    in parentheses; see [EVIDENCE.md](./EVIDENCE.md));
*   **argued only**;
*   **approximation** (first-order or asymptotic);
*   **cited** (from the literature, with reference).

| Label | One-line statement | Section | Status |
|---|---|---|---|
| Lemma V | Variance of the sample mean of cluster totals under `P1_c(κ)` and `(P2v)` | §2.3 | argued only |
| Lemma R | Conditional coverage on design $D$ when the partition and check adjacency are functions of $D$ alone | §2.4 | argued only |
| Claim 4 | Finite-sample Cantelli coverage of the cluster interval $[\hat\theta/(1+c), \hat\theta/(1-c)]$ under `(P2v)` and `P1_c(κ)` | §3 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md) §3.1 blocks T1–T3, G1; `benchmark/coverage_dependent_test.py`, `cluster_bound_test.py`) |
| Lemma I | Per-cluster second-moment bound $\mathbb{E}[S_c^2] \le M\mu_c^2$ under a per-cluster item premise | §4 | argued only |
| Lemma I′ | Pooled item premise `P2*_pool(M)` implies `(P2v)` with $M_c - 1 = r(M - 1)$ and `P2*_c(M_c)` at $M_c^{\text{default}}$ for arbitrary per-item means | §4 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md) §3.1; `benchmark/coverage_dependent_test.py`, `cluster_sketch_test.py`, `benchmark/coverage_dependent_truth_test.py`) |
| Lemma W | Midpoint quantile assignment of pre-aggregated time buckets to $G$ windows bounds count imbalance by $|n_j - \bar m| < b_{\max}$ and $M_c \le (1 + b_{\max}/\bar m)M$ | §5 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md) §4 block V2; `benchmark/validation_r2_buckets_test.py`, `temporal_test.py`) |
| Claim 5 | Per-side miss probability bound $\Pr(\text{side miss}) \le \rho\alpha' / (1 - \alpha' + \rho\alpha') \le \rho\alpha'$ when true variance exceeds declared variance by factor $\rho$ | §6 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md) §3.1; `benchmark/coverage_dependent_test.py`, `benchmark/coverage_dependent_truth_test.py`) |
| Lemma T′ | Topological bound $\kappa_T = 1 + \lambda_{\max}(\Phi^{\mathrm{cut}}) \le 1 + \beta_{\max}$ under cross-cluster correlation dominance `(T1)` and within-cluster non-repulsion `(T2)` | §7 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md) §4 block V3; `benchmark/validation_r2_buckets_test.py`, `graph_test.py`) |
| Gebelein–Lancaster remark | Residual-to-loss correlation bound $|\mathrm{Corr}(f(X), g(Y))| \le |\rho|$ (and $\le \rho^2$ for centred even functions) for Gaussian residuals | §7 | cited (Gebelein, 1941; Lancaster, 1957) |
| Claim 7 | Finite-sample coverage of the correlated $R^2$ interval from cluster totals without pairing, under `(P2v)`, `P1_c(κ)`, and `P1_c(κ_y)` | §8 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md) §4 blocks V1a–V1b; `benchmark/validation_r2_buckets_test.py`, `benchmark/theory_sims_test.py`, `cluster_bound_test.py`) |
| Lemma J | Decomposition of the raw-total lag-1 autocorrelation into noise lag-1 autocorrelation and count-imbalance bias $\tilde\rho\rho_\delta$ | §9.3 | argued, checked by simulation (`benchmark/theory_sims_test.py`; [EVIDENCE.md](./EVIDENCE.md) §3.1 block T2b, `benchmark/coverage_dependent_test.py`) |
| Lemma J′ | Negative lag-1 autocorrelation $\rho_\delta = -f/(1-f) \le 0$ of the internal equal-count window sizes `(i·G)//n` | §9.3 | argued, checked by simulation (`benchmark/theory_sims_test.py`) |
| Lemma K(1) | Residualised totals $e_c = S_c - n_c\hat\theta$ remove the deterministic count component identically and have first-order null mean $-1/G + O(1/G^2)$ | §9.3 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md) §3.1 block T2 and §4 block V2; `benchmark/coverage_dependent_test.py`, `benchmark/validation_r2_buckets_test.py`, `benchmark/theory_sims_test.py`, `temporal_test.py`) |
| Lemma K(2) | Asymptotic null variance $h_\sigma / G$ of $r_1(e)$ under non-dominance and bounded fourth moments | §9.3 | approximation |
| Claim 6(a), (c), (d) | Exact null mean and variance identities for $\sum_E \zeta_c\zeta_d$ and $\sum_E z_c z_d = \zeta^\top A\zeta$ with rank-1 centring correction | §9.4 | argued only |
| Claim 6(b), (e) | Asymptotic $\mathcal{N}(0, 1)$ null distribution and asymptotic level $1 - \text{level}$ of the self-normalised graph edge test $T$ under `(E1)–(E4)` (via de Jong, 1987) | §9.4 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md) §3.1 block G1 and §4 block V2; `benchmark/coverage_dependent_test.py`, `benchmark/validation_r2_buckets_test.py`, `refutation_test.py`, `benchmark/theory_sims_test.py`) |
| Proposition B | First-order expectation $\mathbb{E}[\hat\kappa_B] \approx \mathrm{VIF} - 2\sum_{E_{\mathrm{cut}}} \mathrm{Cov} / \sum \mathrm{Var}$ of the super-batch estimator under $W$-local dependence | §9.5 | approximation |
| Proposition D | False-warning probability of the split-half temporal drift check is at most $\alpha$ when both halves share the same mean and satisfy Claim 4's premises | §10 | argued, checked by simulation ([EVIDENCE.md](./EVIDENCE.md) §3.1 blocks T1, T3; `benchmark/coverage_dependent_test.py`, `temporal_test.py`) |

---

**What this document argues.** The library issues a confidence interval for
the average expected loss of a model over a fixed set of evaluated items whose
losses are correlated: nodes of a graph, or items observed over time. It groups
the items into disjoint clusters (graph communities or contiguous time windows),
sums the losses within each cluster, and applies a one-sided Cantelli bound to
the cluster totals. This document argues that the interval covers the target
with the stated probability under two declared premises: a bound `M_c` on the
relative second moment of the totals, and a bound `κ` on their variance
inflation factor (Claim 4). It shows how to derive `M_c` from an item-level
declaration (Lemma I′), how much coverage degrades when a declaration is wrong
(Claim 5), how to derive `κ` from a graph and a declared edge correlation
bound (Lemma T′), how to bound R² without pairing (Claim 7), and how to build
windows from pre-aggregated time buckets (Lemma W). It also states what each
refutation check controls and what it does not (Claim 6, Proposition B,
Lemmas J, J′, K, Proposition D). The checks never change an interval.

**Who it is for.** Statisticians and ML engineers who want to know exactly what
an interval certifies before relying on it, and reviewers who need to check
the arguments. It assumes familiarity with Cantelli's inequality and basic
variance algebra.

**Scope.** This document covers the cluster, temporal, time-bucket and graph
entry points of the library, for the metrics MAE,
MSE, RMSE, accuracy and R². Space-time product partitions are covered in
[THEORY_SPACETIME.md](THEORY_SPACETIME.md), and range-envelope derivations of
κ in [THEORY_RANGE_ENVELOPE.md](THEORY_RANGE_ENVELOPE.md). The design
rationale is in [DESIGN.md](DESIGN.md); where the two disagree, this document
is normative. The single-sample results it builds on (Lemma A′, Claim 1′,
Lemma D and others) are in [THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md); the one that every
argument here uses is restated in Section 1.

**How each result is presented.** Every branch that issues an interval states
its claim, its premises, its rate, and why the bound is tight. Checks that
only warn or refute state what they control and what they do not. Section 12
summarises the empirical check-size measurements not covered by the table above.

---

## Notation

All symbols used in more than one section are listed here. Symbols local to an
argument are defined where they appear.

| Symbol / term | Definition |
|---|---|
| $n$ | Total number of evaluated items ($n = \sum_{c=1}^G n_c$). |
| $G$ | Number of disjoint clusters or contiguous time windows ($c \in \{1, \dots, G\}$). |
| $n_c,\; \bar m,\; n_{\max}$ | Item count $n_c$ in cluster $c$, mean cluster size $\bar m := n/G$, and maximum cluster size $n_{\max} := \max_c n_c$. |
| $r,\; \mathrm{CV}_n^2$ | Cluster-size ratio $r := n_{\max}/\bar m \ge 1$ and squared coefficient of variation of counts $\mathrm{CV}_n^2 := \frac{G\sum_c n_c^2}{n^2} - 1$ (Section 4). |
| $s_i \ge 0,\; \sigma_i^2$ | Non-negative per-item loss summand ($|e_i|$ for MAE, $e_i^2$ for MSE/RMSE, $\mathbf{1}\{\text{correct}_i\}$ for accuracy; see Section 8 for $R^2$), and its variance $\sigma_i^2 := \mathrm{Var}(s_i)$. |
| $S_c,\; \bar S$ | Cluster total $S_c := \sum_{i \in c} s_i$ and sample mean of cluster totals $\bar S := \frac{1}{G}\sum_{c=1}^G S_c$. |
| $\mu_c,\; \theta,\; \bar\theta_c$ | Expected cluster total $\mu_c := \mathbb{E}[S_c]$, pooled item mean $\theta := \frac{1}{n}\sum_{i=1}^n \mathbb{E}[s_i]$, and mean expected cluster total $\bar\theta_c := \frac{1}{G}\sum_{c=1}^G \mu_c = \bar m \theta$. |
| $\hat\theta$ | Sample pooled mean $\hat\theta := \sum_c S_c / n$. |
| $\alpha,\; \alpha'$ | Two-sided failure probability $\alpha \in (0, 1)$ (target coverage $1 - \alpha$) and per-side budget $\alpha' := \alpha/2$. |
| $c(\alpha', M_c, \kappa)$ | One-sided Cantelli constant (Section 1). |
| $n_A(M_c, \alpha)$ | Existence threshold $(M_c - 1)(1 - \alpha')/\alpha'$ ([THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md#59-the-sample-size-thresholds)). |
| $M$ | Declared item-level pooled relative second moment, $\frac{1}{n}\sum_{i=1}^n \mathbb{E}[s_i^2] \le M\theta^2$ (premise `P2*_pool(M)`). |
| $M_c,\; M_c^{\text{default}}$ | Declared cluster-level relative second moment ($M_c \ge 1$) and its Lemma I′ default computed from $(M, r, \mathrm{CV}_n)$ (Section 4). |
| $\hat M_G$ | Empirical cluster-level relative second moment, $\hat M_G := G\sum_c S_c^2 / (\sum_c S_c)^2$, used by the status check (Section 2). |
| $\kappa,\; \mathrm{VIF}$ | Declared between-cluster variance inflation factor ($\kappa \ge 1$, default $1$) and true $\mathrm{VIF} := \mathrm{Var}(\sum_c S_c) / \sum_c \mathrm{Var}(S_c)$. |
| $\kappa_y$ | Declared between-cluster VIF of the label totals $Y_c = \sum_{i \in c} y_i$ (`kappa_labels` in code, default $\kappa$; Section 8). |
| $e_c,\; z_c$ | Residualised cluster total $e_c := S_c - n_c \hat\theta$ and count-standardised residual $z_c := e_c / \sqrt{n_c}$ (Section 9). |
| $\varepsilon_c,\; \zeta_c,\; \tau_c^2$ | Noise of a total in the no-drift model $S_c = n_c\theta + \varepsilon_c$, its standardised form $\zeta_c := \varepsilon_c/\sqrt{n_c}$, and $\tau_c^2 := \mathrm{Var}(\zeta_c)$ (Section 9). |
| $E,\; \lvert E\rvert,\; W,\; \bar d$ | Cluster adjacency (edge set) used by a check, its number of edges, its symmetric 0/1 adjacency matrix, and mean cluster degree $\bar d := 2\lvert E\rvert/G$ (Section 9.4). |
| $\Phi,\; \Phi^{\text{cut}},\; \kappa_T$ | Declared item-pair correlation bound matrix $\Phi$, its cross-cluster restriction $\Phi^{\text{cut}}$, and topological bound $\kappa_T := 1 + \lambda_{\max}(\Phi^{\text{cut}})$ (Section 7). |
| $\lambda_{\max},\; \rho(\cdot)$ | Largest eigenvalue and spectral radius of a matrix. |
| $D$ | Design tuple: item set, timestamps, partition, adjacency, bucket counts (Section 2). |
| $b_k,\; b_{\max}$ | Count of time bucket $k$ and largest bucket count (Section 5). |
| `V1`–`V6` | Known limitation identifiers catalogued in Section 11. |

## Premises at a glance

Each premise is stated in full where it is first used. The one-line meanings
are:

| Premise | Meaning | Used by | Checked by |
|---|---|---|---|
| `P2*_pool(M)` | Items' pooled second moment is at most `M` times the squared mean. | Lemma I′ (default `M_c`) | Declared out of sample. |
| `P2*_c(M_c)` | Cluster totals' pooled second moment is at most `M_c θ̄_c²`. | Status threshold | `M̂_G > M_c` gives `TAIL_UNRESOLVED`. |
| `(P2v)` | Sum of cluster-total variances is at most `G(M_c − 1)θ̄_c²`. | Coverage (Lemma V, Claims 4 and 7) | Implied by `P2*_c(M_c)`. |
| `P1_c(κ)` | Variance of the sum of totals is at most `κ` times the sum of their variances. | Coverage (Lemma V, Claims 4 and 7) | Refutation checks (Section 9); never verified. |
| `P1_c(κ_y)` | Same as `P1_c(κ)`, for the label totals `Y_c`. | Claim 7 | Check on `Y_c` (Section 8). |
| `(T1)`, `(T2)` | Cross-cluster covariances are bounded by a declared `Φ`; no net negative within-cluster covariance. | Lemma T′ | `(T2)` is not checkable on one draw. |
| Design exogeneity | Partition, timestamps, buckets and adjacency do not depend on the losses or labels. | Every result (Section 2, Lemma R) | Not checkable from the losses. |
| Independent totals | The noise terms of the totals are independent (stronger than `P1_c(1)`). | Calibration of the κ = 1 checks only | Not needed for coverage. |

---

## 1. Background: the Cantelli bound for correlated totals

This section restates, for self-containment, the result of
[THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md) (Lemma A′ / Claim 1′) that every interval in
this document reduces to.

**Recap of Lemma A′ / Claim 1′ (Cantelli under dependence).** For any
collection of $G$ non-negative cluster totals $S_1, \dots, S_G \ge 0$ with sample
mean $\bar S = \frac{1}{G}\sum_c S_c$, expected mean $\bar\theta_c = \mathbb{E}[\bar S]$,
and variance bound $\mathrm{Var}(\bar S) \le \frac{(M_c - 1)\bar\theta_c^2}{G/\kappa}$,
define the one-sided Cantelli constant at side level $\alpha' = \alpha/2$:

$$c(\alpha', M_c, \kappa) \;:=\; \sqrt{\frac{(M_c - 1)\,\kappa\,(1 - \alpha')}{\alpha'\, G}}.$$

Then Cantelli's inequality gives $\Pr\bigl(\bar\theta_c < \bar S / (1 + c)\bigr) \le \alpha'$
unconditionally, and $\Pr\bigl(\bar\theta_c > \bar S / (1 - c)\bigr) \le \alpha'$
whenever $c < 1$ (if $c \ge 1$, the library refuses with `ASSUMPTION_REQUIRED`).

In independent terms, this is the correlated path (Algorithm 1 with
`effective_n = G/κ`), run on the `G` totals in place of the `n` items.

---

## 2. Premises, the variance bound, and conditioning on the design

### 2.1 The estimand

The library bounds the **average expected loss over the evaluated items** when their
losses are correlated: nodes of a graph, or items of a time window. Formally,
conditional on the design $D$ (the evaluated item set, timestamps, and cluster
partition), the target estimand is $\theta = \frac{1}{n}\sum_{i=1}^n \mathbb{E}[s_i \mid D]$
(referred to as estimand `E2` in [DESIGN.md](DESIGN.md) decision 7). It
says nothing about items outside that set, and in particular nothing about
future items when the distribution drifts; the drift warning (Section 10) can
refute stationarity but never certifies it.

The items are grouped into `G` disjoint **clusters** `c` with `n_c` items
each (`Σ_c n_c = n`): contiguous time windows on the temporal path, the
output of the partitioner on the graph path. For a metric with per-item
summand `s_i ≥ 0` the **cluster total** is `S_c = Σ_{i∈c} s_i`, with
`μ_c = E[S_c]`, `θ̄_c = (1/G)Σ_c μ_c = nθ/G` and `θ = (1/n)Σ_i E[s_i]`.

### 2.2 The two declared premises

**Premise P2\*_c(M_c) (cluster-level second moment).** The cluster totals
satisfy the pooled second-moment premise of [THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md):

$$\tfrac1G\textstyle\sum_c E[S_c^2] \;\le\; M_c\,\bar\theta_c^{\,2}.$$

`M_c` is declared (default in Section 4). It is refutable:
`M̂_G = G·ΣS_c²/(ΣS_c)²` estimates the left side divided by `θ̄_c²`
(Proposition F in [THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md)), and `M̂_G > M_c` is
reported as `TAIL_UNRESOLVED`.

**Premise P1_c(κ) (between-cluster dependence).** The declaration, with `κ ≥ 1`
(default 1: uncorrelated totals), is

$$\mathrm{Var}\big(\textstyle\sum_c S_c\big) \;\le\; \kappa \textstyle\sum_c \mathrm{Var}(S_c).$$

When `Σ_c Var(S_c) > 0` this says that the variance inflation factor
`VIF = Var(ΣS_c)/ΣVar(S_c)` is at most κ; the multiplicative form avoids
`0/0` for constant totals. A correlation row-sum bound
`max_c Σ_d |ρ_cd| ≤ κ` (with `ρ_cd := 0` when `Var(S_c)Var(S_d) = 0`) is a
sufficient condition, not the definition. Premise P1_c(κ) is **not**
estimable from `M̂_G`, which depends only on marginals ([DESIGN.md](DESIGN.md)
§4). The checks of Section 9 test it; they do not verify it. Section 7 shows
how to derive κ from a graph instead of declaring it directly.

### 2.3 Lemma V — the variance bound

**Lemma V (the variance bound).** Suppose premise P1_c(κ) (between-cluster
dependence) and premise `(P2v)` (cluster-level variance bound):

$$\textstyle\sum_c \mathrm{Var}(S_c) \;\le\; G\,(M_c - 1)\,\bar\theta_c^{\,2}. \tag{P2v}$$

Then

$$\mathrm{Var}(\bar S) \;\le\; \frac{(M_c - 1)\,\bar\theta_c^{\,2}}{G/\kappa}, \qquad \bar S = \tfrac1G\textstyle\sum_c S_c .$$

Premise P2\*_c(M_c) implies premise (P2v): `Σ_c Var(S_c) = Σ_c E[S_c²] − Σ_c μ_c² ≤
G M_c θ̄_c² − G θ̄_c²`, using Cauchy–Schwarz (`Σ_c μ_c² ≥ G θ̄_c²`).

*Argument.* `G²Var(S̄) = Var(ΣS_c) ≤ κ Σ_c Var(S_c)`; apply (P2v). ∎

So the totals satisfy premise P1′(`n_eff`) (variance bound for
correlated data) with `n_eff = G/κ` and the variance part of premise
P2\*_pool with `M_c`, **without** any assumption of independence, identical
distribution, equal cluster sizes or equal cluster means. Only premise (P2v)
enters the coverage argument; premise P2\*_c is what the status check tests
(Section 4 explains why the default `M_c` must satisfy both).

### 2.4 Conditioning on the design and Lemma R

**What the probabilities are over.** All probabilities in this document are
conditional on the design: the item set, the partition into clusters
(`cluster_ids`, or windows from `timestamps` or bucket counts), and any
adjacency passed to a check (`cluster_edges`, `edges`, `batch_ids`). The
claims hold for that fixed design, and they require it to be chosen
without looking at the losses. Ways this fails in practice:

*   timestamps, retries or deduplication that depend on the outcome (for
    example, failed requests retried into a later window);
*   clusters, windows or buckets whose sizes correlate with the errors;
*   a partition, edge list or batch assignment tuned on the residuals;
*   for R² (Section 8), a partition stratified by the labels `y`.

> **Lemma R (conditioning on the design).** Let `D` collect the design
> variables: item indices, timestamps, bucket counts, and any graph or
> cluster adjacency. Take a partition (and any check adjacency) that is a
> function of `D` only. Conditionally on `D`, Claims 4 and 7 hold for the
> conditional estimand `θ(D) = (1/n)Σ_i E[s_i | D]`, provided their premises
> hold under the conditional law `P(· | D)`.

*Argument.*

1.  Under the regular conditional distribution `P(· | D)`, `D` and any
    partition computed from `D` alone are fixed constants.
2.  Any branch choice computed from `D` and the declarations alone (for
    example the issuance condition `c < 1`, whose inputs `G`, `κ`, `α` and the
    default `M_c` depend only on the counts) is also fixed under
    `P(· | D)`. By Lemma C in [THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md) §5.3, such a
    selection costs no union-bound penalty.
3.  Applying Claims 4 and 7 directly to the conditional law `P(· | D)`,
    with conditional expectations `μ_c(D) = E[S_c | D]` and variances
    `Var(S_c | D)`, yields the stated coverage for `θ(D)`. ∎

The lemma always holds formally. What can fail is its meaning. Suppose `D`
is selected on the outcome, for example failed requests retried into a later
bucket. Then `θ(D)` is a shifted estimand, not the one the user wants,
and the conditional premises can differ from the unconditional ones. So the
real requirement is on the design itself: timestamps and counts must not
depend on the losses.

---

## 3. Claim 4 — the cluster interval

> **Claim 4.** Let `α ∈ (0,1)` and assume premise (P2v) (cluster-level variance bound) with `M_c` and premise P1_c(κ) (between-cluster dependence). Run the
> correlated path (Algorithm 1 with `effective_n = G/κ`, i.e. Lemma A′ on both
> sides) on the `G` totals with `M = M_c`, obtaining `[L, U]` for `θ̄_c`. Then
> `[L/m̄, U/m̄]` with `m̄ = n/G` satisfies
>
> $$\Pr(\theta < L/\bar m) \le \tfrac{\alpha}{2}, \qquad \Pr(\theta > U/\bar m) \le \tfrac{\alpha}{2}.$$
>
> RMSE (square root) and the accuracy clamp at 1 transfer as in
> Corollaries 1 and 2 ([THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md)).

*Argument.*

1.  The correlated path is Cantelli on each side with
    `c² = (M_c−1)κ(1−α′)/(α′G)` (Section 1). Its argument uses only the
    variance bound of Lemma V, not the uncentred second moment. So
    Claim 1′ applies to the totals.
2.  The branch `ΣS_c = 0 → [0, 0]` adds no miss: for `θ > 0` and `c < 1`
    the event `S̄ = 0` lies inside the upper-side Cantelli event, and for
    `θ = 0` the interval `[0, 0]` covers.
3.  `θ̄_c = m̄ θ` with `m̄` known, so dividing by `m̄` is a fixed monotone map
    (Lemma D in [THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md)). ∎

*   **Premises.** (P2v) and P1_c(κ) only. The status is a verdict on the
    stronger P2\*_c(M_c) ([THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md) §3); κ has its own
    checks (Section 9).
*   **Rate.** The Cantelli constant is `c² = (M_c−1)κ(1−α′)/(α′G)`. With a
    stationary per-item law, window size `b`, within-window VIF `v_b` and
    `M_c − 1 ≈ (M−1)v_b/b`, this is `(M−1)·v_b·κ·(1−α′)/(α′ n)`: item-level
    Cantelli with `n_eff = n/(v_b κ)` ([DESIGN.md](DESIGN.md) decision 12).
    Clustering buys refutability, not width.
*   **Tightness.** Each side is Cantelli's inequality, which is attained by a
    two-point law of the mean of the totals (limitation L10 in
    [THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md)). The only other slack is the declared
    `M_c` against the true pooled ratio, which the status reports.
*   **Existence threshold.** The interval is issued only if `c < 1`, i.e.
    `G/κ > n_A(M_c, α)` (where $n_A(M_c, \alpha) := (M_c - 1)(1 - \alpha')/\alpha'$; see [THEORY_INDEPENDENT.md](THEORY_INDEPENDENT.md#59-the-sample-size-thresholds)). With the default `M_c` this is the binding
    constraint at realistic sizes ([DESIGN.md](DESIGN.md) decision 8).

---

## 4. The default `M_c`: Lemmas I and I′

The user declares the item-level `M`: a **pooled** premise over
all items, `(1/n)Σ_i E[s_i²] ≤ Mθ²`. The default `M_c` must follow from that
premise alone.

**Lemma I (per cluster).** If the items of cluster `c` satisfy the pooled
premise with the item-level `M` and the cluster's own mean
(`(1/n_c)Σ_{i∈c}E[s_i²] ≤ M·(μ_c/n_c)²`), then `E[S_c²] ≤ M·μ_c²`, for any
correlation inside the cluster, since `S_c² ≤ n_c Σ_{i∈c} s_i²` by
Cauchy–Schwarz. ([DESIGN.md](DESIGN.md) §4 proves the same conclusion,
written `Var(S_c)/μ_c² ≤ M − 1`, under the weaker condition
`(1/n_c)Σ_{i∈c}Var(s_i) ≤ (M − 1)(μ_c/n_c)²`; the worst case is perfect
correlation.) Pooling over clusters gives `(1/G)Σ_c E[S_c²] ≤ M·θ̄_c²·(1 +
CV_μ²)`, where `CV_μ` is the coefficient of variation of the cluster means
`μ_c`.

Lemma I's premise is per cluster, which is stronger than what the user
declares: it forbids concentrating the variance in some clusters. The
default therefore rests on the following lemma. Write `r = n_max/m̄` with
`n_max = max_c n_c`, and `CV_n² = G Σ_c n_c²/n² − 1` for the squared
coefficient of variation of the counts.

**Lemma I′ (pooled).** Under the pooled item premise with `M`, for any
per-item means and any correlation inside and between clusters:

1.  (variance, what coverage needs) `Σ_c Var(S_c) ≤ G·r(M − 1)·θ̄_c²`, i.e.
    (P2v) holds with `M_c − 1 = r(M − 1)`;
2.  (uncentred, what the status check tests) P2\*_c holds with
    $$M_c \;=\; \min\Big\{\, rM,\;\; 1 + r(M-1) + CV_n^2 + 2\,CV_n\sqrt{r(M-1)} \,\Big\}.$$

*Argument.*

1.  **Part (1).** By Cauchy–Schwarz, `Var(S_c) ≤ n_c Σ_{i∈c} Var(s_i)` for
    any correlation, so `Σ_c Var(S_c) ≤ n_max Σ_i Var(s_i)`. Also
    `Σ_i Var(s_i) = Σ_i E[s_i²] − Σ_i (E s_i)² ≤ nMθ² − nθ²`. With
    `n = G m̄` and `θ̄_c = m̄θ` this is part (1).
2.  **Part (2), first branch.** Pointwise, `S_c² ≤ n_c Σ_{i∈c} s_i²`, so
    `Σ_c E[S_c²] ≤ n_max Σ_i E[s_i²] ≤ n_max·nMθ²`; divide by `G θ̄_c²`.
3.  **Part (2), second branch.** Write the cluster means per item as
    `θ + δ_c` (`Σ_c n_c δ_c = 0`), and set `V = Σ_i Var(s_i)` and
    `D = Σ_c n_c δ_c²`. Then `V + D ≤ n(M−1)θ²`, because
    `Σ_i (E s_i)² ≥ Σ_c n_c(θ+δ_c)² = nθ² + D`. Next,
    `E[S_c²] ≤ n_c Σ_{i∈c}Var(s_i) + n_c²(θ+δ_c)²`, so
    `Σ_c E[S_c²] ≤ n_max V + θ²Σn_c² + 2θ Σ_c n_c(n_c − m̄)δ_c + n_max D`. By
    Cauchy–Schwarz the middle term is at most
    `2θ √(n_max Σ_c (n_c − m̄)²)·√D`. Put `V + D ≤ n(M−1)θ²` and divide by
    `G θ̄_c²`. ∎

(In step 3, `D` denotes the mean-dispersion sum, not the design.)

*Tightness.* The first branch is attained whenever the size-`n_max` clusters
can carry all of the mean: put `s_i = X` on `k` clusters of size `n_max` and
`s_i = 0` elsewhere. That needs `E X²/(E X)² = M k n_max/n ≥ 1`, i.e.
`k n_max ≥ n/M`. Example: sizes `(1, 3)` and `M ≥ 4/3`, with `X` on the
large cluster, give `M_c = 1.5M`. The second branch is smaller only for `M`
near 1 (for `r = 1.5` and `CV_n² = 0.05`, only when `M < 1.675`). At `M = 1`
it gives `1 + CV_n²`, the exact value for constant summands. When few
clusters have size `n_max`, neither branch need be attained. The gap is small
in the cases checked, about 1% at `r = 1.5`, `CV_n = 0.1`, `M = 4`.

**The default.** The library uses

$$M_c^{\text{default}} \;=\; \min\Big\{\, rM,\;\; 1 + r(M-1) + CV_n^2 + 2\,CV_n\sqrt{r(M-1)} \,\Big\},$$

which the sketch computes from `n_max` (merged by `max`) and `Σ_c n_c²`
(merged by sum). Equal counts give exactly `M`, so reusing the item-level `M`
unchanged is valid only for equal cluster sizes. For equal-count time
windows (`n_c` within ±1), `r ≤ 1 + 1/m̄`, so the default exceeds `M` by at
most `M/m̄`. For the built-in graph partitioner the size cap gives
`n_max ≤ ⌈1.5·cs⌉` (`cs` is the partitioner's target cluster size, so
`n_max/m̄ ≤ ⌈1.5·cs⌉/m̄`), and at most one cluster has size below `⌈cs/2⌉`. On the real test subgraphs `n_max/m̄`
measured 1.26–1.55 and `CV_n² ≤ 0.08` ([DESIGN.md](DESIGN.md) decision 20).

**Why the default must satisfy both parts.** One value serves two roles,
because the library uses a single `M` for both: the width constant `c`, which needs
only part (1), and the `TAIL_UNRESOLVED` threshold, which needs part (2). The
value `1 + r(M−1) + CV_n²` satisfies part (1), so its intervals cover, but it
does not satisfy part (2) and would flag valid data. Example: sizes
alternating `(1, 2)`, `s = 0` on the small clusters and `s = 1` on the large
ones, `M = 3/2`. The true `M_c` is 2, that value gives 1.78, and the status
check would report `TAIL_UNRESOLVED`. With the default above, the population
target `G Σ E S_c²/(Σ μ_c)²` never exceeds `M_c` under the item premise.
The sample `M̂_G` still fluctuates and can exceed it by chance, like any
test statistic.

Lemma I′ holds for any per-item means, so clusters with different accuracy
(θ varying over the graph or over time) are covered by the pooled `M`. They
need no separate allowance. In practice the default has wide margin,
because the worst case of Lemma I′ (perfect within-cluster correlation)
overstates `M_c − 1` by roughly the factor `b/v_b`.

---

## 5. Windows from time buckets: Lemma W

Some time series arrive pre-aggregated: one count and one loss total per time
bucket (for example, per hour), with no per-item records. The entry point
`temporal_interval_from_buckets` groups contiguous buckets into `G` windows.
Lemma W shows that the windows are nearly equal in size, so the Lemma I′
default stays close to `M`, and that the assignment is a valid design.

> **Lemma W (bucket-quantile windows).** Buckets `k = 1..K` arrive in time
> order with counts `b_k ≥ 1`. Let `C_k = Σ_{l≤k} b_l`, `C_0 = 0`,
> `n = C_K`, `b_max = max_k b_k` and `m̄ = n/G`. Assign bucket `k` to
> window `w(k) = ⌊G(C_{k−1} + b_k/2)/n⌋`. Then:
>
> 1. the windows are contiguous in time, and `w(k) ∈ {0, …, G−1}`;
> 2. every window count satisfies `|n_j − m̄| < b_max`;
> 3. if `m̄ ≥ b_max`, every window is nonempty;
> 4. `r = n_max/m̄ ≤ 1 + b_max/m̄`, `CV_n ≤ b_max/m̄`, and the Lemma I′
>    default satisfies `M_c ≤ (1 + b_max/m̄)M`.

*Argument.*

1.  **Contiguity.** The midpoints `m_k = C_{k−1} + b_k/2` increase strictly
    and lie in `(0, n)`. So `w` is nondecreasing with values in
    `{0, …, G−1}`.
2.  **Boundaries.** Window `j` begins at the first bucket `k*` with
    `m_{k*} ≥ jn/G`. Let `B_j = C_{k*−1}` be the count before it, with
    `B_j = n` if there is no such bucket, and `B_0 = 0`, `B_G = n`.
    *   From `m_{k*} ≥ jn/G`: `B_j ≥ jn/G − b_{k*}/2`.
    *   If `k* > 1`, bucket `k*−1` has `m_{k*−1} < jn/G`, so
        `B_j = m_{k*−1} + b_{k*−1}/2 < jn/G + b_max/2`.
    *   If `k* = 1`, then `B_j = 0 ≤ jn/G`.
    *   If there is no such bucket, `n = m_K + b_K/2 < jn/G + b_max/2`, and
        `n ≥ jn/G`.

    In every case `−b_max/2 ≤ B_j − jn/G < b_max/2`, and the ends `B_0`
    and `B_G` are exact.
3.  **Counts.** `n_j = B_{j+1} − B_j`, so `m̄ − b_max < n_j < m̄ + b_max`.
    This gives part (2), and part (3) because then `n_j > 0`.
4.  **Constants.** Part (4) follows from part (2): Lemma I′ is bounded by its
    first branch `rM`.
5.  **Design.** The assignment is deterministic in the counts, so Lemma R
    applies. ∎

*   **Implementation.** The library first drops buckets with `b_k = 0`
    (they carry no items and do not change any window count), then computes
    `w(k)` in integers as `⌊G(2C_{k−1} + b_k)/(2n)⌋`. It refuses bucket
    inputs with `m̄ < b_max`, which is exactly the complement of part (3)'s
    premise, and reports `r`. `m̄ = b_max` is allowed; it covers `K = G`
    equal buckets, one per window. Caller-side bucket partials (per-bucket
    counts and loss totals) merge exactly by addition per bucket id, and
    coarsening into windows happens at query time in
    `temporal_interval_from_buckets`.
*   **Checks and drift.** The κ checks on bucket windows are described in
    Section 9.7. The drift warning is Proposition D (Section 10), unchanged,
    splitting at window `⌊G/2⌋`.

---

## 6. Claim 5 — what a wrong declaration costs

The certificate of Claim 4 is conditional on the declarations. This
section bounds how fast it degrades when they are wrong. It is what makes
"declare, then check" coherent: small errors cost little by this claim,
and large errors are what the checks of Section 9 have power against.

> **Claim 5 (sensitivity).** Suppose Claim 4 is run with declarations
> `(M_c, κ)`, `M_c > 1`, on data with `θ > 0`, and the interval is issued
> (`c < 1`). Suppose the true variance of `S̄` is
> `Var(S̄) = ρ · (M_c − 1)κθ̄_c²/G` for some `ρ > 0`. (For instance
> `ρ = κ_true/κ` when only κ is wrong, or
> `ρ = κ_true(M_c,true − 1)/(κ(M_c − 1))` in general. `ρ` is not the size
> ratio `r = n_max/m̄` of Section 4.) Then each side misses
> with probability at most
>
> $$\frac{\rho\,\alpha'}{1 - \alpha' + \rho\,\alpha'} \;\le\; \rho\,\alpha', \qquad \alpha' = \alpha/2 .$$

*Argument.*

1.  The correlated path returns `[S̄/(1+c), S̄/(1−c)]` for `θ̄_c`, with
    `c² = (M_c−1)κ(1−α′)/(α′G)`.
2.  The lower side misses iff `S̄ > (1+c)θ̄_c`.
3.  By Cantelli, with `σ² = Var(S̄)`,
    `Pr(S̄ − θ̄_c ≥ cθ̄_c) ≤ σ²/(σ² + c²θ̄_c²)`. Substituting σ² and c² gives
    `ρ/(ρ + (1−α′)/α′)`.
4.  The upper side is the same with `S̄ < (1−c)θ̄_c`. ∎

*   **Reading.** A κ understated by 2× at most doubles the miss rate; at
    `α = 0.05` a 20× understatement still misses at most 34% per side. It
    never fails abruptly. This is worst case over laws: realised misses are
    far lower ([DESIGN.md](DESIGN.md) decision 15 measured `miss/α ≤ 0.27`
    for mild understatement before any check ran).
*   **Tightness.** Attained by the same two-point law as Cantelli.
*   **Combined with a check.** For any check,
    `Pr(miss ∧ not refuted) ≤ min{ρα′/(1−α′+ρα′), Pr(not refuted)}` per side.
    The first term controls small `ρ`; the second needs the check's power
    at large `ρ`, which Section 9 states only asymptotically. The
    finite-sample joint rate is evaluated by simulation in
    [EVIDENCE.md](./EVIDENCE.md) §3.1.

---

## 7. Deriving κ from the graph topology: Lemma T′

Instead of declaring κ directly, a user can declare how strongly the losses
of individual item pairs may correlate across clusters, and let the library
compute κ from the partition. In this section `σ_i² = Var(s_i)`.

> **Lemma T′.** Let `Φ` be a symmetric, non-negative n×n matrix, declared
> without looking at the evaluated losses. Let `Φ^cut` be `Φ` with every
> entry whose two items share a cluster set to 0, including the diagonal.
> Assume
>
> (T1) `|Cov(s_i, s_j)| ≤ Φ_ij σ_i σ_j` whenever `i` and `j` are in
> different clusters, and
>
> (T2) `Σ_c Var(S_c) ≥ Σ_i σ_i² > 0`.
>
> Then P1_c(κ_T) holds with
>
> $$\kappa_T \;=\; 1 + \lambda_{\max}(\Phi^{\text{cut}}) \;\le\; 1 + \beta_{\max}, \qquad \beta_{\max} = \max_i \textstyle\sum_j \Phi^{\text{cut}}_{ij}.$$

*Argument.*

1.  Write `X = Var(Σ_c S_c) − Σ_c Var(S_c)`. This is the sum of
    `Cov(s_i, s_j)` over ordered cross-cluster pairs. By (T1),
    `X ≤ Σ_{ij} Φ^cut_ij σ_i σ_j = σᵀΦ^cut σ`.
2.  `Φ^cut` is symmetric, so by the Rayleigh quotient
    `σᵀΦ^cut σ ≤ λ_max(Φ^cut)‖σ‖²`.
3.  `Φ^cut` is non-negative, so `λ_max(Φ^cut) = ρ(Φ^cut) ≥ 0`
    (Perron–Frobenius).
4.  If `X ≤ 0`, then `Var ΣS_c ≤ Σ_c Var S_c ≤ κ_T Σ_c Var S_c`.
5.  Otherwise, by (T2),
    `X / Σ_c Var S_c ≤ λ_max‖σ‖² / Σ_i σ_i² = λ_max`.
6.  The bound `ρ ≤ β_max` holds because the spectral radius is at most the
    maximum row sum. ∎

*   **Tightness.** Suppose `λ_min(Φ^cut) ≥ −1`. This holds whenever
    `λ_max(Φ^cut) ≤ 1`, because `λ_min ≥ −ρ`, and on some graphs well beyond
    that: `K_m` on singleton clusters has `λ_min = −φ`. Then
    `Σ = D_σ(I + Φ^cut)D_σ` is PSD for every `σ` (`D_σ` is the diagonal
    matrix of the `σ_i`).
    *   Take `σ = v ≥ 0`, the Perron vector. If `Φ^cut` is reducible, `v`
        may have zero entries. Those items then have zero variance, which is
        harmless.
    *   Such a Σ is realised by bounded summands: `s = a + Σ^{1/2}ξ` with
        Rademacher `ξ` and a large enough shift `a`.
    *   (T1) holds with equality on cross pairs, and within-cluster
        covariance is 0, so (T2) holds with equality.
    *   Then `VIF = (‖v‖² + vᵀΦ^cut v)/‖v‖² = 1 + λ_max` exactly.

    So `κ_T` cannot be improved from `Φ` and the partition alone whenever
    `λ_min(Φ^cut) ≥ −1`, and in particular whenever `κ_T ≤ 2`.
*   **Edge-local declaration.** Set `Φ_ij = φ_e` on the item edges and 0
    elsewhere. Then `Φ^cut = A_cut` is the weighted adjacency restricted to
    cut edges, and `κ_T = 1 + λ_max(A_cut)`.
    *   This declares that cross-cluster items **not** joined by an edge are
        uncorrelated, i.e. 1-dependence along the graph. That is a strong
        premise. Longer-range correlation must be declared by adding the
        corresponding pairs, for example 2-hop pairs, to `edges`.
    *   Let every `φ_e ≤ φ₁`, and let `d_i` be the cut degree. Then
        `φ₁·max(√d_max, 2|E_cut|/n) ≤ λ_max`. Both lower bounds need
        `φ_e = φ₁`. They come from the star subgraph (monotonicity of ρ) and
        from the all-ones Rayleigh quotient.
    *   Upper bounds: `λ_max ≤ φ₁ d_max` (row sums), and
        `λ_max ≤ φ₁ max_{ij ∈ E_cut} √(d_i d_j)`.

        *Argument for the second.* Restrict to the items with `d_i > 0`. The
        others form zero rows and columns and do not change ρ. Apply
        Collatz–Wielandt with `x_i = √d_i`:
        `(A_cut x)_i/x_i ≤ φ₁ Σ_{j∼i} √d_j/√d_i
        ≤ φ₁ √d_i max_{j∼i} √d_j`. ∎
    *   So a hub of cut degree `k` whose neighbours have cut degree at most
        `d` costs at most `φ₁√(kd)`, not the `φ₁k` that the row-sum bound
        `β_max` charges. With `d = 1` it costs exactly `φ₁√k`. When the
        neighbours are hubs too, the row-sum bound can be attained, for
        example by `K_{k,k}` across clusters.
*   **Numerics (never underestimate κ).**
    *   For any non-negative `A` and any `x > 0`,
        `ρ(A) ≤ max_i (Ax)_i/x_i` (Collatz–Wielandt). Argument: `ρ` is at
        most the ∞-norm of `D_x^{-1} A D_x`, which has the same spectrum as
        `A`.
    *   `graph.topological_kappa` evaluates this at `x = v + 10⁻³` and at
        `x = max(v, 10⁻¹²)` (where `v` is the shifted power iterate,
        normalised to maximum 1), takes the minimum of those bounds and the
        row-sum bound, and inflates the result by a relative
        `max(10⁻⁹, 4(d_max + 2)·ε₆₄)` against rounding, where `d_max` is the
        maximum cut row degree and `ε₆₄` the float64 machine epsilon.
    *   The result is an upper bound for every `x`. The iteration only
        makes it tight.
*   **Premises in practice.**
    *   (T1) is a statement about summands (for example `e_i²`), not about
        residuals. The remark below converts a residual-level bound `φ_res`
        into a summand bound. It needs each residual pair to be jointly
        bivariate Gaussian and the functions to have finite second moments.
        *   The bound is `φ_res` for any such functions.
        *   It is `φ_res²` for `e²` and `|e|` only if the residuals are
            also centred.
    *   (T2) is not verifiable on one draw. It fails only under net negative
        within-cluster covariance.
    *   `Φ` must come from outside the evaluated sample, for example from
        archive models or edge-level measurements on other models.
    *   The partition is a design variable (Section 2.4), so `κ_T` is a
        legitimate declaration once `Φ` is.

**Remark (from residual correlation to summand correlation; cited results).**
If each pair of residuals is bivariate Gaussian with `|ρ_ij| ≤ φ_res`,
Gebelein's inequality gives `|Corr(f(e_i), g(e_j))| ≤ φ_res` for any functions
(with finite second moments). If the residuals are also **centred**
(`E e_i = 0` for every item), then `e²` and `|e|` are even functions of a
centred Gaussian. Only even Hermite terms survive, and the bound becomes
`φ_res²` (Lancaster 1957). With a biased model (`E e_i = μ_i ≠ 0`), `e_i²`
has a first Hermite term `2μ_iσ_iZ_i` (here `σ_i` is the standard deviation
of the residual `e_i` and `Z_i` its standardised form). The correlation then
approaches `|ρ_ij|` as `|μ_i|/σ_i` grows, and only the unsquared bound holds.
So a residual-level spatial correlation measured on other models converts
into a summand-level φ, squared for MSE and MAE only under the
centred-Gaussian premise. For classification indicators only the unsquared
bound holds, and only under a Gaussian-copula premise.

---

## 8. Correlated R² without pairing: Claim 7

R² is a ratio of two means, so Claim 4 does not apply to it directly.
Claim 7 bounds the numerator and the denominator separately, each through
cluster totals, and combines them with a union bound.

**Setup.**

*   Predictions `ŷ_i` are fixed (conditioning, Section 2.4), and
    `e_i = y_i − ŷ_i`. `μ_i = E y_i` and `μ̄ = (1/n)Σ_i μ_i`.
*   `θ_A = (1/n)Σ_i E e_i²`, and
    `θ_B = (1/n)Σ_i E(y_i − μ̄)² = (1/n)Σ_i Var y_i + (1/n)Σ_i(μ_i − μ̄)²`.
*   The estimand is `R² = 1 − θ_A/θ_B`, with `θ_B > 0`.
*   Per cluster: `E2_c = Σ_{i∈c} e_i²`, `Y_c = Σ_{i∈c} y_i`, and
    `Q_c = Σ_{i∈c} q_i` with `q_i = (y_i − μ̄)²`. `Q_c` is unobservable.
*   `ȳ = ΣY_c/n` and `V̂ = (1/n)Σ_i(y_i − ȳ)²`.
*   `c(α′, M_c, κ) = √((M_c − 1)κ(1 − α′)/(α′G))`, the one-sided constant of
    Claim 4 at side level `α′`.

> **Claim 7.** Assume:
>
> *   premise (P2v) with `M_c^A` and premise P1_c(κ) for the squared-error totals `E2_c`;
> *   premise (P2v) with `M_c^q` and premise P1_c(κ) for the centred squared-label totals `Q_c`;
> *   premise P1_c(κ_y) for the label totals `Y_c`, where `κ_y ≥ 1` is the VIF of the label sequence (`kappa_labels` in code);
> *   `θ_B > 0`.
>
> Let `[L_A, U_A]` be the Claim 4 interval for `θ_A` at level `1 − α/2`,
> and
>
> $$L_B = \frac{\hat V}{1 + c(\alpha/4, M_c^q, \kappa)}, \qquad U_B = \frac{\hat V}{D},\quad D = 1 - c(\alpha/8, M_c^q, \kappa) - \frac{\kappa_y\, n_{\max}}{n\,\alpha/8},$$
>
> with `U_B = ∞` if `D ≤ 0` or `V̂ = 0`, and `U_A = ∞` if the Claim 4
> constant for `θ_A` is at least 1. Then, with probability at least `1 − α`,
>
> $$R^2 \in \Big[\,1 - \frac{U_A}{L_B},\; 1 - \frac{L_A}{U_B}\,\Big],$$
>
> with the conventions `U_A/0 = ∞` (lower endpoint `−∞`), also when
> `U_A = 0`, and `L_A/∞ = 0`.

(In Claim 7, `D` denotes the denominator margin above, not the design.)

*Argument.* We bound five miss events: three with probability at most `α/4`
each and two with probability at most `α/8` each. Outside their union,
`L_A ≤ θ_A ≤ U_A` and `L_B ≤ θ_B ≤ U_B`. Since all four bounds are
non-negative, `θ_A/θ_B ∈ [L_A/U_B, U_A/L_B]`.

1.  **The two A-sides.** Claim 4 at level `1 − α/2` gives `α/4` per side.
2.  **B lower side.**
    *   The `q_i` are non-negative summands with mean `θ_B`, so Claim 4
        applies to the `Q_c`.
    *   Its lower side gives `Pr(θ_B < q̄/(1 + c₁)) ≤ α/4`, with
        `c₁ = c(α/4, M_c^q, κ)` and `q̄ = (1/n)Σ_i q_i`. The Cantelli step
        does not need `c₁ < 1`.
    *   `q̄ = V̂ + (ȳ − μ̄)² ≥ V̂`, so `{θ_B < L_B} ⊆ {θ_B < q̄/(1+c₁)}`.
3.  **B upper side.**
    *   **Cantelli.** Let `c₂ = c(α/8, M_c^q, κ)`. Cantelli's lower tail on
        `q̄` gives `Pr(q̄ < (1 − c₂)θ_B) ≤ α/8`. The inequality inside is
        strict so that the case `M_c^q = 1` is covered: then `c₂ = 0`,
        `Var q̄ = 0`, `q̄ = θ_B` almost surely and the event is empty.
    *   **Bound on Var ȳ.** By P1_c(κ_y) and Cauchy–Schwarz within each
        cluster (`Var Y_c ≤ n_c Σ_{i∈c} Var y_i`),
        `Var ȳ ≤ κ_y Σ_c Var Y_c / n² ≤ κ_y n_max Σ_i Var y_i / n²
        ≤ κ_y n_max θ_B/n =: B`.
    *   **Chebyshev.** `B > 0` because `θ_B > 0`. With `δ = α/8`,
        Chebyshev gives `Pr((ȳ − μ̄)² ≥ B/δ) ≤ Var ȳ·δ/B ≤ δ`.
    *   **Combine.** Outside both events,
        `(1 − c₂)θ_B ≤ q̄ = V̂ + (ȳ − μ̄)² < V̂ + κ_y n_max θ_B/(nδ)`.
        That is, `D θ_B < V̂`.
        *   If `D > 0`, this gives `θ_B < U_B`.
        *   If `D ≤ 0`, then `U_B = ∞`.
4.  **Union bound.** The events have total probability at most
    `α/4 + α/4 + α/4 + (α/8 + α/8) = α`. ∎

**What the claim is about.** The interval covers the population R² of the
evaluated items defined by `θ_A` and `θ_B` above, with `θ_B` the label
variance around the pooled mean `μ̄`. It is not about R² on other items, and it
does not pair errors with labels.

**Remarks.**

*   **`V̂ = 0`.** All labels are equal, so `L_B = 0`, `U_B = ∞` by
    convention, and the interval is `(−∞, 1]`.
    *   The convention costs nothing. If `D > 0`, the good event would need
        `θ_B < V̂/D = 0`, which is impossible, so a miss event has already
        occurred.
    *   In floating point, the shard formula for `V̂` can come out slightly
        negative. The library clamps it at 0.
*   **Refusal.** For consistency with the independent estimator, the interval is refused
    (`ASSUMPTION_REQUIRED`) if `c(α/4, ·)` is at least 1 for either
    sequence. The claim does not need this, given the convention
    `U_A = ∞`. A refusal issues no interval, so it cannot cause a miss.
*   **Where the premise sits.** The default `M_c^q` is Lemma I′ applied to
    the pooled item premise `(1/n)Σ E q_i² ≤ M_q θ_B²`, which is centred at
    `μ̄`. For identically distributed labels, `M_q` is the label kurtosis.
    The default is `M_q = M = 16`, the R² row of the metric table.
*   **Declarations and checks.** One κ is declared for `E2` and `Q`, and
    `κ_y` defaults to κ. Each of the three total sequences (`E2_c`,
    `Q̂_c`, `Y_c`) gets its own check.
    *   The plug-in `Q̂_c = Σ_{i∈c}(y_i − ȳ)²` is used only for checks and
        for `M̂_G`. It never enters the interval.
    *   Reporting "refuted if any of three refutes" raises the
        false-refutation rate to at most three times the level. It never
        affects coverage.
*   **Shards.** An R² cluster shard stores `(E2_c, Y_c, Y2_c, n_c)`, with
    `Y_c = Σ_{i∈c}(y_i − ref)` and `Y2_c = Σ_{i∈c}(y_i − ref)²` shifted by a
    scalar reference `ref` (default: the shard label mean) for numerical
    precision. Because `V̂` and `Q̂_c` are shift-invariant,
    `V̂ = ΣY2_c/n − (ΣY_c/n)²` and `Q̂_c = Y2_c − 2(ΣY_d/n)Y_c + n_c(ΣY_d/n)²`
    hold for any `ref`, and merging shards with different `ref` values is
    algebraically exact via the parallel-axis shift.
*   **Rate.** The lower R² endpoint uses only Cantelli terms. The Chebyshev
    term is `κ_y r/(Gδ)` with `r = n_max/m̄`, of order `1/G` against
    Cantelli's `1/√G`. It affects only the upper endpoint.
*   **Two thresholds.** Both need `G/κ` large. The figures below take
    `M_c = 16` (the MSE and R² rows of the metric table), `α = 0.05` and
    `κ = κ_y = r = 1`. With the Lemma I′ default they scale roughly with `r`.
    *   **Issuance** needs `c(α/4) < 1`, i.e. `G/κ > 15·79 = 1185`. MSE,
        at `α/2` per side, needs `G/κ > 15·39 = 585`.
    *   **A finite upper side** (`U_B < ∞`, so an upper R² endpoint below 1)
        needs `D > 0`: `√(2385/G) + 160/G < 1`, i.e. `G ≳ 2700`.
    *   For `1185 < G ≲ 2700` the interval is issued, but its upper
        endpoint is 1.
*   **Design.** The partition must not depend on `y` (Section 2.4):
    stratifying by label breaks the moments of `Q_c` and `Y_c`.
*   **Tightness.** Each side is attained only up to the union bound and the
    `V̂ ≤ q̄` substitution. The latter costs `O_p(κ_y r/G)` relative. The
    interval is conservative, as every Cantelli branch is.

---

## 9. The refutation checks

The checks test the declaration κ. None of them enters the certificate; each
can only flag. This section first describes the temporal checks, then shows
why they run on residualised totals (Lemmas J, J′, K), then describes the
graph checks (Claim 6, Proposition B) and the checks on bucket windows.

### 9.1 What the checks see

Both temporal checks take one value per cluster in cluster order (time order
on the temporal path). That value is the residualised total
`e_c = S_c − n_c θ̂` (Section 9.3, Lemma K), not the raw total `S_c`. The
statements below are about the noise terms `ε_c` in `S_c = n_c θ + ε_c`,
which is what the residuals carry once the count component is removed.

### 9.2 Temporal checks

**κ = 1: lag-1 test** (`refutation.check_uncorrelated`). One-sided
Fisher-z test of the lag-1 sample autocorrelation `r₁` of the residualised
totals, `√(G−4)·artanh(r₁) > z_{level}` (`UNTESTABLE` when `G < 5`; `G − 4 = (G − 1) − 3` is Fisher's `n − 3` with the `G − 1` lag-1 pairs).

*   **Controls.** Under independent, equal-variance totals with finite
    fourth moments, `√G·r₁ → N(0,1)`, so the false-refutation rate tends to
    `1 − level`. With unequal variances the limit variance is `h_σ`
    (Lemma K), which the count guard of Section 9.3 controls only in two
    regimes. This is asymptotic; measured size is 0.03–0.06
    ([DESIGN.md](DESIGN.md) decision 14).
    Independence is needed here, not only the uncorrelatedness that
    P1_c(1) states: with `ε_c = ξ_c ξ_{c−1}` for i.i.d. standard normal `ξ`
    the totals are uncorrelated, yet `Var(√G·r₁) → E ξ⁴ = 3` and a nominal
    5% test has asymptotic size `1 − Φ(1.645/√3) ≈ 17%` (here `Φ` is the
    standard normal distribution function). The certificate needs only
    P1_c; the stronger premise is the check's calibration. For totals over
    disjoint item sets, independence across windows is the natural null;
    conditionally heteroscedastic totals are not covered.
*   **Power.** Consistent against any fixed positive lag-1 correlation of the
    totals. It is blind to correlation that skips adjacent windows and to
    negative correlation, and it tests the totals at the interval's own `G`,
    which is the null the certificate needs ([DESIGN.md](DESIGN.md)
    decision 16).

**κ > 1: batch-means bound** (`refutation.check_kappa`). A χ² lower
confidence bound on the VIF from `B = 30` super-batches; refutes when it
exceeds κ.

*   **Controls (approximate).** The χ² bound treats the `B` batch means as
    independent. That is approximately right when the super-batches are
    longer than the correlation length. With shorter batches, and with
    non-negative autocovariances, the estimate is biased low, so a
    refutation is conservative and the check is weak. Negative
    autocovariances can bias it either way. Correlation between adjacent
    batches also makes the nominal χ² degrees of freedom too large. So the
    level is approximate and is evaluated by simulation in
    [EVIDENCE.md](./EVIDENCE.md) §3.1.
*   **Blind corner.** Small `G` with long memory ([DESIGN.md](DESIGN.md)
    decision 15: `n = 20000`, `L = 1000`, `G = 100`, where `L` is the
    correlation length). There, protection comes from the `M_c` check and
    from refusal. [EVIDENCE.md](./EVIDENCE.md) §3.1 includes this corner.

### 9.3 Windows with unequal counts: Lemmas J, J′ and K

When the totals come from a merged summary, window counts may differ by more
than one. This subsection shows that raw totals then carry a deterministic
count component that biases the lag-1 statistic (Lemma J), that the internal
equal-count windows bias it in the masking direction (Lemma J′), and that
residualising removes the component exactly (Lemma K).

Write `S_c = n_c θ + ε_c` with `E ε_c = 0` (no drift), and
`δ_c = n_c − m̄`. The checks see `S_c − S̄ = (ε_c − ε̄) + θ(δ_c − δ̄)`: a
known, deterministic **count component** added to the noise. Let

$$\tilde\rho \;=\; \frac{\theta^2 \sum_c (\delta_c - \bar\delta)^2}{\sum_c (\varepsilon_c - \bar\varepsilon)^2}$$

be its energy relative to the noise.

> **Lemma J.** With `ρ_δ` the lag-1 autocorrelation of the counts and `r₁^ε`
> that of the noise alone, the lag-1 statistic on the totals satisfies
>
> $$r_1 \;=\; \frac{r_1^{\varepsilon} + \tilde\rho\,\rho_\delta + X}{1 + \tilde\rho + Y},$$
>
> where the cross terms obey `|X| ≤ 2√ρ̃` and `|Y| ≤ 2√ρ̃` deterministically
> and have approximately mean 0 and standard deviation `O(√(ρ̃/G))` when the
> `ε_c` are uncorrelated.

*Argument.*

1.  Write `e_c = ε_c − ε̄`, `d_c = θ(δ_c − δ̄)` and `D = Σ_c e_c²`, so
    that `Σ_c d_c² = ρ̃D`. The totals, centred, are `e_c + d_c`. (In this
    argument `e_c` is the centred noise, not the residual of Section 9.1.)
2.  The lag-1 numerator is
    `Σ_c e_c e_{c+1} + Σ_c d_c d_{c+1} + Σ_c (e_c d_{c+1} + d_c e_{c+1})`,
    and the denominator is `D + ρ̃D + 2Σ_c e_c d_c`.
3.  Divide both by `D`. The first two numerator terms give `r₁^ε` and
    `ρ̃ρ_δ`; the cross terms give `X`, and `2Σe_c d_c / D` is `Y`.
4.  By Cauchy–Schwarz each cross sum is at most `√(D · ρ̃D) = √ρ̃·D` in
    absolute value, so `|X| ≤ 2√ρ̃` and `|Y| ≤ 2√ρ̃`.
5.  If the `ε_c` are uncorrelated with common variance `σ²`, a cross sum is
    a fixed linear combination of them with weights `d`, so it has mean
    approximately 0 (exactly, up to the centring by `ε̄`) and variance about
    `σ²Σd_c² = ρ̃σ²D`. Since `D ≈ Gσ²`, its size relative to `D` is
    `O(√(ρ̃/G))`. ∎

**Consequences for checks on raw totals.**

*   **Bias.** The count component shifts `r₁` by `ρ̃·ρ_δ` in the numerator
    and shrinks it by the factor `1/(1 + ρ̃)`. Counts that trend over time
    give `ρ_δ` near `+1` and raise the false-refutation rate; for the
    internal windows the sign is always the other one (Lemma J′).
*   **Size of ρ̃.** With a stationary per-item law, `Var(S_c) ≈ m̄·v_b·
    (M−1)θ²` (window size `m̄`, within-window VIF `v_b`), so ±1 counts give
    `ρ̃ ≤ 1/(4 m̄ v_b (M−1))`. For accuracy `M − 1 = (1−θ)/θ`, so ρ̃ grows
    without bound as θ → 1: the bound gives 0.25 at θ = 0.99 and 2.50 at
    θ = 0.999 for `m̄ = 100`, `v_b = 1` (measured mean ρ̃ of 0.23 and 2.4 in
    simulation with `n = 20070`, `G = 200`).

> **Lemma J′ (internal counts).** Let `x = n/G`, `f₀ = frac(x)` and
> `f = min(f₀, 1−f₀)`. The counts of the windows `(i·G)//n` are
> `⌊x⌋ + β_c` with `β_c = ⌈(c+1)f₀⌉ − ⌈c f₀⌉ ∈ {0,1}`, and their lag-1
> autocorrelation is `ρ_δ = −f/(1−f) ≤ 0`, up to one wrap-around term of
> order `1/G`.

Here `(i·G)//n` is the window index of item `i = 0, …, n−1` (integer
division), which is how `temporal_interval` forms windows.

*Argument.*

1.  `⌈c x⌉ = c⌊x⌋ + ⌈c f₀⌉`. `β_c = 1` iff
    `frac(c f₀) ∈ (1−f₀, 1) ∪ {0}`.
2.  Since `G f₀ = n mod G` is an integer, `c f₀ mod 1` visits the multiples
    of `1/q` (`q = G / gcd(n mod G, G)`) equally often over `c = 0 … G−1`,
    so `Pr(β_c = 1) = f₀` and `Pr(β_c = β_{c+1} = 1) = max(0, 2f₀ − 1)`.
3.  The covariance is `−f₀²` for `f₀ ≤ ½` and `−(1−f₀)²` otherwise; divide
    by `f₀(1−f₀)`. ∎

So on raw totals the internal windows can only **mask** correlation, and at
high accuracy they mask it completely: at θ = 0.99, `m̄ ≈ 100`, `G = 200`,
the raw lag-1 test refutes 0% of the time under the null and 0.7% under an
AR alternative that the residualised test below refutes 13% of the time; at
θ = 0.999 it never refutes (simulation, 4000 replications). The checks
therefore do not use raw totals.

> **Lemma K (residualised totals).** Let `θ̂ = ΣS_c / Σn_c` and
> `e_c = S_c − n_c θ̂`. Without drift (`S_c = n_c θ + ε_c`, `E ε_c = 0`),
> `e_c = ε_c − n_c ε̄` with `ε̄ = Σ_c ε_c / n`: the count component is
> identically zero, for any counts. If the `ε_c` are independent with
> variances `σ_c²`:
>
> 1.  the lag-1 numerator has mean
>     `Σ_c [n_c n_{c+1} Σσ²/n² − (n_{c+1}σ_c² + n_cσ_{c+1}²)/n]`, which for
>     `σ_c² ∝ n_c` and ±1 counts is about `−1/G` in `r₁` (a first-order
>     ratio of expectations), the same as ordinary mean centring;
> 2.  if in addition the standardised fourth moments `E ε_c⁴/σ_c⁴` are
>     bounded and no window dominates (`max_c σ_c²/Σ_c σ_c² → 0`), `r₁` has
>     asymptotic variance `h_σ/G` with
>     `h_σ = [ (1/(G−1)) Σ_c σ_c² σ_{c+1}² ] / [ (1/G) Σ_c σ_c² ]²`.

(In Lemma K, `σ_c² = Var(ε_c)` is the variance of a window's noise, not an
item variance.)

*Argument.*

1.  Expand `e_c e_{c+1}` and use independence for part (1).
2.  For part (2), the cross terms of `Var(Σ ε_c ε_{c+1})` vanish under
    independence, leaving `Σ σ_c² σ_{c+1}²`.
3.  The fourth-moment and non-dominance conditions give the law of large
    numbers `Σ e_c² = Σ σ_c² (1 + o_p(1))` for the denominator and the
    (1-dependent) CLT for the numerator. ∎

*   **Count guard.** Only heteroscedasticity remains. It depends on the
    counts and on the unknown within-window VIF `v(n)` through
    `σ_c² ∝ n_c v(n_c)`, with `1 ≤ v(n) ≤ n`. The implementation evaluates
    the two power-law regimes `v ≡ 1` (`σ_c² ∝ n_c`) and `v(n) = n`
    (`σ_c² ∝ n_c²`). With
    $$h_a = \frac{\tfrac{1}{G-1}\sum_c n_c^a n_{c+1}^a}{\big(\tfrac1G\sum_c n_c^a\big)^2}, \qquad a \in \{1, 2\},$$
    the checks run only if `max(h₁, h₂) ≤ 1.1`. In those two regimes the
    Fisher-z standard error is then off by at most 4.9%, and a nominal
    one-sided 5% test runs at no more than 5.84%. Otherwise the κ check
    reports `UNTESTABLE` with the reason. The guard reads only the counts,
    never the losses.
    *   **Not a bracket.** `h₁` and `h₂` do not bound `h_σ` for every
        `1 ≤ v(n) ≤ n`, even a monotone one. Take 22 windows with counts
        `(4, 4)` followed by `(3, 1)` ten times, and `v(1) = v(3) = 1`,
        `v(4) = 4`. Then `h₁ = 0.850` and `h₂ = 0.755`, but
        `h_σ = 1.605`. We know of no count-only guard that covers every
        `v` and still passes typical merged summaries. Outside the two
        regimes the guard is therefore a heuristic, and its size is
        evaluated by simulation (Limitation V5, Section 11). This affects only
        the check, never the interval.
    *   Equal and ±1 counts give `h ≤ 1 + O(1/m̄²)` and always pass, so the
        raw-data and summary entry points agree exactly on the same data.
    *   Independent count jitter gives `h ≈ 1` even when the counts vary a
        lot. Trending counts give `h ≈ 1 + CV_n²` and are refused once
        `CV_n² > 0.1`.
*   **Measured.** With the internal windows (`n = 20070`, `G = 200`, 4000
    replications) the residualised lag-1 test has size 0.041, 0.042 and
    0.053 at θ = 0.9, 0.99 and 0.999. Its power against AR(1)
    log-error rates (`φ = 0.5`, scale 0.6) at θ = 0.99 is 0.54; the
    raw-totals test refutes in 0 of 4000 null runs at θ = 0.99 and 0.999
    ([EVIDENCE.md](./EVIDENCE.md) §8.2, suite S2). The high-accuracy temporal
    cells of [EVIDENCE.md](./EVIDENCE.md) §3.1 evaluate it as well
    (Limitation V5, Section 11).
*   **Coverage.** None of this affects the interval: Claim 4 needs no equal
    sizes, and unequal counts enter only through the default `M_c`
    (Lemma I′, Section 4). The checks only flag, and Claim 5 bounds what a
    missed refutation costs.

### 9.4 Graph checks for κ = 1: the edge test and Claim 6

A graph gives no order, but it gives an adjacency: clusters joined by a cut
edge. Both graph checks run on the residuals `e_c = S_c − n_c θ̂` and
`z_c = e_c/√n_c` (`graph.graph_interval`).

**The edge test** (`refutation.check_uncorrelated_edges`). With
`q_k = √(n_k/N)` (`N = Σ_k n_k = n`), `P = I − qqᵀ` and `W` the cluster
adjacency, `z = Pζ` for the independent `ζ_c = ε_c/√n_c`, so
`Σ_E z_c z_d = ζᵀAζ` with `A = ½PWP`. The statistic is

$$T = \frac{\sum_E z_c z_d - \sum_k A_{kk} z_k^2}{\sqrt{2\sum_{k,l} A_{kl}^2 z_k^2 z_l^2}},$$

the quadratic form centred and scaled by its plug-in null mean and variance
(computed in `O(|E| + G)`). Refute if `T > z_level`; `UNTESTABLE` if
`|E| < 30` or if fewer than 10 effective edges carry the statistic
(dominance guard, below). Claim 6 gives its null mean and variance.

> **Claim 6.** Assume H₀: `S_c = n_c θ + ε_c` with the `ε_c` independent and
> mean zero, with arbitrary variances. Write `ζ_c = ε_c/√n_c`,
> `N = Σ_E ζ_c ζ_d` and `V = Σ_E ζ_c² ζ_d²`.
>
> (a) *(exact)* `E N = 0` and `Var N = E V`.
>
> (b) *(asymptotic)* If the standardised `ζ_c` have uniformly bounded fourth
> moments and `λ_max(A_σ)² / ‖A_σ‖_F² → 0`, where `(A_σ)_{cd} = σ_c σ_d`
> on edges, then `N/√V → N(0,1)`.
>
> (c) *(plug-in, uncorrected)* Let `q`, `P` and `A` be as in (d) and
> `τ_k² := Var(ζ_k)`. Replacing θ by θ̂ turns the edge sum into
> `Σ_E z_c z_d`, whose null mean is, exactly,
>
> $$\mathbb{E}\sum_E z_c z_d \;=\; \sum_k A_{kk}\tau_k^2 \;=\; \sum_E q_c q_d\,\bigl(\bar\tau^2 - \tau_c^2 - \tau_d^2\bigr), \qquad \bar\tau^2 := \sum_k q_k^2 \tau_k^2,$$
>
> whereas `E Σ_E ζ_c ζ_d = 0` by (a). In units of
> `sd₀ := (Σ_E τ_c² τ_d²)^{1/2}`, the null standard deviation of the
> known-θ edge sum by (a), the bias is at most
> `2(τ_max/τ_min)² · Σ_E q_c q_d / √|E|` in absolute value. When
> `Var(S_c) ∝ n_c` (so `τ_k² ≡ τ²`) it equals `−Σ_E q_c q_d / √|E| < 0`,
> which is `−√(d̄/(2G))` for equal counts, where `d̄ = 2|E|/G`.
>
> (d) *(corrected)* Let `q_k = √(n_k/N)` and `P = I − qqᵀ`. Then
> `z = Pζ` exactly, so `Σ_E z_c z_d = ζᵀAζ` with `A = ½PWP`. Under H₀,
> `E ζᵀAζ = Σ_k A_kk τ_k²` and, up to the fourth-cumulant term
> `Σ_k A_kk² κ₄,k`, `Var ζᵀAζ = 2Σ_{k,l} A_kl² τ_k² τ_l²`. The
> statistic plugs in `τ_k² ← z_k²`:
>
> $$T \;=\; \frac{\sum_E z_c z_d - \sum_k A_{kk} z_k^2}{\sqrt{2\sum_{k,l} A_{kl}^2 z_k^2 z_l^2}},$$
>
> computed in `O(|E| + G)` through the rank-2 structure of `PWP − W`. In
> e-scale the mean term is
> `Σ_E [n_c n_d Q/N² − (n_c e_d² + n_d e_c²)/N]/√(n_c n_d)`. (b) applies
> with `D_τ^{1/2} A D_τ^{1/2}` in place of `A_σ`.
>
> (e) *(asymptotic level of the corrected test)* Consider a sequence of
> designs (graph, counts, cluster variances) under H₀ with `|E| → ∞`, and
> write `B̂ := Σ_k A_kk z_k²` and `V′ := 2Σ_{k,l} A_kl² z_k² z_l²` for the
> centring and the variance in (d). Let `A_τ := D_τ^{1/2} A D_τ^{1/2}`, with
> entries `a_kl = A_kl τ_k τ_l`, let `λ := ‖A_τ‖_op` be its largest
> *absolute* eigenvalue, and let `F := Σ_{k≠l} a_kl²`. Assume:
>
> *   **(E1)** `τ_k > 0`, and the standardised `ξ_k := ζ_k/τ_k` satisfy
>     `E ξ_k⁴ ≤ μ₄` for a constant `μ₄` not depending on the design.
> *   **(E2)** `λ² / ‖A_τ‖_F² → 0` (no few directions dominate).
> *   **(E3)** `Σ_k a_kk² / ‖A_τ‖_F² → 0` (the diagonal is negligible).
> *   **(E4)** `δ := F^{-1/2} [ τ̄ R₁ + τ̄² (R₂ + |Σ_k A_kk q_k²|) ] → 0`
>     (the projection is negligible), where `R₁² := Σ_{k,l} q_k² A_kl² τ_l²`
>     and `R₂² := Σ_{k,l} q_k² q_l² A_kl²`.
>
> Then `V′/(2F) → 1` in probability, `T → N(0,1)` in distribution, and the
> edge test refutes a true H₀ with probability at most
> `Pr(V′ > 0, T > z_level) → 1 − level`. The library checks none of
> (E1)–(E4).

Symbols local to Claim 6: in (a) and (b), `N` is the uncorrected edge sum
and `σ_c² = Var(ζ_c)` (that is, `σ_c = τ_c`); in (c)–(e), `N = Σ_k n_k`
is the total item count. `κ₄,k` is the fourth cumulant of `ζ_k`,
`Q = Σ_c e_c²`, `D_τ` is the diagonal matrix of the `τ_k²`, and
`τ_max`, `τ_min` are the largest and smallest `τ_k`. `z_level` is the
`level` quantile of `N(0,1)`, so `1 − level` is the nominal
false-refutation rate. `w := Wq` and `s := qᵀWq = 2Σ_E q_c q_d`.

*Argument.*

1.  **(a)** For two distinct edges `(c,d) ≠ (c′,d′)`, at least one vertex
    appears exactly once in `ζ_c ζ_d ζ_{c′} ζ_{d′}`, so the expectation
    vanishes by independence and mean zero. Only the diagonal terms
    `E ζ_c² E ζ_d²` survive, and these are `E V` term by term. `E N = 0`
    for the same reason.
2.  **(b)** This is the special case of (e) with `P` replaced by `I`, so
    that `A = ½W`, `z = ζ`, `D = 0` (step 9) and the projection terms in
    step 10 vanish. `W` has zero diagonal, so (E3) holds trivially, and
    `A_τ = ½A_σ`, so (E2) is the eigenvalue condition of (b). Steps 6–8
    then give `N/√(E V) → N(0,1)`, and step 10 (where `2F = E V` and
    `V′ = V`) gives `V/E V → 1`; Slutsky's lemma gives `N/√V → N(0,1)`. On
    the path, if `ε` is only a martingale difference sequence, then
    `ε_c ε_{c+1}` is one too, and the self-normalised martingale CLT
    applies (Hall and Heyde 1980, Thm 3.2).
3.  **(c)** By (d), `Σ_E z_c z_d = ζᵀAζ`. The `ζ_k` are independent with
    mean zero, so `E ζᵀAζ = Σ_k A_kk τ_k²`. Expanding
    `PWP = W − q wᵀ − w qᵀ + s qqᵀ` and using `W_kk = 0` gives
    `A_kk = s q_k²/2 − q_k w_k`. Now `Σ_k q_k w_k τ_k² = Σ_E q_c q_d (τ_c² + τ_d²)`,
    because each edge contributes `q_c q_d` to both `w_c q_c` and `w_d q_d`,
    and `(s/2) Σ_k q_k² τ_k² = Σ_E q_c q_d τ̄²`. Subtracting gives the stated
    identity. Each bracket satisfies `|τ̄² − τ_c² − τ_d²| ≤ 2τ_max²`, and
    `sd₀ ≥ τ_min² √|E|`, which gives the bound. If `τ_k² ≡ τ²`, every bracket
    equals `−τ²` and `sd₀ = τ²√|E|`. With equal counts `q_c q_d = 1/G`, so
    `Σ_E q_c q_d / √|E| = √|E|/G = √(d̄/(2G))`.
4.  **(d)** `e = ε − n(Σε)/N` gives `z = Pζ`. The mean and variance are
    those of a quadratic form in independent mean-zero variables:
    `E ζᵀAζ = Σ_k A_kk τ_k²`, and `Var ζᵀAζ = 2Σ_{k,l} A_kl² τ_k² τ_l²`
    plus the fourth-cumulant term `Σ_k A_kk² κ₄,k`. Whether the plug-in
    `τ_k² ← z_k²` preserves the asymptotic null law is part (e).
5.  **(e), the evaluated quantities.** The code's `B̂` and `V′` are exactly
    `Σ_k A_kk z_k²` and `2Σ_{k,l} A_kl² z_k² z_l²`:
    *   `B̂`. By step 3, `Σ_k A_kk z_k² = (s/2)Σ_k q_k² z_k² − Σ_E q_c q_d (z_c² + z_d²)`.
        With `q_c q_d = √(n_c n_d)/N` and `z_c² = e_c²/n_c`, this is the
        e-scale expression of (d):
        `q_c q_d z_c² = n_d e_c² / (N√(n_c n_d))` and
        `(s/2)Σ_k q_k² z_k² = Σ_E n_c n_d Q / (N² √(n_c n_d))`.
    *   `V′`. Write `A = ½(W + R)` with `R := −q wᵀ − w qᵀ + s qqᵀ`, and
        `x_k := z_k²`. Then
        `2Σ_{k,l} A_kl² x_k x_l = ½Σ_{k,l}(W_kl² + 2W_kl R_kl + R_kl²) x_k x_l`.
        The first two terms give `Σ_E x_c x_d (1 + 2R_cd)`. With
        `Q_ab := Σ_k a_k b_k x_k`, the last gives
        `½[2Q_qq Q_ww + s² Q_qq² + 2Q_qw² − 4s Q_qq Q_qw]`. This is the
        closed form evaluated in `refutation.check_uncorrelated_edges`.

    Split the numerator as `ζᵀAζ − B̂ = U + D`, where
    `U := Σ_{k≠l} A_kl ζ_k ζ_l = Σ_{k≠l} a_kl ξ_k ξ_l` is a quadratic form
    with zero diagonal, and `D := Σ_k A_kk (ζ_k² − z_k²)`.
6.  **(e), variance of `U`.** In `E[ξ_k ξ_l ξ_{k′} ξ_{l′}]` with `k ≠ l`
    and `k′ ≠ l′`, an index that appears once makes the term vanish, so
    only `{k′, l′} = {k, l}` survives. Hence `E U = 0` and
    `Var U = 2F` exactly. Write `r_k := Σ_{l≠k} a_kl²`, so that
    `F = Σ_k r_k`, and `m := max_k r_k`. Then
    `r_k ≤ ‖A_τ e_k‖² ≤ λ²`, so `m ≤ λ²`. Because
    `F = ‖A_τ‖_F² − Σ_k a_kk²`, (E3) gives `F/‖A_τ‖_F² → 1`, and with (E2)
    `λ²/F → 0`.
7.  **(e), de Jong's theorem.** de Jong (1987, Theorem 2.1; restated in
    Döbler and Peccati 2017) states the
    following. Let `X_1, …, X_G` be independent and `W_G = Σ_{k<l} W_kl`
    with `W_kl` a function of `(X_k, X_l)`, square-integrable and *clean*:
    `E[W_kl | X_k] = E[W_kl | X_l] = 0`. Let `σ_kl² := E W_kl²` and
    `σ² := Var W_G = Σ_{k<l} σ_kl²`. If
    `max_k Σ_{l≠k} σ_kl² / σ² → 0` and `E W_G⁴ / σ⁴ → 3`, then
    `W_G/σ → N(0,1)`. Apply it with `X_k = ξ_k` and
    `W_kl = b_kl ξ_k ξ_l`, `b_kl := 2a_kl`: the terms are clean because
    `E ξ_k = 0`, `σ_kl² = 4a_kl²` and `σ² = 2F`. The first condition reads
    `2m/F ≤ 2λ²/F → 0` (step 6). Step 8 verifies the second.
8.  **(e), fourth moment.** Index unordered pairs `e = {k, l}` and write
    `b_e = b_kl` and `ξ_e = ξ_k ξ_l`, so `E U⁴ = Σ b_{e₁}b_{e₂}b_{e₃}b_{e₄} E[ξ_{e₁}ξ_{e₂}ξ_{e₃}ξ_{e₄}]`
    over ordered 4-tuples. The expectation vanishes unless every index
    appears at least twice in the eight factors. The non-vanishing
    configurations are:
    *   `{e,e,f,f}` with `e ≠ f`, 3 orderings per ordered pair: expectation 1
        if `e, f` are disjoint, and `μ₄,k := E ξ_k⁴` if they share the index `k`.
    *   `{e,e,e,e}`: expectation `μ₄,k μ₄,l`.
    *   A 4-cycle `{k,l},{l,p},{p,r},{r,k}` on distinct indices: expectation
        1, and summing over orderings gives
        `3 Σ_{k,l,p,r distinct} b_kl b_lp b_pr b_rk`.
    *   A triangle with one doubled edge `{k,l},{k,l},{l,p},{p,k}`:
        expectation `μ₃,k μ₃,l` (`μ₃,k := E ξ_k³`), and summing over
        orderings gives `6 Σ_{k,l,p distinct} b_kl² b_lp b_pk μ₃,k μ₃,l`.

    No other multiset of four pairs puts every index at least twice. Since
    `3σ⁴ = 3Σ_{e≠f} b_e² b_f² + 3Σ_e b_e⁴`,

    $$\mathbb{E}U^4 - 3\sigma^4 = \sum_e b_e^4(\mu_{4,k}\mu_{4,l} - 3) + 3\!\!\sum_{e \ne f \text{ share } k}\!\! b_e^2 b_f^2(\mu_{4,k} - 1) + 3C_4 + 6C_3,$$

    with `C₄`, `C₃` the cycle and triangle sums above. Let `ρ_k := Σ_l b_kl² = 4r_k ≤ 4m`,
    so `Σ_k ρ_k = 4F`. Use `1 ≤ μ₄,k ≤ μ₄` and `|μ₃,k| ≤ μ₄^{3/4}`:
    *   `|Σ_e b_e⁴(…)| ≤ (μ₄² + 3) · max_e b_e² · σ² ≤ (μ₄² + 3) · 4m · 2F`.
    *   The shared-index sum is at most `Σ_k ρ_k² ≤ 4m · 4F`, so its term is at most `48 μ₄ m F`.
    *   With `B := (b_kl)` (zero diagonal),
        `C₄ = tr(B⁴) − 2Σ_k ρ_k² + Σ_{k,l} b_kl⁴`, so
        `|C₄| ≤ tr(B⁴) + 3Σ_k ρ_k²`. Here `tr(B⁴) ≤ ‖B‖_op² ‖B‖_F²`,
        `‖B‖_F² = 4F`, and
        `‖B‖_op ≤ 2(λ + max_k |a_kk|) ≤ 4λ`. So `3|C₄| ≤ (192λ² + 144m)F`.
    *   `|C₃| ≤ μ₄^{3/2} Σ_{k,l} b_kl² Σ_p |b_lp||b_pk| ≤ μ₄^{3/2} · 4F · 4m`
        by Cauchy–Schwarz on the inner sum, so `6|C₃| ≤ 96 μ₄^{3/2} m F`.

    With `m ≤ λ²`, `|E U⁴ − 3σ⁴| ≤ C λ² F`, where
    `C := 8(μ₄² + 3) + 48μ₄ + 336 + 96μ₄^{3/2}`. Dividing by `σ⁴ = 4F²`
    gives `|E U⁴/σ⁴ − 3| ≤ Cλ²/(4F) → 0` (step 6). By step 7,
    `U/√(2F) → N(0,1)`.
9.  **(e), the projection remainder.** Let `L := qᵀζ`, so `z = ζ − Lq` and
    `ζ_k² − z_k² = 2L q_k ζ_k − L² q_k²`. Hence
    `D = 2L Σ_k A_kk q_k ζ_k − L² Σ_k A_kk q_k²`. Here `E L² = τ̄²` and
    `E(Σ_k A_kk q_k ζ_k)² = Σ_k A_kk² q_k² τ_k² ≤ R₁²`. By Chebyshev, for
    every `K > 0`,
    `Pr(|D| > 2K²τ̄R₁ + K²τ̄²|Σ_k A_kk q_k²|) ≤ 2/K²`. Since
    `τ̄R₁ + τ̄²|Σ_k A_kk q_k²| ≤ δ√F`, (E4) gives `D/√(2F) → 0` in probability.
10. **(e), the variance estimator.** `V′ = 2‖D_z A D_z‖_F²` with
    `D_z := diag(z)`. Compare `V_ζ := ‖D_ζ A D_ζ‖_F² = Y + Σ_k A_kk² ζ_k⁴`,
    where `Y := Σ_{k≠l} a_kl² ξ_k² ξ_l²`:
    *   Let `η_k := ξ_k² − 1`, so `E η_k = 0` and `Var η_k ≤ μ₄ − 1`. Then
        `Y − F = 2Σ_k r_k η_k + Σ_{k≠l} a_kl² η_k η_l`. The two parts are
        uncorrelated, and
        `Var Y ≤ 4(μ₄−1)Σ_k r_k² + 2(μ₄−1)² Σ_{k≠l} a_kl⁴ ≤ [4(μ₄−1) + 2(μ₄−1)²] m F`,
        using `a_kl² ≤ r_k ≤ m`. Since `m/F → 0`, Chebyshev gives `Y/F → 1`.
    *   `E Σ_k A_kk² ζ_k⁴ ≤ μ₄ Σ_k a_kk² = o(F)` by (E3), so by Markov
        `V_ζ/F → 1`.
    *   `D_z = D_ζ − L D_q`, so
        `D_z A D_z − D_ζ A D_ζ = −L(D_q A D_ζ + D_ζ A D_q) + L² D_q A D_q`.
        Its Frobenius norm is at most `2|L| ‖D_q A D_ζ‖_F + L² R₂`, where
        `E‖D_q A D_ζ‖_F² = R₁²`. As in step 9 this is
        `O_p(τ̄R₁ + τ̄²R₂) = O_p(δ√F)`.

    By the triangle inequality for `‖·‖_F`,
    `|√(V′/2) − √V_ζ| = O_p(δ√F)`. Hence `V′/(2F) → 1` in probability,
    and in particular `Pr(V′ ≤ 0) → 0`.
11. **(e), level.** On `{V′ > 0}`,
    `T = (U/√(2F) + D/√(2F)) / √(V′/(2F))`. By steps 8–10 and Slutsky's
    lemma, `T → N(0,1)`. The test refutes only if every guard passes,
    `V′ > 0` and `T > z_level`. Therefore
    `lim sup Pr(REFUTED) ≤ lim Pr(V′ > 0, T > z_level) = 1 − Φ(z_level) = 1 − level`,
    with equality when the guards pass with probability tending to 1. ∎

**Sufficient condition for (E2)–(E4).** Suppose the counts are equal
(`q_k = G^{-1/2}`), `τ_max/τ_min` stays bounded, and
`d_max²/|E| → 0`, where `d_max` is the largest cluster degree. Then
(E2)–(E4) hold, so under (E1) the conclusion of (e) holds.

*Argument.*

1.  `d_max ≥ 1` and `d_max²/|E| → 0` give `|E| → ∞`, and `|E| ≤ G²/2`
    gives `G → ∞`.
2.  `‖W‖_op ≤ d_max` (maximum row sum), so `‖A‖_op ≤ d_max/2` and
    `λ ≤ τ_max² d_max/2`.
3.  Decomposing `W` along `q` and `q^⊥` gives
    `‖PWP‖_F² = 2|E| − 2‖Wq‖² + s²`. With `‖Wq‖ ≤ d_max`, this gives
    `‖A‖_F² ≥ (|E| − d_max²)/2`, and `‖A_τ‖_F² ≥ τ_min⁴ ‖A‖_F²`.
4.  (E2): `λ²/‖A_τ‖_F² ≤ (τ_max/τ_min)⁴ d_max² / (2(|E| − d_max²)) → 0`.
5.  With equal counts, `w_k = d_k/√G` and `s = 2|E|/G = d̄ ≤ d_max`, so
    `A_kk = (d̄/2 − d_k)/G` and `|A_kk| ≤ d_max/G`. Hence
    `Σ_k a_kk² ≤ τ_max⁴ d_max²/G`, and (E3) follows:
    `Σ_k a_kk²/‖A_τ‖_F² ≤ 2(τ_max/τ_min)⁴ d_max² / (G(|E| − d_max²)) → 0`.
6.  (E4): `τ̄² = G⁻¹Σ_k τ_k² ≤ τ_max²`, `R₁² ≤ τ_max² ‖A‖_F²/G` and
    `R₂² = ‖A‖_F²/G²`. Also `Σ_k A_kk q_k² = tr(A)/G = −s/(2G)`, because
    `tr(PWP) = tr(W) − qᵀWq = −s`. By steps 3 and 5, `F ≥ ½ τ_min⁴ ‖A‖_F²` for
    all large designs. Therefore
    `δ ≤ √2 (τ_max/τ_min)² [G^{-1/2} + G^{-1} + d̄/(2G‖A‖_F)]`, and
    `d̄/(2G‖A‖_F) = |E|/(G²‖A‖_F) ≤ √2 |E| / (G² √(|E| − d_max²))`, which is
    `O(1/G)` because `|E| ≤ G²/2`. All three terms tend to 0 by step 1. ∎

*Power (heuristic, not a guarantee).* Under homoscedastic edge correlation ρ,
`T ≈ ρ√|E|`. On the path that is `ρ√G`, the same as the lag-1 Fisher-z test.

The properties of the test are:

*   **Controls.** Under independent `ε_c` with variances
    `τ_k² = Var(ζ_k)`, `T` is asymptotically `N(0,1)` and the one-sided
    test has asymptotic false-refutation rate at most `1 − level`
    (Claim 6(e)) under four premises that the library does not check:
    bounded standardised fourth moments (E1); no few eigenvalues of the
    variance-weighted matrix `A_τ = D_τ^{1/2} A D_τ^{1/2}` dominate,
    `‖A_τ‖_op²/‖A_τ‖_F² → 0` (E2); a negligible diagonal (E3); and a
    negligible projection term (E4). Equal counts, bounded `τ_max/τ_min`
    and `d_max²/|E| → 0` imply (E2)–(E4) (the sufficient condition after
    Claim 6). Measured
    size (2000 replications): 0.053 on a random geometric cluster graph,
    0.051 on a partitioned Barabási–Albert graph (`G = 348`, 71% of cluster
    pairs adjacent), 0.054 on a hub graph.
*   **`hub_ratio`.** The result reports the design-only ratio
    `λ_max(A)²/‖A‖_F²`: the de Jong ratio at `τ_k² ≡` const (`Var(S_c) ∝
    n_c`). Near 0 is the CLT regime; near 1 means one direction dominates.
    It is only a proxy: if the variance sits on a few clusters, `A_τ` can be
    dominated by a few directions while `hub_ratio` is `O(1/G)`. For example,
    take a perfect matching with non-zero variance on `k` edges only. Up to
    `O(k/G)` projection terms, `T = Σ_j ε_j a_j / ‖a‖`, where
    `a_j = |z z′|` and `ε_j = sign(z z′)` on edge `j`, so `T ≤ √k`, and
    `T` is not asymptotically normal.
    *   If the `ζ` are **symmetric**, the `ε_j` are i.i.d. Rademacher and
        independent of the `a_j`. Then, as `G → ∞`, for `k ≤ 2` the
        one-sided test never refutes, and for any `k` Hoeffding gives
        `Pr(T > z) ≤ exp(−z²/2) = 0.26` at `z = 1.645`. At finite `G` the
        `O(k/G)` terms can push `T` slightly above `√k`. For example,
        `k = 2`, `G = 10` gives 1.690, but that graph has 5 edges and is
        below the 30-edge floor.
    *   The null does **not** assume symmetry, and sparse binary losses are
        skewed. With `n_c = 1` accuracy losses at θ = 0.999, a residual is
        `+0.001` with probability 0.999. Take `G = 1000` clusters in a
        perfect matching (500 edges), with variance on `k = 3` edges only.
        All six active residuals are positive with probability
        `0.999⁶ = 0.994`, and then `T = 1.746 > 1.645`, while
        `hub_ratio = 0.002`. **The false-refutation rate is 99.4%.**
    *   So a few dominant, skewed clusters can make the edge test refute a
        true κ = 1 almost surely. This does not affect coverage.
*   **Dominance guard.** With `y_j = z_c z_d` on the edges, the result
    reports `edge_n_eff = ‖y‖₁²/‖y‖₂²`. To leading order,
    `T = Σy/‖y‖₂ ≤ √edge_n_eff` by Cauchy–Schwarz. The test reports
    `UNTESTABLE` when `edge_n_eff < 10`.
    *   **Threshold.** The threshold 10 comes from the equal-weight
        symmetric lattice, where the size is 0.090 at 9 edges and at most
        0.073 for 10–32 edges.
    *   **Synthetic sweep**
        ([EVIDENCE.md](./EVIDENCE.md), `benchmark/edge_guard_sweep.py`,
        2000 replications per cell):
        *   Gaussian nulls: size 0.051–0.054, and the guard never fires.
        *   Power: unchanged.
        *   The counterexample (`k = 3`, `edge_n_eff = 3.04`) and `k = 5`
             are refused every time.
        *   Near the 30-edge floor: at 32 edges the guard fires 25% of the
            time and the size among tested runs is 0.060; at 84 edges it
            fires 0.6% and the size is 0.068. That 0.068 is the same
            without the guard, so it is a small-graph effect.
    *   **Sparse binary losses with a common θ are conservative even
        without the guard.** The size is at most 0.048 in every cell, and
        0.000 at θ = 0.999, `n_c = 1`. At θ = 0.999, `n_c = 1` the test
        returns `UNTESTABLE` in 64–99.5% of runs, either through the guard
        or because every item is correct and the edge statistic has zero
        variance, so the κ = 1 edge test is effectively
        silent there. It is informative again from `n_c ≈ 10` at θ = 0.99
        and `n_c ≈ 50` at θ = 0.999.
    *   **Not covered.** Many edges whose residuals are all skewed the same
        way around near-zero-variance "anchor" clusters still false-alarm:
        98% at `k = 10` (`edge_n_eff = 10.4`, just above the threshold) and
        90% at `k = 50`. No guard that reads only the residuals can see
        this, because the sample looks like a real positive shift.
        Anchors need `Var(S_c) ≈ 0` at the common mean `n_cθ`. That is
        impossible for binary losses with `n_c = 1` (all such clusters are
        Bernoulli(θ)). Otherwise it needs negative within-cluster
        dependence, or a near-constant loss equal to θ.
*   **Fourth cumulants.** The plug-in variance drops a fourth-cumulant
    term. That term is negligible only when the standardised fourth
    cumulants `κ_{4,k}/τ_k⁴` are bounded. For sparse binary losses they grow
    like `1/(n_k(1−θ))`, and the approximation must be measured, not
    assumed.
*   **Why the corrections.** Without them the numerator is biased by about
    `−√(d̄/(2G))` standard errors (Claim 6(c)) and `Σ_E z_c² z_d²`
    overstates the variance, because `P` removes the near-top eigenvector of
    a dense `W`. On the Barabási–Albert graph the uncorrected test refuted 0
    of 2000 null runs.
*   **Power.** 0.99 at a true VIF of 1.7 (random geometric, `G = 500`). The
    same blind spots as the lag-1 test: non-adjacent and negative
    correlation.

### 9.5 Graph checks for κ > 1: the super-batch bound and Proposition B

**The super-batch bound** (`refutation.check_kappa_batches`). The cluster
graph is partitioned by the built-in partitioner into about 30 super-batches,
and a χ² lower bound on the VIF is computed from their totals; `UNTESTABLE`
if fewer than 10 super-batches or `G < 2B`. With super-batch residual totals
`T_b = Σ_{c∈b} e_c` and counts `N_b = Σ_{c∈b} n_c`, the count-normalised
estimator (`refutation.kappa_lower_bound_batches`) is

$$\hat\kappa_B \;=\; \frac{\sum_b T_b^2/N_b\,/\,(B-1)}{\sum_c e_c^2/n_c\,/\,(G-1)},$$

and the check refutes when `(B−1)κ̂_B / χ²_{level,B−1} > κ`, where
`χ²_{level,B−1}` is the `level` quantile of the χ² distribution with `B − 1`
degrees of freedom. (In this subsection `T_b` is a super-batch total, not the
edge statistic `T`.)

Dependence is called **W-local** when only clusters joined by an edge of `W`
are correlated. Write `E_in` for the edges inside super-batches and `E_cut`
for the edges between them.

> **Proposition B (first order).** Assume W-local dependence, balanced
> super-batches and `Var(S_c) ∝ n_c`. Then, to first order (a ratio of
> expectations standing in for the expectation of a ratio, and ignoring
> the `O(1/B)` centring terms),
> `E Σ_b T_b² ≈ Σ_c Var(S_c) + 2 Σ_{E_in} Cov`, where `E_in` are the edges
> inside super-batches. Hence
> `κ̂_B ≈ VIF − 2 Σ_{E_cut} Cov / Σ Var`. If every edge covariance is
> non-negative, `κ̂_B` is biased **low**, so refuting when
> `(B−1)κ̂_B / χ²_{level,B−1} > κ` is conservative to first order. With
> homogeneous edge covariances, `κ̂_B − 1 ≈ (VIF − 1)(1 − |E_cut|/|E|)`.

*Argument sketch (first order).*

1.  Ignoring the centring by `θ̂`, `T_b ≈ Σ_{c∈b} ε_c`, so
    `E T_b² ≈ Σ_{c∈b} Var(S_c) + 2 Σ_{(c,d) ∈ E_in, c,d ∈ b} Cov(S_c, S_d)`
    under W-local dependence. Summing over `b` gives the stated
    `E Σ_b T_b²`.
2.  Under W-local dependence, `VIF = 1 + 2 Σ_E Cov(S_c, S_d) / Σ_c Var(S_c)`
    exactly, because only adjacent pairs contribute to
    `Var(Σ_c S_c) − Σ_c Var(S_c)`.
3.  With balanced super-batches (`N_b = n/B`) and `Var(S_c) = τ²n_c` (so
    `Σ_c Var(S_c) = τ²n`), the numerator has expectation
    `(B/(B−1)) · (1/n) E Σ_b T_b²` and each denominator term has
    `E[e_c²/n_c] ≈ τ²`, giving denominator expectation
    `(G/(G−1))τ² = (G/(G−1)) · (1/n) Σ_c Var(S_c)`. Up to the `O(1/B)` and
    `O(1/G)` degrees-of-freedom factors, `1/n` cancels in the ratio, giving
    `κ̂_B ≈ E Σ_b T_b² / Σ_c Var(S_c) = 1 + 2 Σ_{E_in} Cov / Σ Var`.
    Subtracting from step 2 gives `κ̂_B ≈ VIF − 2 Σ_{E_cut} Cov / Σ Var`.
4.  With non-negative edge covariances the subtracted term is non-negative,
    so `κ̂_B` is biased low; with homogeneous covariances it is the fraction
    `|E_cut|/|E|` of `VIF − 1`. ∎

With unbalanced batches or `Var(S_c)` not proportional to `n_c`, the count
weights change the estimand, and the statement holds only approximately. With
negative covariances it can fail in either direction. On the path,
`|E_cut| = B − 1` out of `G − 1`, which states the bias of the temporal
`check_kappa` quantitatively. On graphs the cut fraction of a 30-way partition
of the cluster graph is the price; it is known from topology alone and is
reported as `batch_cut_fraction`. The χ² step assumes roughly normal,
equal-variance, independent `T_b`. That needs balanced super-batches, and it
is approximate like the temporal version. The denominator uses the
across-cluster variance, not the within-batch variance: under positive
within-batch correlation the within-batch variance is biased low, which would
inflate `κ̂_B` and refute a correct κ.

*   **Controls (first order).** With non-negative adjacent covariances,
    balanced super-batches and `Var(S_c) ∝ n_c`, the estimate is biased low
    by the cut edges, so refutations are conservative to first order;
    otherwise approximate. Measured size 0.051 (independent, homoscedastic)
    and 0.038 (declared κ = true VIF 1.60). Under strong between-batch
    heteroscedasticity (`Var(T_b)/N_b` varying across super-batches) the
    check over-refutes: with `B = 20` super-batches, one of which has 100
    times the variance of the other 19, the measured false-refutation rate
    is about 0.19 at nominal 0.05. A heuristic explanation:
    `Σ_b T_b²/N_b` is then a weighted sum of χ₁² variables whose effective
    degrees of freedom are smaller than `B − 1`, so the unweighted
    `χ²_{level,B−1}` threshold is too low. This affects only the check, never
    the interval.
*   **Weakness.** When most cluster edges are cut between super-batches
    (`batch_cut_fraction` near 1, typical of dense cluster graphs), the
    check sees little of the dependence. Measured power 0.49 at true VIF
    1.85 against declared 1.2.

### 9.6 The κ̂_W diagnostic and results on real graphs

`κ̂_W = 1 + 2(Σ_E e_c e_d − B̂_e)/Σ_c e_c²`, where
`B̂_e = Σ_E [n_c n_d Q/N² − (n_c e_d² + n_d e_c²)/N]` is the unweighted
counterpart of the Claim 6(d) centering term (without the `1/√(n_c n_d)` edge
weights), is reported as a point estimate of the VIF under adjacent-only
(W-local) dependence. It estimates
the exact W-local identity of step 2 in the Proposition B argument sketch. It is
a diagnostic, never an input ([DESIGN.md](DESIGN.md) §6.4): a κ estimated
from the evaluated data would make the design depend on the losses.

*Real graphs*
(the tasks are GraphLand datasets and `ogbn-arxiv`, cited in the package
[README](README.md#references)). At cluster size 10 and 50 the edge test refutes
κ = 1 on every task. At cluster size 200 it does not refute on the hm-*
tasks, tolokers or artnet-exp, and still refutes on avazu, artnet-views,
city-*, twitch and arxiv. `κ̂_W` ranges from about 1 to 84; for example
hm-prices MAE at `t = 10` has adjacent-pair correlation 0.03 but 1.7M cluster
edges, so `κ̂_W ≈ 28`. Small per-edge correlation on a dense cluster graph
still gives large VIF. The super-batch bound is much lower than `κ̂_W` where
`batch_cut_fraction ≥ 0.9`, as Proposition B predicts.

### 9.7 Checks on bucket windows

Windows built from time buckets (Section 5) can have unequal counts, so the
count guard of Section 9.3 would often refuse them. The κ = 1 check on these
windows therefore uses the edge test instead.

*   κ = 1 uses the self-normalised edge test (Section 9.4) on the path over
    windows, with **no count guard**. It handles unequal window variances by
    construction.
*   Its null distribution is asymptotic (Claim 6(e), whose premises the
    library does not check). The path has
    `G − 1` edges, so the 30-edge floor needs `G ≥ 31`.
*   Its size on bucket windows at `G ∈ {31, 50, 100, 300}` (0.045–0.057 across
    both metrics and both count patterns) is evaluated by simulation in
    [EVIDENCE.md](./EVIDENCE.md) §4.
*   κ > 1 uses `check_kappa` with the count guard. Lemma W claim 2 keeps the
    guard satisfied when `m̄ ≫ b_max`.

---

## 10. Proposition D — the drift warning

The estimand is the evaluated window (Section 2.1). A user who extrapolates
to later data needs stationarity; this check can refute it, never confirm it.

**Procedure.** Split the `G` windows into the first `⌊G/2⌋` and the last
`⌈G/2⌉`. Run Claim 4 on each half at level `1 − α/2`, with the same
declarations and each half's own default `M_c`. Warn if the two intervals
are disjoint.

> **Proposition D.** Suppose each half satisfies the premises of Claim 4
> with the declared `(M_c, κ)`, and the two halves have the same mean
> per-item loss. Then the probability of a warning is at most `α`.

*Argument.*

1.  Each half-interval covers its half's mean with probability at least
    `1 − α/2` (Claim 4).
2.  If the two means are equal and both intervals cover it, the intervals
    intersect.
3.  A warning therefore needs at least one miss; the union bound gives
    `α`. ∎

*   **Premises.** Per half. Neither declaration is inherited from the whole
    window. The item-level `M` is a pooled premise, and the variance may
    sit in one half; the VIF of a subset is not bounded by the VIF of the
    whole. So `M` and κ per half are premises. Each half uses its own
    default `M_c` (Lemma I′ with its own counts), reports its own `M̂_G`
    status, and runs its own κ check.
*   **Power.** Each half has `G/2` windows and runs at `α/2`, and the
    Cantelli constant grows as `√((1−α′)/(α′G))`, so each half-interval is
    about twice as wide as the full-window interval (for small `c`). The
    warning fires once `|θ_early − θ_late|` exceeds roughly that
    half-interval width, i.e. about twice the full-window width. Smaller
    drift is not detected.
*   **Existence.** The warning is evaluated only when both halves return
    status `UNREFUTED`. A half with `c ≥ 1` gets no interval (the library returns
    `ASSUMPTION_REQUIRED`; Section 3, existence threshold, here at `G/(2κ)`
    and `α/2`). A half whose `M̂_G` exceeds its `M_c` (`TAIL_UNRESOLVED`) has
    refuted its own premise. In both cases the warning reports
    `UNTESTABLE`. This only lowers the warning rate, so the bound `α` still
    holds. Using the finite lower endpoint of such a half alone would be
    valid but is left out for simplicity. Because the threshold already
    binds for the full window at realistic sizes, `UNTESTABLE` is common on
    short series; the warning is most useful for long series.
*   **What it is not.** A trend inside a half, a shift after the window, or
    drift in the tails with an unchanged mean are all invisible. The result
    is reported as a warning next to the interval and never changes the
    interval or its status.

---

## 11. Known limitations

*   **V1** (design; high). κ is a declaration. On the temporal path the
    checks of Section 9 have power against the errors that matter except in
    the blind corner. On the graph path the edge test has power against
    adjacent positive correlation; the κ > 1 super-batch check is weak on
    dense cluster graphs, where `κ̂_W` is the more informative diagnostic.
*   **V2** (analytic; medium). The joint rate `Pr(miss ∧ not refuted)` under
    wrong declarations is bounded by Claim 5 for small errors and relies
    on measured power for large ones; see [EVIDENCE.md](./EVIDENCE.md) §3.1.
*   **V3** (analytic; medium). Merging cluster sketches takes
    `max(κ_a, κ_b)`. That is valid when the shards are mutually
    uncorrelated. For correlated shards (adjacent time ranges, overlapping
    graph regions) no merge rule is valid, and κ must be declared for the
    union.
*   **V4** (analytic; low). The lag-1 test, the edge test and the χ² step of
    the super-batch bounds are asymptotic. Their finite-sample size
    (0.03–0.06) is evaluated by simulation.
*   **V5** (design; medium). The checks run on residualised totals
    (Lemma K), which removes the count component for any counts. The
    count-only guard `max(h₁, h₂) ≤ 1.1` controls the remaining
    heteroscedasticity only in the regimes `σ_c² ∝ n_c` and `σ_c² ∝ n_c²`.
    It is not a bracket for general within-window VIF (Section 9.3 has a
    counterexample). Elsewhere the check's size is evaluated in
    [EVIDENCE.md](./EVIDENCE.md) §3.1. Counts that fail the guard (e.g.
    trending time buckets) leave κ
    unchecked. On bucket windows (Section 5) the κ = 1 check is the
    self-normalised path test (Section 9.7), which needs no count guard and
    whose size is evaluated by simulation.
*   **V6** (design; medium). Correlated R² is supported (Section 8). Average
    precision is not covered by this document (see "Not covered" below).

---

## 12. Empirical check-size summary

The status of each theorem, lemma and proposition is in the table at the top
of this document. The table below summarises the empirical check-size regimes
from Section 9 that are not rows of the top table:

| Item | Status |
|---|---|
| κ > 1 batch-means and super-batch χ² bounds | approximate; size measured (0.038–0.051 under homoscedastic batches; super-batch check up to ≈ 0.19 under 100:1 between-batch heteroscedasticity, Section 9.5) |
| Edge test size on hub graphs | measured: 0.054 on a hub graph (Section 9.4); at most 4.66% on the hub cells of [EVIDENCE.md](./EVIDENCE.md) §3.1; up to 98% false refutation with skewed anchor clusters above the dominance guard (Section 9.4). Refutation only; coverage unaffected |

## Not covered

*   **Average precision and other ranking metrics.** They
    are not means of per-item summands, so the second-moment route of
    Lemma V does not apply to them.

## References

*   P. de Jong (1987). A central limit theorem for generalized quadratic
    forms. *Probability Theory and Related Fields* 75, 261–277.
*   C. Döbler and G. Peccati (2017). Quantitative de Jong theorems in any
    dimension. *Electronic Journal of Probability* 22.
*   P. Hall and C. C. Heyde (1980). *Martingale Limit Theory and Its
    Application*, Theorem 3.2.
*   H. Gebelein (1941). Das statistische Problem der Korrelation als
    Variations- und Eigenwertproblem.
*   H. O. Lancaster (1957). Some properties of the bivariate normal
    distribution considered in the form of a contingency table.
*   GraphLand and `ogbn-arxiv` datasets: see the references in the package
    [README](README.md#references).

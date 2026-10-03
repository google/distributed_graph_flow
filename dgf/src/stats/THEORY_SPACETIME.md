# Space-time graphs: theory of the space-time path

> These arguments have not been peer-reviewed; see [EVIDENCE.md](EVIDENCE.md) for simulation checks.

## Status of claims, lemmas, and propositions

Each result in this document carries one of four statuses:

*   **argued, checked by simulation** (with the corresponding test or benchmark
    in parentheses; see [EVIDENCE.md](EVIDENCE.md));
*   **argued only**;
*   **approximation** (first-order or asymptotic);
*   **cited** (from the literature, with reference).

| Label | One-line statement | Section | Status |
|---|---|---|---|
| Claim S1 | Under Kronecker-separable item covariance `(S1)` on a complete grid, block totals have covariance $K_s \otimes K_t$ and $\mathrm{VIF} = \mathrm{VIF}_s \cdot \mathrm{VIF}_t \le \kappa_s\kappa_t$ | §9.1 | argued, checked by simulation (`benchmark/validation_spacetime_test.py`, [EVIDENCE.md](EVIDENCE.md) §5 block ST1; `spacetime_test.py`) |
| Proposition S1B | Counterexample: block-level dropout on an incomplete grid $O \subsetneq V_s \times T_{\mathrm{grid}}$ can inflate $\mathrm{VIF}(S^O)$ to $\Omega(\kappa_s\kappa_t / f_{\mathrm{obs}})$ | §9.1.2 | argued only |
| Proposition S1C | Counterexample: within-block dilution from sparse observations can inflate $\mathrm{VIF}(S^O)$ to $\Omega(\kappa_s\kappa_t / f_{\min}^2)$ even when every block is non-empty | §9.1.2 | argued only |
| Claim S1D | Mask-adjusted variance-inflation bounds `(S1D-1)`, `(S1D-2)` (under non-dilution) and `(S1D-3)` (general positive-definite) under Kronecker separability on an incomplete grid $O \subsetneq V_s \times T_{\mathrm{grid}}$ | §9.1.2 | argued only |
| Proposition S2A | Counterexample: $1 + \lambda_{\max}(\Phi^{\mathrm{cut}})$ can strictly exceed $\kappa_{T,s}\kappa_{T,t}$ when spatial communities contain internal edges ($\Phi_s^{\mathrm{in}} \ne I$) | §9.2.1 | argued only |
| Claim S2B(1)–(3) | Closed-form one-dimensional factored Weyl upper bounds `(S3)` for $1 + \lambda_{\max}(\Phi^{\mathrm{cut}})$ under `(T1-prod)` and `(T2)`, with exact equality in all three boundary regimes | §9.2.2 | argued only |
| Claim S2B(4) | Missing-data monotonicity $\lambda_{\max}(\Phi_O^{\mathrm{cut}}) \le \lambda_{\max}(\Phi^{\mathrm{cut}})$ and topological $\kappa_T$ validity on observed items $O \subseteq V_s \times T_{\mathrm{grid}}$ under `(T1)` and `(T2)` on $O$ | §9.2.2 | argued, checked by simulation (`benchmark/validation_spacetime_test.py`, [EVIDENCE.md](EVIDENCE.md) §5 blocks ST2, ST3a, ST3b; `spacetime_test.py`) |
| Proposition S3 | Temporal boundary localization: $\Phi_t^{\mathrm{cut}}$ is the direct sum of $G_t - 1$ identical $2L \times 2L$ bipartite boundary blocks, so $\lambda_{\max}(\Phi_t^{\mathrm{cut}})$ is independent of window length $b \ge 2L$ (and equals $\phi_t$ for $L = 1$) | §9.3 | argued only |
| Lemma K′(1), (4) | Per-window centring $e_{a,w}^{\mathrm{win}} = S_{a,w} - n_{a,w}\hat\theta_w$ removes pure temporal drift $\theta_w$ and count bias identically, making the spatial marginal check on $e_{a,\bullet}^{\mathrm{win}} = \sum_w e_{a,w}^{\mathrm{win}}$ drift-invariant | §9.4.2 | argued, checked by simulation (`benchmark/validation_spacetime_test.py`, [EVIDENCE.md](EVIDENCE.md) §5 block ST5; `spacetime_test.py`) |
| Lemma K′(2)–(3) | Residual bias $O(\sqrt{G_t\bar d_s}/G_s^{3/2})$ of plug-in $\hat B_{\mathrm{win}}$ and exact zero mean of Sherman–Morrison $\hat B_{\mathrm{win}}^{\mathrm{unb}}$ for the unimplemented joint statistic $T_{\mathrm{win}}$ | §9.4.2 | argued only |
| Proposition S5 | Selecting the partition maximizing size-adjusted $N_{\mathrm{eff}}(\Pi) = G(\Pi)/(\kappa(\Pi)r_{\mathrm{eff}}(\Pi))$ over pre-fixed design-only candidates preserves Lemma R conditional coverage | §9.5.2 | argued, checked by simulation (`benchmark/validation_spacetime_test.py`, [EVIDENCE.md](EVIDENCE.md) §5 blocks ST1–ST3; `spacetime_test.py`) |
| Proposition D′ | False-warning probability of the window-split temporal drift check on product partitions ($G_t \ge 2$) is at most $\alpha$ under stationary window means | §9.6 | argued, checked by simulation (`benchmark/validation_spacetime_test.py`, [EVIDENCE.md](EVIDENCE.md) §5 blocks ST1–ST3, ST5; `spacetime_test.py`) |
| Gebelein–Lancaster bounds | Maximal correlation of functions of jointly Gaussian variables ($|\mathrm{Corr}(f(X), g(Y))| \le |\rho|$, and $\le \rho^2$ for centred even functions) | Appendix B | cited (Gebelein, 1941; Lancaster, 1957) |

---

## What this document argues

This document extends the cluster-interval framework of
[THEORY.md](THEORY.md) to **space-time graphs**, where each evaluated item
is a `(node, time)` pair `(i, t)`. It argues when the variance inflation factor
of space-time block totals factors into a spatial and a temporal part
(Claim S1), shows by counterexample that this factorization fails under
missing observations (Propositions S1B and S1C) and gives mask-adjusted bounds
that remain valid (Claim S1D). Without separability, it bounds the
topological inflation factor of [THEORY.md](THEORY.md) Lemma T′ by
quantities computed separately on the spatial and the temporal graph
(Proposition S2A, Claim S2B, Proposition S3). It then shows that globally
centred refutation checks falsely refute under temporal drift and that
per-window centring removes the drift exactly (Lemma K′), states which of the
checks are exposed to drift (§9.4.4), shows that choosing among
design-only candidate partitions costs
no coverage (Proposition S5), and gives a drift warning on space-time blocks
with false-warning rate at most `α` (Proposition D′). Every result is about
the average expected loss over the **observed** items, conditional on the
design; nothing is claimed about unobserved `(node, time)` pairs or about
future data.

**Audience.** An ML engineer or statistician who evaluates a model on
spatio-temporal data (for example traffic sensors over time) and wants to know
what the space-time interval of the library certifies and under which
declarations. The reader should know [THEORY.md](THEORY.md) Claim 4 (the
cluster interval) and Lemma T′ (κ from topology); both are summarized in
[Results used from THEORY.md](#results-used-from-theorymd).

**Scope.** This document describes the space-time path (`spacetime.py`).
Sections are numbered 9.0–9.6; the numbers are stable identifiers that the
code and [README.md](README.md) refer to.

**Not covered.**

*   The asymptotic null distribution of the per-window joint edge statistic
    `T_win` on space-time blocks (not implemented in the library).

Limitations, counterexamples, and failure modes are stated before positive
results.

**Related documents.**

*   [THEORY.md](THEORY.md): the cluster interval (Claim 4), the default
    cluster-level moment (Lemma I′), the refutation checks (Claim 6,
    Proposition B), Lemma T′ and Lemma R.
*   [THEORY_RANGE_ENVELOPE.md](THEORY_RANGE_ENVELOPE.md): one way to obtain
    the item-level correlation bound `Φ` used by premise `(T1)` from the model
    configuration (message-passing hops, lookback, horizon) and a declared data
    dependence range.

### Library functions referred to

The space-time path in `spacetime.py` consists of the following functions:

| Function | Role |
|---|---|
| `spacetime_items(node_ids, times)` | Maps observed `(node, time)` pairs to dense item indices and reports `is_complete_grid` (observed items equal the product of the observed nodes and the observed time steps). |
| `spacetime_edges(index, spatial_edges, *, max_lag, phi_spatial, phi_temporal, phi_cross=None, max_edges=20_000_000)` | Builds the item-level strong product (§9.3) on observed items: same-time spatial edges, same-node temporal edges, and cross-lagged edges with default weight `phi_spatial * phi_temporal[l - 1]`. |
| `candidate_partitions(index, spatial_edges, *, target_sizes, window_lengths, seed=0)` | Builds community-only, window-only, and product candidate partitions (§9.5). A window of length `wl` is a run of `wl` consecutive **observed** time steps (`index.unique_times`), not a run of `wl` time units. |
| `separable_kappa(kappa_s, kappa_t)` | Candidate κ under Claim S1: `κ_s` for community-only, `κ_t` for window-only, `κ_s κ_t` for product candidates. |
| `topological_kappa_fn(num_items, edges, phi)` | Candidate κ by `graph.topological_kappa` (Lemma T′, Claim S2B(4)). |
| `choose_partition(candidates, *, kappa_fn, m_item, is_complete_grid, metric="mse", level=0.95, fallback_kappa_fn=None, edges=None)` | Selects a candidate (Proposition S5). |
| `spacetime_interval(data, index, choice, *, metric="mse", level=0.95, m_item=None, kappa=None, **kwargs)` | Runs the cluster interval and refutation checks on the chosen partition, the marginal checks (§9.1), and the drift check (§9.6). |
| `graph.topological_kappa(num_nodes, edges, cluster_ids, phi)` | Upper bound on `1 + λ_max(Φ^cut)` (Lemma T′); returns `kappa` and `lambda_upper`. |

---

## Notation

The following symbols are used throughout. A symbol introduced only inside one
argument is defined there.

| Symbol | Meaning |
|---|---|
| $V_s = \{0,\dots,N_s-1\}$, $N_s$, $E_s$ | Spatial node set, number of spatial nodes, undirected spatial edge set. |
| $T_{\mathrm{grid}} = \{0,\dots,T-1\}$ | Discrete time grid of `T` steps (or ordered time buckets). |
| $\mathcal{O} \subseteq V_s \times T_{\mathrm{grid}}$, $n$ | Observed `(i, t)` index set and the number $n$ of observed items. |
| $V_s^{\mathrm{obs}}$, $T_{\mathrm{grid}}^{\mathrm{obs}}$ | Active marginal sets: nodes observed at least once, and time steps at which at least one node is observed (§9.1.2). |
| $s_{i,t} \ge 0$, $\mu_{i,t}$, $\sigma_{i,t}^2$ | Per-item loss summand, its mean $\mathbb{E}[s_{i,t}]$, its variance $\operatorname{Var}(s_{i,t})$. |
| $e_{i,t} = y_{i,t} - \hat y_{i,t}$ | Raw prediction residual (label minus prediction). |
| $D$ | Design: the mask $\mathcal{O}$, spatial edges, timestamps, partition, and declared $\Phi$ (§9.5.1). |
| $\theta$, $\theta(D)$ | Estimand: the average expected loss over observed items, $\theta(D) = \frac{1}{n}\sum_{(i,t)\in\mathcal{O}}\mathbb{E}[s_{i,t}\mid D]$. |
| $\Pi = \Pi_s \times \Pi_t$ | Product partition into $G_s$ spatial communities $C_a$ and $G_t$ contiguous time windows $W_w$. |
| $G \le G_s G_t$ | Number of non-empty space-time blocks $c = (a, w)$. |
| $a(i)$, $w(t)$ | Community of node $i$ and window of time step $t$. |
| $B_{a,w}$, $n_{a,w}$ | Block item set $(C_a \times W_w) \cap \mathcal{O}$ and its count. |
| $N_{a,w}$, $f_{a,w}$ | Block capacity and observed fraction $f_{a,w} = n_{a,w}/N_{a,w}$. In Propositions S1B–S1C, $N_{a,w} = \lvert C_a\rvert\cdot\lvert W_w\rvert$ (full grid). From Claim S1D on, $N_{a,w} = \lvert C_a \cap V_s^{\mathrm{obs}}\rvert\cdot\lvert W_w \cap T_{\mathrm{grid}}^{\mathrm{obs}}\rvert$ (active grid). |
| $f_{\mathrm{obs}}$, $f_{\min}$ | Global observed fraction $n/(N_s T)$ and smallest block fraction $\min_{a,w} f_{a,w}$. |
| $S_{a,w}$, $S_{a,\bullet}$, $S_{\bullet,w}$ | Block total $\sum_{(i,t)\in B_{a,w}} s_{i,t}$, marginal community total $\sum_w S_{a,w}$, marginal window total $\sum_a S_{a,w}$. |
| $N_w$, $\hat\theta_w$ | Observed items in window $w$, $\sum_a n_{a,w}$, and the window sample mean $\frac{1}{N_w}\sum_a S_{a,w}$. |
| $\hat\theta$ | Global sample mean $\sum S_c / n$. |
| $e^{\mathrm{glob}}_{a,w}$, $e^{\mathrm{win}}_{a,w}$, $z^{\mathrm{win}}_{a,w}$ | Globally centred block residual $S_{a,w} - n_{a,w}\hat\theta$; per-window centred residual $S_{a,w} - n_{a,w}\hat\theta_w$; standardised residual $e^{\mathrm{win}}_{a,w}/\sqrt{n_{a,w}}$. |
| $\mathrm{VIF}$ | Variance inflation factor of a set of totals: $\operatorname{Var}(\sum S)/\sum\operatorname{Var}(S)$. |
| $\kappa$, $\kappa_s$, $\kappa_t$ | Declared inflation factors: joint, between communities, between windows. |
| $\kappa_{\mathcal{O}}$, $\kappa_{\mathcal{O}}^{\mathrm{adv}}$, $\kappa_{\mathcal{O}}^{\mathrm{nd}}$ | Mask-adjusted inflation factors of Claim S1D. |
| $\kappa_T$, $\kappa_T(\mathcal{O})$ | Topological factor $1 + \lambda_{\max}(\Phi^{\mathrm{cut}})$ of Lemma T′, and the same factor computed on the observed items. |
| $\rho_{\mathrm{mask}}$ | Multiplicative mask-inflation factor applied to an inflation factor; $\rho_{\mathrm{mask}} = 1$ means no inflation. |
| $h_\sigma$ | Declared item-variance spread $\sigma_{\max}^2/\sigma_{\min}^2$ over the whole active grid, including missing items. |
| $h_{\mathrm{blk}}$ | Declared spread $c_{\max}/c_{\min}$ of full-block mean pair covariances. |
| $C_s$, $C_t$; $K_s$, $K_t$ | Spatial and temporal item covariance factors; community and window covariance matrices (Claim S1). |
| $\Phi$, $\Phi_s$, $\Phi_t$ | Declared non-negative item-pair correlation bounds; in the factored case $\Phi = \Phi_s \otimes \Phi_t$. |
| $\Phi^{\mathrm{in}}$, $\Phi^{\mathrm{cut}}$ | Within-cluster part (including the unit diagonal) and cross-cluster part of a $\Phi$ matrix. |
| $\lambda_{\max}(A)$ | Largest eigenvalue of a symmetric matrix $A$; for non-negative symmetric $A$ it equals the spectral radius. |
| $\otimes$, $\circ$, $A^{\circ 2}$ | Kronecker product, Hadamard (entrywise) product, Hadamard square $A \circ A$. |
| $\phi_s$, $\phi_t(\ell)$, $\phi_{\mathrm{cross}}$, $L$ | Spatial edge bound, temporal bound at lag $\ell$, cross-lagged bound, maximum lag. |
| $b$ | Window length $T/G_t$ (§9.3). |
| $\bar d_s$ | Average degree of the spatial block graph within a window, $2\lvert E_w\rvert / G_s$ (§9.4). |
| $M$, $m_{\mathrm{item}}$, $M_c$, $M_c^{\mathrm{default}}$, $\hat M_G$ | Item-level relative second moment, its code name, the cluster-level moment, its Lemma I′ default, and the empirical cluster ratio $G\sum S_c^2/(\sum S_c)^2$. |
| $r$, $\bar m$, $n_{\max}$, $\mathrm{CV}_n$ | Cluster-size ratio $n_{\max}/\bar m$, mean cluster size $n/G$, largest cluster size, coefficient of variation of cluster counts. |
| $\alpha$, $\alpha'$, $\alpha_{\mathrm{ref}}$ | Two-sided failure probability, per-side budget $\alpha/2$, and the level of a refutation check (one minus its confidence level). |
| PSD, SE | Positive semidefinite; standard error. |

## Premises

The results below combine the following premises. Each is a declaration about
the data-generating process; none is verified by the library.

| Premise | One-line meaning |
|---|---|
| `P1_c(κ)` | Between-cluster dependence: $\operatorname{Var}(\sum_{a,w} S_{a,w}) \le \kappa \sum_{a,w}\operatorname{Var}(S_{a,w})$ with $\kappa \ge 1$. |
| `P2*_c(M_c)` | Cluster-level second moment: $\frac{1}{G}\sum_{a,w}\mathbb{E}[S_{a,w}^2] \le M_c\,\bar\theta_c^2$ with $\bar\theta_c = (n/G)\theta$. Tested by $\hat M_G$. |
| `(P2v)` | Cluster-level variance bound: $\frac{1}{G}\sum_{a,w}\operatorname{Var}(S_{a,w}) \le (M_c - 1)\,\bar\theta_c^2$. Implied by `P2*_c(M_c)`; sufficient for coverage ([THEORY.md](THEORY.md) Lemma V). |
| `(S1)` | Separable item covariance $\operatorname{Cov}(s_{i,t}, s_{j,s}) = C_s(i,j)\,C_t(t,s)$ (§9.1). |
| `(T1)` | Cross-block correlation dominance $\lvert\operatorname{Cov}(s_{i,t}, s_{j,s})\rvert \le \Phi_{(i,t),(j,s)}\sigma_{i,t}\sigma_{j,s}$ for items in different blocks. |
| `(T1-prod)` | `(T1)` with factored bound $\Phi = \Phi_s \otimes \Phi_t$ (§9.2). |
| `(T2)` | No net within-block repulsion: $\sum_{a,w}\operatorname{Var}(S_{a,w}) \ge \sum_{(i,t)\in\mathcal{O}}\sigma_{i,t}^2$. |
| Non-dilution | Missingness inside a block does not selectively remove high-covariance pairs: $\operatorname{Var}(S^{\mathcal{O}}_{a,w}) \ge f_{a,w}^2\operatorname{Var}(S^{\mathrm{full}}_{a,w})$ (Claim S1D(2)). |
| $h_\sigma$ declaration | $0 < \sigma_{\min}^2 \le \sigma_{i,t}^2 \le \sigma_{\max}^2$ on the whole active grid, including missing items (Claim S1D(3)). |
| Exogenous design | The mask, edges, timestamps, and partition do not depend on the losses (§9.1.2 part 4, §9.5.1). |

## Results used from THEORY.md

| Result in [THEORY.md](THEORY.md) | Content used here |
|---|---|
| Claim 4 | Cluster Cantelli certificate $[\hat\theta/(1+c), \hat\theta/(1-c)]$ with $c = \sqrt{(M_c - 1)\kappa(1-\alpha')/(\alpha' G)}$ (Lemma A′ in [`THEORY_INDEPENDENT.md`](THEORY_INDEPENDENT.md)), valid under `(P2v)` and `P1_c(κ)` for any fixed partition. |
| Lemma V | `P1_c(κ)` and `(P2v)` give the variance bound that Claim 4 needs. |
| Lemma I′ | Transfer from the item-level $M$ to $M_c^{\mathrm{default}} = \min\{rM,\ 1 + r(M-1) + \mathrm{CV}_n^2 + 2\mathrm{CV}_n\sqrt{r(M-1)}\}$; its argument shows $\sum_i\operatorname{Var}(s_i) + \sum_c n_c\delta_c^2 \le n(M-1)\theta^2$ for per-cluster mean offsets $\delta_c$. |
| Lemma R | Claims 4 and 7 hold conditionally on a design $D$ that is chosen without the losses, for the conditional estimand $\theta(D)$. |
| Lemma T′ | Under `(T1)` and `(T2)`, `P1_c(κ_T)` holds with $\kappa_T = 1 + \lambda_{\max}(\Phi^{\mathrm{cut}})$. |
| Lemma K | Residualised totals $e_c = S_c - n_c\hat\theta$ remove the count component. |
| Claim 6 | Null mean and variance of the edge test `check_uncorrelated_edges`; part (d) is the corrected statistic (Appendix A). |
| Proposition B | First-order behaviour of the super-batch bound `check_kappa_batches` (Appendix A). |
| Proposition D | Drift warning by split-half intervals. |
| Claim 7 | Correlated $R^2$ interval. |
| Gebelein–Lancaster remark (under Lemma T′) | Converts a residual-level correlation bound into a summand-level bound; stated with argument in [Appendix B](#appendix-b-gebelein-and-lancaster-bounds). |

---

## 9.0 Setup, estimand, and what can go wrong first

### Setting and product partitions

Let `V_s = {0, …, N_s − 1}` be a set of `N_s` spatial nodes with undirected
spatial edge set `E_s`, and let `T_grid = {0, …, T − 1}` be `T` discrete time
steps (or ordered time buckets). The evaluated dataset consists of `n` observed
items indexed by pairs `u = (i, t) ∈ O ⊆ V_s × T_grid`, with non-negative
per-item loss summands `s_{i,t} ≥ 0`, means `μ_{i,t} = E[s_{i,t}]`, variances
`σ_{i,t}² = Var(s_{i,t})`, and pooled estimand:

$$\theta \;=\; \frac{1}{n}\sum_{(i,t) \in \mathcal{O}} \mathbb{E}\bigl[s_{i,t} \mid D\bigr].$$

A **space-time product partition** `Π = Π_s × Π_t` partitions `V_s` into `G_s`
disjoint spatial communities `{C_a}_{a=1}^{G_s}` and `T_grid` into `G_t`
contiguous time windows `{W_w}_{w=1}^{G_t}`, forming `G ≤ G_s G_t` non-empty
space-time blocks `c = (a, w)` with item sets `B_{a,w} = (C_a × W_w) ∩ O`,
counts `n_{a,w} = |B_{a,w}| ≥ 1`, and block totals:

$$S_{a,w} \;=\; \sum_{(i,t) \in B_{a,w}} s_{i,t}.$$

Two boundary cases of `Π = Π_s × Π_t` are:

*   **Community-only (`G_t = 1`):** `G = G_s` spatial clusters, each aggregating
    a community over the entire time horizon.
*   **Window-only (`G_s = 1`):** `G = G_t` temporal windows, each aggregating all
    spatial nodes within a time slice.

Claim 4 and Lemma I′ of [THEORY.md](THEORY.md) apply to any fixed
partition of `O`. What is non-trivial in space-time—and where naive reductions
to the static graph path fail—are four structural issues:

1.  **Non-factoring observation masks or non-separable variances (§9.1).**
    Even if the latent process is separable across space and time, missing
    observations (`O ⊊ V_s × T_grid`) or non-multiplicative heteroscedasticity
    (`σ_{i,t}² ≠ σ_{s,i}² σ_{t,t}²`) break the exact Kronecker factorization of
    the block covariance matrix.
2.  **Cut-matrix coupling and boundary localization in Lemma T′ (§§9.2–9.3).**
    For a product partition (`G_s ≥ 2, G_t ≥ 2`), the space-time cut matrix
    `Φ^cut` is **not** `Φ_s^cut ⊗ Φ_t^cut`: cutting a dense spatial community
    `C_a` between window `w` and window `w + 1` turns every internal spatial
    edge of `C_a` near the window boundary into a cross-block spatio-temporal
    pair. Moreover, Lemma T′'s spectral radius `λ_max(Φ^cut)` is localized on
    the `L`-step boundary layer of each window and does not decay as window
    length `b = T / G_t` grows (Proposition S3).
3.  **Global-mean residualisation under temporal drift (§9.4).**
    On the static graph path, residuals are centred against the grand mean
    `θ̂ = Σ S_c / n`. Under pure temporal drift (`θ_w ≠ θ`) with independent
    noise (`κ = 1`), global centring injects a positive deterministic drift
    product `n_{a,w} n_{b,w} (θ_w − θ)² > 0` onto every same-window spatial edge
    and `n_{a,w} n_{a,w+1} (θ_w − θ)(θ_{w+1} − θ)` onto every temporal edge,
    causing `check_uncorrelated_edges` and `check_kappa_batches` to falsely
    refute a true `κ = 1` with probability tending to `1`.
4.  **Endogenous space-time design (§9.5).**
    Conditioning on the full space-time graph over `[0, T − 1]` conditions on
    future edges and counts `D_{t+1..T-1}` when evaluating `E[s_{i,t} | D]`. If
    past losses trigger future edges, retries, or node churn, `D_{t+1..T-1}` is
    a post-outcome collider.

---

## 9.1 Separable covariance: Claim S1

### Exact scope of separability

A loss covariance `Cov(s_{i,t}, s_{j,s}) = C_s(i, j) C_t(t, s)` requires two
structural conditions to transfer cleanly to space-time block totals `S_{a,w}`:

*   **Multiplicative heteroscedasticity only:** Setting `(j, s) = (i, t)` forces
    the item variances to factor as `σ_{i,t}² = C_s(i, i) C_t(t, t) =
    σ_{s,i}² σ_{t,t}²`. Node-dependent variance (`σ_{s,i}²`) and time-of-day /
    window-dependent variance (`σ_{t,t}²`) are both allowed, provided their
    interaction is multiplicative.
*   **Product observation set `O = V_s × T_grid` (or factoring observation
    weights):** Every active node in `V_s` must be observed on the same set of
    timestamps `T_grid` (or, more generally, item weights inside each block must
    factor as `w_{i,t} = w_{s,i} w_{t,t}`). If node `i_1 ∈ C_a` is observed only
    in window `1` and node `i_2 ∈ C_a` only in window `2`, then `Cov(S_{a,1},
    S_{a,2})` depends on `C_s(i_1, i_2)` while `Var(S_{a,1})` and `Var(S_{a,2})`
    depend on `C_s(i_1, i_1)` and `C_s(i_2, i_2)`, breaking `K = K_s ⊗ K_t`.

> **Claim S1 (Separable space-time VIF).**
> Suppose `O = V_s × T_grid` and the item-level loss covariance factors as
> $$\mathrm{Cov}(s_{i,t},\, s_{j,s}) \;=\; C_s(i, j)\,C_t(t, s) \qquad \forall\, i, j \in V_s,\;\; t, s \in T_{\mathrm{grid}}, \tag{S1}$$
> for symmetric positive semidefinite matrices `C_s ∈ ℝ^{N_s × N_s}` and
> `C_t ∈ ℝ^{T × T}` (so item variances take the multiplicative form
> `σ_{i,t}² = C_s(i, i) C_t(t, t)`). For any product partition
> `Π = {C_a}_{a=1}^{G_s} × {W_w}_{w=1}^{G_t}`, define the `G_s × G_s` community
> covariance matrix `K_s` and the `G_t × G_t` window covariance matrix `K_t` by
> $$K_s(a, b) \;:=\; \sum_{i \in C_a}\sum_{j \in C_b} C_s(i, j), \qquad K_t(w, v) \;:=\; \sum_{t \in W_w}\sum_{s \in W_v} C_t(t, s).$$
> Then:
>
> 1. *(Exact Kronecker block covariance)* For all `(a, w), (b, v)`:
>    $$\mathrm{Cov}(S_{a,w},\, S_{b,v}) \;=\; K_s(a, b)\,K_t(w, v), \qquad \text{i.e. } \mathrm{Cov}(S) = K_s \otimes K_t.$$
>
> 2. *(Exact factorization of the Variance Inflation Factor)*
>    $$\mathrm{Var}\!\left(\sum_{a=1}^{G_s}\sum_{w=1}^{G_t} S_{a,w}\right) \;=\; \left(\sum_{a,b=1}^{G_s} K_s(a, b)\right)\!\left(\sum_{w,v=1}^{G_t} K_t(w, v)\right),$$
>    $$\sum_{a=1}^{G_s}\sum_{w=1}^{G_t} \mathrm{Var}(S_{a,w}) \;=\; \left(\sum_{a=1}^{G_s} K_s(a, a)\right)\!\left(\sum_{w=1}^{G_t} K_t(w, w)\right).$$
>    Whenever `K_s` satisfies premise `P1_c(κ_s)` (between-community dependence:
>    `Σ_{a,b} K_s(a,b) ≤ κ_s Σ_a K_s(a,a)`) and `K_t` satisfies premise `P1_c(κ_t)`
>    (between-window dependence: `Σ_{w,v} K_t(w,v) ≤ κ_t Σ_w K_t(w,w)`), the
>    space-time block totals `S_{a,w}` satisfy premise `P1_c(κ_s κ_t)`:
>    $$\mathrm{Var}\!\left(\sum_{a,w} S_{a,w}\right) \;\le\; \kappa_s\,\kappa_t \sum_{a,w} \mathrm{Var}(S_{a,w}),$$
>    with **exact equality** `VIF(S) = VIF_s · VIF_t` whenever `Σ_{a,w} Var(S_{a,w}) > 0`.
>
> 3. *(Exact alignment with marginal community and window totals)* Let
>    `S_{a,•} := Σ_{w=1}^{G_t} S_{a,w}` be the marginal community totals and
>    `S_{•,w} := Σ_{a=1}^{G_s} S_{a,w}` be the marginal window totals. Then
>    $$\mathrm{Cov}(S_{a,\bullet}, S_{b,\bullet}) \;=\; \Bigl(\textstyle\sum_{w,v} K_t(w,v)\Bigr)\,K_s(a, b), \qquad \mathrm{Cov}(S_{\bullet,w}, S_{\bullet,v}) \;=\; \Bigl(\textstyle\sum_{a,b} K_s(a,b)\Bigr)\,K_t(w, v).$$
>    Hence, whenever `Var(Σ S_{a,w}) > 0`, `VIF_s` is the exact VIF of the `G_s`
>    marginal community totals `S_{a,•}` and `VIF_t` is the exact VIF of the
>    `G_t` marginal window totals `S_{•,w}`.

*Argument.*
**(1)** By bilinearity of covariance and `(S1)`:
$$\mathrm{Cov}(S_{a,w}, S_{b,v}) \;=\; \sum_{i \in C_a}\sum_{t \in W_w}\sum_{j \in C_b}\sum_{s \in W_v} C_s(i, j)\,C_t(t, s) \;=\; \left(\sum_{i \in C_a, j \in C_b} C_s(i, j)\right)\!\left(\sum_{t \in W_w, s \in W_v} C_t(t, s)\right) \;=\; K_s(a, b)\,K_t(w, v).$$
**(2)** Summing `(1)` over all `(a, w), (b, v)` gives the first identity; setting
`(b, v) = (a, w)` and summing over `(a, w)` gives the second. Since `K_s` and
`K_t` are covariance matrices, `Σ_a K_s(a,a) ≥ 0`, `Σ_w K_t(w,w) ≥ 0`,
`Σ_{a,b} K_s(a,b) ≥ 0`, and `Σ_{w,v} K_t(w,v) ≥ 0`. Multiplying the two
non-negative inequalities `Σ_{a,b} K_s(a,b) ≤ κ_s Σ_a K_s(a,a)` and
`Σ_{w,v} K_t(w,v) ≤ κ_t Σ_w K_t(w,w)` gives `P1_c(κ_s κ_t)`.
**(3)** Summing `(1)` over `w, v ∈ {1, …, G_t}` (resp. `a, b ∈ {1, …, G_s}`)
scales `K_s(a, b)` (resp. `K_t(w, v)`) by a common positive scalar across all
pairs, leaving the ratio of the sum of all entries to the trace invariant. ∎

**Consequence for refutation under separability.** Because `S_{a,•}` has the
exact covariance `∝ K_s` and `S_{•,w}` has the exact covariance `∝ K_t`, the
separable path can run **both** existing refutation checks directly on the
marginals (`check_uncorrelated_edges` / `check_kappa_batches` on `(S_{a,•})_{a=1}^{G_s}`
against `κ_s`, and `check_uncorrelated` / `check_kappa` on `(S_{•,w})_{w=1}^{G_t}`
against `κ_t`), in addition to the joint check on the space-time block graph.

In `spacetime_interval`, these marginal checks run only on the separable route
for product partitions (`G_s > 1` and `G_t > 1`). The joint check and the
temporal marginal check centre residuals at the global mean `θ̂`; the spatial
marginal check uses the per-window centred community residuals of §9.4.2
(item 3 of the remarks on missingness). §9.4.4 states the consequences under
temporal drift; none of the checks changes the interval or its status.

### 9.1.1 Losses versus residuals

Claim S1 is a statement about the covariance of the **losses** `s_{i,t}`, not
the raw prediction residuals `e_{i,t} = y_{i,t} − ŷ_{i,t}`:

*   **Squared error (`MSE` / `RMSE`, `s_{i,t} = e_{i,t}²`) under mean-zero
    Gaussian residuals:** If `e` is jointly Gaussian with **mean zero**
    (`E[e_{i,t}] = 0`) and separable residual covariance
    `Cov(e_{i,t}, e_{j,s}) = Σ_s(i, j) Σ_t(t, s)`, then by Isserlis' theorem
    $$\mathrm{Cov}(e_{i,t}^2,\, e_{j,s}^2) \;=\; 2\,\bigl(\mathrm{Cov}(e_{i,t}, e_{j,s})\bigr)^2 \;=\; 2\,\Sigma_s(i, j)^2\,\Sigma_t(t, s)^2 \;=\; 2\,\bigl(\Sigma_s^{\circ 2} \otimes \Sigma_t^{\circ 2}\bigr)_{(i,t),(j,s)},$$
    where `Σ^{∘2} = Σ ∘ Σ` is the Hadamard (entrywise) square. By the Schur
    product theorem, `Σ_s^{∘2} ⪰ 0` and `Σ_t^{∘2} ⪰ 0`, so **MSE inherits exact
    Kronecker separability `(S1)`**, with loss-level `κ_s, κ_t` equal to the
    VIFs of the squared covariances `Σ_s^{∘2}` and `Σ_t^{∘2}`. *(Caveat: if
    `μ_{i,t} = E[e_{i,t}] ≠ 0`, `Cov(e_{i,t}², e_{j,s}²) = 2 Σ_s² Σ_t² +
    4 μ_{i,t} μ_{j,s} Σ_s Σ_t` is a sum of two Kronecker terms and is not
    separable in general.)*
*   **Absolute error (`MAE`, `s_{i,t} = |e_{i,t}|`):** For mean-zero jointly
    Gaussian residuals with correlation `ρ = ρ_s(i, j) ρ_t(t, s)`, Nabeya's
    identity gives `Cov(|e_{i,t}|, |e_{j,s}|) = (2/π) σ_{i,t} σ_{j,s} (ρ arcsin ρ +
    √(1 − ρ²) − 1) = (σ_{i,t} σ_{j,s} / π)(ρ² + ρ⁴/12 + …)`, which is not a
    multiplicative function of `(ρ_s, ρ_t)` because of the higher-order `O(ρ⁴)`
    terms. Thus MAE is only approximately separable at small correlations,
    though by the Gebelein–Lancaster bounds ([Appendix B](#appendix-b-gebelein-and-lancaster-bounds))
    it satisfies pointwise correlation dominance `(T1-prod)` with
    `Φ_s = R_s^{∘2}, Φ_t = R_t^{∘2}` (centred) or `|R_s|, |R_t|` (general), so
    Claim S2B (§9.2.2) applies rigorously. Here `R_s` and `R_t` are the
    spatial and temporal residual correlation factors, with entries
    `ρ_s(i, j)` and `ρ_t(t, s)`.

### 9.1.2 Missing observations (`O ⊊ V_s × T_grid`): counterexamples, mask-adjusted `κ_O` bounds, and fallback bound

Real traffic and sensor datasets (for example METR-LA and PEMS-BAY, where
dropouts are encoded as `0` and excluded by the loader) routinely have missing
`(i, t)` pairs so that `O ⊊ V_s × T_grid` is not a Cartesian product. Because
missing data is the norm in practice, the library does not refuse
incomplete grids; instead, it estimates the observed-item mean

$$\theta(D) \;=\; \frac{1}{n}\sum_{(i,t) \in \mathcal{O}} \mathbb{E}\bigl[s_{i,t} \mid D\bigr], \qquad n \;=\; |\mathcal{O}|,$$

and widens the variance inflation factor to account for the mask `O`. This
section first shows that unadjusted `κ_s κ_t` can fail under both block-level
and within-block missingness (Propositions S1B–S1C). It then establishes
computable mask-adjusted bounds `κ_O` from `(κ_s, κ_t)` on the active
marginals together with `h_σ` (for `(S1D-3)`) or non-dilution and `h_blk` (for
`(S1D-1)–(S1D-2)`) (Claim S1D), the exact `φ`-based fallback
(Claim S2B(4)), and the Lemma R exogeneity premise on the mask.

#### 1. Why unadjusted `κ_s κ_t` fails under missingness (two counterexamples)

Let `N_{a,w} := |C_a| · |W_w|` be the full block size on `V_s × T_grid`,
`n_{a,w} := |(C_a × W_w) ∩ O|` be the observed block count, `f_{a,w} :=
n_{a,w} / N_{a,w} ∈ [0, 1]` be the per-block observed fraction, and `f_obs :=
n / (N_s T)` be the global observed fraction. Under a latent separable
covariance `Cov(s_{i,t}, s_{j,s}) = C_s(i, j) C_t(t, s)` on `V_s × T_grid` with
`C_s ≥ 0, C_t ≥ 0`, masking cannot increase the numerator (`Var(Σ S_{a,w}^O) ≤
Var(Σ S_{a,w}^{full})`), but it can shrink the denominator
`Σ_{a,w} Var(S_{a,w}^O)` relative to `Σ_{a,w} Var(S_{a,w}^{full})` in two ways:

> **Proposition S1B (Block-level dropout: `VIF(S^O) = Ω(κ_s κ_t / f_obs)`).**
> For any perfect square `G_s = G_t = K = m²` with integer `m ≥ 2` and
> `|C_a| = |W_w| = 1` (`N_{a,w} = 1`), there exist entrywise non-negative
> correlation matrices `C_s, C_t ∈ ℝ^{K × K}` with unit diagonals and an
> observation set `O ⊂ V_s × T_grid` in which every community and every window
> is observed (`f_obs = (2K − √K) / K² ≈ 2 / K`) such that
> `κ_s = VIF_s^{full} < 2` and `κ_t = VIF_t^{full} < 2` (`κ_s κ_t < 4`),
> whereas on `O`:
> $$\mathrm{VIF}(S^{\mathcal{O}}) \;>\; \frac{1}{f_{\mathrm{obs}}} \;>\; \max\Bigl\{\frac{K}{2},\; \frac{\kappa_s\,\kappa_t}{4\,f_{\mathrm{obs}}}\Bigr\}.$$

*Argument.* Premises: `K = m²`, `m ≥ 2`, singleton communities and windows, so
block `(a, w)` is the single item `(a, w)` and
`Cov(S_{a,w}, S_{b,v}) = C_s(a, b) C_t(w, v)`.

1.  **Construction.** Let `C_s = C_t` be the direct sum of `1_m 1_mᵀ` on
    `{1, …, m}` and `I_{K−m}` on `{m + 1, …, K}`; both are PSD, entrywise
    non-negative, with unit diagonal. Let
    `O = ({1, …, m} × {1, …, m}) ∪ {(a, a) : m < a ≤ K}`. Every index
    `a ∈ {1, …, K}` occurs as a community and as a window of some observed
    item.
2.  **Full-grid factors.** The entries of `C_s` sum to `m² + (K − m)` and its
    trace is `K`, so `κ_s = κ_t = (m² + K − m)/K = (2m² − m)/m² = 2 − 1/m < 2`.
3.  **Observed fraction.** `|O| = m² + K − m = 2K − √K`, so
    `f_obs = |O|/K² = (2K − √K)/K²`.
4.  **VIF on `O`.** Pairs inside `{1, …, m}²` have covariance `1·1 = 1`,
    giving `m⁴ = K²`. A diagonal item `(a, a)` with `a > m` is uncorrelated
    with every other item, because `C_s(a, b) = 0` for `b ≠ a` and
    `C_t(a, v) = 0` for `v ≠ a`; these items contribute `K − m`. Hence
    `Var(Σ_O S_{a,w}) = K² + K − √K`, `Σ_O Var(S_{a,w}) = |O| = 2K − √K`, and
    `VIF(S^O) = (K² + K − √K)/(2K − √K)`.
5.  **First inequality.** `VIF(S^O) − 1/f_obs = (K² + K − √K − K²)/(2K − √K)
    = (K − √K)/(2K − √K) > 0` because `K > √K` for `K ≥ 4`.
6.  **Second inequality.** `1/f_obs = K²/(2K − √K) > K²/(2K) = K/2`, and
    `1/f_obs > κ_s κ_t/(4 f_obs)` because `κ_s κ_t < 4` (step 2). ∎

For `K = 4`: `VIF(S^O) = 18/6 = 3`, `1/f_obs = 16/6 ≈ 2.67`, `K/2 = 2`,
`κ_s κ_t/(4 f_obs) = (9/4)(16/6)/4 = 1.5`.

> **Proposition S1C (Within-block correlation dilution at uniform `f_{a,w} ≡ f_min`: `VIF(S^O) = Ω(κ_s κ_t / f_min²)`).**
> For any `G_s ≥ 4`, `G_t ≥ 1`, and integer `B ≥ 2`, there exist `N_s = G_s B`
> nodes partitioned into `G_s` communities of size `|C_a| = B`, windows of size
> `|W_w| = 1`, entrywise non-negative correlation matrices `C_s, C_t` with unit
> diagonals, and a mask `O` with **identical observed fraction** `f_{a,w} = 1 / B
> =: f_min` in every block `(a, w)` such that `VIF(S^O) = G_s`. When
> `B − 1 = ⌈√G_s⌉`, moreover `κ_s < 2`, `κ_t = 1`, and
> $$\mathrm{VIF}(S^{\mathcal{O}}) \;=\; G_s \;>\; \frac{\kappa_s\,\kappa_t}{6\,f_{\min}^2}.$$

*Argument.* Premises: `G_s ≥ 4`, `|W_w| = 1`, and the construction below.

1.  **Construction.** In each community `C_a = {i_{a,1}, …, i_{a,B}}`, let
    node `i_{a,1}` be observed in every window `w` (`n_{a,w} = 1`,
    `f_{a,w} = 1/B`), and let the remaining `B − 1` nodes `i_{a,2}, …, i_{a,B}`
    be missing. Set `C_t = I_{G_t}` (`κ_t = 1`). Define `C_s` as the direct sum
    of `1_{G_s} 1_{G_s}ᵀ` on the `G_s` observed nodes `{i_{a,1}}_{a=1}^{G_s}`
    and `G_s` diagonal blocks `1_{B-1} 1_{B-1}ᵀ` on the missing nodes
    `{i_{a,2}, …, i_{a,B}}` of each community. Then `C_s ⪰ 0` (a direct sum of
    PSD blocks) and has unit diagonal.
2.  **VIF on `O`.** In each window the `G_s` observed block totals have unit
    variance and pairwise covariance `1`, and totals in different windows are
    uncorrelated (`C_t = I`). Hence `Var(Σ_O S_{a,w}) = G_t G_s²`,
    `Σ_O Var(S_{a,w}) = G_t G_s`, and `VIF(S^O) = G_s`.
3.  **Full-grid factor.** `K_s(a, a) = 1 + (B − 1)²` (the observed node plus
    the missing block; the two are uncorrelated) and `K_s(a, b) = 1` for
    `a ≠ b`, so `κ_s = VIF_s^{full} = 1 + (G_s − 1)/(1 + (B − 1)²)`.
4.  **Choice of `B`.** Let `k := B − 1 = ⌈√G_s⌉`, so `(k − 1)² < G_s ≤ k²`
    and `k ≥ 2`. Then `G_s − 1 < 1 + k²`, so `κ_s < 2`, and
    `κ_s = (k² + G_s)/(1 + k²)`, `1/f_min² = (k + 1)²`.
5.  **Ratio.** `G_s / (κ_s κ_t / f_min²) = [G_s/(k² + G_s)] · [(1 + k²)/(k + 1)²]`.
    *   `G_s ≥ (k − 1)² + 1 ≥ k²/2`, because
        `2((k − 1)² + 1) − k² = (k − 2)² ≥ 0`. Since `x ↦ x/(1 + x)` is
        increasing, `G_s/(k² + G_s) ≥ (1/2)/(1 + 1/2) = 1/3`.
    *   `(1 + k²)/(k + 1)² ≥ 1/2`, because `2(1 + k²) − (k + 1)² = (k − 1)² > 0`
        for `k ≥ 2`; the inequality is strict.
    Hence the ratio exceeds `1/6`, that is, `G_s > κ_s κ_t/(6 f_min²)`. ∎

For `G_s = 4` (`k = 2`, `B = 3`): `κ_s = 8/5`, `1/f_min² = 9`,
`κ_s κ_t/f_min² = 14.4`, and `VIF(S^O) = 4 > 14.4/6 = 2.4`. The constant
cannot be raised to `1/2`: for every `G_s ≥ 4`, using `k² ≥ G_s`,
`κ_s κ_t/(2 f_min²) = (k² + G_s)(k + 1)²/(2(1 + k²)) ≥ G_s (k + 1)²/(1 + k²) > G_s`
(for `G_s = 4` it is `7.2 > 4`). At `G_s = 5` the ratio of step 5 is
`50/224 ≈ 0.223`; it tends to `1/2` as `G_s → ∞`.

NOTE: In Proposition S1C the missing nodes are never observed, so they lie
outside the active marginal set `V_s^{obs}` defined below. The failure comes
from declaring `κ_s` on the full node set, including permanently absent nodes.
Claim S1D requires `κ_s` and `κ_t` to hold on the active marginals.

#### 2. Claim S1D: Computable bounds `κ_O` from `(κ_s, κ_t)` and auxiliary mask premises (`h_σ`, or non-dilution and `h_blk`)

To avoid paying for sensors or timestamps that are completely absent across the
entire dataset, first define the **active marginal sets**
`V_s^{obs} := {i ∈ V_s : ∃ t, (i, t) ∈ O}` and
`T_grid^{obs} := {t ∈ T_grid : ∃ i, (i, t) ∈ O}`, with active block capacity
`N_{a,w} := |C_a ∩ V_s^{obs}| · |W_w ∩ T_grid^{obs}|` and active block fraction
`f_{a,w} := n_{a,w} / N_{a,w}` (for `N_{a,w} ≥ 1`). *(If `O = V_s^{obs} ×
T_grid^{obs}` is a rectangular subgrid—for instance, when missingness consists
entirely of global timestamp outages and/or permanently offline sensors—then
`f_{a,w} ≡ 1` on `V_s^{obs} × T_grid^{obs}` and `κ_O = κ_s κ_t` with no
inflation whenever `(κ_s, κ_t)` holds on the active marginals `V_s^{obs}` and
`T_grid^{obs}`.)*

In the code, `spacetime_items` computes `is_complete_grid` against the active
marginals: it is `True` exactly when the number of observed items equals the
number of distinct observed nodes times the number of distinct observed time
steps.

For general non-product masks `O ⊊ V_s^{obs} × T_grid^{obs}`:

> **Claim S1D (Mask-adjusted separable VIF bounds `κ_O`).**
> Suppose the latent process on the active grid `V_s^{obs} × T_grid^{obs}` has
> entrywise non-negative separable covariance `Cov(s_{i,t}, s_{j,s}) =
> C_s(i, j) C_t(t, s) ≥ 0` whose **active marginal partitions** on `V_s^{obs}`
> and `T_grid^{obs}` satisfy `VIF_s^{full}(V_s^{obs}) ≤ κ_s` and
> `VIF_t^{full}(T_grid^{obs}) ≤ κ_t`. Let `S_{a,w}^{full}` denote the latent
> block total on `(C_a ∩ V_s^{obs}) × (W_w ∩ T_grid^{obs})` (size `N_{a,w}`) and
> `S_{a,w}^O` the observed block total on `B_{a,w} = (C_a × W_w) ∩ O` (size
> `n_{a,w} = f_{a,w} N_{a,w}`). Then:
>
> 1. *(Master ratio bound)* For any mask `O`:
>    $$\mathrm{VIF}(S^{\mathcal{O}}) \;=\; \frac{\mathrm{Var}\!\left(\sum_{a,w} S_{a,w}^{\mathcal{O}}\right)}{\sum_{a,w} \mathrm{Var}\!\bigl(S_{a,w}^{\mathcal{O}}\bigr)} \;\le\; \kappa_s\,\kappa_t \cdot \frac{\sum_{a=1}^{G_s}\sum_{w=1}^{G_t} \mathrm{Var}\!\bigl(S_{a,w}^{\mathrm{full}}\bigr)}{\sum_{(a,w):\,n_{a,w}\ge 1} \mathrm{Var}\!\bigl(S_{a,w}^{\mathcal{O}}\bigr)}. \tag{S1D-0}$$
>
> 2. *(Bounds under within-block non-diluting missingness — matching
>    Propositions S1B and S1C)* Suppose missingness within each non-empty block
>    does not selectively remove the higher-covariance pairs, so the mean pair
>    covariance on `B_{a,w}²` is at least that on the full block:
>    `(1/n_{a,w}²) Σ_{u,v ∈ B_{a,w}} Cov(s_u, s_v) ≥ (1/N_{a,w}²) Σ_{u,v} Cov(s_u, s_v)`,
>    i.e. `Var(S_{a,w}^O) ≥ f_{a,w}² Var(S_{a,w}^{full})` (which holds in
>    expectation under uniform within-block dropouts and deterministically under
>    exchangeable/constant within-block correlations). Then:
>    * If every block is non-empty (`f_min := min_{a,w} f_{a,w} > 0`):
>      $$\mathrm{VIF}(S^{\mathcal{O}}) \;\le\; \frac{\kappa_s\,\kappa_t}{f_{\min}^2}. \tag{S1D-1}$$
>    * Allowing empty blocks (`n_{a,w} = 0`), if the full-block mean pair
>      covariance `c_{a,w} := Var(S_{a,w}^{full}) / N_{a,w}²` satisfies
>      `c_{a,w} ∈ [c_min, c_max]` with `h_blk := c_max / c_min ≥ 1` (`h_blk = 1`
>      for homogeneous blocks):
>      $$\mathrm{VIF}(S^{\mathcal{O}}) \;\le\; \kappa_s\,\kappa_t\,h_{\mathrm{blk}}\,\frac{\sum_{a=1}^{G_s}\sum_{w=1}^{G_t} N_{a,w}^2}{\sum_{a=1}^{G_s}\sum_{w=1}^{G_t} n_{a,w}^2} \;=\; \frac{\kappa_s\,\kappa_t\,h_{\mathrm{blk}}}{\sum_{a,w} w_{a,w}\,f_{a,w}^2}, \qquad w_{a,w} := \frac{N_{a,w}^2}{\sum_{b,v} N_{b,v}^2}. \tag{S1D-2}$$
>      *(Note: `(S1D-2)` is tighter than `(S1D-1)` if and only if
>      `Σ_{a,w} w_{a,w} f_{a,w}² ≥ h_blk f_min²`, which always holds when
>      `h_blk = 1` or when some block is empty `f_min = 0`, but can fail when
>      `h_blk > 1` and `f_{a,w}` is nearly uniform—for instance, `f_{a,w} ≡ 0.5`
>      and `h_blk = 2` gives `8 κ_s κ_t` in `(S1D-2)` vs. `4 κ_s κ_t` in
>      `(S1D-1)`. Under the convention `1 / f_min² = +∞` when `f_min = 0`, the
>      certified non-dilution bound combining both is their minimum
>      `κ_O^{nd} := κ_s κ_t min{1 / f_min², h_blk / Σ_{a,w} w_{a,w} f_{a,w}²}`.)*
>
> 3. *(Worst-case bound under arbitrary adversarial within-block missingness)*
>    Without assuming `Var(S_{a,w}^O) ≥ f_{a,w}² Var(S_{a,w}^{full})`, suppose
>    only that item variances across the **entire active grid `V_s^{obs} ×
>    T_grid^{obs}`, including the unobserved/missing latent items `(V_s^{obs} ×
>    T_grid^{obs}) \setminus O`**, satisfy `0 < σ_min² ≤ σ_{i,t}² ≤ σ_max²`
>    (`σ_min > 0`) with `h_σ := σ_max² / σ_min² ≥ 1` *(a prior declaration across
>    observed and missing items; under exact multiplicative variances `σ_{i,t}² =
>    σ_{s,i}² σ_{t,t}²` the marginal factors are identifiable when the bipartite
>    observation graph `O` is connected, but in finite samples or under
>    approximate separability `h_σ` is hard to estimate and must be declared
>    conservatively)*. Define the **all-block Minkowski factor** `R_Mink` and the
>    **per-block mediant + empty-block factor** `R_max`:
>    $$R_{\mathrm{Mink}} \;:=\; \left(1 + \sqrt{\frac{h_\sigma \sum_{a=1}^{G_s}\sum_{w=1}^{G_t} (N_{a,w} - n_{a,w})^2}{\sum_{a=1}^{G_s}\sum_{w=1}^{G_t} n_{a,w}}}\right)^{\!2},$$
>    $$R_{\max} \;:=\; \max_{(a,w):\,n_{a,w}\ge 1} \frac{\min\!\Bigl\{h_\sigma N_{a,w}^2,\; \bigl(\sqrt{n_{a,w}} + \sqrt{h_\sigma}(N_{a,w} - n_{a,w})\bigr)^2\Bigr\}}{n_{a,w}} \;+\; \frac{h_\sigma \sum_{(a,w):\,n_{a,w}=0} N_{a,w}^2}{\sum_{(a,w):\,n_{a,w}\ge 1} n_{a,w}}.$$
>    Then
>    $$\mathrm{VIF}(S^{\mathcal{O}}) \;\le\; \kappa_{\mathcal{O}}^{\mathrm{adv}} \;:=\; \kappa_s\,\kappa_t\,\min\bigl\{R_{\mathrm{Mink}},\, R_{\max}\bigr\}, \tag{S1D-3}$$
>    which equals `κ_s κ_t` when `n_{a,w} = N_{a,w}` for all `(a, w)` (`f_{a,w} ≡ 1`,
>    no empty blocks) and is finite for every non-empty `O`.

*Argument.*
**(1)** Because `C_s ≥ 0` and `C_t ≥ 0` entrywise and `O ⊆ V_s^{obs} × T_grid^{obs}`,
$$\mathrm{Var}\!\left(\sum_{a,w} S_{a,w}^{\mathcal{O}}\right) \;=\; \sum_{(i,t) \in \mathcal{O}}\sum_{(j,s) \in \mathcal{O}} C_s(i, j)\,C_t(t, s) \;\le\; \sum_{(i,t), (j,s) \in V_s^{\mathrm{obs}} \times T_{\mathrm{grid}}^{\mathrm{obs}}} C_s(i, j)\,C_t(t, s) \;=\; \mathrm{Var}\!\left(\sum_{a,w} S_{a,w}^{\mathrm{full}}\right).$$
By Claim S1(2) on `V_s^{obs} × T_grid^{obs}`, `Var(Σ S_{a,w}^{full}) ≤
κ_s κ_t Σ_{a,w} Var(S_{a,w}^{full})`. Dividing by `Σ_{a,w} Var(S_{a,w}^O)` gives
`(S1D-0)`.

**(2)** If `Var(S_{a,w}^O) ≥ f_{a,w}² Var(S_{a,w}^{full}) ≥ f_min² Var(S_{a,w}^{full})`
for all `(a, w)`, summing over `(a, w)` in `(S1D-0)` yields `(S1D-1)`. If some
blocks are empty (`f_{a,w} = 0`) and `Var(S_{a,w}^{full}) = c_{a,w} N_{a,w}² ∈
[c_min N_{a,w}², c_max N_{a,w}²]`, then `Σ_{a,w} Var(S_{a,w}^{full}) ≤
c_max Σ_{a,w} N_{a,w}²` while `Σ_{a,w} Var(S_{a,w}^O) ≥ Σ_{a,w} f_{a,w}² c_{a,w}
N_{a,w}² ≥ c_min Σ_{a,w} n_{a,w}²`, giving `(S1D-2)`. Taking the minimum of the
two valid upper bounds gives `κ_O^{nd}`.

**(3)** For every block `(a, w)` (including empty blocks `n_{a,w} = 0`, where
`S_{a,w}^O = 0`), non-negativity of `C_s, C_t` gives `Var(S_{a,w}^O) ≥
Σ_{(i,t) ∈ B_{a,w}} σ_{i,t}² ≥ n_{a,w} σ_min²`, so `x_{a,w} :=
√Var(S_{a,w}^O) / σ_min ≥ √n_{a,w}` (`x_{a,w} = 0` if `n_{a,w} = 0`).
Cauchy–Schwarz on the `N_{a,w} − n_{a,w}` missing latent items in block `(a, w)`
gives `√Var(S_{a,w}^{miss}) ≤ (N_{a,w} − n_{a,w}) σ_max = b_{a,w} σ_min` with
`b_{a,w} := √h_σ (N_{a,w} − n_{a,w}) ≥ 0`, and the `L²(P)` triangle inequality
on `S_{a,w}^{full} = S_{a,w}^O + S_{a,w}^{miss}` gives `√Var(S_{a,w}^{full}) ≤
σ_min (x_{a,w} + b_{a,w})`.

*   **Argument for `R_Mink`:** Summing `Var(S_{a,w}^{full})` over all `(a, w)` and
    applying Minkowski's inequality `‖x + b‖_2 ≤ ‖x‖_2 + ‖b‖_2` in
    `ℝ^{G_s G_t}` gives
    $$\frac{\sum_{a,w} \mathrm{Var}(S_{a,w}^{\mathrm{full}})}{\sum_{a,w} \mathrm{Var}(S_{a,w}^{\mathcal{O}})} \;\le\; \frac{\|\mathbf{x} + \mathbf{b}\|_2^2}{\|\mathbf{x}\|_2^2} \;\le\; \left(1 + \frac{\|\mathbf{b}\|_2}{\|\mathbf{x}\|_2}\right)^{\!2}.$$
    Because `t ↦ (1 + ‖b‖_2 / t)²` is strictly decreasing in the scalar norm
    `t = ‖x‖_2 ≥ √(Σ_{a,w} n_{a,w})`, substituting `‖x‖_2 ≥ √(Σ_{a,w} n_{a,w})`
    and `‖b‖_2 = √(h_σ Σ_{a,w} (N_{a,w} − n_{a,w})²)` yields `R_Mink`. *(Note:
    whereas `(Σ (x_k + b_k)²) / (Σ x_k²)` is not monotone in the vector `(x_k)`
    when `b_k` varies, `(1 + ‖b‖_2 / ‖x‖_2)²` depends on `(x_k)` only through
    the scalar `‖x‖_2` and is unconditionally decreasing in `‖x‖_2`.)*
*   **Argument for `R_max`:** For each non-empty block (`n_{a,w} ≥ 1`), the
    per-block ratio `Var(S_{a,w}^{full}) / Var(S_{a,w}^O) ≤ min{h_σ N_{a,w}² /
    x_{a,w}², (1 + b_{a,w} / x_{a,w})²}` is non-increasing in `x_{a,w} ≥
    √n_{a,w}`, so it is maximized at `x_{a,w} = √n_{a,w}`. Applying the mediant
    inequality over non-empty blocks and bounding empty blocks by
    `Var(S_{a,w}^{full}) ≤ h_σ N_{a,w}² σ_min²` over `Σ_{n_{a,w} ≥ 1}
    Var(S_{a,w}^O) ≥ σ_min² Σ_{n_{a,w} ≥ 1} n_{a,w}` yields `R_max`. Taking the
    minimum of `R_Mink` and `R_max` in `(S1D-0)` yields `(S1D-3)`. ∎

#### 3. Required declarations and default hierarchy on incomplete grids (`O ⊊ V_s^{obs} × T_grid^{obs}`)

Each bound in Claim S1D and Claim S2B(4) requires a specific prior
declaration on `O ⊊ V_s^{obs} × T_grid^{obs}`:

1.  **Primary default on incomplete grids — Claim S2B(4) (`φ` path,
    `ρ_mask = 1`):** Whenever the caller provides item-level correlation
    dominance bounds `(φ_s, φ_t(ℓ), φ_cross)` satisfying `(T1)` and `(T2)` on
    `O`, Lemma T′ / Claim S2B(4) (§9.2.2) on the observed items `O` of the
    strong product `G_s ⊠ G_t` is **valid under any exogenous mask
    `O ⊆ V_s × T_grid` provided premises `(T1)` and `(T2)` hold on `O`**,
    without any mask-inflation factor (`ρ_mask = 1`): the observed cut matrix
    `Φ_O^cut` is a principal submatrix of `Φ^cut`, so
    `1 + λ_max(Φ_O^cut) ≤ 1 + λ_max(Φ^cut)` by Perron–Frobenius monotonicity
    (and computing `topological_kappa` directly on `O` can only be smaller).
2.  **Default when only `(κ_s, κ_t)` (on the active marginals `V_s^{obs},
    T_grid^{obs}`) and latent item-variance spread `h_σ = σ_max² / σ_min² ≥ 1`
    are declared — `(S1D-3)` (`κ_O^{adv} = κ_s κ_t min{R_Mink, R_max}`):**
    Requires no assumption on how missingness interacts with within-block
    correlation; valid under arbitrary adversarial within-block correlation
    dilution (including Proposition S1C).
3.  **Opt-in under non-diluting within-block missingness (`Var(S_{a,w}^O) ≥
    f_{a,w}² Var(S_{a,w}^{full})`) — `κ_O^{nd} = min{(S1D-1), (S1D-2)}`:**
    Requires the caller to declare the non-dilution premise (and `h_blk =
    c_max / c_min ≥ 1` for `(S1D-2)`). When `h_blk = 1` or when some blocks are
    empty (`f_min = 0`), `(S1D-2)` always dominates `(S1D-1)`; when `h_blk > 1`
    and `f_min > 0`, `(S1D-2)` dominates `(S1D-1)` iff `Σ_{a,w} w_{a,w} f_{a,w}² ≥
    h_blk f_min²`, so taking `min{(S1D-1), (S1D-2)}` achieves the tighter bound
    in all regimes.
4.  **Combining multiple simultaneous declarations (theory versus the
    `spacetime.py` implementation):**
    Mathematically, when multiple valid declaration sets are supplied
    simultaneously—for example, both `φ = (φ_s, φ_t(ℓ), φ_cross)` satisfying
    `(T1)–(T2)` on `O` **and** `(κ_s, κ_t, h_σ)` satisfying `(S1)` on the active
    marginals—every corresponding bound holds simultaneously, so the tightest
    certified inflation factor is their **minimum** (`min{κ_T(O), κ_O^{adv}}`,
    or `min{κ_T(O), κ_O^{adv}, κ_O^{nd}}` if non-dilution and `h_blk` are also
    declared). In the `spacetime.py` implementation (`choose_partition`),
    when `is_complete_grid=False` and `kappa_fn` is `separable_kappa`, the
    library requires `fallback_kappa_fn` and routes deterministically to
    `fallback_kappa_fn` (`topological_kappa_fn` via Claim S2B(4),
    `route="fallback_topological"`); `(S1D-2)` and `(S1D-3)` are argued in this
    document and are not exposed as runtime functions in `spacetime.py`.

#### 4. Lemma R condition on the mask `O`

Under missingness, the mask `O` is part of the conditioning design `D = (O, E_s,
timestamps, Π, Φ)` (§9.5.1), and the estimand is the conditional mean over the
observed items `θ(D) = (1/n) Σ_{(i,t) ∈ O} E[s_{i,t} | D]`. By Lemma R,
Claim S1D and Claim S2B(4) hold conditionally on `D` if and only if the
mask `O` is **exogenous to the losses**—that is, conditioning on `O` does not
alter the conditional second-moment premise `(P2v)` or the conditional
covariance bound `P1_c(κ_O)`:

*   **Admissible (exogenous outages):** Scheduled sensor maintenance, power or
    telemetry packet drops unrelated to traffic conditions, or fixed sensor
    commission/decommission timestamps.
*   **Inadmissible (outcome-dependent / endogenous outages):** Congestion-driven
    sensor failures (for example loop detectors reporting `0` / missing
    precisely when traffic jams cause large prediction errors) or
    error-triggered retries/dropouts. When `O` depends on the realized losses
    `s_{i,t}`, conditioning on `O` induces both **selection bias** in `θ(D)`
    relative to the full-grid mean and **collider bias** in
    `Cov(s_{i,t}, s_{j,s} | O)`.

---

## 9.2 Non-separable correlation dominance: counterexample and factored spectral bounds

Non-linear loss functions (`s_{i,t} = e_{i,t}²` or `|e_{i,t}|`), non-separable
variances, or incomplete observation sets `O ⊊ V_s × T_grid` need not satisfy
`(S1)`. Suppose instead that the item correlations satisfy the **pointwise
correlation-dominance premise**:

$$|\mathrm{Cov}(s_{i,t},\, s_{j,s})| \;\le\; \Phi_s(i, j)\,\Phi_t(t, s)\,\sigma_{i,t}\,\sigma_{j,s} \qquad \text{whenever } (a(i), w(t)) \ne (a(j), w(s)), \tag{T1-prod}$$

where `Φ_s ∈ ℝ_{\ge 0}^{N_s × N_s}` and `Φ_t ∈ ℝ_{\ge 0}^{T × T}` are symmetric
non-negative matrices with unit diagonals `Φ_s(i, i) = 1`, `Φ_t(t, t) = 1`.
Decompose `Φ_s` and `Φ_t` into their within-cluster (block-diagonal,
**including the unit diagonal**) and cross-cluster (cut) parts:

$$\Phi_s \;=\; \Phi_s^{\mathrm{in}} + \Phi_s^{\mathrm{cut}}, \qquad \Phi_t \;=\; \Phi_t^{\mathrm{in}} + \Phi_t^{\mathrm{cut}},$$

where `Φ_s^{in}(i, j) = Φ_s(i, j) 1{a(i) = a(j)}` (`Φ_s^{in} ≥ I_{N_s}`
entrywise) and `Φ_s^{cut}(i, j) = Φ_s(i, j) 1{a(i) ≠ a(j)}`. Let
`κ_{T,s} := 1 + λ_max(Φ_s^cut)` and `κ_{T,t} := 1 + λ_max(Φ_t^cut)` be the
separate one-dimensional Lemma T′ bounds for the spatial and temporal
partitions.

A declaration of `Φ` must come from outside the evaluated sample. One
construction from the model's receptive field and a declared data dependence
range is given in [THEORY_RANGE_ENVELOPE.md](THEORY_RANGE_ENVELOPE.md).

### 9.2.1 Counterexample: why `κ_{T,s} κ_{T,t}` is NOT an upper bound on `1 + λ_max(Φ^cut)` when blocks have internal edges

> **Proposition S2A (Refutation of naive `κ_{T,s} κ_{T,t}` on product cuts).**
> For a product partition `Π = Π_s × Π_t`, two items `(i, t)` and `(j, s)` share
> the same space-time block `(a, w)` if and only if `a(i) = a(j)` **and**
> `w(t) = w(s)`. Therefore, the within-block part of `Φ_s ⊗ Φ_t` is
> `Φ_s^{in} ⊗ Φ_t^{in}`, and the space-time cut matrix is:
> $$\Phi^{\mathrm{cut}} \;=\; \Phi_s \otimes \Phi_t \;-\; \Phi_s^{\mathrm{in}} \otimes \Phi_t^{\mathrm{in}} \;=\; \Phi_s^{\mathrm{cut}} \otimes \Phi_t^{\mathrm{in}} \;+\; \Phi_s^{\mathrm{in}} \otimes \Phi_t^{\mathrm{cut}} \;+\; \Phi_s^{\mathrm{cut}} \otimes \Phi_t^{\mathrm{cut}}. \tag{S2}$$
> Whenever `Φ_s^{in} ≠ I_{N_s}` (communities contain internal spatial edges) and
> `Φ_t^{cut} ≠ 0` (at least two time windows), `1 + λ_max(Φ^cut)` can **strictly
> exceed** `κ_{T,s} κ_{T,t} = (1 + λ_max(Φ_s^cut))(1 + λ_max(Φ_t^cut))`.

*Argument (minimal `2 × 2` counterexample).* Take `N_s = 2` connected spatial
nodes in a **single** spatial community (`G_s = 1`, `C_1 = {0, 1}`) with
`Φ_s = [[1, φ_s], [φ_s, 1]]` (`φ_s ∈ (0, 1]`), and `T = 2` time steps in
`G_t = 2` windows (`W_1 = {0}, W_2 = {1}`) with `Φ_t = [[1, φ_t], [φ_t, 1]]`
(`φ_t ∈ (0, 1]`).

1.  Because `G_s = 1`, there are no spatial cut edges: `Φ_s^cut = 0`,
    `Φ_s^in = Φ_s`, and **`κ_{T,s} = 1`**.
2.  Because each window has one timestamp, `Φ_t^in = I_2`,
    `Φ_t^cut = [[0, φ_t], [φ_t, 0]]`, and **`κ_{T,t} = 1 + φ_t`**.
3.  Hence `κ_{T,s} κ_{T,t} = 1 + φ_t`.
4.  By `(S2)`, the space-time cut matrix between block `(1, 1)` and block
    `(1, 2)` is
    $$\Phi^{\mathrm{cut}} \;=\; \Phi_s^{\mathrm{in}} \otimes \Phi_t^{\mathrm{cut}} \;=\; \begin{pmatrix} 1 & \phi_s \\ \phi_s & 1 \end{pmatrix} \otimes \begin{pmatrix} 0 & \phi_t \\ \phi_t & 0 \end{pmatrix},$$
    whose largest eigenvalue is `λ_max(Φ^cut) = (1 + φ_s) φ_t`.
5.  Choosing unit item variances `σ = (1, 1, 1, 1)ᵀ`, zero within-window
    spatial correlation `Cov(s_{0,t}, s_{1,t}) = 0` (which is valid because
    `(T1-prod)` constrains only cross-block pairs, and `(T2)` holds with
    equality `Var(S_{1,w}) = 2 = Σ_{i} σ_{i,w}²`), and cross-window covariances
    equal to `Φ^cut` (PSD whenever `(1 + φ_s) φ_t ≤ 1`) achieves the exact
    variance inflation
    $$\mathrm{VIF} \;=\; 1 + (1 + \phi_s)\phi_t \;>\; 1 + \phi_t \;=\; \kappa_{T,s}\kappa_{T,t}. \quad \blacksquare$$

*Why this happens (and why S1 and S2A do not contradict each other):* Slicing a
spatial community `C_a` across time windows turns every **internal** spatial
edge `{i, j} ⊆ C_a` into two cross-block spatio-temporal pairs
`((i, t), (j, s))` across the window boundary, represented by the middle term
`Φ_s^{in} ⊗ Φ_t^{cut}` in `(S2)`. Proposition S2A compares the **topological
bounds** `1 + λ_max(Φ^cut)` and `κ_{T,s} κ_{T,t}` under `(T1-prod)` (where
within-window spatial correlation can be `0` while cross-window correlation is
`(1 + φ_s)φ_t`); under exact Kronecker separability `(S1)`, within-window
spatial correlation is forced to be `φ_s` (inflating the denominator
`Var(S_{1,w}) = 2(1 + φ_s)`), so Claim S1 still gives
`VIF = VIF_s · VIF_t = 1 · (1 + φ_t)`.

### 9.2.2 Claim S2B: computable one-dimensional spectral upper bounds for `1 + λ_max(Φ^cut)`

Although `κ_{T,s} κ_{T,t}` fails when `Φ_s^{in} ≠ I_{N_s}`, identity `(S2)`
yields both an **exact factored algorithm** and **closed-form one-dimensional
spectral upper bounds** that require running `topological_kappa` only on the
separate `N_s`-node spatial graph and `T`-step temporal graph (never
materializing the `N_s T × N_s T` matrix):

> **Claim S2B (Factored spectral bounds under correlation dominance).**
> Assume `(T1-prod)` and `(T2)` (`Σ_{a,w} Var(S_{a,w}) ≥ Σ_{i,t} σ_{i,t}² > 0`)
> on `O = V_s × T_grid`. Let:
>
> * `λ_s^cut := λ_max(Φ_s^cut)`, `λ_s^in := λ_max(Φ_s^in) ≥ 1`, and
>   `λ_s := λ_max(Φ_s) ≥ 1`;
> * `λ_t^cut := λ_max(Φ_t^cut)`, `λ_t^in := λ_max(Φ_t^in) ≥ 1`, and
>   `λ_t := λ_max(Φ_t) ≥ 1`.
>
> Then `P1_c(κ_T)` holds with `κ_T = 1 + λ_max(Φ^cut)`, which satisfies:
>
> 1. *(Global Kronecker product bound — partition-free)*
>    $$1 + \lambda_{\max}(\Phi^{\mathrm{cut}}) \;\le\; \lambda_{\max}(\Phi_s)\,\lambda_{\max}(\Phi_t) \;=\; \lambda_s\,\lambda_t.$$
>
> 2. *(Partition-aware factored Weyl bound)*
>    $$1 + \lambda_{\max}(\Phi^{\mathrm{cut}}) \;\le\; 1 + \min\Bigl\{\,\lambda_s^{\mathrm{cut}}\,\lambda_t + \lambda_s^{\mathrm{in}}\,\lambda_t^{\mathrm{cut}},\;\; \lambda_s\,\lambda_t^{\mathrm{cut}} + \lambda_s^{\mathrm{cut}}\,\lambda_t^{\mathrm{in}},\;\; \lambda_s\,\lambda_t - 1\,\Bigr\}. \tag{S3}$$
>
> 3. *(Exact equality in all three boundary regimes)* Inequality `(S3)` holds
>    with **exact equality** in each of the following cases:
>    * **Community-only partition (`G_t = 1`):** `λ_t^cut = 0`, `λ_t^in = λ_t`,
>      and `1 + λ_max(Φ^cut) = 1 + λ_s^cut λ_t`.
>    * **Window-only partition (`G_s = 1`):** `λ_s^cut = 0`, `λ_s^in = λ_s`, and
>      `1 + λ_max(Φ^cut) = 1 + λ_s λ_t^cut`.
>    * **Singleton blocks (`Φ_s^in = I_{N_s}`, `Φ_t^in = I_T`):**
>      `1 + λ_max(Φ^cut) = (1 + λ_s^cut)(1 + λ_t^cut) = κ_{T,s} κ_{T,t} = λ_s λ_t`.
>
> 4. *(Missing observations `O ⊊ V_s × T_grid`)* If only `O ⊆ V_s × T_grid` is
>    observed and premises `(T1)` and `(T2)` hold on `O`, the observed cut
>    matrix `Φ_O^cut` is a principal submatrix of `Φ^cut`, so
>    `λ_max(Φ_O^cut) ≤ λ_max(Φ^cut)` by Perron–Frobenius monotonicity, and
>    `(1)–(2)` remain valid upper bounds for the observed items without
>    modification. `(T2)` on `O` holds automatically for singleton blocks
>    (each `S_{a,w}^O` is a single item) and whenever all within-block
>    covariances of observed items are non-negative; it is not implied by
>    `(T2)` on `V_s × T_grid`.

*Argument.*
**(1)** Because `Φ_s^in ≥ I_{N_s} ≥ 0` and `Φ_t^in ≥ I_T ≥ 0` entrywise, their
Kronecker product satisfies `Φ_s^in ⊗ Φ_t^in ≥ I_{N_s} ⊗ I_T = I_{N_s T}`
entrywise. By `(S2)`:
$$0 \;\le\; I_{N_s T} + \Phi^{\mathrm{cut}} \;=\; \Phi_s \otimes \Phi_t - \bigl(\Phi_s^{\mathrm{in}} \otimes \Phi_t^{\mathrm{in}} - I_{N_s T}\bigr) \;\le\; \Phi_s \otimes \Phi_t \qquad \text{entrywise}.$$
By Perron–Frobenius monotonicity of the spectral radius for non-negative
symmetric matrices, `1 + λ_max(Φ^cut) = λ_max(I_{N_s T} + Φ^cut) ≤
λ_max(Φ_s ⊗ Φ_t) = λ_max(Φ_s) λ_max(Φ_t)`.

**(2)** Grouping `(S2)` in two ways gives:
$$\Phi^{\mathrm{cut}} \;=\; \Phi_s^{\mathrm{cut}} \otimes \Phi_t \;+\; \Phi_s^{\mathrm{in}} \otimes \Phi_t^{\mathrm{cut}} \;=\; \Phi_s \otimes \Phi_t^{\mathrm{cut}} \;+\; \Phi_s^{\mathrm{cut}} \otimes \Phi_t^{\mathrm{in}}.$$
Since each summand is symmetric and non-negative, Weyl's inequality
`λ_max(A + B) ≤ λ_max(A) + λ_max(B)` and the Kronecker spectrum identity
`λ_max(A ⊗ B) = λ_max(A) λ_max(B)` yield both branches of `(S3)`.

**(3)** When `G_t = 1`, `Φ_t^cut = 0` so `Φ^cut = Φ_s^cut ⊗ Φ_t` has exact
largest eigenvalue `λ_s^cut λ_t`. When `G_s = 1`, `Φ_s^cut = 0` so
`Φ^cut = Φ_s ⊗ Φ_t^cut` has exact largest eigenvalue `λ_s λ_t^cut`. When
`Φ_s^in = I_{N_s}` and `Φ_t^in = I_T`, `I + Φ^cut = Φ_s ⊗ Φ_t =
(I + Φ_s^cut) ⊗ (I + Φ_t^cut)`.

**(4)** Restricting a non-negative symmetric matrix to a subset of indices `O`
zeroes out the rows and columns outside `O`, which cannot increase the spectral
radius. Lemma T′ ([THEORY.md](THEORY.md)) applied to the observed items
needs `(T1)` and `(T2)` on `O`; with these premises it completes the argument.
For singleton blocks `Var(S_{a,w}^O) = σ_{i,t}²`, so `(T2)` on `O` holds with
equality; with non-negative within-block covariances
`Var(S_{a,w}^O) = Σ σ² + Σ_{i≠j} Cov ≥ Σ σ²`. `(T2)` does not pass to subsets
in general: one block of three unit-variance items with
`Cov(1,2) = −1/2`, `Cov(1,3) = Cov(2,3) = 1/2` has `Var = 3 + 2·(1/2) = 4 ≥ 3`,
but on `O = {1, 2}` `Var = 2 − 1 = 1 < 2`. ∎

**Note on one-dimensional computation.** Every quantity in `(S3)` is bounded
above by an `O(|E_s|)` or `O(T L)` call to `topological_kappa`:

*   `λ_s^cut ≤ topological_kappa(N_s, E_s, c_s, φ_s).lambda_upper`,
*   `λ_s ≤ topological_kappa(N_s, E_s, np.arange(N_s), φ_s).kappa` (singleton
    clusters make every edge a cut edge, so `1 + λ_max(Φ_s − I) = λ_max(Φ_s)`),
*   `λ_s^in ≤ max_a topological_kappa(|C_a|, E_s|_{C_a}, np.arange(|C_a|), φ_s).kappa`
    (or via one call on `E_s^in` with singleton node IDs, since `Φ_s^in − I` is
    the internal adjacency).

Here `c_s` is the vector of spatial community labels, and `E_s|_{C_a}` and
`E_s^in` are the spatial edges inside community `C_a` and inside any community.
`graph.topological_kappa` returns a certified upper bound on the largest
eigenvalue (`lambda_upper`, and `kappa = 1 + lambda_upper`), not the exact
value. Every branch of `(S3)` is non-decreasing in each `λ`, so substituting
these upper bounds keeps `(S3)` a valid upper bound. The library does not
implement this factored computation: `topological_kappa_fn` evaluates
Lemma T′ directly on the materialized item graph built by `spacetime_edges`
(§9.3).

---

## 9.3 The item graph as a strong product on the general `φ` path, and boundary localization of Lemma T′

When the caller uses the general item-level `topological_kappa` path (item
graph from `spacetime_edges`, κ from `topological_kappa_fn`) with spatial edge
bound `φ_s(u, v)` on `{u, v} ∈ E_s` and temporal lag bound `φ_t(ℓ)` for
`1 ≤ ℓ ≤ L`:

1.  **Why the strong product `G_s ⊠ G_t` is required for `(T1)`.**
    The Cartesian product `G_s □ G_t` contains only same-time spatial edges
    `((u, t), (v, t))` (`{u, v} ∈ E_s`) and same-node temporal edges
    `((u, t), (u, t + ℓ))` (`1 ≤ ℓ ≤ L`), setting `Φ_{(u,t),(v,t+ℓ)} = 0` for
    `u ≠ v` and `ℓ ≥ 1`. Under any spatio-temporal propagation (graph
    diffusion, message passing, or cascades), a shock at `(u, t)` correlates
    with spatial neighbour `v ~ u` at time `t + ℓ`. As shown in
    Proposition S2A, slicing a community `C_a` across window boundary
    `w → w + 1` places `(u, t) ∈ B_{a,w}` and `(v, t + ℓ) ∈ B_{a,w+1}` in
    **different** clusters. Thus `(T1)` requires including the cross-lagged
    edges `((u, t), (v, t ± ℓ))` (`{u, v} ∈ E_s`, `1 ≤ ℓ ≤ L`) with weight
    `φ_{st}(u, v, ℓ)` (default `φ_s(u, v) φ_t(ℓ)`), which is the **strong
    product** `G_s ⊠ G_t`. With `φ_{st}(u, v, ℓ) = φ_s(u, v) φ_t(ℓ)`, the
    strong-product matrix is `Φ_s ⊗ Φ_t − I`, so Claim S2B applies directly;
    mathematically, its bounds need no materialization of the `2 L |E_s| T`
    cross-lagged edges.

    In the code, `spacetime_edges` materializes this strong product on the
    observed items (same-time spatial edges with weight `phi_spatial`,
    same-node temporal edges with weight `phi_temporal[l - 1]`, cross-lagged
    edges with weight `phi_cross[l - 1]`, default
    `phi_spatial * phi_temporal[l - 1]`), and `topological_kappa_fn` evaluates
    `graph.topological_kappa` on it; `spacetime_items` only builds the item
    index. `phi_spatial` is a single scalar, so `spacetime_edges` realizes a
    constant `φ_s(u, v) ≡ φ_s`; a per-edge `φ_s(u, v)` requires building the
    edge list and its per-edge `phi` array directly and passing them to
    `topological_kappa_fn`. A lag `l` connects time steps whose values differ
    by exactly `l` (in time units), and only if both are observed.

2.  **Why Lemma T′ does not decay with window length `b` (Proposition S3).** On a stationary time series with window length
    `b = T / G_t ≫ L`, only `2L` of the `b` timestamps in a window touch its
    boundary, so the true temporal window VIF is `1 + O(L / b)`. Lemma T′,
    however, bounds the worst-case Rayleigh quotient over **all** item variance
    vectors `σ ∈ ℝ_{\ge 0}^n`, including vectors concentrated entirely on the
    window boundaries:

> **Proposition S3 (Boundary localization of `λ_max(Φ_t^cut)`).**
> Let `T = G_t b` with `G_t ≥ 2` contiguous windows of length `b ≥ 2L`, and let
> `Φ_t^cut` connect timestamps `(t, s)` in different windows with `1 ≤ |t − s| ≤ L`
> and weight `φ_t(|t − s|) ≥ 0`. Then `Φ_t^cut` is the direct sum of `G_t − 1`
> identical `2L × 2L` bipartite boundary blocks (plus zero rows/columns on the
> `b − 2L` interior timestamps of each window). Consequently:
> $$\lambda_{\max}(\Phi_t^{\mathrm{cut}}) \;\text{is completely independent of the window length } b \text{ for all } b \ge 2L.$$
> In particular, for `L = 1` with lag-1 correlation `φ_t`, `λ_max(Φ_t^cut) = φ_t`
> for **every** window length `b ≥ 2` (and on the strong product with `G_s = 1`,
> `λ_max(Φ^cut) = λ_max(Φ_s) φ_t` for all `b ≥ 2`).

*Argument.* Any pair `(t, s)` in different windows with `|t − s| ≤ L` must have
one endpoint in the last `L` steps of some window `w` and the other in the
first `L` steps of window `w + 1`. For `b ≥ 2L`, the `L`-step right boundary of
window `w` is disjoint from the `L`-step left boundary of window `w`. Hence
`Φ_t^cut` decomposes into `G_t − 1` vertex-disjoint `2L × 2L` bipartite blocks
across the `G_t − 1` window boundaries, together with isolated interior
vertices. Its largest eigenvalue equals the largest eigenvalue of a single
`2L × 2L` boundary block, which does not depend on `b`. For `L = 1`, each
boundary block is a single edge `[[0, φ_t], [φ_t, 0]]` with largest eigenvalue
`φ_t`. ∎

*Design implication:* Proposition S3 explains why the cluster-level separable
declaration `κ = κ_s κ_t` (§9.1) is the primary default and item-level
`topological_kappa` (§§9.2–9.3) is the conservative fallback: `κ_t` at the
window level decays to `1` as window length `b` exceeds the correlation memory,
whereas item-level `λ_max(Φ_t^cut)` protects against an adversary that places
all variance on the boundary timestamps of each window.

---

## 9.4 Per-window centring for refutation checks, and behaviour under drift

### 9.4.1 Why global centring fails on space-time graphs under temporal drift

Suppose block totals follow the drifting-mean model without error correlation
(`κ = 1`):

$$S_{a,w} \;=\; n_{a,w}\,\theta_w + \varepsilon_{a,w}, \qquad \mathbb{E}[\varepsilon_{a,w}] = 0, \qquad \mathrm{Cov}(\varepsilon_{a,w}, \varepsilon_{b,v}) = 0 \;\; ((a,w) \ne (b,v)),$$

where `θ_w` is the per-item expected loss in window `w` and
`θ = (1/n) Σ_w N_w θ_w` (`N_w := Σ_{a=1}^{G_s} n_{a,w}`). Under global-mean
residualisation `e_{a,w}^{glob} := S_{a,w} − n_{a,w} θ̂`, the expected residual
is `E[e_{a,w}^{glob}] = n_{a,w}(θ_w − θ)`. Along every **same-window spatial
edge** `((a, w), (b, w))`, the standardized residual product has expectation

$$\mathbb{E}\bigl[z_{a,w}^{\mathrm{glob}}\,z_{b,w}^{\mathrm{glob}}\bigr] \;=\; \sqrt{n_{a,w} n_{b,w}}\,(\theta_w - \theta)^2 + O(1/n) \;>\; 0 \qquad (\text{whenever } \theta_w \ne \theta),$$

and along every **temporal edge** `((a, w), (a, w+1))` it has expectation
`√(n_{a,w} n_{a,w+1}) (θ_w − θ)(θ_{w+1} − θ) + O(1/n)`, which is positive at
every window boundary under a monotone trend except the single sign crossing.
Thus a purely temporal shift (`θ_w ≠ θ`) produces `O(n)` positive bias in the
edge statistic `T` (`check_uncorrelated_edges`) and in the super-batch variance
(`check_kappa_batches`), falsely refuting `κ = 1` with probability `→ 1`. Here
`z_{a,w}^{glob} := e_{a,w}^{glob} / √n_{a,w}`.

### 9.4.2 Lemma K′ — per-window centring and rank-`G_t` projection bias

For any product partition with `G_s ≥ 2` spatial communities per window, define
the **per-window sample mean** and **per-window residualised total**:

$$\hat\theta_w \;:=\; \frac{1}{N_w}\sum_{a=1}^{G_s} S_{a,w}, \qquad e_{a,w}^{\mathrm{win}} \;:=\; S_{a,w} - n_{a,w}\hat\theta_w, \qquad z_{a,w}^{\mathrm{win}} \;:=\; \frac{e_{a,w}^{\mathrm{win}}}{\sqrt{n_{a,w}}}.$$

**Rank-`G_t` centring bias.** Passing `e_{a,w}^{win}` into an
unmodified `check_uncorrelated_edges` (which subtracts only the rank-1 global
centring bias `O(|E| / (G_s G_t))`) leaves a negative numerator bias of order
`−G_t d̄_s τ² / 2` because `P_w` projects out one degree of freedom **per
window** (`O(1/G_s)` per spatial edge, which is `G_t` times larger than
`O(1/(G_s G_t))`). On a balanced grid that unadjusted bias shifts `T` by
`−O(√(G_t d̄_s / G_s))` standard errors. Lemma K′ gives both (a) the
exact residual bias of the plug-in `B̂_win` (`O(√(G_t d̄_s) / G_s^{3/2})`
standard errors, a factor of `G_s` smaller than the rank-1 error) and (b) the
exact finite-sample unbiased Sherman–Morrison correction `B̂_win^{unb}`
whenever `max_a n_{a,w} < N_w / 2`. The library does not implement the
per-window edge statistic `T_win := (Σ_E z_c^{win} z_d^{win} − B̂_win)/√V̂_win` of Lemma K′(2) on the block graph, and its
asymptotic null distribution is not covered here; the checks that run are
listed in §9.4.4.

In Lemma K′, `W` is the adjacency matrix of the block graph (one vertex per
non-empty block `c = (a, w)`), `E` its edge set, `τ_c² = Var(ζ_c)` the
per-block noise variance on the `z` scale, and
`V_win := 2 Σ_{c,d} (A_win)_{cd}² τ_c² τ_d²` (with `A_win` as in part 2) the
null variance of the
quadratic form `Σ_E z_c^{win} z_d^{win}` up to fourth-cumulant terms, used to
express biases in standard errors.

> **Lemma K′ (Per-window residualised totals and edge test null).**
> Assume `S_{a,w} = n_{a,w} θ_w + ε_{a,w}` where `θ_w ∈ ℝ` is an arbitrary
> window-dependent mean and the noise terms `ε_{a,w}` are independent with
> `E[ε_{a,w}] = 0` and variances `Var(ε_{a,w}) = n_{a,w} τ_{a,w}²`. Within each
> window `w`, let `q_a^{(w)} := √(n_{a,w} / N_w)` and `P_w := I_{G_s} −
> q^{(w)}(q^{(w)})ᵀ`, and let `P_win := ⨁_{w=1}^{G_t} P_w`. Then:
>
> 1. *(Exact invariance to drift and counts)* On every draw,
>    `e_{a,w}^{win} = ε_{a,w} − (n_{a,w}/N_w) Σ_{b=1}^{G_s} ε_{b,w}` and
>    `z_{\cdot,w}^{win} = P_w ζ_{\cdot,w}` (`ζ_{a,w} := ε_{a,w}/√n_{a,w}`). Both
>    the deterministic count component `n_{a,w} θ_w` and the temporal drift
>    `θ_w` cancel identically.
>
> 2. *(Exact null mean, plug-in residual bias, and Sherman–Morrison unbiased
>    correction)* Let `E = E_spatial ⊔ E_temporal` be the block adjacency edges,
>    where `E_spatial = ⊔_{w=1}^{G_t} E_w` connects blocks in the same window
>    `w` and `E_temporal` connects blocks in distinct windows `w ≠ v`. Let
>    `A_win := ½ P_win W P_win`. Then:
>    * **Temporal edges:** For every `((a, w), (b, v)) ∈ E_temporal` (`w ≠ v`),
>      `E[z_{a,w}^{win} z_{b,v}^{win}] = 0` identically.
>    * **Spatial edges:** For every `((a, w), (b, w)) ∈ E_w`,
>      $$B_{a,b,w} \;:=\; \mathbb{E}\bigl[z_{a,w}^{\mathrm{win}} z_{b,w}^{\mathrm{win}}\bigr] \;=\; \frac{\sqrt{n_{a,w} n_{b,w}}}{N_w}\left(\frac{1}{N_w}\sum_{k=1}^{G_s} n_{k,w}\tau_{k,w}^2 - \tau_{a,w}^2 - \tau_{b,w}^2\right),$$
>      so `E[Σ_E z_c^{win} z_d^{win}] = Σ_c (A_win)_{cc} τ_c² = Σ_w Σ_{E_w} B_{a,b,w}`.
>    * **Plug-in correction `B̂_win` and its exact residual bias:**
>      Replacing `τ_{a,w}²` by `(z_{a,w}^{win})²` gives the `O(|E| + G)` plug-in
>      $$\widehat B_{\mathrm{win}} \;:=\; \sum_{w=1}^{G_t} \sum_{((a,w),(b,w)) \in E_w} \frac{\sqrt{n_{a,w} n_{b,w}}}{N_w}\left(\frac{1}{N_w}\sum_{k=1}^{G_s} n_{k,w}(z_{k,w}^{\mathrm{win}})^2 - (z_{a,w}^{\mathrm{win}})^2 - (z_{b,w}^{\mathrm{win}})^2\right) \;=\; \sum_{c} (A_{\mathrm{win}})_{cc} (z_c^{\mathrm{win}})^2.$$
>      Because `E[(z_{a,w}^{win})²] = (1 − 2n_{a,w}/N_w)τ_{a,w}² +
>      (n_{a,w}/N_w²) Σ_{k=1}^{G_s} n_{k,w} τ_{k,w}²`, the centred numerator
>      `Σ_E z_c^{win} z_d^{win} − B̂_win` has exact residual expectation:
>      $$\mathbb{E}\!\left[\sum_E z_c^{\mathrm{win}} z_d^{\mathrm{win}} - \widehat B_{\mathrm{win}}\right] \;=\; \sum_{w=1}^{G_t}\sum_{((a,w),(b,w)) \in E_w} \frac{\sqrt{n_{a,w} n_{b,w}}}{N_w}\left[\frac{2}{N_w^2}\sum_{k=1}^{G_s} n_{k,w}^2\tau_{k,w}^2 - \frac{2}{N_w}\bigl(n_{a,w}\tau_{a,w}^2 + n_{b,w}\tau_{b,w}^2\bigr) + \frac{(n_{a,w} + n_{b,w})N_w - \sum_{k=1}^{G_s} n_{k,w}^2}{N_w^3}\sum_{k=1}^{G_s} n_{k,w}\tau_{k,w}^2\right].$$
>      Under equal counts `n_{a,w} = N_w / G_s` and homoscedastic
>      `τ_{a,w}² ≡ τ_w²`, this residual bias equals `−τ_w² / G_s²` per spatial
>      edge (`−|E_w| τ_w² / G_s² = −d̄_s τ_w² / (2 G_s)` per window). Summing
>      over `G_t` windows and dividing by `√V_win ≍ τ² √(G_t G_s d̄_s / 2)` gives a
>      standardized residual bias of `O(√(G_t d̄_s) / G_s^{3/2})` standard
>      errors—a factor of `G_s` smaller than the unadjusted rank-1 bias
>      `O(√(G_t d̄_s / G_s))`.
>    * **Exact finite-sample unbiased correction `B̂_win^{unb}` via
>      Sherman–Morrison (`max_a n_{a,w} < N_w / 2`):** Whenever
>      `d_{k,w} := 1 − 2n_{k,w}/N_w > 0` for all `k = 1, …, G_s` (which holds
>      whenever no single community holds `≥ 50%` of window `w`'s items, for
>      example `G_s ≥ 3` balanced communities), the expectation matrix
>      `M_w := P_w ∘ P_w = diag(d_{\cdot,w}) + q_w² (q_w²)ᵀ` is invertible in
>      `O(G_s)` time by the Sherman–Morrison formula:
>      $$\hat\tau_{k,w}^2 \;:=\; \frac{(z_{k,w}^{\mathrm{win}})^2 - q_{k,w}^2\,\gamma_w}{1 - 2q_{k,w}^2}, \qquad \gamma_w \;:=\; \frac{\sum_{l=1}^{G_s} \frac{q_{l,w}^2 (z_{l,w}^{\mathrm{win}})^2}{1 - 2q_{l,w}^2}}{1 + \sum_{l=1}^{G_s} \frac{q_{l,w}^4}{1 - 2q_{l,w}^2}}, \qquad (q_{k,w}^2 = n_{k,w}/N_w).$$
>      *(For equal counts `n_{k,w} = N_w / G_s` with `G_s ≥ 3`, `q_{k,w}² = 1/G_s`
>      and `γ_w = (1 / (G_s − 1)) Σ_{l=1}^{G_s} (z_{l,w}^{win})²`, so this
>      simplifies to `τ̂_{k,w}² = (G_s / (G_s − 2))[(z_{k,w}^{win})² − (1 / (G_s(G_s − 1)))
>      Σ_{l=1}^{G_s} (z_{l,w}^{win})²]`.)* Then `E[τ̂_{k,w}²] = τ_{k,w}²`
>      **identically**, and substituting `τ̂_{k,w}²` for `(z_{k,w}^{win})²` in
>      `B̂_win` defines `B̂_win^{unb} := Σ_c (A_win)_{cc} τ̂_c²`, which satisfies
>      $$\mathbb{E}\!\left[\sum_E z_c^{\mathrm{win}} z_d^{\mathrm{win}} - \widehat B_{\mathrm{win}}^{\mathrm{unb}}\right] \;=\; 0$$
>      **with exact finite-sample equality** for any heteroscedastic
>      `τ_{a,w}²`, any counts `n_{a,w} < N_w / 2`, and any drift `θ_w`.

*Argument.*
**(1)** Substitute `S_{a,w} = n_{a,w} θ_w + ε_{a,w}` into `S_{a,w} − n_{a,w}
(Σ_b S_{b,w} / N_w)`; the terms `n_{a,w} θ_w` cancel. Dividing by `√n_{a,w}`
gives `z_{\cdot,w}^{win} = (I − q^{(w)}(q^{(w)})ᵀ) ζ_{\cdot,w} = P_w ζ_{\cdot,w}`.

**(2)** Because `P_win = ⨁_w P_w` is block-diagonal across windows, for `w ≠ v`
the vectors `z_{\cdot,w}^{win}` and `z_{\cdot,v}^{win}` are functions of
disjoint independent noise vectors `ζ_{\cdot,w}` and `ζ_{\cdot,v}`, so
`E[z_{a,w}^{win} z_{b,v}^{win}] = 0`. Within window `w`, expanding
`z_{a,w}^{win} = ζ_{a,w} − q_a^{(w)} Σ_k q_k^{(w)} ζ_{k,w}` and taking
expectations under `E[ζ_{k,w} ζ_{l,w}] = δ_{kl} τ_{k,w}²` gives `B_{a,b,w}` and
`E[(z_{a,w}^{win})²] = ((P_w)_{a,a}²) τ_{a,w}² + Σ_{k ≠ a} (P_w)_{a,k}² τ_{k,w}²
= (1 − 2q_a²) τ_{a,w}² + q_a² Σ_k q_k² τ_{k,w}²`. Substituting
`E[(z_{k,w}^{win})²]` into `B̂_win` and subtracting from `B_{a,b,w}` yields the
exact residual bias formula. Finally, writing `E[(z_{\cdot,w}^{win})²] =
(D_w + u_w u_wᵀ) τ_{\cdot,w}²` with `D_w = diag(1 − 2q_w²)` and `u_w = q_w²`,
the Sherman–Morrison identity
`(D_w + u_w u_wᵀ)⁻¹ = D_w⁻¹ − (D_w⁻¹ u_w u_wᵀ D_w⁻¹) / (1 + u_wᵀ D_w⁻¹ u_w)`
gives `M_w⁻¹ (z_{\cdot,w}^{win})² = τ̂_{\cdot,w}²` whenever `1 − 2q_{k,w}² > 0`,
so `E[τ̂_{k,w}²] = τ_{k,w}²` and `E[B̂_win^{unb}] = Σ_c (A_win)_{cc} τ_c² =
E[Σ_E z_c^{win} z_d^{win}]`. ∎

**No-clipping invariant.** On a finite-sample draw, `τ̂_{k,w}²` can be negative
when `(z_{k,w}^{win})² < q_{k,w}² γ_w` and **must not be clipped at zero** when
forming `B̂_win^{unb}`. Expanding `A_win = ½ P_win W P_win` gives the exact
diagonal coefficient on block `c = (a, w)`:

$$(A_{\mathrm{win}})_{(a,w),(a,w)} \;=\; -q_{a,w}\sum_{b:\,((a,w),(b,w))\in E_w} q_{b,w} \;+\; q_{a,w}^2 \sum_{((u,w),(v,w))\in E_w} q_{u,w}\,q_{v,w},$$

where the first sum is over neighbours of `a` in `E_w` and the second sum runs
over **all** spatial edges in window `w`. On a regular equal-count graph
(`q_{a,w} ≡ 1/√G_s`, degree `d_w`), every diagonal entry is negative
(`(A_win)_{cc} = −d_w / (2 G_s) < 0`), so clipping `τ̂_c² ← max(0, τ̂_c²)` makes
`B̂_win^{unb}` more negative, increases `T_win`, and inflates the Type I
refutation rate under `H_0`. On heterogeneous/non-regular graphs, however,
`(A_win)_{cc}` can be positive on low-degree/pendant blocks—for example, with
`5` equal blocks (`q_{a,w} = 1/√5`), `K_4` on blocks `1..4` (`6` edges), and
block `5` attached only to block `1` (`|E_w| = 7` edges),
`(A_win)_{5,5} = −1/5 + 7/25 = +2/25 > 0` while `(A_win)_{1,1} = −4/5 +
7/25 = −13/25 < 0` (block 1 has degree `4`) and
`(A_win)_{c,c} = −3/5 + 7/25 = −8/25 < 0` for `c = 2, 3, 4`. Clipping on a block with `(A_win)_{cc} > 0` increases its
term in `B̂_win^{unb}` (lowering `T_win`), whereas clipping on a block with
`(A_win)_{cc} < 0` decreases its term (raising `T_win`); thus on mixed graphs
the net sign of the clipping bias on `B̂_win^{unb}` and `T_win` is
indeterminate (the interval formula itself is unaffected). Leaving
`τ̂_{k,w}²` unclipped preserves exact unbiasedness `E[B̂_win^{unb}] =
E[Σ_E z_c^{win} z_d^{win}]` on every graph topology.

**Runtime implementation note (`spacetime.py`, `refutation.py`).** The
library does not compute `T_win`, `B̂_win` or `B̂_win^{unb}` on the block graph.
The only runtime use of per-window centring is the spatial marginal check
(item 3 below), which passes the `G_s` community residuals `e_{a,•}^{win}` and
counts `n_{a,•}` to the unmodified `check_uncorrelated_edges` (`κ_s = 1`) or
`check_kappa_batches` (`κ_s > 1`). On the separable route the grid is
complete, and then this is exactly the static edge test of Claim 6(d) on the
community graph (argument in §9.4.4). Its plug-in residual bias is the `G_t = 1`
instance of Lemma K′(2): under equal counts and homoscedastic `τ²`,
`−τ²/G_s²` per community edge, `−d̄_C τ²/(2 G_s)` in total, where `d̄_C` is
the average degree of the community graph, that is `−O(√d̄_C / G_s^{3/2})`
standard errors. The bias is negative, so the one-sided check is
conservative at level `α_ref` under `H_0`, at the cost of an `O(G_s^{-3/2})`
power loss for small `G_s`. The joint check on the block graph is globally
centred; §9.4.4 states what that implies under drift.

**How Lemma K′ behaves when counts `n_{a,w}` vary because of missingness:**

1.  **Arbitrary positive counts `n_{a,w} ≥ 1` are already exact:** All formulas
    in Lemma K′(1)–(2) are derived for arbitrary heterogeneous `n_{a,w} ≥ 1`
    with `N_w = Σ_{a:\,n_{a,w}\ge 1} n_{a,w}` and `q_{a,w} = √(n_{a,w} / N_w)`;
    they do **not** require `n_{a,w} = |C_a| · |W_w|` to factor as a product.
    Even when missingness varies across communities and windows,
    `e_{a,w}^{win} = S_{a,w} − n_{a,w} θ̂_w` cancels `n_{a,w} θ_w` identically.
2.  **Empty blocks (`n_{a,w} = 0`) and degenerate windows (`G_{s,w} ≤ 2` or
    `max_a n_{a,w} ≥ N_w / 2`):**
    *   Empty blocks (`n_{a,w} = 0`) are omitted from the cluster vector (`G`
        counts only non-empty blocks), so in window `w` the projector `P_w` has
        dimension `G_{s,w} := |{a : n_{a,w} ≥ 1}|`.
    *   If missingness leaves only `G_{s,w} = 1` active community in window `w`,
        then `q_{a,w} = 1`, `P_w = 0`, `z_{a,w}^{win} = 0`, and `E_w = ∅`
        (window `w` contributes `0` to both the numerator and `B̂_win`).
    *   Whenever `max_a n_{a,w} < N_w / 2` in window `w`, the Sherman–Morrison
        estimator `τ̂_{k,w}²` and `B̂_win^{unb}` are well-defined
        (`1 − 2q_{k,w}² > 0`) and remain **exact finite-sample unbiased** under
        arbitrary missingness. If in some window `w` a single community holds
        `≥ 50%` of that window's observed items (`max_a n_{a,w} ≥ N_w / 2`, for
        example when `G_{s,w} = 2`), that window uses the plug-in `B̂_win,w`
        (whose exact expectation is still given by Lemma K′(2)).
3.  **Spatial marginal totals under non-product missingness and temporal
    drift:** When `O` is not a product grid (`n_{a,w} ≠ n_a^s n_w^t`) and `θ_w`
    drifts over time, the uncentred spatial marginal `S_{a,•} = Σ_w S_{a,w}`
    has expectation `Σ_w n_{a,w} θ_w ≠ n_{a,•} θ` (a community with sensor
    dropouts during low-loss windows has a higher marginal mean per item than
    one with complete records). Summing the per-window centred residuals
    instead, `e_{a,•}^{win} := Σ_w e_{a,w}^{win} = Σ_w (S_{a,w} − n_{a,w} θ̂_w)`,
    removes `Σ_w n_{a,w} θ_w` **identically** (`E[e_{a,•}^{win}] = 0`) for any
    missingness pattern `n_{a,w}` and any temporal drift `θ_w`. The spatial
    marginal check in `spacetime_interval` uses `e_{a,•}^{win}`. At runtime
    the marginal checks run only on the separable route, which
    `choose_partition` takes only when called with `is_complete_grid=True`
    (the caller should pass `index.is_complete_grid`); on a complete grid the
    non-product case of this item does not arise. It applies to callers who
    run `check_uncorrelated_edges` on `e_{a,•}^{win}` themselves.

### 9.4.3 Behaviour of `M̂_G` under temporal drift

Unlike the `κ` checks, the tail statistic `M̂_G = G Σ_{a,w} S_{a,w}² /
(Σ_{a,w} S_{a,w})²` tests the **uncentred** second-moment premise `P2*_c(M_c)`
on the non-negative block totals `S_{a,w} ≥ 0` and is **never** per-window
centred:

1.  **When `M` is a valid pooled item-level bound on the evaluated set:**
    By Lemma I′ ([THEORY.md](THEORY.md)), `(1/n) Σ_{i,t} E[s_{i,t}²] ≤ M θ²`
    implies `Σ_{i,t} Var(s_{i,t}) + Σ_{a,w} n_{a,w}(θ_{a,w} − θ)² ≤ n(M − 1)θ²`,
    so `M_c^{default}` remains above the population target
    `G Σ E[S_{a,w}²] / (Σ E[S_{a,w}])²` for **any** temporal drift `θ_w`. Here
    `θ_{a,w}` is the mean per-item loss in block `(a, w)`.
2.  **When `m_item` is declared from a stationary archive without drift:**
    Under equal block sizes `n_{a,w} = m̄`, the same number `G_s` of non-empty
    blocks in every window (`G = G_s G_t`), and window means
    `θ_w = θ(1 + δ_w)` (`Σ_w δ_w = 0`, so that `Σ_{a,w} δ_w = 0`), the
    population target of `M̂_G` decomposes as
    $$\frac{G \sum_{a,w} \mathbb{E}[S_{a,w}^2]}{\bigl(\sum_{a,w} \mathbb{E}[S_{a,w}]\bigr)^2} \;=\; \frac{1}{G}\sum_{a,w} \frac{\mathrm{Var}(S_{a,w})}{\bar\theta_c^2} \;+\; 1 \;+\; \underbrace{\frac{1}{G_t}\sum_{w=1}^{G_t} \delta_w^2}_{\mathrm{CV}_{\theta_w}^2}.$$
    Temporal drift adds `CV_{θ_w}² = Var_w(θ_w) / θ²` to the uncentred second
    moment. If `CV_{θ_w}²` exceeds the slack in an archive-declared `m_item`,
    `M̂_G > M_c` flags `TAIL_UNRESOLVED` (because `P2*_c(M_c^{archive})` is
    genuinely violated on the drifted evaluation set, even though `(P2v)` still
    holds), and the window-split drift check (§9.6) can identify drift as the
    cause when the drift exceeds its detection threshold and `G_t ≥ 4`.

### 9.4.4 What the checks do under temporal drift

`spacetime_interval` runs three kinds of refutation checks. None of them
enters the interval endpoints or the interval status: the interval comes from
`cluster_interval` (or, for `R²`, from the Claim 7 computation, whose status
depends only on the tail checks and the existence condition), and the check
outcomes are only reported ([THEORY.md](THEORY.md): the checks never change
an interval). Therefore none of the findings below affects the coverage of an
issued interval; they affect only how a refutation should be read.

1.  **Joint κ check on the block graph (global centring).**
    `graph.graph_interval_from_shard` forms `e_c = S_c − n_c θ̂` with the
    global mean `θ̂` and runs `check_uncorrelated_edges` (`κ = 1`) or
    `check_kappa_batches` (`κ > 1`) on the cluster graph of the chosen
    partition (`Choice.cluster_edges`, built from the item edges passed to
    `choose_partition`; without them the check is `UNTESTABLE`). Under the
    drifting-mean model of §9.4.1 this check falsely refutes with probability
    tending to `1`. A joint refutation together with a drift `WARNING` (§9.6)
    may be caused by drift alone.
2.  **Temporal marginal check (global centring).** On the separable route
    with `G_s > 1` and `G_t > 1`, the window totals are centred as
    `e_w = S_{•,w} − N_w θ̂` and passed to `check_uncorrelated` (lag-1
    Fisher-z, `κ_t = 1`; `UNTESTABLE` if `G_t < 5`) or `check_kappa` (batch
    means over 30 batches, `κ_t > 1`; `UNTESTABLE` if `G_t < 30`). Under
    drift `E[e_w] = N_w(θ_w − θ)` is a deterministic trend; a smooth (for
    example monotone) trend raises the lag-1 autocorrelation and the
    batch-mean variance, so this
    check can falsely refute `κ_t`. Per-window centring is not available for
    this check: it would set every window total to zero. The temporal
    marginal check cannot separate drift from temporal correlation.
3.  **Spatial marginal check (drift-invariant).** The community residuals
    `e_{a,•}^{win} = Σ_w (S_{a,w} − n_{a,w} θ̂_w)` are free of `θ_w` by
    Lemma K′(1). On a complete grid they coincide with the globally centred
    community marginals, and the check is the static Claim 6(d) edge test
    (`κ_s = 1`) or Proposition B super-batch check (`κ_s > 1`) on the
    community graph:

    *Argument.* Premises: complete grid `O = V_s^{obs} × T_grid^{obs}` and the
    model of Lemma K′.

    1.  On a complete grid, `n_{a,w} = |C_a ∩ V_s^{obs}| · |W_w ∩ T_grid^{obs}|`,
        so `n_{a,w}/N_w = |C_a ∩ V_s^{obs}| / |V_s^{obs}| = n_{a,•}/n` for every
        `w`.
    2.  Hence `Σ_w n_{a,w} θ̂_w = Σ_w (n_{a,•}/n) S_{•,w} = n_{a,•} θ̂`, so
        `e_{a,•}^{win} = S_{a,•} − n_{a,•} θ̂`.
    3.  By Lemma K′(1),
        `e_{a,•}^{win} = ε_{a,•} − (n_{a,•}/n) Σ_b ε_{b,•}` with
        `ε_{a,•} := Σ_w ε_{a,w}`, independent across `a` and mean zero under `H_0`. This
        is the residual structure `z = Pζ`, `P = I − qqᵀ`,
        `q_a = √(n_{a,•}/n)`, assumed by Claim 6(d) (and Proposition B when `κ_s > 1`), so
        `check_uncorrelated_edges` (`κ_s = 1`) and `check_kappa_batches` (`κ_s > 1`) apply to it without modification. ∎

The space-time validation study ([EVIDENCE.md](EVIDENCE.md) §5, block ST5)
measures the spatial marginal check under drift; the joint and temporal
marginal checks under drift are not measured there.

---

## 9.5 Design choice (`choose_partition`) and the space-time Lemma R exogeneity premise

### 9.5.1 Premise: strict exogeneity of the space-time design

Let `D = (O, E_s, timestamps, Π, Φ)` collect all design variables over the
evaluated horizon `[0, T − 1]`. By Lemma R ([THEORY.md](THEORY.md)),
Claim 4 and Claim 7 hold conditionally on `D` for
`θ(D) = (1/n) Σ_{(i,t)∈O} E[s_{i,t} | D]` provided premise `(P2v)`
(cluster-level variance bound) and premise `P1_c(κ)` (between-cluster
dependence) hold under the conditional law `P(· | D)`.

In space-time data, conditioning on the **entire** trajectory `D = D_{0..T-1}`
at time `t < T − 1` conditions on future graph structure and future item
arrivals `D_{t+1..T-1}`. For `E[s_{i,t} | D]` to equal the causal/filtered
expectation `E[s_{i,t} | D_{0..t}]` and for conditional independence /
`P1_c(κ)` to hold without collider bias, the design must satisfy **strict
exogeneity**:

*   Future spatial edges, node degrees, and community structure `E_s(t')`
    (`t' > t`) must not depend on realized losses or residuals `s_{i,t}` at
    time `t` (no error-triggered quarantines, blocks, or re-routing).
*   Future observation counts and timestamps `O_{t'}` (`t' > t`) must not
    depend on past outcomes `s_{i,t}` (no failed requests retried into window
    `w + 1`, no error-driven sensor drop-out).
*   If `partition_graph` is run on the time-aggregated spatial graph
    `∪_{t} E_s(t)`, the entire edge-arrival process over `[0, T − 1]` must be
    exogenous to the losses.

This is a structural premise on the data-generating mechanism; it cannot be
verified from the losses alone.

### 9.5.2 Proposition S5 — design-only partition selection (`choose_partition`)

**Why score by size-adjusted `N_eff(Π)` rather than `G / κ`.** When comparing
candidate partitions `Π` with different cluster-size imbalance
`r(Π) = n_max(Π) / m̄(Π)` (for example, balanced time windows with `r ≈ 1`
versus a skewed community × window product where some cells have small or
large counts), the Cantelli half-width factor
`c(Π) = √((M_c(Π) − 1) κ(Π) (1 − α′) / (α′ G(Π)))` with
`M_c(Π) = M_c^{default}(Π)` is a monotone function of the **size-adjusted
effective sample size**:

$$N_{\mathrm{eff}}(\Pi) \;:=\; \frac{G(\Pi)}{\kappa(\Pi)\,r_{\mathrm{eff}}(\Pi)}, \qquad r_{\mathrm{eff}}(\Pi) \;:=\; \frac{M_c^{\mathrm{default}}(\Pi) - 1}{M - 1} \qquad (M > 1).$$

Indeed `M_c^{default}(Π) − 1 = r_eff(Π)(M − 1)` gives, for `M > 1`,

$$c(\Pi)^2 \;=\; \frac{(M - 1)(1 - \alpha')}{\alpha'\, N_{\mathrm{eff}}(\Pi)},$$

so `c(Π) < 1` if and only if `N_eff(Π) > (M − 1)(1 − α′)/α′`, a threshold
common to all candidates. Bounds on `r_eff`: both branches of Lemma I′ are at
least `1 + r(M − 1)` (because `rM = 1 + r(M − 1) + (r − 1)` and `r ≥ 1`), so
`r_eff(Π) ≥ r(Π)`. On the `rM` branch,
`r_eff(Π) = (rM − 1)/(M − 1) = r(Π) + (r(Π) − 1)/(M − 1)`, which equals `r(Π)`
only when `r(Π) = 1`; on the other branch,
`r_eff(Π) = r(Π) + (CV_n² + 2 CV_n √(r(M − 1)))/(M − 1)`. For equal-size
clusters (`r = 1`, `CV_n = 0`) `r_eff(Π) = 1` and `N_eff(Π) = G(Π) / κ(Π)`.
When `M = m_item = 1`, the formula is `0/0`; `choose_partition` then uses
`r_eff(Π) = r(Π)` (code: `n_max / m_bar`), and the identity for `c(Π)` above
does not apply. Scoring by raw `G(Π) / κ(Π)` without `r_eff(Π)` can select a
skewed product partition that fails the issuance check `c(Π) < 1` over a
balanced window partition that passes.

**Pre-evaluation specification and deterministic tie-breaking:**

1.  Both the **candidate grid** `P_cand` (the list of target spatial cluster
    sizes and temporal window lengths) and the **declarations** (`κ_s, κ_t` or
    `φ_s, φ_t, L`, and `M` / `m_item`) must be fixed before inspecting the
    evaluated losses, residuals, or labels `(s_{i,t}, y_{i,t}, ŷ_{i,t})`.
    Tuning `κ_s, κ_t`, `φ`, or the candidate grid on the evaluated residuals or
    on refutation outcomes violates Lemma R.
2.  Candidates are ranked by `N_eff(Π)` rounded to 12 decimal places. Ties
    must be broken **deterministically** and depend only on `D`: prefer
    (i) candidates with `G_t ≥ 2` over `G_t = 1` (so that the marginal checks
    and, when `G_t ≥ 4`, the drift check of §9.6 can run), then (ii) **fewer
    clusters `G(Π)`** (so each block is larger and better absorbs
    within-block correlation and variance at equal `N_eff`), and finally
    (iii) lexicographically earlier `Candidate.name`.

> **Proposition S5 (Zero selection bias of `choose_partition`).**
> Let `P_cand = {Π_1, …, Π_J}` be a finite family of `J` candidate partitions
> (including community-only, window-only, and product partitions
> `Π_s × Π_t` over a fixed grid of target cluster sizes and window lengths)
> constructed solely from `D = (O, E_s, timestamps)` and a fixed PRNG seed, with
> both `P_cand` and all declarations fixed prior to inspecting `(s_{i,t}, y_{i,t},
> ŷ_{i,t})`. For each `Π ∈ P_cand`, let `κ(Π)` be computed solely from `D` and the
> prior declarations (`κ_O(Π)` from Claim S1D via `κ_s, κ_t`, the auxiliary
> mask premises (`h_σ` for `(S1D-3)`, or non-dilution and `h_blk` for
> `(S1D-1)–(S1D-2)`), and the mask counts `(n_{a,w}, N_{a,w})`, or
> `topological_kappa` / Claim S2B from `φ_s, φ_t`), and let `M_c^{default}(Π)`
> be computed from the cluster counts `n_c(Π)` via Lemma I′. Let `Π* ∈ P_cand` be
> selected by maximizing `N_eff(Π) = G(Π) / (κ(Π) r_eff(Π))` over all of
> `P_cand`, using the deterministic tie-breaking rule above.
>
> Then `Π*` is a deterministic function of `(D, declarations, α)` that never
> inspects `(s_{i,t}, y_{i,t}, ŷ_{i,t})`. By Lemma R ([THEORY.md](THEORY.md)),
> running Claim 4 (or Claim 7) on `Π*` achieves nominal coverage `≥ 1 − α`
> conditionally on `D`, with **no multiple-testing correction** across the `J`
> candidates.

*Argument.* Premises: `P_cand`, the declarations, the PRNG seed and `α` are
fixed before the losses are seen, and the design is exogenous (§9.5.1).

1.  Conditional on `D`, the candidate set `P_cand`, every candidate's cluster
    counts `n_c(Π)`, `M_c^{default}(Π)`, `κ(Π)`, `N_eff(Π)`, and the
    deterministically tie-broken argmax `Π*` are non-random constants.
2.  Applying Lemma R to the single fixed partition `Π*(D)` yields
    `Pr(θ(D) ∉ [L(Π*), U(Π*)] | D) ≤ α`. ∎

**Issuance is not a selection constraint.** `choose_partition` ranks all
candidates and does not exclude those with `c(Π) ≥ 1`; it reports
`clears_issuance` (`G/κ > n_A` with `n_A = (M_c^{default} − 1)(1 − α′)/α′`,
equivalent to `c(Π) < 1`) per candidate in `Choice.candidates_table`. The
consequence concerns efficiency only, never coverage, because the rule
remains a function of `D`, the declarations and `α`:

1.  When `m_item > 1` and the interval is computed with the same `κ(Π*)` and
    `M_c^{default}(Π*)` that were scored, `c(Π) < 1` is equivalent to
    `N_eff(Π) > (M − 1)(1 − α′)/α′` (identity above). The argmax therefore
    clears issuance whenever any candidate does, except when this threshold
    falls between the winner's `N_eff` and a larger `N_eff` that rounds to
    the same 12 decimal places.
2.  When `m_item = 1`, or when the interval uses a larger `kappa` override
    than the scored one, the winner can fail issuance while another candidate
    clears it.
3.  If no candidate clears issuance, the winner is still returned, and for
    metrics other than `R²` `spacetime_interval` reports no interval (status
    `ASSUMPTION_REQUIRED`).

---

## 9.6 Proposition D′ — window-split drift warning on product partitions

For a product partition `Π = Π_s × Π_t` with `G_t ≥ 2` time windows (or a
window-only partition `G_s = 1, G_t ≥ 2`), split the time windows into the
early half `W_early = {0, …, ⌊G_t/2⌋ − 1}` and the late half
`W_late = {⌊G_t/2⌋, …, G_t − 1}`. Let `S_early = {S_{a,w} : w ∈ W_early}`
(`G_1` non-empty blocks, `n_1` items, mean
`θ_early = (1/n_1) Σ_{w ∈ W_early} Σ_a E[S_{a,w}]`) and
`S_late = {S_{a,w} : w ∈ W_late}` (`G_2` non-empty blocks, `n_2` items, mean
`θ_late`).

Run Claim 4 on `S_early` at level `1 − α/2` with its own Lemma I′ default
`M_{c,1}` (computed from the same declared `m_item`) and half-horizon inflation factor
`κ_1`, and on `S_late` at level `1 − α/2` with `M_{c,2}` and `κ_2`, obtaining
`[L_1, U_1]` and `[L_2, U_2]`.

*   **Choice and half-horizon premise of `κ_1, κ_2`:**
    Just as in Proposition D ([THEORY.md](THEORY.md): *"Neither declaration
    is inherited from the whole window... the VIF of a subset is not bounded by
    the VIF of the whole. So `M` and `κ` per half are premises"*),
    `(M_{c,1}, κ_1)` on `S_early` and `(M_{c,2}, κ_2)` on `S_late` are
    **half-horizon premises**:
    *   On the **separable path (`κ = κ_O` from Claim S1 / S1D)**: setting
        `κ_1 = κ_2 = κ_s κ_t` on a complete grid (or `κ_{O,half}` from
        Claim S1D on each half) requires the half-horizon temporal VIFs
        `VIF_{t,early}^{full}` and `VIF_{t,late}^{full}` to be at most `κ_t`.
        *(Note: non-negativity `K_t(w, v) ≥ 0` and equal diagonal variances
        `K_t(w, w) = σ_t²` alone do **not** imply `VIF_{t,half}^{full} ≤
        VIF_{t,full}^{full}`—for example, with `G_t = 4`, unit variances,
        correlation `1` between windows `1` and `2`, and independent windows
        `3` and `4`, `VIF_{t,full} = 6/4 = 1.5` whereas
        `VIF_{t,early} = 4/2 = 2.0 > 1.5` because all off-diagonal covariance
        sits in the early half. By contrast, if `K_t(w, v) = σ_t² ρ_t(|w − v|) ≥ 0`
        is also **stationary/Toeplitz** across windows, then for
        `G_{t,half} ≤ G_t`
        `VIF_{t,half}^{full} = 1 + 2 Σ_{h=1}^{G_{t,half}-1} (1 − h/G_{t,half}) ρ_t(h) ≤
        1 + 2 Σ_{h=1}^{G_t-1} (1 − h/G_t) ρ_t(h) = VIF_{t,full}^{full} ≤ κ_t`
        holds automatically.)*
    *   On the **general `φ` path (`topological_kappa`)**: either set
        `κ_1 = κ_2 = κ_T` (valid whenever `(T1)` and `(T2)` hold on each half,
        since each half's cut matrix `Φ_{half}^cut` is a principal submatrix of
        `Φ^cut` so `λ_max(Φ_{half}^cut) ≤ λ_max(Φ^cut)` by Perron–Frobenius
        monotonicity), or recompute `κ_{T,1}, κ_{T,2} ≤ κ_T` on each half's item
        subgraph.
    *   *(Runtime implementation note (`spacetime.py`): the runtime drift
        check passes the joint `resolved_kappa` (`κ_s κ_t` on the separable
        route when `G_s > 1, G_t > 1`, `κ_s` or `κ_t` on one-dimensional
        candidates, or `κ_T` on the topological route) to both half-window
        `cluster_interval` calls. Passing the joint `κ_s κ_t` rather than `κ_t`
        alone is required because the half-horizon clusters `S_early` and
        `S_late` are the `G_1, G_2` space-time blocks `(a, w)`, whose sums carry
        both spatial between-community correlation `VIF_s` and half-horizon
        temporal correlation `VIF_{t,half}`; under the half-horizon premise
        `VIF_s ≤ κ_s` and `VIF_{t,half} ≤ κ_t`, Claim S1(2) gives
        `VIF_{half} = VIF_s · VIF_{t,half} ≤ κ_s κ_t = resolved_kappa`.)*
*   **Verdict** (`spacetime_interval` returns a `temporal.DriftResult` whose
    `outcome` is `Drift.UNTESTABLE`, `Drift.WARNING` or `Drift.NO_WARNING`):
    *   If `G_t = 1` (**community-only partition**): every cluster spans the
        entire time horizon, so no disjoint split of cluster totals separates
        early and late time. The outcome is `UNTESTABLE` with message
        `"drift check untestable (community-only partition G_t=1 spans all
        time)."`.
    *   If `2 ≤ G_t < 4`: the runtime reports `UNTESTABLE` (`"drift check
        untestable (fewer than 4 windows)."`). Proposition D′ itself holds for
        every `G_t ≥ 2`; the threshold `4` is a runtime design choice.
    *   If one half contains no items, the outcome is `UNTESTABLE`.
    *   If either half returns status other than `UNREFUTED`
        (`ASSUMPTION_REQUIRED` or `TAIL_UNRESOLVED`) or a non-finite endpoint,
        the outcome is `UNTESTABLE`.
    *   Otherwise, the outcome is `WARNING` iff
        `[L_1, U_1] ∩ [L_2, U_2] = ∅` (`max(L_1, L_2) > min(U_1, U_2)`), and
        `NO_WARNING` otherwise. `first_half` and `second_half` hold
        `(L_1, U_1)` and `(L_2, U_2)`.
    *   No per-half κ check is run: `first_kappa_check` and
        `second_kappa_check` are set to `UNTESTABLE`. (Proposition D in
        [THEORY.md](THEORY.md) describes each half running its own κ check;
        the space-time path does not.) Each half does report its own `M̂_G`
        status through `TAIL_UNRESOLVED`. For `metric="r2"`, the drift check
        runs on the squared errors `e²` with `metric="mse"`.
    These `UNTESTABLE` cases only lower the warning rate, so the bound below
    is unaffected by them.

> **Proposition D′ (Window-split drift warning on space-time blocks).**
> Suppose `G_t ≥ 2`, `S_early` and `S_late` satisfy the premises of Claim 4
> with `(M_{c,1}, κ_1)` and `(M_{c,2}, κ_2)`, and `θ_early = θ_late`. Then
> $$\Pr\bigl(\text{outcome} = \texttt{WARNING} \mid D\bigr) \;\le\; \alpha.$$

*Argument.* Premises: as stated, with `θ_early = θ_late =: θ*`.

1.  By Claim 4 at level `1 − α/2`,
    `Pr(θ_early ∉ [L_1, U_1] ∧ early UNREFUTED | D) ≤ α/2` and
    `Pr(θ_late ∉ [L_2, U_2] ∧ late UNREFUTED | D) ≤ α/2`.
2.  A warning requires both halves `UNREFUTED` and
    `[L_1, U_1] ∩ [L_2, U_2] = ∅`. If both intervals covered `θ*`, then
    `θ* ∈ [L_1, U_1] ∩ [L_2, U_2] ≠ ∅`; hence a warning implies that at least
    one `UNREFUTED` half misses `θ*`.
3.  The union bound over the two halves gives `≤ α/2 + α/2 = α`. ∎

**What the drift warning is not.** The proposition bounds false warnings when
the two halves have equal means. It does not certify stationarity, and it does
not detect drift smaller than roughly the half-interval width (see
Proposition D in [THEORY.md](THEORY.md)).

---

## Appendix A. Edge-test and super-batch results

The static graph checks used at runtime (§9.4.4) are
those of [THEORY.md](THEORY.md): the edge test `check_uncorrelated_edges`
with its null mean and variance (Claim 6, in particular part (d), the
corrected statistic) and the super-batch bound
`check_kappa_batches` (Proposition B, first order, with its argument sketch).
Statements, arguments and evidence levels are given there; this document uses
them unchanged.

## Appendix B. Gebelein and Lancaster bounds

§9.1.1 uses the following classical bounds to turn a residual-level correlation
bound into a summand-level bound for `(T1-prod)`.

> **Gebelein–Lancaster bounds.** Let `(X, Y)` be jointly Gaussian with
> correlation `ρ`, and let `f, g` have finite, non-zero variance under the
> marginals. Then `|Corr(f(X), g(Y))| ≤ |ρ|` (Gebelein). If in addition `X` and
> `Y` are centred and `f`, `g` are even functions (for example `x²` or `|x|`),
> then `|Corr(f(X), g(Y))| ≤ ρ²` (Lancaster).

*Argument.* Standardize `X` and `Y`; this changes neither correlation nor
evenness when the means are zero. Expand `f` and `g` in normalized Hermite
polynomials, `f = Σ_k a_k H_k` and `g = Σ_k b_k H_k`. Mehler's formula gives
`E[H_j(X) H_k(Y)] = δ_{jk} ρ^k`, so `Cov(f(X), g(Y)) = Σ_{k≥1} a_k b_k ρ^k`,
`Var f(X) = Σ_{k≥1} a_k²` and `Var g(Y) = Σ_{k≥1} b_k²`. The expansions
converge in `L²` of the standard normal law and the covariance is a bounded
bilinear form, so it may be evaluated term by term.

1.  By Cauchy–Schwarz, `|Σ_{k≥1} a_k b_k ρ^k| ≤ |ρ| (Σ a_k²)^{1/2}(Σ b_k²)^{1/2}`,
    since `|ρ|^k ≤ |ρ|` for `k ≥ 1`. This is Gebelein's bound.
2.  For even `f`, `g` and centred variables, the odd Hermite coefficients
    vanish, so the sum runs over `k ≥ 2` and `|ρ|^k ≤ ρ²`. The same
    Cauchy–Schwarz step gives Lancaster's bound. ∎

This is the content of the Gebelein–Lancaster remark under Lemma T′ in
[THEORY.md](THEORY.md): the bound is `φ_res` for any functions, and
`φ_res²` for `e²` and `|e|` only if the residuals are also centred.

## References

*   H. Gebelein (1941); H. O. Lancaster (1957): maximal correlation of
    functions of Gaussian variables.
*   L. Isserlis (1918): moments of the multivariate normal distribution.
*   S. Nabeya (1951): absolute moments of the bivariate normal distribution.
*   Sherman–Morrison formula for rank-one updates of an inverse.

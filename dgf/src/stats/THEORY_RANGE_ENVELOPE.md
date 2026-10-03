# Range-envelope dependence declarations: theory

> The arguments in this document were written by the library authors, who are
> not specialist statisticians, and have not been peer-reviewed; see
> [EVIDENCE.md](EVIDENCE.md) for simulation evidence.

## Status of claims, lemmas, and propositions

Each result in this document carries one of four statuses:

*   **argued, checked by simulation** (with the corresponding test or benchmark
    in parentheses; see [EVIDENCE.md](EVIDENCE.md));
*   **argued only**;
*   **approximation** (first-order or asymptotic);
*   **cited** (from the literature, with reference).

| Label | One-line statement | Section | Status |
|---|---|---|---|
| Claim RE0 (§0.1) | End-to-end held-out conditional coverage $\ge 1 - \alpha$ of `spacetime_interval` under `(P1)`, `(T1-range)`, `(T2)`, and `(P2*_pool(M))` | §0.1 | argued, checked by simulation ([EVIDENCE.md §6](EVIDENCE.md#6-range-envelope-coverage): `choose_partition` + `cluster_interval`, the interval computation used by `spacetime_interval`, without its edge check; `benchmark/range_coverage_test.py`) |
| Lemma RE1 (§1.1) | Loss-level dependence range bounding box $\bar R_s = 2K + R_s$, $\bar R_t = L + H + R_t$ and conditional independence of item losses outside it at $\gamma = 0$ | §1.1 | argued, checked by simulation (`range_envelope_test.py`; `benchmark/range_generators_test.py`) |
| Proposition RE1 (§2) | Training–test separation via temporal embargo gap $> \bar R_t$ or spatial buffer gap $> \bar R_s$ preserves the unconditional data ranges given the trained model at $\gamma = 0$ | §2 | argued only |
| Claim RE1 | Range-envelope variance inflation bound $\mathrm{VIF}(\Pi) \le \kappa_{\mathrm{range}}(\Pi) = 1 + \lambda_{\max}(A_{\mathrm{range}}^{\mathrm{cut}}(\Pi)) + \gamma$ and certified `direct` and `kronecker` upper bounds $\hat\kappa(\Pi) \ge \kappa_{\mathrm{range}}(\Pi)$ | §3 | argued, checked by simulation (`range_envelope_test.py`; [EVIDENCE.md](EVIDENCE.md) "Range-envelope coverage"; `benchmark/range_coverage_test.py`) |
| Proposition RE2 | Inconsistency ($\mathrm{Var}(\hat\theta \mid D) \ge \rho_0\bar\sigma^2$, $\nu \le 1/\rho_0$) under a persistent common shock vs. $O(n^{-1/2})$ width contraction under diffuse $O(1/n)$ global dependence | §4 | argued, checked by simulation ([EVIDENCE.md](EVIDENCE.md) "Range-envelope coverage"; `benchmark/range_generators_test.py`) |
| Proposition RE3 | Componentwise monotonicity of $\kappa_{\mathrm{range}}(\Pi)$ in $(K, L, H, R_s, R_t, \gamma)$ and preservation of `(T1-range)` and $\lambda_{\max}(A_{\mathrm{range}}^{\mathrm{cut}}(\Pi|_{\mathcal{O}'})) \le \lambda_{\max}(A_{\mathrm{range}}^{\mathrm{cut}}(\Pi))$ under exogenous missing data $\mathcal{O}' \subseteq \mathcal{O}$ | §5 | argued, checked by simulation (`range_envelope_test.py`) |
| Proposition REU1 | Structural unverifiability of design exogeneity `(P0)`, population relative second moment `(P2*_pool(m_item))` (via Bahadur and Savage, 1956), and multi-item within-cluster `(T2)` from a single evaluation realization | §6 | argued only |
| Proposition RE4 | Upward sampling-noise bias $\Omega(N_{\mathrm{far}}(u)/\sqrt{T_{\mathrm{train}}})$ of naive out-of-range absolute-correlation summing $\sum_{v \notin B_{\mathrm{range}}(u)} |\hat\rho(u, v)|$ under $H_0: \gamma_{\mathrm{true}} = 0$ | §7 | argued only |

---

**What this document argues.** The space-time entry point issues a
confidence interval for the average expected loss of a fixed trained model over
a set of evaluated (node, time) items whose losses are correlated. This
document argues that the interval covers that estimand with probability at
least $1 - \alpha$ when the user declares, before looking at the evaluation
losses, three kinds of quantities: the model's receptive field ($K$ message-passing hops,
$L$ lookback steps, $H$ horizon steps), the range of dependence in the data
($R_s$ hops, $R_t$ steps), and a budget $\gamma$ for correlation outside that
range (Claim RE0). The core step is a certified bound $\hat\kappa$ on the
variance inflation factor of the cluster totals, computed from the declarations
and the graph for any candidate partition (Claim RE1). The document also
derives the loss-level dependence range from the declarations (Lemma RE1), states
the training–test separation the coverage bound needs (Proposition RE1), shows when an
interval cannot shrink with more data (Proposition RE2), argues monotonicity in
the declarations and validity under missing data (Proposition RE3), lists which
premises can be refuted and on what data (Proposition REU1), and explains why the
library does not estimate $\gamma$ from data (Proposition RE4).

**Who it is for.** ML engineers and statisticians who evaluate forecasting
models on spatio-temporal graphs (for example traffic sensors) and must choose
the declarations, and reviewers who check the arguments. It assumes familiarity
with Cantelli's inequality and with the results of
[THEORY.md](THEORY.md) listed below; each is restated in one line where it
is used.

**Scope.** This document describes the
space-time entry point `spacetime_interval` with partitions scored by the
range-envelope bound (`range_kappa_fn`), for the metrics `mae`, `mse`, `rmse`
and `accuracy`, and for the residual part of `r2` (Section 3.1). It covers a
single held-out split. General space-time product partitions and separable or
topological κ declarations are covered in
[THEORY_SPACETIME.md](THEORY_SPACETIME.md).

**Results used from [THEORY.md](THEORY.md):**

*   **Claim 4** (cluster interval): under premises `(P2v)` and
    `P1_c(κ)` on $G$ cluster totals, the Cantelli interval covers
    $\theta(D)$ with probability at least $1 - \alpha$.
*   **Lemma I′** (default $M_c$): the pooled item-level premise
    `P2*_pool(M)` implies `(P2v)` and `P2*_c(M_c)` with
    $M_c = \text{default\_m\_c\_from\_counts}(\dots)$, for any per-item means.
*   **Lemma T′** (κ from a graph): the Collatz–Wielandt upper bound on the
    largest eigenvalue of a non-negative symmetric matrix.
*   **Lemma R** (conditioning on the design): if the partition is a function
    of the design $D$ only, the claims hold conditionally on $D$ for
    $\theta(D)$, provided their premises hold under the conditional law.
*   **Claim 7** (correlated $R^2$): the $R^2$ interval needs a separate
    variance inflation factor $\kappa_y$ (`kappa_labels`) for the label totals.

**Library functions referred to:**

| Function | Module | Role |
| :--- | :--- | :--- |
| `spacetime_items(node_ids, times)` | `spacetime.py` | Builds the item index $\mathcal{O}$; rejects empty input, unequal lengths, non-integer times and duplicate `(node, time)` pairs. |
| `RangeDeclaration(k_hops, lookback, horizon, data_range_s, data_range_t, gamma)` | `range_envelope.py` | Holds $(K, L, H, R_s, R_t, \gamma)$; rejects non-integer or negative $K, L, H, R_s, R_t$ and non-finite or negative $\gamma$ (no default for $\gamma$). |
| `range_candidates(index, spatial_edges, declaration, *, target_sizes=None, window_multipliers=None, seed=0)` | `range_envelope.py` | Builds the candidate partition family $\mathcal{F}_{\mathrm{cand}}$. |
| `range_kappa(index, spatial_edges, cluster_ids, declaration, *, method="kronecker", max_edges=20_000_000)` | `range_envelope.py` | Computes the certified bound $\hat\kappa(\Pi)$ of Claim RE1(3). |
| `range_kappa_fn(index, spatial_edges, declaration, *, method="kronecker", max_edges=20_000_000)` | `range_envelope.py` | Wraps `range_kappa` as the scoring function for `choose_partition`. |
| `choose_partition(candidates, *, kappa_fn, m_item, is_complete_grid, metric="mse", level=0.95, ...)` | `spacetime.py` | Selects $\Pi^\star$ from counts and $\hat\kappa(\Pi)$ only; reports `route="range"` for a range-envelope `kappa_fn`. |
| `spacetime_interval(data, index, choice, *, metric="mse", level=0.95, m_item=None, kappa=None, ...)` | `spacetime.py` | Issues the interval on $\Pi^\star$. |
| `default_m_c_from_counts(m_item, n, g, n_max, s)` | `cluster_sketch.py` | Computes $M_c$ from Lemma I′. |
| `topological_kappa(num_nodes, edges, cluster_ids, phi)` | `graph.py` | Collatz–Wielandt bound used by `method="direct"`. |

---

## Notation

| Symbol | Meaning |
| :--- | :--- |
| $G_s = (V_s, E_s)$, $N_s$ | Full spatial graph, including unobserved nodes; $N_s = \lvert V_s\rvert$. |
| $d_s(i, j)$ | Shortest-path hop distance in $G_s$, in $\mathbb{Z}_{\ge 0} \cup \{\infty\}$. |
| $B_{G_s}(i, r)$ | Hop ball $\{a \in V_s : d_s(i, a) \le r\}$. |
| $t \in \{0, \dots, T-1\}$ | Discrete model time step (integer counts of the model's step size). |
| $\mathcal{O}$, $n$ | Observed evaluation index set $\mathcal{O} \subseteq V_s \times \{0, \dots, T-1\}$, at most one item per `(node, time)` pair; $n = \lvert\mathcal{O}\rvert \ge 1$. |
| $u = (i, t)$, $V_{\mathcal{O}}$ | An item at node $i$ and time $t$; $V_{\mathcal{O}}$ is the set of observed nodes. |
| $D = (D_{\mathrm{design}}, \hat w)$ | Conditioning design: design variables and the trained model $\hat w$ (Section 2). |
| $s_u \ge 0$, $\mu_u$, $\sigma_u^2$ | Item loss, $\mu_u := \mathbb{E}[s_u \mid D]$, $\sigma_u^2 := \mathrm{Var}(s_u \mid D)$. |
| $\theta(D)$, $\hat\theta$ | Estimand $\theta(D) := \frac{1}{n}\sum_{u \in \mathcal{O}} \mu_u$; sample mean $\hat\theta := \frac{1}{n}\sum_{u \in \mathcal{O}} s_u$. |
| $\Pi = \{B_1, \dots, B_G\}$, $c(u)$ | Partition of $\mathcal{O}$ into $G$ clusters; $c(u) \in \{1, \dots, G\}$ is the cluster of $u$. |
| $S_c$, $n_c$, $n_{\max}$ | Cluster total $S_c = \sum_{u \in B_c} s_u$, count $n_c = \lvert B_c\rvert$, $n_{\max} = \max_c n_c$. |
| $r$, $\mathrm{CV}_n^2$ | $r = n_{\max}/(n/G)$; $\mathrm{CV}_n^2 = G\sum_c n_c^2/n^2 - 1$. |
| $\bar\theta_c$ | Mean expected cluster total $\frac{1}{G}\sum_c \mathbb{E}[S_c \mid D] = n\,\theta(D)/G$. |
| $\Pi_{\mathrm{singletons}}$ | The partition into single items ($n_c \equiv 1$, $G = n$). |
| $K, L, H$ | Model message-passing hops, lookback steps, prediction horizon steps. |
| $R_s, R_t$ | Data dependence range in hops and steps (Definition RE1). |
| $\gamma$ | Out-of-range absolute-correlation row-sum budget (premise `(T1-range)`). |
| $M = m_{\mathrm{item}}$ | Declared pooled item-level relative second moment (premise `(P2*_pool(M))`). |
| $\alpha$ | Miscoverage, $\alpha = 1 - \text{level} \in (0, 1)$. |
| $\bar R_s$, $\bar R_t$ | Loss-level range $\bar R_s := 2K + R_s$, $\bar R_t := L + H + R_t$ (Lemma RE1). |
| $G_{\mathrm{range}} = (\mathcal{O}, E_{\mathrm{range}})$ | Loss-level range graph (Section 1.1). |
| $d_{\mathrm{range}}(u)$, $d_{\mathrm{range}}^{\min}$, $\bar d_{\mathrm{range}}$, $d_{\mathrm{range}}^{\max}$ | Degree of $u$ in $G_{\mathrm{range}}$, and its minimum, mean and maximum over $u \in \mathcal{O}$. |
| $B_{\mathrm{range}}(u)$ | Range neighbourhood $\{u\} \cup \{v : \{u, v\} \in E_{\mathrm{range}}\}$; "$v \notin B_{\mathrm{range}}(u)$" means $v$ is out of range of $u$. |
| $A_{\mathrm{range}}$, $A_{\mathrm{range}}^{\mathrm{cut}}(\Pi)$ | 0/1 adjacency of $G_{\mathrm{range}}$ and its cross-cluster restriction (equation (3.1)). |
| $d_{\mathrm{range}}^{\mathrm{cut}}(\Pi)$ | Maximum row sum of $A_{\mathrm{range}}^{\mathrm{cut}}(\Pi)$. |
| $\mathrm{VIF}(\Pi)$ | $\mathrm{Var}(\sum_c S_c \mid D) / \sum_c \mathrm{Var}(S_c \mid D)$. |
| $\kappa_{\mathrm{range}}(\Pi)$, $\hat\kappa(\Pi)$ | Exact bound $1 + \lambda_{\max}(A_{\mathrm{range}}^{\mathrm{cut}}(\Pi)) + \gamma$ and its certified numerical upper bound from `range_kappa`. |
| $\lambda_{\max}(B)$, $\lVert B\rVert_\infty$ | Largest eigenvalue; maximum absolute row sum. |
| $M_c$, $n_A(M_c, \alpha)$ | Cluster-level moment bound from Lemma I′; issuance threshold $(M_c - 1)(1 - \alpha/2)/(\alpha/2)$. |
| $c(\Pi)$ | Cantelli constant $\sqrt{(M_c(\Pi)-1)\kappa(1-\alpha/2)/((\alpha/2)G(\Pi))}$ ([THEORY.md](THEORY.md), Section 1). |
| $\nu(\Pi)$, $r_{\mathrm{eff}}(\Pi)$, $N_{\mathrm{eff}}(\Pi)$ | $\nu(\Pi) = G(\Pi)/\kappa(\Pi)$; $r_{\mathrm{eff}}(\Pi) = (M_c(\Pi) - 1)/(M - 1)$; $N_{\mathrm{eff}}(\Pi) = G(\Pi)/(\kappa(\Pi)\,r_{\mathrm{eff}}(\Pi))$. |
| $\widehat M_G$ | Empirical cluster relative second moment $G\sum_c S_c^2/(\sum_c S_c)^2$; $\widehat M_G > M_c$ gives status `TAIL_UNRESOLVED`. |
| $\mathcal{F}_{\mathrm{cand}}$, $\Pi^\star$ | Finite candidate partition family; selected partition. |
| $[L, U]$ | Interval endpoints. (The lower endpoint $L$ is unrelated to the lookback $L$; context distinguishes them.) |
| `UNREFUTED`, `TAIL_UNRESOLVED`, `ASSUMPTION_REQUIRED` | Statuses: interval issued with $\widehat M_G \le M_c$; interval issued with $\widehat M_G > M_c$; no interval issued ($G/\hat\kappa \le n_A(M_c, \alpha)$). |
| $\mathcal{T}_{\mathrm{train}}$, $T_{\mathrm{train}}$, $\mathcal{T}_{\mathrm{eval}}$ | Training data, its number of time steps, and the evaluation data (Section 6). |
| $n_{\mathrm{train}}$ | Number of training items (Section 6). |
| $Z$; $\xi$, $r_s$, $r_t$ | Node-time data process (features and targets); in the example of Definition RE1, i.i.d. innovations and moving-average radii in hops and steps. |
| $\mathcal{S}(i, t)$, $\mathcal{S}_{\mathrm{feat}}(i, t)$, $\mathcal{S}_{\mathrm{lab}}(i, t)$ | Support set of the loss of item $(i, t)$ (equation (1.2)) and its feature and label parts (Lemma RE1). |
| $\mathcal{O}_{\mathrm{train}}$, $\mathcal{S}_{\mathrm{train}}$, $\mathcal{S}_{\mathrm{eval}}$, $\mathcal{A}$ | Training items, training and evaluation support sets, and the training algorithm (Section 2). |
| $t_{\mathrm{tr,max}}$, $t_{\mathrm{ev,min}}$, $\delta$ | Last training step, first evaluation step, and the offset $t' - t_{\mathrm{tr,max}}$ of an evaluation item $(j, t')$ (Proposition RE1(2)). |
| `(P0)` | Design exogeneity (Section 6). |
| $R^{\mathrm{cut}}(\Pi)$ | Cross-cluster, out-of-range absolute correlation matrix (argument for Claim RE1). |
| $\varepsilon_{64}$ | Float64 machine epsilon, $2^{-52}$. |
| $d_{\max}$, $\mathrm{ub}_{\mathrm{rows}}$, $\mathrm{ub}_{\mathrm{cw}}$, $\mathrm{ub}_{\mathrm{cw,best}}$ | Maximum cut row degree as computed in code ($= d_{\mathrm{range}}^{\mathrm{cut}}(\Pi)$); the exact row-sum bound; the Collatz–Wielandt bounds (Claim RE1(3)). |
| $N_{s,\mathcal{O}}$, $T_{\mathcal{O}}$, $x_{\mathrm{pert}}$, $v_{\mathrm{norm}}$, $y_{\mathrm{pert}}$, $P_{\mathrm{pert}}$, $\mathrm{err}$ | Quantities of the Kronecker method (Claim RE1(3)). |
| $\kappa_y$ | Variance inflation factor of the label totals, declared as `kappa_labels` (Section 3.1). |
| $\rho_0$, $\delta_0$, $d_0$ | Common-shock correlation floor and degree conditions of Proposition RE2. |
| $\theta_{\mathcal{O}'}(D)$, $S_c^{\mathcal{O}'}$ | Estimand and cluster totals on a sub-mask $\mathcal{O}' \subseteq \mathcal{O}$ (Proposition RE3(2)). |
| $\mathrm{TV}$, $\psi$, $\alpha_{\mathrm{ref}}$, $\varepsilon$, $\varepsilon_M$, $M_{\mathrm{true}}$, $m_0$, $\theta_0$, $x_0$ | Quantities of Proposition REU1(2). |
| $\Delta_c$, $W_c$, $\eta_u$, $\mathcal{M}_+$, $\mathcal{M}_-$ | Quantities of Proposition REU1(3). |
| $\hat\rho(u, v)$, $\rho_{\mathrm{true}}(u, v)$, $H_0$, $N_{\mathrm{far}}(u)$, $v_{uv}$, $v_{\min}$, $\rho_u(k)$, $\hat\gamma_{\mathrm{naive}}(u)$ | Quantities of Proposition RE4. |

**Symbols with a second, local meaning.** The section fixes the meaning:

*   $L$: lookback steps; the lower interval endpoint in $[L, U]$.
*   $c$: cluster index in $c(u)$, $S_c$, $n_c$; the Cantelli constant $c(\Pi)$
    (always written with its argument); a correlation scale in
    Proposition RE2(2).
*   $T$: number of grid time steps; $T_{\mathrm{train}}$ and $T_{\mathcal{O}}$
    are different counts.
*   $G$: number of clusters; $G_s$ and $G_{\mathrm{range}}$ (always
    subscripted) are graphs.
*   $B$: cluster $B_c$; hop ball $B_{G_s}(i, r)$; range neighbourhood
    $B_{\mathrm{range}}(u)$; a generic non-negative matrix $B$ in
    Claim RE1(3) and Proposition RE3(1).
*   $v$: a second item ($u, v \in \mathcal{O}$); the power-iteration vector
    $v$, $v_{\mathrm{pos}}$, $v_{\mathrm{norm}}$ in Claim RE1(3); the
    asymptotic variances $v_{uv}$, $v_{\min}$ in Proposition RE4.
*   $d$: degrees and the hop distance $d_s$; a second cluster index in
    $\sum_{c \ne d}$ (argument for Claim RE1). The code quantity $d_{\max}$ (maximum
    cut degree) differs from $d_{\mathrm{range}}^{\max}$ (maximum degree of
    $G_{\mathrm{range}}$); they coincide at $\Pi_{\mathrm{singletons}}$.
*   $r$: the count ratio $r = n_{\max}/(n/G)$; a radius in $B_{G_s}(i, r)$;
    a time index in $\xi_{q, r}$ and the radii $r_s$, $r_t$ (Definition RE1).
*   $\varepsilon$: spike probability (Proposition REU1(2)); noise
    $\varepsilon_{i,t}$ (Section 3.1); $\varepsilon_{64}$ is the machine
    epsilon.
*   $\delta$: offset of an evaluation item (Proposition RE1(2)); the inflation
    of $\hat\kappa$ above the exact bound (argument for Proposition RE2);
    $\delta_0$ is a fixed fraction (Proposition RE2(1)).
*   $\sigma$: the vector $(\sigma_u)$ (Claim RE1); a common scalar standard
    deviation (Proposition RE2, Proposition REU1(3)).
*   $\theta$: the estimand $\theta(D)$; a Gaussian mean parameter in
    Proposition REU1(3).
*   $H$: horizon steps; $H_0$ is the null hypothesis in Proposition RE4.
*   $x$: a positive vector in Collatz–Wielandt quotients; an unobserved node
    in Section 1.2; $x_0$ is the spike level in Proposition REU1(2).
*   $\mathcal{S}$ (calligraphic) is a support set; $S_c$ (italic) is a
    cluster total.

---

## 0. The coverage bound in one statement

### 0.1 Claim RE0

The coverage bound applies to a single held-out split and to the metrics `mae`,
`mse`, `rmse` and `accuracy`.

> **Claim RE0 (End-to-End Held-Out Range-Envelope Coverage Bound).**
> Fix the conditioning design $D = (D_{\mathrm{design}}, \hat w)$—consisting of the trained model $\hat w$, the observed evaluation index set $\mathcal{O}$ ($n = |\mathcal{O}| \ge 2$), the full spatial graph $G_s = (V_s, E_s)$, the integer time grid in units of the model step, the finite candidate partition family $\mathcal{F}_{\mathrm{cand}}$, and any partitioner random seed—together with the user declarations $(K, L, H, R_s, R_t, \gamma, M, \alpha)$ (with $K, L, H, R_s, R_t \in \mathbb{Z}_{\ge 0}$, $\gamma \ge 0$, $M = m_{\mathrm{item}} \ge 1$, and $\alpha \in (0, 1)$) **before inspecting the evaluation losses $\{s_u\}_{u \in \mathcal{O}}$**.
>
> Suppose that, conditionally on $D$:
>
> 1. **Non-negativity `(P1)`:** $s_u \ge 0$ almost surely for all $u \in \mathcal{O}$.
> 2. **Loss-level range envelope `(T1-range)`:** For every $u = (i, t) \in \mathcal{O}$, the sum of $|\mathrm{Corr}(s_u, s_v \mid D)|$ over all $v = (j, t') \in \mathcal{O} \setminus \{u\}$ outside the loss-level box $\{d_s(i, j) \le \bar R_s := 2K + R_s,\; |t - t'| \le \bar R_t := L + H + R_t\}$ is at most $\gamma$ (equation (1.4)).
> 3. **Within-cluster non-negative net covariance `(T2)` on the selected partition $\Pi^\star = \Pi^\star(D)$:** $\sum_{c=1}^G \mathrm{Var}(S_c \mid D) \ge \sum_{u \in \mathcal{O}} \sigma_u^2$ (which holds **automatically with exact identity** if $\Pi^\star = \Pi_{\mathrm{singletons}}$ has $n_c \equiv 1$, and is a required premise on the observed sub-clusters of $\Pi^\star$ when $\max_c n_c \ge 2$).
> 4. **Pooled item second-moment bound `(P2*_pool(M))` on $\mathcal{O}$:** $\frac{1}{n}\sum_{u \in \mathcal{O}} \mathbb{E}[s_u^2 \mid D] \le M\,\theta(D)^2$ (which by Lemma I′ in [THEORY.md](THEORY.md) implies the cluster-level variance premise `(P2v)` and moment bound `P2*_c(M_c)` at $M_c$ for **arbitrary** heterogeneous item means $(\mu_u)_{u \in \mathcal{O}}$ and cluster counts $(n_c)_{c=1}^G$, without requiring equal item means or equal cluster sizes).
>
> Let $\Pi^\star = \text{choose\_partition}(\text{candidates}, \text{kappa\_fn}=\text{range\_kappa\_fn}(\text{index}, \text{spatial\_edges}, \text{declaration}), m_{\mathrm{item}}=M).\text{winner} \in \mathcal{F}_{\mathrm{cand}}$ be the partition selected from counts $n_c$ and $\hat\kappa(\Pi) = \text{range\_kappa}(\text{index}, \text{spatial\_edges}, \Pi, \text{declaration}).\text{kappa}$ alone, let $\hat\kappa := \hat\kappa(\Pi^\star)$, and let $M_c := \text{default\_m\_c\_from\_counts}(M, n, G, n_{\max}, \sum_{c=1}^G n_c^2) = \min\{rM,\, 1 + r(M-1) + \mathrm{CV}_n^2 + 2\,\mathrm{CV}_n\sqrt{r(M-1)}\}$ with $r = n_{\max}/(n/G)$.
>
> Then, whenever `spacetime_interval` issues an interval $[L, U]$ (`status != ASSUMPTION_REQUIRED`, i.e., $G / \hat\kappa > n_A(M_c, \alpha) := (M_c - 1)(1 - \alpha/2)/(\alpha/2)$, covering both `UNREFUTED` and `TAIL_UNRESOLVED`),
> $$\mathbb{P}\bigl(\theta(D) \in [L, U] \;\big|\; D\bigr) \;\ge\; 1 - \alpha, \qquad \theta(D) \;:=\; \frac{1}{n}\sum_{u \in \mathcal{O}} \mathbb{E}[s_u \mid D]$$
> (with $[L, U]$ square-rooted for `rmse` and clamped to $[0, 1]$ for `accuracy`).

*Argument.*

1.  **Dependence premise.** By `(T2)` on $\Pi^\star$,
    $\sum_c \mathrm{Var}(S_c \mid D) \ge \sum_u \sigma_u^2$. If
    $\sum_u \sigma_u^2 > 0$, then $\sum_c \mathrm{Var}(S_c \mid D) > 0$ and
    Claim RE1 (Section 3) gives
    $\hat\kappa \ge 1 + \lambda_{\max}(A_{\mathrm{range}}^{\mathrm{cut}}(\Pi^\star)) + \gamma \ge \mathrm{VIF}(\Pi^\star)$,
    so the cluster premise
    `P1_c(κ̂)`, $\mathrm{Var}(\sum_{c=1}^G S_c \mid D) \le \hat\kappa\sum_{c=1}^G \mathrm{Var}(S_c \mid D)$,
    holds. If $\sum_u \sigma_u^2 = 0$, every $s_u$ is almost surely equal to
    $\mu_u$ given $D$, so both sides of `P1_c(κ̂)` are $0$ and it holds with
    equality.
2.  **Moment premise.** By Lemma I′ ([THEORY.md](THEORY.md)),
    $M_c - 1 \ge r(M - 1)$ and
    $\sum_{c=1}^G \mathrm{Var}(S_c \mid D) \le G\,r(M-1)\bar\theta_c^2 \le G(M_c - 1)\bar\theta_c^2$,
    so the cluster variance premise `(P2v)` with $M_c$ (as well as
    `P2*_c(M_c)`) holds for any per-item means $(\mu_u)_{u \in \mathcal{O}}$.
3.  **No selection effect.** Both $\Pi^\star$ and the issuance event
    $G / \hat\kappa > n_A(M_c, \alpha)$ are deterministic functions of $D$
    (Lemma R, [THEORY.md](THEORY.md)).
4.  **Coverage.** By steps 1–3, Claim 4 ([THEORY.md](THEORY.md)) applies
    to the totals of $\Pi^\star$ conditionally on $D$ and gives the stated
    coverage whenever the interval is issued. ∎

**Sample size and zero variance.** Claim RE0 assumes $n \ge 2$. The Notation
allows $n \ge 1$ because Lemma RE1, Claim RE1 and Propositions RE1–RE3 hold for every
$n \ge 1$. Claim RE0 needs no condition $\sum_u \sigma_u^2 > 0$: step 1 covers
the case $\sum_u \sigma_u^2 = 0$, whereas Claim RE1(1)–(2) state their
variance-ratio bounds only for $\sum_u \sigma_u^2 > 0$, where the ratio
$\mathrm{VIF}(\Pi)$ is defined.

**What the estimand $\theta(D)$ is—and is not.** $\theta(D) = \frac{1}{n}\sum_{u \in \mathcal{O}} \mathbb{E}[s_u \mid D]$
is the average expected loss of **this fixed trained model $\hat w$** over the
evaluated index set $\mathcal{O}$ under hypothetical redraws of the evaluation
data given $D$. It is **not** the realized sample loss
$\hat\theta = \frac{1}{n}\sum_{u \in \mathcal{O}} s_u$ (which is observed
without error), and it is **not** the unconditional expected risk of the
training algorithm over random redraws of the training set. For
`metric="r2"`, Claim RE1 bounds only the residual numerator; the label
denominator requires a separate declaration `kappa_labels` (Section 3.1).

### 0.2 Which conditions the code enforces, which you declare, and which cannot be verified

Every condition of Claim RE0 falls into one of three groups.

**1. Computed and enforced by the code:**

| Condition | Where |
| :--- | :--- |
| Evaluation `times` are integers, `node_ids` and `times` have equal non-zero length, and `(node, time)` pairs in $\mathcal{O}$ are unique. | `spacetime_items` |
| Consecutive differences of the sorted unique times are positive integers with $\gcd = 1$ (necessary but not sufficient for unit alignment; Section 1.2). | `range_kappa`, `range_kappa_fn` |
| $K, L, H, R_s, R_t \in \mathbb{Z}_{\ge 0}$ and $\gamma \in \mathbb{R}_{\ge 0}$ finite. | `RangeDeclaration` |
| $M \ge 1$ finite and $\alpha = 1 - \text{level} \in (0, 1)$. | `choose_partition` |
| Certified upper bound $\hat\kappa(\Pi) \ge 1 + \lambda_{\max}(A_{\mathrm{range}}^{\mathrm{cut}}(\Pi)) + \gamma \ge 1$, via shifted power iteration and a Collatz–Wielandt quotient capped by the exact integer row degree $d_{\mathrm{range}}^{\mathrm{cut}}(\Pi)$ (Claim RE1(3)). | `range_kappa` |
| Exact cluster-size moment inflation $M_c = \text{default\_m\_c\_from\_counts}(M, n, G, n_{\max}, \sum_c n_c^2)$. | `ClusterSketch`, `choose_partition` |
| Loss-free partition selection $\Pi^\star = \text{choose\_partition}(\dots).\text{winner}$: it inspects only $n_c$ and $\hat\kappa(\Pi)$, never loss values; the returned `Choice` has `route="range"`. | `choose_partition` |
| Deterministic refusal `ASSUMPTION_REQUIRED` when $G/\hat\kappa \le n_A(M_c, \alpha)$. | `spacetime_interval` |
| Consistency with the selection: if `m_item` is omitted, `choice.m_item` is used; a different `m_item` is rejected; a `kappa` override smaller than `choice.winner_kappa` is rejected. | `spacetime_interval` |
| For `metric="r2"`, `kappa_labels` must be passed explicitly. | `spacetime_interval` |

**2. User declarations** (checkable from the architecture or protocol, or
assumed from domain knowledge):

| Declaration | Meaning |
| :--- | :--- |
| $(K, L, H)$ | $K$ spatial message-passing hops, $L$ lookback steps, $H$ prediction horizon steps. Checkable from the model and the task definition. |
| $(R_s, R_t, \gamma, M, \alpha)$ | Data dependence ranges $(R_s, R_t)$, out-of-range correlation row-sum budget $\gamma$, pooled item second-moment bound $M = m_{\mathrm{item}}$, and miscoverage $\alpha = 1 - \text{level}$ (plus `kappa_labels` when `metric="r2"`). |
| Full spatial graph `spatial_edges` | The edge list passed to `range_candidates`, `range_kappa_fn` and `range_kappa` must be the **entire** spatial graph $G_s = (V_s, E_s)$ on which $K$-hop propagation and $R_s$-hop dependence act, **including unobserved nodes** that can lie on shortest paths between observed nodes (Section 1.2). |
| Time-step unit alignment | Evaluation `times` are integer counts of the model's discrete step size, the same unit in which $L, H, R_t$ are expressed (Section 1.2). |
| Consistent inputs across calls | The same `index`, `spatial_edges` and `RangeDeclaration` are used for `range_candidates` and `range_kappa_fn`, and the `Choice` returned by `choose_partition` is passed to `spacetime_interval` unchanged. |

**3. Statistical premises that cannot be verified from the evaluation data:**

| Premise | Meaning |
| :--- | :--- |
| Conditional range envelope `(T1-range)`, equation (1.4) | Out-of-range absolute correlation row sums under $\mathbb{P}(\cdot \mid D)$ do not exceed $\gamma$. |
| Within-cluster `(T2)` when $\max_c n_c \ge 2$ | $\sum_{c=1}^G \mathrm{Var}(S_c \mid D) \ge \sum_{u \in \mathcal{O}} \sigma_u^2$ on the observed sub-clusters of $\Pi^\star$ (an identity when $\Pi^\star = \Pi_{\mathrm{singletons}}$). |
| Pooled second-moment bound `(P2*_pool(M))` on the observed set $\mathcal{O}$ | $\frac{1}{n}\sum_{u \in \mathcal{O}} \mathbb{E}[s_u^2 \mid D] \le M\,\theta(D)^2$. Unrefutable in finite samples by Bahadur–Savage (Proposition REU1(2)); only the one-sided empirical refutation $\widehat M_G > M_c \implies \texttt{TAIL\_UNRESOLVED}$ is observable. |
| Design exogeneity `(P0)`, including the observed mask $\mathcal{O}$ (Lemma R) | The design, and in particular missingness or node-time selection in $\mathcal{O}$, does not depend on the evaluation losses given $D$. |
| Training-test separation and measurability of $\hat w$ | $\hat w$ is fitted exclusively on training data (no scalers, normalizers, or hyperparameters fitted on evaluation data; Section 2), and all declarations $(K, L, H, R_s, R_t, \gamma, M, \alpha, \mathcal{F}_{\mathrm{cand}})$ are fixed before viewing evaluation losses. |

---

## 1. Declarations, the data premise, and the loss-level range

The user declares six non-negative dependence quantities (plus $M \ge 1$ and
$\alpha \in (0, 1)$) before inspecting evaluation losses:

*   **Model configuration $(K, L, H)$:** $K \in \mathbb{Z}_{\ge 0}$ spatial
    message-passing hops, $L \in \mathbb{Z}_{\ge 0}$ temporal lookback steps,
    and $H \in \mathbb{Z}_{\ge 0}$ prediction horizon steps.
*   **Data dependence range $(R_s, R_t)$:** $R_s \in \mathbb{Z}_{\ge 0}$
    spatial hops and $R_t \in \mathbb{Z}_{\ge 0}$ temporal steps beyond which
    the underlying data noise field is conditionally independent given $D$
    when $\gamma = 0$.
*   **Global tail budget $\gamma \in \mathbb{R}_{\ge 0}$:** For every item
    $u \in \mathcal{O}$, the sum of $|\mathrm{Corr}(s_u, s_v \mid D)|$ over
    all items $v \in \mathcal{O} \setminus \{u\}$ outside the loss-level range
    of $u$ is at most $\gamma$.

> [!IMPORTANT]
> Definition RE1(1) and Lemma RE1 derive the loss-level support range
> $(\bar R_s, \bar R_t)$ from raw-data independence **only in the pure
> finite-range case $\gamma = 0$**. Non-linear loss maps $\ell(\hat Y, Y)$ can
> distort weak raw-data correlations, so when $\gamma > 0$ the coverage bound rests
> directly on the loss-level premise `(T1-range)` in equation (1.4).

In the code, $(K, L, H, R_s, R_t, \gamma)$ are the fields `k_hops`,
`lookback`, `horizon`, `data_range_s`, `data_range_t` and `gamma` of
`RangeDeclaration`; its properties `spatial_range` and `temporal_range` return
$\bar R_s = 2K + R_s$ and $\bar R_t = L + H + R_t$.

### 1.1 The data-field premise and receptive-field arithmetic ($\gamma = 0$)

Let $Z = (Z_{a, \tau})_{(a, \tau) \in V_s \times \mathbb{Z}}$ denote the
underlying node-time data process (input features $X_{a, \tau}$ and target
signals $Y_{a, \tau}$).

> **Definition RE1 (`(R_s, R_t)`-Dependent Data Field and Receptive-Field Support).**
> Conditionally on the design $D$:
>
> 1. **Pure finite-range baseline ($\gamma = 0$):** Two space-time index sets $A, B \subset V_s \times \mathbb{Z}$ are conditionally independent given $D$ ($Z_A \perp\!\!\!\perp Z_B \mid D$) whenever for every $(a, \tau_a) \in A$ and $(b, \tau_b) \in B$,
>    $$d_s(a, b) > R_s \qquad \text{or} \qquad |\tau_a - \tau_b| > R_t. \tag{1.1}$$
>    *(Convention matching moving-average innovation fields: $(R_s, R_t)$ denotes the **pairwise data-dependence range** in (1.1). For a data field $Z_{a,\tau} = g_{a,\tau}(\{\xi_{q, r} : d_s(a, q) \le r_s, \tau - r_t \le r \le \tau\})$ formed by a spatial moving average of radius $r_s$ hops and a backward-only causal temporal moving average of lag $r_t$ steps over i.i.d. innovations $\xi_{q, r}$, two data points share an innovation only up to $R_s = 2 r_s$ hops and $R_t = r_t$ lags—or $R_t = 2 r_t$ for a two-sided temporal filter.)*
>
> 2. **Model and loss support:** For an evaluation item $u = (i, t) \in \mathcal{O}$, the predictor $\hat Y_{i, t+H}$ fixed in $D$ is a measurable function of features in the $K$-hop ball $B_{G_s}(i, K) := \{a \in V_s : d_s(i, a) \le K\}$ over the lookback window $[t - L, t]$ (spanning $L$ lag steps / $L + 1$ timestamps $\{t - L, \dots, t\}$), while the target label $Y_{i, t+H}$ is read at node $i$ and time $t + H$. Thus $s_{i,t} = \ell(\hat Y_{i, t+H}, Y_{i, t+H})$ is a measurable function of $Z_{\mathcal{S}(i, t)}$ on the support set
>    $$\mathcal{S}(i, t) \;:=\; \bigl(B_{G_s}(i, K) \times [t - L,\, t]\bigr) \;\cup\; \bigl\{(i,\, t + H)\bigr\} \;\subset\; B_{G_s}(i, K) \times [t - L,\, t + H], \tag{1.2}$$
>    with temporal projection $\mathcal{S}_t(i, t) := \{\tau \in \mathbb{Z} : \exists a \in V_s \text{ with } (a, \tau) \in \mathcal{S}(i, t)\} = [t - L, t] \cup \{t + H\} \subset [t - L, t + H]$.
>    *(Indexing note: if "$L$ lookback steps" denotes $L$ timestamps $\{t - L + 1, \dots, t\}$, replace $L$ by $L - 1$ in (1.2)–(1.4); using $L$ is then conservative by 1 step.)*

> **Lemma RE1 (Loss-Level Dependence Range Bounding Box $\bar R_s = 2K + R_s$ and $\bar R_t = L + H + R_t$ at $\gamma = 0$).**
> Under Definition RE1 with $\gamma = 0$, two item losses $s_{i,t}$ and $s_{j,t'}$ are conditionally independent given $D$ ($s_{i,t} \perp\!\!\!\perp s_{j,t'} \mid D$, hence $\mathrm{Cov}(s_{i,t}, s_{j,t'} \mid D) = 0$) whenever $(i, t)$ and $(j, t')$ lie outside the rectangular **bounding box (valid, possibly conservative)**:
> $$d_s(i, j) \;>\; \bar R_s \;:=\; 2K + R_s \qquad \text{or} \qquad |t - t'| \;>\; \bar R_t \;:=\; L + H + R_t \tag{1.3}$$
> *(in terms of moving-average filter radii $(r_s, r_t)$ with $R_s = 2 r_s$ and $R_t = r_t$, $\bar R_s = 2K + 2 r_s$ and $\bar R_t = L + H + r_t$)*. Decomposing $\mathcal{S}(i, t) = \mathcal{S}_{\mathrm{feat}}(i, t) \cup \mathcal{S}_{\mathrm{lab}}(i, t)$ into $\mathcal{S}_{\mathrm{feat}}(i, t) = B_{G_s}(i, K) \times [t - L, t]$ and $\mathcal{S}_{\mathrm{lab}}(i, t) = \{(i, t + H)\}$, the three pairwise interaction supports are:
>
> * **Feature–feature ($\mathcal{S}_{\mathrm{feat}}(i, t)$ vs. $\mathcal{S}_{\mathrm{feat}}(j, t')$):** $d_s(i, j) \le 2K + R_s$ and $|t - t'| \le L + R_t$.
> * **Feature–label ($\mathcal{S}_{\mathrm{feat}}(i, t)$ vs. $\mathcal{S}_{\mathrm{lab}}(j, t')$ or vice versa):** $d_s(i, j) \le K + R_s$ and $|t - t'| \le L + H + R_t$.
> * **Label–label ($\mathcal{S}_{\mathrm{lab}}(i, t)$ vs. $\mathcal{S}_{\mathrm{lab}}(j, t')$):** $d_s(i, j) \le R_s$ and $|t - t'| \le R_t$.
>
> The rectangle $\{d_s(i, j) \le 2K + R_s\} \times \{|t - t'| \le L + H + R_t\}$ is the smallest axis-aligned bounding box containing all three regions (it is conservative on the corner $K + R_s < d_s(i, j) \le 2K + R_s$, $L + R_t < |t - t'| \le L + H + R_t$ when $K > 0$ and $H > 0$).

*Argument.* Premises: Definition RE1 with $\gamma = 0$.

1.  **Labels correlate only through the data, never through the model.** The
    target label $Y_{i, t+H}$ is an observation of the data field at the
    single coordinate $(i, t + H)$. Conditionally on $D$ (which fixes the
    deterministic predictor map $\hat Y_{i, t+H}(\cdot)$), the model does not
    alter the joint law of $(Y_{i, t+H}, Y_{j, t'+H})$. Thus
    $\mathcal{S}_{\mathrm{lab}}(i, t) = \{(i, t + H)\}$ has spatial radius $0$
    around $i$, and two labels interact only when $d_s(i, j) \le R_s$ and
    $|(t + H) - (t' + H)| = |t - t'| \le R_t$.
2.  **Spatial bound $\bar R_s = 2K + R_s$.** For any
    $(a, \tau_a) \in \mathcal{S}(i, t)$ and $(b, \tau_b) \in \mathcal{S}(j, t')$,
    equation (1.2) gives $d_s(i, a) \le K$ and $d_s(j, b) \le K$. By the
    triangle inequality,
    $d_s(i, j) \le d_s(i, a) + d_s(a, b) + d_s(b, j) \le 2K + d_s(a, b)$, so
    $d_s(i, j) > 2K + R_s \implies d_s(a, b) > R_s$.
3.  **Temporal bound $\bar R_t = L + H + R_t$.** Assume without loss of
    generality that $t' \ge t$. For any $(a, \tau_a) \in \mathcal{S}(i, t)$ and
    $(b, \tau_b) \in \mathcal{S}(j, t')$, we have
    $\tau_a \in \mathcal{S}_t(i, t) \subset [t - L, t + H]$ so
    $\tau_a \le t + H$, and
    $\tau_b \in \mathcal{S}_t(j, t') \subset [t' - L, t' + H]$ so
    $\tau_b \ge t' - L$, yielding $\tau_b - \tau_a \ge (t' - t) - (L + H)$.
    Thus $|t - t'| > L + H + R_t \implies |\tau_b - \tau_a| > R_t$.
4.  **Conclusion.** By steps 2–3 and (1.1),
    $Z_{\mathcal{S}(i,t)} \perp\!\!\!\perp Z_{\mathcal{S}(j,t')} \mid D$, so by
    Definition RE1(2) $s_{i,t} \perp\!\!\!\perp s_{j,t'} \mid D$. ∎

**The loss-level range graph and the range premise.** With
$\bar R_s = 2K + R_s$ and $\bar R_t = L + H + R_t$, define the **loss-level
range graph** $G_{\mathrm{range}} = (\mathcal{O}, E_{\mathrm{range}})$ by
placing an undirected edge $\{u, v\} \in E_{\mathrm{range}}$ between
$u = (i, t) \ne v = (j, t') \in \mathcal{O}$ iff $d_s(i, j) \le \bar R_s$ and
$|t - t'| \le \bar R_t$. The **$\gamma$-relaxed range premise** (stated
directly on the conditional law of the losses given $D$, and serving as the
primary premise whenever $\gamma > 0$) is:

$$\mathrm{(T1\text{-}range)}:\qquad \max_{u \in \mathcal{O}} \sum_{\substack{v \in \mathcal{O} \setminus \{u\} \\ \{u, v\} \notin E_{\mathrm{range}}}} \bigl|\mathrm{Corr}(s_u, s_v \mid D)\bigr| \;\le\; \gamma, \tag{1.4}$$

where $\mathrm{Corr}(s_u, s_v \mid D) := \mathrm{Cov}(s_u, s_v \mid D) / (\sigma_u \sigma_v)$
if $\sigma_u \sigma_v > 0$ and $0$ otherwise.

### 1.2 Required graph and time-unit declarations (`spatial_edges` and `times`)

Two inputs must be prepared by the user exactly as described here, because the
code cannot fully check them:

1.  **`spatial_edges` must be the full spatial graph $G_s = (V_s, E_s)$,
    including unobserved nodes.** In Lemma RE1, $d_s(i, j)$ is the shortest-path
    hop distance in the **full** graph $G_s = (V_s, E_s)$ over which the
    model's $K$-hop propagation and the data's $R_s$-hop dependence operate.
    Unobserved nodes $x \in V_s \setminus V_{\mathcal{O}}$ can lie on shortest
    paths between observed nodes $a, b \in V_{\mathcal{O}}$. For example, on
    the path $a - x - b$ with $x$ unobserved and $\bar R_s = 2$, the true
    distance is $d_{G_s}(a, b) = 2 \le \bar R_s$, whereas on the subgraph
    induced by observed nodes $\{a, b\}$ alone $d(a, b) = \infty > \bar R_s$,
    which would omit $\{a, b\}$ from $A_{\mathrm{range}}^{\mathrm{cut}}$ and
    understate $\kappa_{\mathrm{range}}$. In
    [range_envelope.py](range_envelope.py), `range_kappa` and `range_kappa_fn`
    traverse all nodes present in `spatial_edges` (including unobserved
    nodes) as breadth-first-search intermediates. Passing the full edge list
    `spatial_edges` of $G_s$ is therefore a **required user declaration**: the
    library cannot detect unobserved intermediate nodes if edges incident to
    them are stripped before calling `range_candidates`, `range_kappa_fn` or
    `range_kappa`.
2.  **Evaluation `times` must be integer counts of the model's discrete time
    step ($\gcd = 1$ is necessary, not sufficient).** The temporal range
    $\bar R_t = L + H + R_t$ is measured in integer counts of the model's
    discrete time step, so `times` passed to `spacetime_items` must be
    expressed in that same unit. In [range_envelope.py](range_envelope.py),
    `range_kappa` and `range_kappa_fn` reject inputs where the $\gcd$ of
    consecutive differences of the sorted unique times exceeds $1$. This catches
    uniform scaling (e.g., regular $5$-minute steps passed as
    $\{0, 5, 10, 15\}$, where $\gcd = 5 > 1$), but it is **necessary and not
    sufficient**: if one timestamp is jittered or irregular (e.g.,
    $\{0, 5, 10, 11\}$ in raw minutes while $L, H, R_t$ are declared in
    $5$-minute steps), the differences $(5, 5, 1)$ have $\gcd = 1$ and pass the
    check while $\bar R_t$ is applied in minutes ($5\times$ too small). Hence
    time-unit alignment is a **required user declaration**.

---

## 2. The trained model as part of the design: training–test separation

In held-out evaluation, the trained model
$\hat w = \mathcal{A}(Z_{\mathcal{S}_{\mathrm{train}}})$, where $\mathcal{A}$
is the training algorithm, is part of the design
$D = (D_{\mathrm{design}}, \hat w)$ (Lemma R, [THEORY.md](THEORY.md)). This
requires that $\hat w$ be **measurable with respect to the training data
$Z_{\mathcal{S}_{\mathrm{train}}}$ and $D_{\mathrm{design}}$ alone**: no
feature scalers, normalizers, or hyperparameters may be fitted on evaluation
or pooled train+test data.

> **Proposition RE1 (Training-Test Separation and Conditional Law Given $D$).**
> Let $\mathcal{O}_{\mathrm{train}}$ and $\mathcal{O}_{\mathrm{eval}} = \mathcal{O}$ be the training and evaluation item sets, reading data coordinates $\mathcal{S}_{\mathrm{train}} := \bigcup_{(i, t) \in \mathcal{O}_{\mathrm{train}}} \mathcal{S}(i, t)$ and $\mathcal{S}_{\mathrm{eval}} := \bigcup_{(i, t) \in \mathcal{O}_{\mathrm{eval}}} \mathcal{S}(i, t)$, and assume $\hat w = \mathcal{A}(Z_{\mathcal{S}_{\mathrm{train}}})$ is $\sigma(Z_{\mathcal{S}_{\mathrm{train}}}, D_{\mathrm{design}})$-measurable.
>
> 1. **Exact independence justification when $\gamma = 0$:** Under Definition RE1(1) ($\gamma = 0$), $Z_{\mathcal{S}_{\mathrm{train}}} \perp\!\!\!\perp Z_{\mathcal{S}_{\mathrm{eval}}} \mid D_{\mathrm{design}}$ holds whenever either:
>    * **Temporal embargo gap:** $\min_{(j, t') \in \mathcal{O}_{\mathrm{eval}}} t' - \max_{(i, t) \in \mathcal{O}_{\mathrm{train}}} t \;>\; \bar R_t = L + H + R_t$; or
>    * **Spatial buffer gap:** $\min_{(i,t)\in\mathcal{O}_{\mathrm{train}},\, (j,t')\in\mathcal{O}_{\mathrm{eval}}} d_s(i, j) \;>\; \bar R_s = 2K + R_s$.
>
>    Under this separation and $\gamma = 0$, $\mathbb{P}(Z_{\mathcal{S}_{\mathrm{eval}}} \in \cdot \mid D_{\mathrm{design}}, \hat w) = \mathbb{P}(Z_{\mathcal{S}_{\mathrm{eval}}} \in \cdot \mid D_{\mathrm{design}})$, so conditioning on $\hat w$ preserves the unconditional data ranges $(R_s, R_t)$ and $\gamma = 0$.
> 2. **Conditional status when $\gamma > 0$ or without the separation gap:**
>    * **When $\gamma > 0$:** The embargo gap alone justifies Proposition RE1(1) **only for $\gamma = 0$**. When $\gamma > 0$, $Z_{\mathcal{S}_{\mathrm{train}}}$ and $Z_{\mathcal{S}_{\mathrm{eval}}}$ can be weakly coupled across the embargo gap, so conditioning on $\hat w = \mathcal{A}(Z_{\mathcal{S}_{\mathrm{train}}})$ can alter the conditional law of $Z_{\mathcal{S}_{\mathrm{eval}}}$. Therefore, for $\gamma > 0$, the declared ranges $(R_s, R_t)$ and tail budget $\gamma$ in `(T1-range)` are **premises directly on the conditional law $\mathbb{P}(\cdot \mid D = (D_{\mathrm{design}}, \hat w))$**.
>    * **Boundary leakage ($0 \le t_{\mathrm{ev,min}} - t_{\mathrm{tr,max}} \le \bar R_t$):** Let $(j, t') \in \mathcal{O}_{\mathrm{eval}}$ be an evaluation item in the first $\bar R_t - (t_{\mathrm{ev,min}} - t_{\mathrm{tr,max}}) + 1$ steps, that is, $\delta := t' - t_{\mathrm{tr,max}} \le \bar R_t$, and suppose node $j$ carries training items at every step of $[t_{\mathrm{tr,max}} - H,\, t_{\mathrm{tr,max}}]$, that is, $(j, t) \in \mathcal{O}_{\mathrm{train}}$ for all such $t$ (as in a temporal split of the same nodes). Then its support coordinates $\mathcal{S}(j, t')$ **lie within the data dependence range $(R_s, R_t)$ of the training support $\mathcal{S}_{\mathrm{train}}$**: there is direct overlap $\mathcal{S}(j, t') \cap \mathcal{S}_{\mathrm{train}} \ne \emptyset$ when $\delta \le L + H$, and a pair of coordinates at node $j$ at time distance between $1$ and $R_t$, where Definition RE1(1) does not exclude dependence, when $L + H < \delta \le L + H + R_t$. The independence argument of part (1) therefore does not apply to these boundary items: their losses can depend on $\hat w$ (collider dependence through $\hat w$), which is the mechanism of optimism bias.
>
> Here $t_{\mathrm{tr,max}} := \max_{(i, t) \in \mathcal{O}_{\mathrm{train}}} t$ and $t_{\mathrm{ev,min}} := \min_{(j, t') \in \mathcal{O}_{\mathrm{eval}}} t'$.

*Argument.* Premises: Definition RE1 with $\gamma = 0$ for part (1); measurability
of $\hat w$.

1.  Under part (1) with $\gamma = 0$, every
    $(a, \tau_a) \in \mathcal{S}_{\mathrm{train}}$ and
    $(b, \tau_b) \in \mathcal{S}_{\mathrm{eval}}$ satisfies
    $|\tau_a - \tau_b| > R_t$ or $d_s(a, b) > R_s$, by the argument of
    Lemma RE1 (steps 2–3).
2.  Hence $Z_{\mathcal{S}_{\mathrm{train}}} \perp\!\!\!\perp Z_{\mathcal{S}_{\mathrm{eval}}} \mid D_{\mathrm{design}}$
    by (1.1).
3.  Since $\hat w$ is $(Z_{\mathcal{S}_{\mathrm{train}}}, D_{\mathrm{design}})$-measurable,
    $\hat w \perp\!\!\!\perp Z_{\mathcal{S}_{\mathrm{eval}}} \mid D_{\mathrm{design}}$.
4.  **Part (2), a coupling changes the law given $\hat w$.** Let
    $a \in \mathcal{S}_{\mathrm{train}}$ and $b \in \mathcal{S}_{\mathrm{eval}}$,
    and let $(Z_a, Z_b)$ be jointly Gaussian given $D_{\mathrm{design}}$ with
    zero means, unit variances and correlation $\rho \in (0, 1]$. Take the
    measurable training algorithm $\hat w = \mathcal{A}(Z_{\mathcal{S}_{\mathrm{train}}}) := Z_a$.
    Given $D_{\mathrm{design}}$ alone, $Z_b \sim \mathcal{N}(0, 1)$; given
    $(D_{\mathrm{design}}, \hat w)$,
    $Z_b \sim \mathcal{N}(\rho\hat w, 1 - \rho^2)$. For instance, with
    predictor $\hat Y_{j, t'+H} = Z_b$, label identically $0$ and squared
    loss, $\mathbb{E}[s_{j,t'} \mid D_{\mathrm{design}}, \hat w] = \rho^2\hat w^2 + 1 - \rho^2$,
    which depends on $\hat w$ (collider dependence through $\hat w$).
5.  **Part (2), $\gamma > 0$.** When $\gamma > 0$, Definition RE1(1) is not
    assumed, so a pair $(a, b)$ as in step 4 with $\rho > 0$ is allowed at any
    distance, in particular across an embargo or buffer gap. By step 4,
    conditioning on $\hat w$ can then change the law of
    $Z_{\mathcal{S}_{\mathrm{eval}}}$, so the declared $(R_s, R_t, \gamma)$
    must be premises on $\mathbb{P}(\cdot \mid D)$ with $D = (D_{\mathrm{design}}, \hat w)$.
6.  **Part (2), boundary leakage.** Write $t_0 := t_{\mathrm{tr,max}}$. Since
    $t' \ge t_{\mathrm{ev,min}}$, $\delta = t' - t_0 \ge t_{\mathrm{ev,min}} - t_0 \ge 0$.
    By (1.2), $(j, \tau) \in \mathcal{S}(j, t')$ for every
    $\tau \in [t' - L, t']$, and $(j, \tau) \in \mathcal{S}(j, t) \subseteq \mathcal{S}_{\mathrm{train}}$
    for every training item $(j, t)$ and $\tau \in [t - L, t] \cup \{t + H\}$.
    *   If $\delta \le L + H$, let $\tau := t' - L = t_0 + \delta - L$. If
        $\tau \le t_0$, then $\tau \in [t_0 - L, t_0]$ (as $\delta \ge 0$),
        so $(j, \tau) \in \mathcal{S}(j, t_0)$. If $t_0 < \tau \le t_0 + H$,
        then $t := \tau - H \in (t_0 - H, t_0]$, so $(j, t) \in \mathcal{O}_{\mathrm{train}}$
        by assumption and $(j, \tau) = (j, t + H) \in \mathcal{S}(j, t)$. In
        both cases $(j, \tau) \in \mathcal{S}(j, t') \cap \mathcal{S}_{\mathrm{train}}$.
    *   If $L + H < \delta \le L + H + R_t$, the coordinates
        $(j, t_0 + H) \in \mathcal{S}(j, t_0) \subseteq \mathcal{S}_{\mathrm{train}}$
        and $(j, t' - L) \in \mathcal{S}(j, t')$ are at the same node and at
        time distance $\delta - L - H \in [1, R_t]$. Condition (1.1) asserts
        independence only for $d_s > R_s$ or time distance $> R_t$, so it
        does not exclude dependence of this pair.
    In both cases the separation used in steps 1–3 fails for $(j, t')$, and
    step 4 (with $\rho = 1$ for a shared coordinate, or any $\rho > 0$ allowed
    within range) shows that the loss of $(j, t')$ can depend on $\hat w$. ∎

---

## 3. Claim RE1: the range-envelope κ bound

For any partition $\Pi = \{B_1, \dots, B_G\}$ of $\mathcal{O}$, let
$A_{\mathrm{range}} \in \{0, 1\}^{n \times n}$ be the $0/1$ adjacency matrix
of $G_{\mathrm{range}} = (\mathcal{O}, E_{\mathrm{range}})$, and let
$A_{\mathrm{range}}^{\mathrm{cut}}(\Pi)$ be its cross-cluster restriction:

$$\bigl(A_{\mathrm{range}}^{\mathrm{cut}}(\Pi)\bigr)_{uv} \;:=\; \begin{cases}
1 & \text{if } c(u) \ne c(v),\; d_s(u, v) \le \bar R_s,\; \text{and } |t_u - t_v| \le \bar R_t,\\
0 & \text{otherwise.}
\end{cases} \tag{3.1}$$

Here $d_s(u, v)$ is the hop distance between the nodes of $u$ and $v$, and
$t_u$, $t_v$ are their times.

> **Claim RE1 (Range-Envelope Variance Inflation Bound — Cluster and Singleton Versions).**
> Suppose `(T1-range)` holds conditionally on $D$ with $\bar R_s = 2K + R_s$, $\bar R_t = L + H + R_t$, and tail budget $\gamma \ge 0$.
>
> 1. **Cluster Version (arbitrary partition $\Pi = \{B_1, \dots, B_G\}$):**
>    If premise `(T2)` holds on $\Pi$ ($\sum_{c=1}^G \mathrm{Var}(S_c \mid D) \ge \sum_{u \in \mathcal{O}} \sigma_u^2 > 0$), then for **any** per-item standard deviations $\sigma = (\sigma_u)_{u \in \mathcal{O}} \in \mathbb{R}_{\ge 0}^n$, the cluster-level variance inflation factor satisfies `P1_c(κ_range(Π))` with
>    $$\mathrm{VIF}(\Pi) \;:=\; \frac{\mathrm{Var}(\sum_{c=1}^G S_c \mid D)}{\sum_{c=1}^G \mathrm{Var}(S_c \mid D)} \;\le\; \kappa_{\mathrm{range}}(\Pi) \;:=\; 1 \;+\; \lambda_{\max}\!\bigl(A_{\mathrm{range}}^{\mathrm{cut}}(\Pi)\bigr) \;+\; \gamma \;\le\; 1 \;+\; d_{\mathrm{range}}^{\mathrm{cut}}(\Pi) \;+\; \gamma, \tag{3.2}$$
>    where $d_{\mathrm{range}}^{\mathrm{cut}}(\Pi) := \max_{u \in \mathcal{O}} \sum_{v \in \mathcal{O}} (A_{\mathrm{range}}^{\mathrm{cut}}(\Pi))_{uv}$ is the maximum cross-cluster range degree.
> 2. **Singleton Version ($\Pi_{\mathrm{singletons}}$, $n_c \equiv 1$, $G = n$):**
>    For singleton clusters $B_u = \{u\}$, premise `(T2)` holds with **exact algebraic equality** ($\sum_{c=1}^n \mathrm{Var}(S_c \mid D) = \sum_{u \in \mathcal{O}} \sigma_u^2$) and $A_{\mathrm{range}}^{\mathrm{cut}}(\Pi_{\mathrm{singletons}}) = A_{\mathrm{range}}$. Hence, **without assuming `(T2)`**, for any $\sigma \in \mathbb{R}_{\ge 0}^n$ with $\|\sigma\|_2 > 0$:
>    $$\mathrm{VIF}(\Pi_{\mathrm{singletons}}) \;\le\; 1 \;+\; \lambda_{\max}(A_{\mathrm{range}}) \;+\; \gamma \;\le\; 1 \;+\; d_{\mathrm{range}}^{\max} \;+\; \gamma \;\le\; 1 \;+\; \Bigl((2\bar R_t + 1)\max_{i \in V_s}|B_{G_s}(i, \bar R_s)| - 1\Bigr) \;+\; \gamma. \tag{3.3}$$
> 3. **Certified Upper Bounds Implemented in Code ([graph.py](graph.py) and [range_envelope.py](range_envelope.py)):**
>    For any non-negative symmetric matrix $B \ge 0$ and any strictly positive vector $x > 0$, the Collatz–Wielandt inequality gives $\lambda_{\max}(B) \le \max_{1 \le u \le n} (Bx)_u / x_u$ (Lemma T′, [THEORY.md](THEORY.md)), while Gershgorin's theorem gives $\lambda_{\max}(B) \le \|B\|_\infty = \max_u \sum_v B_{uv}$. In [range_envelope.py](range_envelope.py), `range_kappa(index, spatial_edges, cluster_ids, declaration, method=...)` computes a certified upper bound $\hat\kappa(\Pi) \ge \kappa_{\mathrm{range}}(\Pi)$ by two methods:
>    * **`method="direct"` (via `graph.topological_kappa`):** Materializes the edge list $E_{\mathrm{range}}$ on $\mathcal{O}$ (up to `max_edges`) and calls `graph.topological_kappa(num_nodes=n, edges=E_range, cluster_ids=c_ids, phi=1.0)` in [graph.py](graph.py). That function computes $\mathrm{ub}_{\mathrm{rows}} = d_{\mathrm{range}}^{\mathrm{cut}}(\Pi)$, runs $200$ shifted power iterations $v \leftarrow (A_{\mathrm{range}}^{\mathrm{cut}} v + \mathrm{ub}_{\mathrm{rows}} v)/\|\cdot\|_\infty$ from $\mathbf{1}$, evaluates the Collatz–Wielandt quotient at both $x = v + 10^{-3}\mathbf{1}$ and $v_{\mathrm{pos}} = \max(v, 10^{-12})$ to obtain $\mathrm{ub}_{\mathrm{cw,best}} = \min(\mathrm{ub}_{\mathrm{cw}}, \mathrm{ub}_{\mathrm{cw},v})$, and returns $\hat\kappa_{\mathrm{top}} = 1 + \min(\mathrm{ub}_{\mathrm{cw,best}}, \mathrm{ub}_{\mathrm{rows}})\cdot\bigl(1 + \max(10^{-9},\, 4(d_{\max} + 2)\,\varepsilon_{64})\bigr)$, where $d_{\max}$ is the maximum cut row degree ($= d_{\mathrm{range}}^{\mathrm{cut}}(\Pi)$ here) and $\varepsilon_{64}$ is the float64 machine epsilon; `range_kappa` adds $\gamma$. Each computed quotient $(A_{\mathrm{range}}^{\mathrm{cut}} x)_u / x_u$ sums at most $d_{\max}$ positive terms with weight $1$ (exact in float64) and divides once, so its relative rounding error is at most about $(d_{\max} + 1)\,\varepsilon_{64}/2$; the inflation factor therefore certifies the bound for any degree.
>    * **`method="kronecker"` (the default; matrix-free; requires a product partition $\Pi = \Pi_s \times \Pi_t$ of spatial communities and contiguous time windows, and raises `ValueError` otherwise):** Never materializes $E_{\mathrm{range}}$ or calls `topological_kappa`. It computes exact integer row sums via the factored spatial-ball and temporal-band operators (requiring $\mathrm{ub}_{\mathrm{rows}} = d_{\mathrm{range}}^{\mathrm{cut}}(\Pi) < 2^{53}$ so float64 integer arithmetic is exact), runs $50$ shifted power iterations of $A_{\mathrm{range}}^{\mathrm{cut}}(\Pi) + \mathrm{ub}_{\mathrm{rows}} I$ from $\mathbf{1}$, evaluates the Collatz–Wielandt quotient at $x_{\mathrm{pert}} = v_{\mathrm{norm}} + 10^{-3}\mathbf{1}$ with an explicit IEEE-754 float64 rounding margin $\mathrm{err} = 4(N_{s,\mathcal{O}} + T_{\mathcal{O}} + 2)\,\varepsilon_{64}\,P_{\mathrm{pert}}$ ($\mathrm{ub}_{\mathrm{cw}} = \max_u (y_{\mathrm{pert},u} + \mathrm{err})/x_{\mathrm{pert},u}$), and returns $\hat\kappa = 1 + \min(\mathrm{ub}_{\mathrm{cw}}, \mathrm{ub}_{\mathrm{rows}})\cdot(1 + 10^{-9}) + \gamma$.
>
>    In the Kronecker method, the operator works on the observed node-by-time grid: $N_{s,\mathcal{O}} := |V_{\mathcal{O}}| \le N_s$ is the number of observed unique nodes (`index.num_nodes`) and $T_{\mathcal{O}} \le T$ is the number of observed unique time steps (`index.num_times`); unobserved nodes enter only through the breadth-first search, which is exact integer arithmetic. $v_{\mathrm{norm}}$ is the power-iteration vector scaled to maximum $1$, $y_{\mathrm{pert}} = A_{\mathrm{range}}^{\mathrm{cut}}(\Pi)\,x_{\mathrm{pert}}$ as evaluated in floating point, $\varepsilon_{64}$ is the float64 machine epsilon, and $P_{\mathrm{pert}}$ is the largest per-node total, over all time steps, of the spatial-ball sums of $x_{\mathrm{pert}}$ (the largest partial sum the operator forms).

*Argument.* Premises: `(T1-range)`; `(T2)` on $\Pi$ for part (1).

1.  **Entrywise covariance bound.** Let $R^{\mathrm{cut}}(\Pi)$ have entries
    $R_{uv}^{\mathrm{cut}} = |\mathrm{Corr}(s_u, s_v \mid D)|\,\mathbf{1}\{c(u) \ne c(v) \wedge \{u, v\} \notin E_{\mathrm{range}}\}$.
    For a cross-cluster pair $\{u, v\} \in E_{\mathrm{range}}$, Cauchy–Schwarz
    gives $|\mathrm{Cov}(s_u, s_v \mid D)| \le \sigma_u \sigma_v = (A_{\mathrm{range}}^{\mathrm{cut}}(\Pi))_{uv}\,\sigma_u\sigma_v$.
    For a cross-cluster pair $\{u, v\} \notin E_{\mathrm{range}}$,
    $|\mathrm{Cov}(s_u, s_v \mid D)| = R_{uv}^{\mathrm{cut}}\,\sigma_u\sigma_v$
    (both sides are $0$ when $\sigma_u\sigma_v = 0$).
2.  **Cross-cluster covariance.** Expanding
    $\mathrm{Var}(\sum_{c=1}^G S_c \mid D) - \sum_{c=1}^G \mathrm{Var}(S_c \mid D)$
    over cross-cluster pairs and applying step 1 and the Rayleigh quotient
    bound for symmetric matrices gives
    $$\sum_{c \ne d} \mathrm{Cov}(S_c, S_d \mid D) \;\le\; \sigma^\top A_{\mathrm{range}}^{\mathrm{cut}}(\Pi)\,\sigma \;+\; \sigma^\top R^{\mathrm{cut}}(\Pi)\,\sigma \;\le\; \Bigl(\lambda_{\max}\!\bigl(A_{\mathrm{range}}^{\mathrm{cut}}(\Pi)\bigr) + \lambda_{\max}\!\bigl(R^{\mathrm{cut}}(\Pi)\bigr)\Bigr)\|\sigma\|_2^2.$$
3.  **Tail term.** Since $R^{\mathrm{cut}}(\Pi)$ is symmetric and
    non-negative, Gershgorin's theorem gives
    $\lambda_{\max}(R^{\mathrm{cut}}(\Pi)) \le \|R^{\mathrm{cut}}(\Pi)\|_\infty$.
    Each row sum of $R^{\mathrm{cut}}(\Pi)$ is a sub-sum of the left side of
    (1.4), so $\|R^{\mathrm{cut}}(\Pi)\|_\infty \le \gamma$.
4.  **Cluster version.** By `(T2)`,
    $\|\sigma\|_2^2 \le \sum_{c=1}^G \mathrm{Var}(S_c \mid D)$. Adding
    $\sum_c \mathrm{Var}(S_c \mid D)$ to both sides of step 2 and dividing by
    $\sum_c \mathrm{Var}(S_c \mid D) > 0$ yields the first inequality of
    (3.2). The second follows from Gershgorin's theorem applied to
    $A_{\mathrm{range}}^{\mathrm{cut}}(\Pi)$.
5.  **Singleton version.** At $n_c \equiv 1$, $S_c = s_u$, so `(T2)` is an
    identity and every pair is a cross-cluster pair, hence
    $A_{\mathrm{range}}^{\mathrm{cut}}(\Pi_{\mathrm{singletons}}) = A_{\mathrm{range}}$.
    Step 4 gives the first inequality of (3.3) and Gershgorin's theorem the
    second. For the third: the range neighbours of $u = (i, t)$ are items
    $(j, t')$ with $j \in B_{G_s}(i, \bar R_s)$ and $|t - t'| \le \bar R_t$;
    with at most one item per `(node, time)` pair there are at most
    $(2\bar R_t + 1)|B_{G_s}(i, \bar R_s)|$ such items including $u$ itself.
6.  **Code bounds.** The Collatz–Wielandt quotient at any strictly positive
    vector, the row-sum bound, and their minimum are upper bounds on
    $\lambda_{\max}$. In exact arithmetic, multiplying by the inflation
    factor ($1 + \max(10^{-9}, 4(d_{\max} + 2)\varepsilon_{64})$ in the direct
    method, $1 + 10^{-9}$ in the Kronecker method) and adding the
    non-negative margin $\mathrm{err}$ only increase them; the inflation and
    the margin absorb floating-point rounding as described in part (3). ∎

### 3.1 Scope: why `metric="r2"` needs a separate `kappa_labels` declaration

Claim RE1 bounds the variance inflation factor of the **residual loss** totals
$S_c = \sum_{u \in B_c} s_u$ (the sole input for `mae`, `mse`, `rmse`, and
`accuracy`). It does **not** bound the variance inflation factor $\kappa_y$ of
the centred squared label totals $\sum_{u \in B_c} (Y_u - \bar y)^2$ in the
denominator of
$R^2 = 1 - (\sum_u (Y_u - \hat Y_u)^2) / (\sum_u (Y_u - \bar y)^2)$
(Claim 7, [THEORY.md](THEORY.md)).

*   **Counterexample (zero-range losses with globally correlated labels).**
    Let $Y_{i,t} = Z_0 + \varepsilon_{i,t}$, where $Z_0$ is a random global
    level shared across all $(i, t) \in \mathcal{O}$ with
    $\mathrm{Var}(Z_0) > 0$, and $\varepsilon_{i,t}$ are i.i.d. zero-mean noise
    independent of $Z_0$. Suppose the predictor tracks the global level
    ($\hat Y_{i,t} = Z_0$). Then the residuals
    $Y_{i,t} - \hat Y_{i,t} = \varepsilon_{i,t}$ are i.i.d., so the
    squared-error losses $s_{i,t} = \varepsilon_{i,t}^2$ satisfy
    `(T1-range)` with $R_s = R_t = 0$ and $\gamma = 0$
    ($\kappa_{\mathrm{range}} = 1$). However, the raw labels $Y_{i,t}$ share
    the common shock $Z_0$ across all $n$ items, so the label denominator has
    $\mathrm{VIF}_y = \Theta(n) \gg \kappa_{\mathrm{range}} = 1$.
*   **Consequence.** For `metric="r2"`, `spacetime_interval` /
    `compute_r2_interval` uses $\hat\kappa$ from Claim RE1 for the residual
    numerator and requires a separate user declaration `kappa_labels`
    ($\kappa_y$) for the label denominator (Claim 7,
    [THEORY.md](THEORY.md)). `spacetime_interval` raises `ValueError` if
    `kappa_labels` is not passed.

---

## 4. A common shock versus diffuse global dependence

> **Proposition RE2 (Inconsistency under a Common Shock vs. $\sqrt{n}$-Contraction under Diffuse Dependence).**
> Suppose $\sigma_u \equiv \sigma > 0$ for all $u \in \mathcal{O}$ (or more generally $\bar\sigma := \frac{1}{n}\sum_{u \in \mathcal{O}} \sigma_u > 0$).
>
> 1. **Persistent common shock ($\mathrm{Corr}(s_u, s_v \mid D) \ge \rho_0 > 0$ for all $u \ne v \in \mathcal{O}$):**
>    The variance of the sample mean $\hat\theta = \frac{1}{n}\sum_{u \in \mathcal{O}} s_u$ satisfies
>    $$\mathrm{Var}(\hat\theta \mid D) \;=\; \frac{1}{n^2}\sum_{u \in \mathcal{O}} \sigma_u^2 + \frac{1}{n^2}\sum_{u \ne v} \sigma_u \sigma_v\,\mathrm{Corr}(s_u, s_v \mid D) \;\ge\; \frac{1 - \rho_0}{n^2}\sum_{u \in \mathcal{O}} \sigma_u^2 + \rho_0\,\bar\sigma^2 \;\ge\; \rho_0\,\bar\sigma^2 \quad \bigl(=\,\rho_0\sigma^2\bigr) \tag{4.1}$$
>    **for every sample size $n \ge 1$**. Because `(T1-range)` holds for every $u \in \mathcal{O}$ and an item of minimum range degree $d_{\mathrm{range}}^{\min} := \min_{u \in \mathcal{O}} d_{\mathrm{range}}(u)$ ($\le \bar d_{\mathrm{range}} \le d_{\mathrm{range}}^{\max}$) has $n - 1 - d_{\mathrm{range}}^{\min}$ out-of-range pairs, any valid tail declaration must satisfy
>    $$\gamma \;\ge\; \bigl(n - 1 - d_{\mathrm{range}}^{\min}\bigr)\,\rho_0 \;\ge\; \bigl(n - 1 - d_{\mathrm{range}}^{\max}\bigr)\,\rho_0 \;=\; \Theta(n), \qquad \nu(\Pi_{\mathrm{singletons}}) \;=\; \frac{n}{\kappa_{\mathrm{range}}} \;\le\; \frac{n}{1 + (n - 1)\rho_0} \;\le\; \frac{1}{\rho_0} \tag{4.2}$$
>    (with strict inequality $\nu < 1/\rho_0$ whenever $\rho_0 < 1$; the order $\Theta(n)$ for the lower bound on $\gamma$ holds whenever $d_{\mathrm{range}}^{\max} \le (1 - \delta_0)(n - 1)$ for a fixed $\delta_0 > 0$). Thus $\nu \le 1/\rho_0$ for all $n$, and no confidence interval can shrink to zero width as $n \to \infty$.
> 2. **Diffuse global dependence ($|\mathrm{Corr}(s_u, s_v \mid D)| \le c / n$ outside $E_{\mathrm{range}}$):**
>    Consider a sequence of problems indexed by $n$ in which $M$ and $\alpha$ are fixed and the range degree is bounded, $d_{\mathrm{range}}^{\max} \le d_0$ for a constant $d_0$ independent of $n$. If out-of-range correlations satisfy $|\mathrm{Corr}(s_u, s_v \mid D)| \le c / n$ (or decay summably so $\sum_{v \notin B_{\mathrm{range}}(u)} |\mathrm{Corr}(s_u, s_v \mid D)| \le c < \infty$ independent of $n$), then `(T1-range)` holds with constant $\gamma = c = O(1)$, $\kappa_{\mathrm{range}}(\Pi_{\mathrm{singletons}}) \le 1 + d_{\mathrm{range}}^{\max} + c \le 1 + d_0 + c = O(1)$, $\nu(\Pi_{\mathrm{singletons}}) = \Theta(n) \to \infty$, and the interval width contracts at rate $O(n^{-1/2})$: on $\Pi_{\mathrm{singletons}}$, with $\kappa = \kappa_{\mathrm{range}}(\Pi_{\mathrm{singletons}})$ or the certified $\hat\kappa(\Pi_{\mathrm{singletons}})$, the interval on $\theta(D)$ (before the `rmse` square root and the `accuracy` clamp) is issued for all sufficiently large $n$ and its relative width $(U - L)/\hat\theta$ is $O(n^{-1/2})$.

In this proposition the constant $c$ is a correlation scale, unrelated to the
cluster index $c$ and the Cantelli constant $c(\Pi)$.

*Argument.*

1.  **Equation (4.1).**
    $\sum_{u \ne v} \sigma_u \sigma_v \mathrm{Corr}(s_u, s_v \mid D) \ge \rho_0((\sum_u \sigma_u)^2 - \sum_u \sigma_u^2) = \rho_0(n^2\bar\sigma^2 - \sum_u \sigma_u^2)$;
    dividing by $n^2$ gives (4.1).
2.  **Equation (4.2).** Taking $u \in \mathcal{O}$ with
    $d_{\mathrm{range}}(u) = d_{\mathrm{range}}^{\min}$ gives
    $n - 1 - d_{\mathrm{range}}^{\min}$ pairs outside $E_{\mathrm{range}}$,
    each contributing $\ge \rho_0$ to $\gamma$, so
    $\gamma \ge (n - 1 - d_{\mathrm{range}}^{\min})\rho_0$. Meanwhile
    $\lambda_{\max}(A_{\mathrm{range}}) \ge \mathbf{1}_n^\top A_{\mathrm{range}}\mathbf{1}_n / n = \bar d_{\mathrm{range}} \ge d_{\mathrm{range}}^{\min} \ge d_{\mathrm{range}}^{\min}\rho_0$
    (Rayleigh quotient at $\mathbf{1}_n$; $\rho_0 \le 1$). Adding gives
    $\kappa_{\mathrm{range}}(\Pi_{\mathrm{singletons}}) = 1 + \lambda_{\max}(A_{\mathrm{range}}) + \gamma \ge 1 + (n - 1)\rho_0$,
    so
    $\nu(\Pi_{\mathrm{singletons}}) \le n / (1 + (n - 1)\rho_0) = 1 / (\rho_0 + (1 - \rho_0)/n) \le 1/\rho_0$
    (strictly $< 1/\rho_0$ for $\rho_0 < 1$).
3.  **Part (2), the premise.** For every $u \in \mathcal{O}$ the out-of-range
    row sum is at most $(n - 1 - d_{\mathrm{range}}(u))(c / n) < c$ (or at
    most $c$ under the summable alternative), so `(T1-range)` holds with
    $\gamma = c$.
4.  **Part (2), the bounds on κ and ν.** At $\Pi_{\mathrm{singletons}}$,
    $A_{\mathrm{range}}^{\mathrm{cut}} = A_{\mathrm{range}}$ and Gershgorin's
    theorem gives
    $\lambda_{\max}(A_{\mathrm{range}}) \le d_{\mathrm{range}}^{\max} \le d_0$,
    so $1 \le \kappa_{\mathrm{range}}(\Pi_{\mathrm{singletons}}) \le 1 + d_0 + c$.
    Both code methods of Claim RE1(3) return
    $\hat\kappa = 1 + \min(\mathrm{ub}_{\mathrm{cw}}, \mathrm{ub}_{\mathrm{rows}})(1 + \delta) + \gamma$
    with $\mathrm{ub}_{\mathrm{rows}} = d_{\mathrm{range}}^{\max} \le d_0$ and
    $\delta \le \max(10^{-9}, 4(d_0 + 2)\varepsilon_{64}) \le 1$, so
    $1 \le \hat\kappa(\Pi_{\mathrm{singletons}}) \le 1 + 2d_0 + c$. Hence, for
    either $\kappa$, $n/(1 + 2d_0 + c) \le \nu(\Pi_{\mathrm{singletons}}) \le n$,
    that is, $\nu = \Theta(n)$.
5.  **Part (2), the width.** At $\Pi_{\mathrm{singletons}}$, $n_c \equiv 1$
    gives $r = 1$ and $\mathrm{CV}_n^2 = 0$, so
    $M_c = \min\{M, 1 + (M - 1)\} = M$ and the Cantelli constant is
    $c(\Pi_{\mathrm{singletons}})^2 = (M - 1)\kappa(1 - \alpha/2)/((\alpha/2)n) \le (M - 1)(1 + 2d_0 + c)(1 - \alpha/2)/((\alpha/2)n) = O(1/n)$.
    Since $c(\Pi)^2 = n_A(M_c, \alpha)\,\kappa / G$, the interval is issued
    ($G/\kappa > n_A$) exactly when $c(\Pi_{\mathrm{singletons}}) < 1$, which
    holds for all sufficiently large $n$. With
    $[L, U] = [\hat\theta/(1 + c(\Pi)), \hat\theta/(1 - c(\Pi))]$
    (Proposition RE3(1)),
    $(U - L)/\hat\theta = 2c(\Pi)/(1 - c(\Pi)^2) \le 4c(\Pi)$ once
    $c(\Pi) \le 1/\sqrt 2$, which is $O(n^{-1/2})$. (At $M = 1$,
    $c(\Pi) = 0$ and the width is $0$.) ∎

---

## 5. Monotonicity in the declarations, and missing data

> **Proposition RE3 (Monotonicity in $(K, L, H, R_s, R_t, \gamma)$ and Missing-Data Validity).**
>
> 1. **Monotonicity in declarations (exact $\kappa_{\mathrm{range}}$ vs. computed Collatz–Wielandt bounds):**
>    * **Exact $\kappa_{\mathrm{range}}(\Pi)$ and exact `choose_partition`:** Fix $\mathcal{O}$ and $\Pi$. If $(K', L', H', R_s', R_t', \gamma') \ge (K, L, H, R_s, R_t, \gamma)$ componentwise, then
>      $$\kappa_{\mathrm{range}}(\Pi; K', L', H', R_s', R_t', \gamma') \;\ge\; \kappa_{\mathrm{range}}(\Pi; K, L, H, R_s, R_t, \gamma). \tag{5.1}$$
>      Consequently, for any fixed $\Pi$, $\nu(\Pi) = G(\Pi) / \kappa_{\mathrm{range}}(\Pi)$ and $N_{\mathrm{eff}}(\Pi) = G(\Pi) / (\kappa_{\mathrm{range}}(\Pi)\,r_{\mathrm{eff}}(\Pi))$ (where $r_{\mathrm{eff}}(\Pi) = (M_c(\Pi) - 1)/(M - 1)$ for $M > 1$, and $r_{\mathrm{eff}}(\Pi) = r(\Pi)$ at $M = 1$) are non-increasing in $(K, L, H, R_s, R_t, \gamma)$. Moreover, for $M > 1$ on the correlated path (Claim 4, [THEORY.md](THEORY.md)), the interval $[L, U] = [\hat\theta/(1 + c(\Pi)),\, \hat\theta/(1 - c(\Pi))]$ depends on the losses **only** through the partition-invariant sample mean $\hat\theta = \frac{1}{n}\sum_{u \in \mathcal{O}} s_u$ and on $\Pi$ **only** through the Cantelli constant $c(\Pi) = \sqrt{\frac{(M_c(\Pi)-1)\kappa_{\mathrm{range}}(\Pi)(1-\alpha/2)}{(\alpha/2)G(\Pi)}} = \sqrt{\frac{(M-1)(1-\alpha/2)}{(\alpha/2)N_{\mathrm{eff}}(\Pi)}}$. Because for $M > 1$ maximizing $N_{\mathrm{eff}}(\Pi)$ over $\Pi \in \mathcal{F}_{\mathrm{cand}}$ is equivalent to minimizing $c(\Pi)$, the winning constant $c(\Pi^\star) = \min_{\Pi \in \mathcal{F}_{\mathrm{cand}}} c(\Pi)$ under exact $\kappa_{\mathrm{range}}$ and exact maximization of $N_{\mathrm{eff}}(\Pi)$ is also non-decreasing in $(K, L, H, R_s, R_t, \gamma)$—even when the winning partition $\Pi^\star$ switches—so increasing any declaration can only widen $[L, U]$ or trigger `ASSUMPTION_REQUIRED` ($c(\Pi^\star) \ge 1$). *(In code, `choose_partition` ranks candidates by $N_{\mathrm{eff}}(\Pi)$ rounded to 12 decimal places, then by deterministic tie-breaks, so the maximization is exact only up to that rounding.)*
>    * **Computed Collatz–Wielandt upper bounds $\hat\kappa(\Pi)$:** The code bounds $\hat\kappa(\Pi)$ in Claim RE1(3) are **certified upper bounds** $\hat\kappa(\Pi) \ge \kappa_{\mathrm{range}}(\Pi)$ at every declaration $(K, L, H, R_s, R_t, \gamma)$, so valid coverage $\ge 1 - \alpha$ holds at every declaration. However, because $\hat\kappa(\Pi)$ is evaluated via a finite number of shifted power iterations and the Collatz–Wielandt quotient $\max_u (Bx)_u/x_u$, the numerical upper bound $\hat\kappa(\Pi)$ itself is not guaranteed to be strictly monotone when adding edges changes the finite-iteration vector $x$.
> 2. **Removing items (missing data $\mathcal{O}' \subseteq \mathcal{O}$):** Let $\mathcal{O}' \subseteq \mathcal{O}$ be an exogenous sub-mask (Lemma R) and let $\Pi|_{\mathcal{O}'}$ be the restricted partition. If `(T1-range)` holds on $\mathcal{O}$ with $(\bar R_s, \bar R_t, \gamma)$, then `(T1-range)` holds on $\mathcal{O}'$ with the **same** $(\bar R_s, \bar R_t, \gamma)$ and
>    $$\lambda_{\max}\!\bigl(A_{\mathrm{range}}^{\mathrm{cut}}(\Pi|_{\mathcal{O}'})\bigr) \;\le\; \lambda_{\max}\!\bigl(A_{\mathrm{range}}^{\mathrm{cut}}(\Pi)\bigr). \tag{5.2}$$
>    * **Pooled second-moment premise `(P2*_pool(M))` must hold on $\mathcal{O}'$ itself:** Unlike `(T1-range)`, the pooled moment inequality $\frac{1}{|\mathcal{O}'|}\sum_{u \in \mathcal{O}'} \mathbb{E}[s_u^2 \mid D] \le M\,\theta_{\mathcal{O}'}(D)^2$ is **not** automatically inherited from $\mathcal{O}$ when low-relative-variance items are dropped (for example, if $\mathcal{O} = \{1, 2\}$ has $\mathbb{E}[s_1] = 1, \mathbb{E}[s_1^2] = 1$ and $\mathbb{E}[s_2] = 1, \mathbb{E}[s_2^2] = 4$, then $\mathcal{O}$ satisfies `(P2*_pool(M))` with $M = (1 + 4)/(2 \cdot 1^2) = 2.5$, whereas $\mathcal{O}' = \{2\}$ has $\mathbb{E}[s_2^2]/(\mathbb{E}[s_2])^2 = 4 > 2.5$). Thus $M$ (`m_item`) must always be declared for the observed evaluation set $\mathcal{O}'$ (or holds on every non-empty $\mathcal{O}' \subseteq \mathcal{O}$ under uniform item-wise `(P2*(M))` $\mathbb{E}[s_u^2 \mid D] \le M\mu_u^2$ when item means are homogeneous).
>    * **Singletons ($n_c \equiv 1$):** Under exogenous $\mathcal{O}'$ and `(P2*_pool(M))` on $\mathcal{O}'$, missing data requires no `(T2)` assumption because `(T2)` holds with equality on $\mathcal{O}'$.
>    * **Multi-item clusters ($n_c \ge 2$):** Premise `(T2)` on the observed sub-clusters $\Pi|_{\mathcal{O}'}$ ($\sum_c \mathrm{Var}(S_c^{\mathcal{O}'} \mid D) \ge \sum_{u \in \mathcal{O}'} \sigma_u^2$) is **not** inherited from `(T2)` on $\mathcal{O}$ (removing positively correlated items from a cluster can leave net negative covariance among the remaining items) and is therefore a **separate required premise** on $\Pi|_{\mathcal{O}'}$ in addition to `(P2*_pool(M))` on $\mathcal{O}'$.
>
> Here $\theta_{\mathcal{O}'}(D) := \frac{1}{|\mathcal{O}'|}\sum_{u \in \mathcal{O}'} \mu_u$ and $S_c^{\mathcal{O}'} := \sum_{u \in B_c \cap \mathcal{O}'} s_u$.

*Argument.*

1.  **Part (1), equation (5.1).** Larger $(K', L', H', R_s', R_t')$ gives
    $\bar R_s' \ge \bar R_s$ and $\bar R_t' \ge \bar R_t$, so
    $0 \le A_{\mathrm{range}}^{\mathrm{cut}}(\Pi; \bar R_s, \bar R_t) \le A_{\mathrm{range}}^{\mathrm{cut}}(\Pi; \bar R_s', \bar R_t')$
    entrywise, and (5.1) follows by Perron–Frobenius monotonicity of
    $\lambda_{\max}$ and $\gamma' \ge \gamma$.
2.  **Part (1), the winning constant.** For $M > 1$, $c(\Pi)$ is a strictly
    decreasing function of $N_{\mathrm{eff}}(\Pi)$; each $c(\Pi)$ is
    non-decreasing in the declarations, so
    $\min_{\Pi \in \mathcal{F}_{\mathrm{cand}}} c(\Pi)$ is also non-decreasing.
3.  **Part (2), `(T1-range)` on $\mathcal{O}'$.** Dropping
    $v \in \mathcal{O} \setminus \mathcal{O}'$ from the non-negative sum in
    (1.4) cannot increase the sum.
4.  **Part (2), equation (5.2).** Let $x \in \mathbb{R}^{|\mathcal{O}'|}$ be any
    unit vector ($\|x\|_2 = 1$) and let $\tilde x \in \mathbb{R}^{|\mathcal{O}|}$
    be its extension by zeros on $\mathcal{O} \setminus \mathcal{O}'$
    ($\|\tilde x\|_2 = 1$). Because
    $A_{\mathrm{range}}^{\mathrm{cut}}(\Pi|_{\mathcal{O}'})$ is the restriction
    of $A_{\mathrm{range}}^{\mathrm{cut}}(\Pi)$ to
    $\mathcal{O}' \times \mathcal{O}'$,
    $x^\top A_{\mathrm{range}}^{\mathrm{cut}}(\Pi|_{\mathcal{O}'}) x = \tilde x^\top A_{\mathrm{range}}^{\mathrm{cut}}(\Pi) \tilde x \le \lambda_{\max}(A_{\mathrm{range}}^{\mathrm{cut}}(\Pi))$.
    Taking the supremum over $\|x\|_2 = 1$ gives (5.2). ∎

---

## 6. Which premises can be refuted, and on what data

The table below states, for each premise, whether it can be refuted on
training data $\mathcal{T}_{\mathrm{train}}$ (with $T_{\mathrm{train}}$ time
steps and $n_{\mathrm{train}}$ items) held out from evaluation, and why.
Proposition REU1 below supplies the impossibility results.

| Premise / Quantity | Refutable on Training Data $\mathcal{T}_{\mathrm{train}}$? | Mathematical Justification |
| :--- | :--- | :--- |
| **Model parameters $(K, L, H)$ & `(P1)` ($s_u \ge 0$)** | **Yes (deterministic)** | Fixed architecture/horizon properties and direct non-negativity check. |
| **Local range $(R_s, R_t)$ & tail $\gamma$ at lags/hops $\ll T_{\mathrm{train}}$** | **Yes (asymptotically as $T_{\mathrm{train}} \to \infty$ under stationarity & bounded 4th moments); no test argued here** | Out-of-range correlations exceeding $\gamma$ on $\mathcal{T}_{\mathrm{train}}$ can be refuted asymptotically via lag/hop shell tests. No such test is stated or argued in this document, and the library implements none. |
| **Within-cluster `(T2)` ($\sum_c \mathrm{Var}(S_c) \ge \sum_u \sigma_u^2$)** | **Exact identity at $n_c \equiv 1$; at $n_c \ge 2$, refutable on $\mathcal{T}_{\mathrm{train}}$ only modulo a mean model** | Separating $\mathrm{Cov}(s_u, s_v)$ from $(\mu_u - \mu_v)^2$ at $n_c \ge 2$ requires repeated draws under a stationary mean model (Proposition REU1(3)). |
| **Training-to-test stationarity & test-mask exogeneity `(P0)`** | **No (impossible from $\mathcal{T}_{\mathrm{train}}$)** | Test-period regime shifts, common shocks, or loss-dependent dropouts occur outside $\mathcal{T}_{\mathrm{train}}$ (Proposition REU1(1)). |
| **Population moment `(P2*_pool(m_item))` on $\mathcal{T}_{\mathrm{eval}}$** | **No in finite samples (Bahadur–Savage); empirically checked on $\mathcal{T}_{\mathrm{eval}}$ by $\widehat M_G \le M_c$** | Rare tail spikes of probability $\varepsilon \ll 1/n_{\mathrm{train}}$ cannot be ruled out from $\mathcal{T}_{\mathrm{train}}$ (Proposition REU1(2)). |

Premise `(P0)` (design exogeneity) means that the design, including the
observation mask $\mathcal{O}$, the graph, the candidate family and the
declarations, is fixed before observing the evaluation losses
$\{s_u\}_{u \in \mathcal{O}}$ and is not a function of them.

> **Proposition REU1 (Structural unverifiability from a single evaluation realization $\{s_u\}_{u \in \mathcal{O}}$).**
> Given a single realization $s = (s_u)_{u \in \mathcal{O}} \in \mathbb{R}_{\ge 0}^n$ with design $D$:
>
> 1. **Exogeneity `(P0)` is distributionally untestable from $s$ alone:** For any law $\mathbb{P}_0(s \mid D)$ satisfying `(P0)` with $\theta(D) = \theta_0$ and any $\tilde\theta \ne \theta_0$, there exists a joint law $\mathbb{P}_1(D, s_{\mathrm{full}})$ whose conditional distribution on the realized $(D, \mathcal{O})$ is identical to $\mathbb{P}_0(\cdot \mid D)$ (so no test on $(D, s)$ can distinguish $\mathbb{P}_0$ from $\mathbb{P}_1$), yet whose full-grid expectation over missing entries $(V_s \times \{0,\dots,T-1\}) \setminus \mathcal{O}$ is $\tilde\theta$.
> 2. **Population relative second moment `(P2*_pool(m_item))` has no distribution-free finite-sample verification test (Bahadur–Savage):** Fix any law $\mathbb{P}_0$ on $\mathbb{R}_{\ge 0}^n$ satisfying `(P2*_pool(m_item))` with mean $\theta_0 > 0$ and relative second moment $m_0 \le m_{\mathrm{item}}$, any level $\alpha_{\mathrm{ref}} \in (0, 1)$, any $\varepsilon \in (0, \alpha_{\mathrm{ref}})$, and any target relative second moment $M_{\mathrm{true}} > m_{\mathrm{item}}$. Set $\varepsilon_M := \min\bigl(\varepsilon,\, \frac{1}{18 M_{\mathrm{true}}}\bigr) > 0$, and let $\mathbb{P}_{\varepsilon_M}$ be the mixture law that draws $s \sim \mathbb{P}_0$ with probability $1 - \varepsilon_M$ and sets $s = x_0\,\mathbf{1}_n$ with probability $\varepsilon_M$, where $x_0 := \theta_0\sqrt{\frac{2 M_{\mathrm{true}}}{\varepsilon_M}}$. *(Note: a single point-mass mixture of fixed weight $\varepsilon$ has relative second moment $\to 1/\varepsilon$ as $x_0 \to \infty$, so reaching $M_{\mathrm{true}} > 1/\varepsilon$ requires spike probability $\varepsilon_M = O(1/M_{\mathrm{true}}) \le \varepsilon$.)* Then $\mathrm{TV}(\mathbb{P}_0, \mathbb{P}_{\varepsilon_M}) \le \varepsilon_M \le \varepsilon$ (so every test $\psi(s) \in [0, 1]$ with $\mathbb{E}_{\mathbb{P}_0}[\psi] \le \alpha_{\mathrm{ref}}$ has power $\mathbb{E}_{\mathbb{P}_{\varepsilon_M}}[\psi] \le \alpha_{\mathrm{ref}} + \varepsilon_M \le \alpha_{\mathrm{ref}} + \varepsilon$), while $\theta_{\mathbb{P}_{\varepsilon_M}} = \theta_0(1 - \varepsilon_M + \sqrt{2\varepsilon_M M_{\mathrm{true}}})$ and
>    $$\frac{\frac{1}{n}\sum_{u \in \mathcal{O}} \mathbb{E}_{\mathbb{P}_{\varepsilon_M}}[s_u^2]}{\theta_{\mathbb{P}_{\varepsilon_M}}^2} \;=\; \frac{(1 - \varepsilon_M)m_0 + 2 M_{\mathrm{true}}}{\bigl(1 - \varepsilon_M + \sqrt{2\varepsilon_M M_{\mathrm{true}}}\bigr)^2} \;>\; \frac{2 M_{\mathrm{true}}}{(1 + 1/3)^2} \;=\; \frac{9}{8}\,M_{\mathrm{true}} \;>\; M_{\mathrm{true}}.$$
>    *(Note: Claim 4's empirical gate $\widehat M_G \le M_c$ is a diagnostic check, not a distribution-free verification test of population `(P2v)`: both endpoints of Claim 4's interval $[\hat\theta/(1+c), \hat\theta/(1-c)]$ depend only on $\hat\theta$ and the declared $(M_c, \kappa, \alpha, G)$ and rely on the population premise `(P2v)` rather than $\widehat M_G$. Under the spike mixture $\mathbb{P}_{\varepsilon_M}$, with probability $1 - \varepsilon_M$ the rare spike is absent from the sample so $\widehat M_G \le M_c$ (`UNREFUTED`), while the unseen spike moves the mean to $\theta_{\mathbb{P}_{\varepsilon_M}} = \theta_0(1 - \varepsilon_M + \sqrt{2\varepsilon_M M_{\mathrm{true}}})$ and $\hat\theta$ keeps its $\mathbb{P}_0$ distribution; the upper endpoint misses whenever $\hat\theta < (1 - c)\,\theta_{\mathbb{P}_{\varepsilon_M}}$, an event whose probability the premise check does not control.)*
> 3. **Within-cluster non-repulsion `(T2)` for $n_c \ge 2$ is unidentifiable without a within-cluster mean model, whereas `(T2)` is an exact algebraic identity at $n_c \equiv 1$:** Suppose $n_c = 2$ for each cluster $B_c = \{u_c, v_c\}$ ($c = 1, \dots, G$). For any deterministic vector of within-cluster mean half-differences $\Delta = (\Delta_1, \dots, \Delta_G) \in \mathbb{R}^G$ fixed by $D$, let $\mathbb{E}[s_{u_c} \mid D] = \theta + \Delta_c$ and $\mathbb{E}[s_{v_c} \mid D] = \theta - \Delta_c$, so the cluster totals $S_c = s_{u_c} + s_{v_c}$ have fixed conditional mean $2\theta$ and the within-cluster differences $W_c = s_{u_c} - s_{v_c} = 2\Delta_c + (\eta_{u_c} - \eta_{v_c})$ have fixed conditional mean $2\Delta_c$, where $\eta_u = s_u - \mathbb{E}[s_u \mid D]$. Consider two parametric families of conditional Gaussian laws $\mathbb{P}_{(\Delta, +)}(\cdot \mid D)$ and $\mathbb{P}_{(\Delta, -)}(\cdot \mid D)$ indexed by the unknown deterministic mean parameter $\Delta \in \mathbb{R}^G$, both having $S_c \stackrel{\mathrm{i.i.d.}}{\sim} \mathcal{N}(2\theta, 2\sigma^2)$ independently of $W_c$:
>    * **Family $\mathcal{M}_+ = \{\mathbb{P}_{(\Delta, +)}(\cdot \mid D) : \Delta \in \mathbb{R}^G\}$ (`(T2)` holds with equality for every $\Delta$):** $\sigma_{u_c}^2 = \sigma_{v_c}^2 = \sigma^2$ and $\mathrm{Cov}(s_{u_c}, s_{v_c} \mid D) = 0$. Then $\mathrm{Var}(\eta_{u_c} - \eta_{v_c} \mid D) = 2\sigma^2$, so conditional on $D$, $W_c \stackrel{\mathrm{ind.}}{\sim} \mathcal{N}(2\Delta_c, 2\sigma^2)$ and $\sum_{c=1}^G \mathrm{Var}(S_c \mid D) = 2G\sigma^2 = \sum_{u \in \mathcal{O}} \sigma_u^2$.
>    * **Family $\mathcal{M}_- = \{\mathbb{P}_{(\Delta, -)}(\cdot \mid D) : \Delta \in \mathbb{R}^G\}$ (`(T2)` violated by $7/4$ for every $\Delta$):** $\sigma_{u_c}^2 = \sigma_{v_c}^2 = \frac{7}{4}\sigma^2$ and $\mathrm{Cov}(s_{u_c}, s_{v_c} \mid D) = -\frac{3}{4}\sigma^2$ (whose $2 \times 2$ covariance matrix has eigenvalues $\sigma^2 > 0$ and $\frac{5}{2}\sigma^2 > 0$, hence is strictly positive definite). Then $\mathrm{Var}(S_c \mid D) = \frac{7}{4}\sigma^2 + \frac{7}{4}\sigma^2 - \frac{6}{4}\sigma^2 = 2\sigma^2$ and $\mathrm{Var}(\eta_{u_c} - \eta_{v_c} \mid D) = \frac{7}{4}\sigma^2 + \frac{7}{4}\sigma^2 + \frac{6}{4}\sigma^2 = 5\sigma^2$, so conditional on $D$, $W_c \stackrel{\mathrm{ind.}}{\sim} \mathcal{N}(2\Delta_c, 5\sigma^2)$ and $\sum_{c=1}^G \mathrm{Var}(S_c \mid D) = 2G\sigma^2 < \frac{7}{2}G\sigma^2 = \sum_{u \in \mathcal{O}} \sigma_u^2$.
>
>    Placing the Gaussian prior $\pi_+ = \mathcal{N}(0, \frac{3}{2}\sigma^2 I_G)$ on $\Delta$ over $\mathcal{M}_+$ and $\pi_- = \mathcal{N}(0, \frac{3}{4}\sigma^2 I_G)$ on $\Delta$ over $\mathcal{M}_-$ yields **identical** Bayes marginal distributions $\int \mathbb{P}_{(\Delta, +)}(\cdot \mid D)\,d\pi_+(\Delta) = \int \mathbb{P}_{(\Delta, -)}(\cdot \mid D)\,d\pi_-(\Delta)$ on $\{(s_{u_c}, s_{v_c})\}_{c=1}^G$, under which $(S_c, W_c) \stackrel{\mathrm{i.i.d.}}{\sim} \mathcal{N}(2\theta, 2\sigma^2) \otimes \mathcal{N}(0, 8\sigma^2)$. Hence for any test $\psi(s) \in [0, 1]$ with level $\sup_{\Delta \in \mathbb{R}^G} \mathbb{E}_{(\Delta, +)}[\psi(s) \mid D] \le \alpha_{\mathrm{ref}}$ on $\mathcal{M}_+$, its worst-case power on $\mathcal{M}_-$ satisfies $\inf_{\Delta \in \mathbb{R}^G} \mathbb{E}_{(\Delta, -)}[\psi(s) \mid D] \le \alpha_{\mathrm{ref}}$. Conversely, when $n_c = 1$ for all $c = 1, \dots, G$, $S_c = s_{u_c}$ so $\sum_{c=1}^G \mathrm{Var}(S_c \mid D) = \sum_{u \in \mathcal{O}} \sigma_u^2$ holds **identically**.
>
> Here $\mathrm{TV}$ is total variation distance, $\psi$ is any (randomized) test with values in $[0, 1]$, and $\alpha_{\mathrm{ref}}$ is its nominal level.

*Argument.*

1.  **Part (1).** Observing $s_{\mathcal{O}} \sim \mathbb{P}(\cdot \mid D)$
    constrains only the marginal on $\mathcal{O}$, leaving the conditional
    distribution of $s_{\mathcal{O}^c} \mid (D, s_{\mathcal{O}})$ completely
    arbitrary.
2.  **Part (2).** By the coupling inequality (Bahadur and Savage, 1956),
    $|\mathbb{E}_{\mathbb{P}_{\varepsilon_M}}[\psi] - \mathbb{E}_{\mathbb{P}_0}[\psi]| \le \mathrm{TV}(\mathbb{P}_0, \mathbb{P}_{\varepsilon_M}) \le \varepsilon_M \le \varepsilon$.
    Since $\varepsilon_M \le \frac{1}{18 M_{\mathrm{true}}}$,
    $\sqrt{2\varepsilon_M M_{\mathrm{true}}} \le 1/3$, so
    $(1 - \varepsilon_M + \sqrt{2\varepsilon_M M_{\mathrm{true}}})^2 < (4/3)^2 = 16/9$,
    while $(1 - \varepsilon_M)m_0 + 2 M_{\mathrm{true}} \ge 2 M_{\mathrm{true}}$,
    giving ratio $> \frac{9}{8} M_{\mathrm{true}} > M_{\mathrm{true}}$.
3.  **Part (3).** Under $\mathbb{P}_{(\Delta, +)}(\cdot \mid D)$ with $\Delta_c \stackrel{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, \frac{3}{2}\sigma^2)$, $W_c = 2\Delta_c + (\eta_{u_c} - \eta_{v_c}) \stackrel{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, 4\cdot\frac{3}{2}\sigma^2 + 2\sigma^2) = \mathcal{N}(0, 8\sigma^2)$; under $\mathbb{P}_{(\Delta, -)}(\cdot \mid D)$ with $\Delta_c \stackrel{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, \frac{3}{4}\sigma^2)$, $W_c \stackrel{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, 4\cdot\frac{3}{4}\sigma^2 + 5\sigma^2) = \mathcal{N}(0, 8\sigma^2)$. Since $(s_{u_c}, s_{v_c}) = \bigl(\frac{S_c + W_c}{2}, \frac{S_c - W_c}{2}\bigr)$ is the same bijective linear transformation of $(S_c, W_c)$, the two Bayes mixtures coincide, so $\inf_{\Delta} \mathbb{E}_{(\Delta, -)}[\psi \mid D] \le \int \mathbb{E}_{(\Delta, -)}[\psi \mid D]\,d\pi_-(\Delta) = \int \mathbb{E}_{(\Delta, +)}[\psi \mid D]\,d\pi_+(\Delta) \le \sup_{\Delta} \mathbb{E}_{(\Delta, +)}[\psi \mid D] \le \alpha_{\mathrm{ref}}$. ∎

---

## 7. Why the library does not estimate γ from data

The library includes **no automated helper that suggests $(R_s, R_t, \gamma)$
from training data**: $\gamma$ is always a required user declaration, and
coverage is conditional on the declared $(K, L, H, R_s, R_t, \gamma)$. The
reason is that the obvious estimator—summing absolute sample correlations over
out-of-range pairs—is biased upward by sampling noise alone, by an amount that
grows with the number of out-of-range pairs (Proposition RE4).

> **Proposition RE4 (Upward Noise Bias of Naive Absolute-Correlation Summing).**
> Suppose on a training split of $N_s$ nodes and $T_{\mathrm{train}}$ steps, for an item $u$ and $N_{\mathrm{far}}(u)$ out-of-range items $v \notin B_{\mathrm{range}}(u)$, the pairwise sample correlations $\hat\rho(u, v)$ satisfy $\sqrt{T_{\mathrm{train}}}\,\hat\rho(u, v) \xrightarrow{d} \mathcal{N}(0, v_{uv})$ under $H_0: \rho_{\mathrm{true}}(u, v) = 0$ ($\gamma_{\mathrm{true}} = 0$), with $v_{uv} \ge v_{\min} > 0$ and $\{\sqrt{T_{\mathrm{train}}}|\hat\rho(u, v)|\}_{T_{\mathrm{train}} \ge 1}$ uniformly integrable for each $v \notin B_{\mathrm{range}}(u)$. Assume further either that $N_{\mathrm{far}}(u)$ is fixed as $T_{\mathrm{train}} \to \infty$ or that the convergence $\mathbb{E}_{H_0}[\sqrt{T_{\mathrm{train}}}|\hat\rho(u, v)|] / \sqrt{2v_{uv}/\pi} \to 1$ is uniform over $v \notin B_{\mathrm{range}}(u)$. *(Note on Bartlett's formula: when the two stationary series $(s_{u,t})_t$ and $(s_{v,t})_t$ are **independent processes**—or jointly fourth-order stationary with zero cross-correlations $\mathrm{Corr}(s_{u,t}, s_{v,t+k}) = 0$ at **all** lags $k \in \mathbb{Z}$ and zero fourth-order cross-cumulants, not merely $\rho_{\mathrm{true}}(u, v) = 0$ at lag $0$—Bartlett's formula gives $v_{uv} = \sum_{k=-\infty}^\infty \rho_u(k)\rho_v(k)$, which need not be $\ge 1$: for example, two independent AR(1) series with parameters $+0.5$ and $-0.5$ have $v_{uv} = 1 + 2\sum_{k=1}^\infty (-0.25)^k = 0.6 < 1$, so only $v_{\min} > 0$ holds in general.)* Then the naive plug-in sum $\hat\gamma_{\mathrm{naive}}(u) := \sum_{v \notin B_{\mathrm{range}}(u)} |\hat\rho(u, v)|$ satisfies
> $$\mathbb{E}_{H_0}\bigl[\hat\gamma_{\mathrm{naive}}(u)\bigr] \;=\; \sqrt{\frac{2}{\pi\,T_{\mathrm{train}}}}\,\sum_{v \notin B_{\mathrm{range}}(u)} \sqrt{v_{uv}}\,\bigl(1 + o(1)\bigr) \;\ge\; \sqrt{\frac{2\,v_{\min}}{\pi}}\,\frac{N_{\mathrm{far}}(u)}{\sqrt{T_{\mathrm{train}}}}\,\bigl(1 + o(1)\bigr). \tag{7.1}$$
>
> Here $\rho_u(k)$ is the lag-$k$ autocorrelation of the series $(s_{u,t})_t$.

*Argument.*

1.  By asymptotic normality and uniform integrability,
    $\lim_{T_{\mathrm{train}}\to\infty} \mathbb{E}_{H_0}[\sqrt{T_{\mathrm{train}}}|\hat\rho(u, v)|] = \mathbb{E}[|\mathcal{N}(0, v_{uv})|] = \sqrt{2 v_{uv} / \pi}$
    for each $v \notin B_{\mathrm{range}}(u)$.
2.  Summing over the $N_{\mathrm{far}}(u)$ pairs (using fixed
    $N_{\mathrm{far}}(u)$ or uniform convergence across
    $v \notin B_{\mathrm{range}}(u)$) gives (7.1). ∎

---

## Not covered

*   Rolling-origin evaluation and grouped holdout.

## References

*   R. R. Bahadur and L. J. Savage (1956). The nonexistence of certain
    statistical procedures in nonparametric problems. *Annals of Mathematical
    Statistics* 27(4), 1115–1122.
*   M. S. Bartlett (1946). On the theoretical specification and sampling
    properties of autocorrelated time-series. *Supplement to the Journal of the
    Royal Statistical Society* 8(1), 27–41.

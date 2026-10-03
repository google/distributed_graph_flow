# Confidence intervals for dependent evaluation data

> The mathematical arguments in the theory documents linked below have not
> been peer-reviewed. Simulations support the main claims (see
> [`EVIDENCE.md`](./EVIDENCE.md)), but errors are possible; please report any
> you find.

`dgf.src.stats` is a Python library that computes confidence intervals for evaluation
metrics: mean absolute error (MAE), mean squared error (MSE), root mean squared
error (RMSE), accuracy, and the coefficient of determination (R²). It provides
a finite-sample bound, argued in [`THEORY.md`](./THEORY.md), that holds at the
sample size you have, not only "for large enough n".
The library covers two kinds of test sets:

*   **Independent items**, such as a test set drawn at random from a pool. The
    `independent` module handles them.
*   **Dependent items**, whose errors can be correlated. Examples are
    forecasts ordered in time, nodes of a graph, and graph nodes observed over
    time. The `temporal`, `graph` and `spacetime` modules handle them.

For dependent items you state a small number of assumptions as numbers, called
**declarations**. The interval covers the true value with at least the
requested probability whenever the declarations hold. The library also checks
the declarations against the data and reports when the data contradict them.
The checks can contradict a declaration, but they can never confirm one.

> **Scope.** This guide describes confidence intervals for independent
> samples and for dependent data across time, graphs, and space-time.

The arguments are in [`THEORY.md`](./THEORY.md) (cluster, temporal and graph
intervals), [`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md) (space-time
intervals) and [`THEORY_RANGE_ENVELOPE.md`](./THEORY_RANGE_ENVELOPE.md) (the
range declaration). The design rationale is in [`DESIGN.md`](./DESIGN.md), the
simulation evidence in [`EVIDENCE.md`](./EVIDENCE.md), and the
release history in [`CHANGELOG.md`](./CHANGELOG.md).

## Which module to use

The right module depends on how your test data were produced. Section 3
explains the two experimental setups. In **Setup A**, the test items are a
random subsample of a pool. In **Setup B**, a fixed test set is evaluated in
full. All dependent entry points below are Setup B; they differ in where κ
(the dependence declaration of §4) comes from.

| Your evaluation | Module and entry point | Section |
| :--- | :--- | :--- |
| Items drawn at random from a fixed pool, without looking at losses (Setup A), for example a random node split | `independent.confidence_interval` | §3, §9 |
| A fixed test set over a graph and a time period, evaluated in full (Setup B, κ derived from model and data ranges) | `spacetime.spacetime_interval`, with κ from `range_envelope.range_kappa_fn` | §3 |
| Time-ordered items: temporal splits, forecasts, sequential logs (Setup B, κ declared) | `temporal.temporal_interval` | §5 |
| Only per-time-bucket counts and loss sums, such as hourly aggregates (Setup B, κ declared) | `temporal.temporal_interval_from_buckets` | §5 |
| Nodes of a graph whose losses may correlate along edges (Setup B, κ declared or computed from an edge bound) | `graph.graph_interval` | §6 |
| Graph nodes observed over time, with declared spatial and temporal κ or edge correlation bounds (Setup B) | `spacetime.spacetime_interval` with `spacetime.separable_kappa` or `spacetime.topological_kappa_fn` | §10 |
| Pre-aggregated totals from a distributed job | `temporal.temporal_interval_from_shard`, `graph.graph_interval_from_shard` | §5 |
| R² on any of the above | the same entry point with `metric="r2"` and `data=(y_true, y_pred)` | §5 |

All dependent entry points use the same idea. Items are grouped into
**clusters**: contiguous time windows for temporal data, parts of a graph
partition for graph data, or space-time product blocks for spatio-temporal
data. The interval is built from the per-cluster loss totals. Correlation
*within* a cluster is allowed without limit. Correlation *between* clusters is
what you declare (§4) and what the checks try to refute.

## Minimal example

A minimal example with time-ordered residuals (prediction minus label):

```python
from dgf.src.stats import temporal

result = temporal.temporal_interval(residuals, num_windows=1000, metric="mse")
print(result.low, result.high, result.status, result.refuted)
```

Complete runnable examples are in §1. The terms are defined in §2.

## Why textbook intervals break under correlation

Textbook intervals, such as the Student-t interval or the bootstrap over
items, assume independent items. Under correlation they are too narrow. The
table below comes from 2000 simulated repetitions per row, with the true value
known exactly. Each method promises at most 5% misses (2.5% per side).

| Data | Student-t misses | i.i.d. bootstrap misses | This package |
| :--- | :---: | :---: | :---: |
| MSE, AR(1) errors, ρ = 0.5, n = 100 000 | **12.1%** | **12.8%** | 0% |
| MSE, AR(1) errors, ρ = 0.8, n = 100 000 | **36.3%** | **36.6%** | 0% |
| MSE, AR(1) errors, ρ = 0.95, n = 100 000 | **66.0%** | **66.5%** | 0% |
| Accuracy 0.99, AR(1), correlation length 4, n = 20 000 | **22.6%** | **22.8%** | 0% |
| Accuracy 0.999, AR(1), correlation length 4, n = 20 000 | **14.5%** | **14.0%** | 0% |

AR(1) is the first-order autoregressive process: each error is ρ times the
previous error plus independent noise. The correlation length is the number
of consecutive items over which errors stay noticeably correlated. "i.i.d."
means independent and identically distributed. [`EVIDENCE.md`](./EVIDENCE.md)
gives the commands that regenerate this table.

The price is width. In the MSE rows, which use the default declaration, the
interval's full width is 0.9–1.2 times the true MSE, against about 0.02 times
for the textbook intervals. The textbook intervals are narrow because they
ignore the correlation. A tighter declaration, justified from earlier data,
narrows this package's interval (§4.1). For the same Gaussian errors, whose
squared-error moment ratio is 3, declaring `m_item=4` instead of the default 16 gives
0.36–0.45 times the true MSE ([EVIDENCE §4](./EVIDENCE.md#4-correlated-r2-midpoint-bucketed-windows-and-topological-kappa)).
That is still 6–16 times wider than a normal interval with the correct
long-run variance. About a factor of 3 is the cost of a distribution-free
tail bound (Cantelli) in place of a normal approximation; the rest comes from
bounding the window-level moment ratio from the item-level `m_item`, which a
window-level `m` avoids.

When correlation is weak, textbook intervals can be adequate. On four real
graph topologies with a simulated loss field, they missed 4.1–6.5% of the
time, close to the promised 5%. In that setting, this package costs width for
little gain in safety.

## 1. Quickstart

### Temporal

```python
import numpy as np
from dgf.src.stats import temporal

# Time-ordered validation residuals (prediction - label).
residuals = np.random.default_rng(0).normal(loc=0.0, scale=1.0, size=20000)

result = temporal.temporal_interval(
    residuals,
    num_windows=1000,
    metric="mse",
    level=0.95,
    kappa=1.0,
)

print(f"Interval: [{result.low:.4f}, {result.high:.4f}]")
print(f"Status: {result.status.value}, Refuted: {result.refuted}")
print(f"Drift: {result.drift.outcome.value}")
print(f"Message: {result.message}")
```

### Graph

```python
import numpy as np
from dgf.src.stats import graph
from dgf.src.stats import partitioners

# edges: (E, 2) int array of node pairs; residuals: one value per node.
rng = np.random.default_rng(0)
num_nodes = 20000
edges = np.column_stack([np.arange(num_nodes - 1), np.arange(1, num_nodes)])
residuals = rng.normal(size=num_nodes)

# Clusters and cluster edges come from the graph only, never from the losses.
cluster_ids = partitioners.partition_graph(num_nodes, edges, target_cluster_size=20)
cluster_edges = graph.cluster_adjacency(num_nodes, edges, cluster_ids)

result = graph.graph_interval(
    residuals, cluster_ids, cluster_edges, metric="mse", level=0.95, kappa=1.0
)

print(f"Interval: [{result.interval.low:.4f}, {result.interval.high:.4f}]")
print(f"Status: {result.interval.status.value}, Refuted: {result.refuted}")
print(f"Edge check: {result.kappa_check.outcome.value}")
print(f"Message: {result.message}")
```

### Independent items and space-time data

*   For independent items, see the example in §9.
*   For a fixed test set over a graph and a time period, see the range-envelope
    example at the end of §3.
*   For space-time data with declared spatial and temporal κ, including
    incomplete grids with dropouts, see
    [§10](#10-space-time-evaluation-v2spacetime).

The result types store endpoints directly (`result.low`, `result.high`,
`result.status` for temporal and space-time) or under `result.interval` (for
graph). §7 lists every field.

## 2. Notation and terms

Each item $i$ contributes a non-negative **summand** $s_i$ to the metric:
$|e_i|$ for MAE, $e_i^2$ for MSE and RMSE, and the 0/1 correctness indicator
for accuracy, where $e_i$ is the residual (prediction minus label). The
metric is the mean summand (RMSE is its square root). The following table
defines the symbols and terms used in the rest of this guide:

| Symbol or term | Meaning |
| :--- | :--- |
| $\theta$ | The **estimand**: the true value of the metric that the interval is meant to cover. §5 and §6 state what it is for each data type. |
| level | The requested coverage probability, such as 0.95. |
| $\alpha$ | The allowed miss probability, $\alpha = 1 - \text{level}$. |
| $M$ | Item-level **relative second moment**: an upper bound on $\mathbb{E}[s^2] / \mathbb{E}[s]^2$ for the item summands. Large $M$ means heavy tails. Defaults: 4 for MAE and accuracy, 16 for MSE, RMSE and R². Passed as `m_item`. |
| cluster, $c$ | A group of items: a time window, or a part of a graph partition. |
| $S_c$ | The **cluster total**: the sum of the summands in cluster $c$. |
| $M_c$ | Cluster-level relative second moment: an upper bound on $\mathbb{E}[S_c^2] / \mathbb{E}[S_c]^2$. Passed as `m`, for every metric including R². If you pass neither `m` nor `m_item`, it is derived from $M$ (the default or `m_item`) and the actual cluster sizes, allowing any correlation inside a cluster: for equal-size clusters $M_c = M$; otherwise $M_c = \min\{rM,\ 1 + r(M-1) + v + 2\sqrt{v\,r(M-1)}\}$, where $r$ is the largest cluster size divided by the mean cluster size and $v$ is the squared coefficient of variation of the cluster sizes (Lemma I′ in [`THEORY.md`](./THEORY.md)). |
| $G$ | The number of clusters (for temporal data, `num_windows`). |
| $\kappa$ | The **variance inflation factor** (VIF) between clusters: how much larger the variance of the sum of all cluster totals is than it would be if the clusters were uncorrelated. $\kappa = 1$ means uncorrelated clusters. Passed as `kappa`. $\kappa = 1$ asserts that **all** pairs of clusters are uncorrelated; the checks (§8) test only adjacent pairs. The largest absolute row sum of the correlation matrix, computed over cluster totals, is a sufficient bound. |
| $\nu$ | The effective number of clusters, $\nu = G / \kappa$. The temporal result reports it as `effective_n`; unlike an item-level `effective_n`, it counts clusters, not items. |
| $n_A$ | The smallest $\nu$ for which an interval can be issued (§5). |
| correlation length $L$ | The number of consecutive items over which errors stay noticeably correlated. (§3 uses $L$ for a model's lookback instead; the meaning is stated where it is used.) |
| relative full width | $(\text{high} - \text{low}) / \theta$: the interval width as a fraction of the true value. |
| certified | The interval covers $\theta$ with at least the requested level, **provided** the declared $M_c$ (or $M$) and $\kappa$ hold. |
| refuted | A check found the data inconsistent with a declaration. |
| super-batch | One of about 30 groups of clusters that the $\kappa > 1$ checks use: consecutive windows for temporal data, neighbouring clusters for graph data (§8). |

## 3. Choosing your experiment and declarations

Coverage depends on how your test data were produced. This library
supports two setups. Pick the one that matches your
experiment, then make the declarations it needs.

### Setup A: a random subsample from a pool

You draw the evaluation items at random, without looking at losses, from a
fixed pool. A common example is a random node split on a graph: given the
training set and the trained model, the test nodes are a simple random sample
of the non-training nodes.

*   **Use:** `stats.independent`.
*   **What the interval states:** the mean loss of this model over the pool lies in the interval.
*   **Declarations:** only the moment bound $M$ (Section 4). Correlation
    between items, for example through shared graph neighbourhoods, does not
    matter here, because the only randomness the interval uses is the random
    draw.

### Setup B: a fixed test set, evaluated in full

You evaluate every item of a fixed test set, for example the whole graph over
the last 10 days. Nothing was drawn at random, so coverage must come from
a model of how the losses depend on each other.

*   **Use:** `stats.spacetime` with the range envelope (`stats.range_envelope`).
*   **What the interval states:** the mean *expected* loss of these items, given the design, lies
    in the interval. The design is everything fixed before looking at test
    losses: the evaluated items and period, the graph, the partition, the
    declarations and the trained model.
*   **What it does not state:** a forecast for future periods. Anything that
    happened in the evaluated period, including one-off events, counts as part
    of the design.
*   **Other Setup B entry points:** temporal evaluation (§5), graph or
    cluster evaluation (§6), and space-time evaluation with declared
    $\kappa_s$, $\kappa_t$ or edge bounds $\phi$ (§10) are also Setup B: a
    fixed test set evaluated in full. They differ only in where $\kappa$
    comes from: you declare it directly (§4) instead of deriving it from the
    range declaration below.

### The range declaration (Setup B)

`RangeDeclaration` takes six numbers:

| Field | Meaning | Where it comes from |
| :--- | :--- | :--- |
| `k_hops` ($K$) | Message-passing hops of the model | Model configuration |
| `lookback` ($L$) | Past time steps the model reads | Model configuration |
| `horizon` ($H$) | Steps ahead the model predicts | Model configuration |
| `data_range_s` ($R_s$) | Hops beyond which the data's own noise is independent | Your knowledge of the data |
| `data_range_t` ($R_t$) | Time steps beyond which the data's own noise is independent | Your knowledge of the data |
| `gamma` ($\gamma$) | For every item, the largest total absolute correlation with all items outside the range | Your knowledge of the data; **required, no default** |

The model's reach alone is not enough. Two items whose model inputs do not
overlap can still have correlated losses, because their labels are correlated
through the data (shared weather, incidents, daily rhythm). That is why $R_s$,
$R_t$ and $\gamma$ exist. They are the real assumptions; the data can refute
them but never confirm them.

From these, beyond the budget $\gamma$, two losses can be correlated only if
they are at most $2K + R_s$ hops and $L + H + R_t$ steps apart (the *range*;
a bounding box, possibly larger than needed).

**Time and graph inputs.**

*   Times must be integer counts of the model's time step. Convert timestamps
    first, for example Unix seconds divided by the step length.
*   The library rejects times whose differences share a common factor larger
    than 1. If you evaluate only every few steps, divide the times by that
    factor. Keeping $L$, $H$ and $R_t$ in the original steps is then
    conservative; converting them to the coarser steps (rounding up) is
    tighter.
*   The library cannot detect times in a finer unit with irregular spacing
    (for example, raw seconds with jitter), so check this yourself.
*   The spatial edges must describe the full graph, including nodes without
    test items, and use the same node identifier type as the items.

**The κ bound.** The library bounds the variance inflation factor by

$$\kappa \le 1 + \lambda_{\max}(\text{cut}) + \gamma,$$

where the cut is the set of item pairs within range that lie in different
clusters, and $\lambda_{\max}(\text{cut})$ is the largest eigenvalue of its
0/1 adjacency matrix. With singleton clusters, $\kappa$ is about the number of
items within range of one item (counting the item itself), plus $\gamma$. The
result is Claim RE1 in [`THEORY_RANGE_ENVELOPE.md`](./THEORY_RANGE_ENVELOPE.md).

**How much data you need.** An interval is issued only when the effective
number of clusters $G / \kappa$ exceeds $39 \, (M_c - 1)$ at the 95% level,
where $M_c$ is the cluster-level moment bound. With singletons and the MSE
default $M = 16$ that is $585$. Example: a model with $K = 1$, $L = 3$,
$H = 1$ on data with $R_s = 2$, $R_t = 2$ reaches $4$ hops and $6$ steps; on a
ring graph each item then has $9 \times 13 = 117$ items within range
(including itself), so you need more than about $585 \times (117 + \gamma)$,
roughly $70{,}000$, test items. Deep models with long lookbacks need much more
data. If the library refuses, the honest options are more test data, a metric
with a smaller $M$ (the default is 4 for MAE and accuracy), or Setup A.

### Requirements (Setup B)

1.  Fix the declarations, the candidate partitions and the confidence level
    before looking at test losses. Choosing among the declared candidates is
    already accounted for; any other choice made after seeing losses is not.
2.  The trained model must depend only on training data. For example, no
    scalers or normalizers fitted on test data.
3.  Leave a gap of more than $L + H + R_t$ time steps between the last
    training item time and the first evaluation item time. Without the gap,
    or when $\gamma > 0$, the ranges and $\gamma$ must hold for the losses
    given the trained model.
4.  Missing items must be missing for reasons unrelated to the losses. For
    example, sensors that drop out exactly when errors are large void
    coverage.

### Global dependence

Dependence that reaches every item comes in two forms, with opposite
consequences:

*   **Diffuse:** each pair of items is only weakly correlated, for example a
    global attention layer with spread-out weights. $\gamma$ stays bounded as
    the data grow, so more data helps.
*   **Common shock:** one random event moves every item together, for example
    weather affecting all sensors. $\gamma$ then grows with the number of
    items, and no amount of data makes the interval shrink: the effective
    sample size stays below $1/\rho_0$, where $\rho_0$ is the shared
    correlation. The library refuses, which is the correct answer. A claim is
    still possible when the shock is part of the design, that is, when the
    claim is about the evaluated period only and the shock is not random
    within it.

### Safe versus dangerous overrides

The left column lists changes after which the coverage bound still holds; the
right column lists changes that can silently void it:

| Safe: the coverage bound still holds | Dangerous: can silently void coverage |
| :--- | :--- |
| Larger $K$, $L$, $H$, $R_s$, $R_t$ or $\gamma$ (wider or refused) | Smaller values than the truth |
| Larger $M$ or $M_c$ (wider or refused) | Smaller $M$ or $M_c$ than the truth |
| More candidate partitions, a higher confidence level | Choosing the partition or level after seeing test losses |
| Singleton clusters, including with missing items | Treating a passed refutation check as confirmation of declarations |
| | Clusters with more than one item: they need an extra premise, that each cluster's variance is at least the sum of its items' variances (true when correlations inside a cluster are non-negative), and it must hold on the observed items |

### Not supported

*   **Rolling-origin evaluation** (train on days $1..n$, test on the next $k$
    days, for many $n$) and **grouped holdout** (holding out whole
    communities or weeks): expanding-window training and models with
    parameters shared across all nodes give no finite range, so these need
    additional assumptions.
*   **K-fold cross-validation:** the folds share training data, and no
    unbiased estimator of the variance of the cross-validation estimate
    exists (Bengio and Grandvalet, 2004). The variance decomposition in that
    paper and the corrected resampled t-test (Nadeau and Bengio, 2003) are
    useful heuristics, but they rest on approximations, not finite-sample
    coverage bounds.
*   **Reusing early-stopping or model-selection data** for the interval: the
    model was chosen on those losses, so the interval would be biased.
*   **Very large spatial graphs:** both `range_kappa` methods build a dense
    nodes-by-nodes matrix.

### Minimal example (Setup B)

Two methods compute the κ bound:

*   `method="kronecker"` (the default) is fast and needs partitions of the
    form node group × time window.
*   `method="direct"` accepts any partition but refuses above an edge limit.

For MSE, `spacetime_interval` takes raw residuals, not squared losses.
`range_candidates` always adds singleton clusters (target size 1, window
length 1) to the candidates you request. `choose_partition` reports the route
`"range"` when it receives a κ function from `range_kappa_fn`.

```python
import numpy as np
from dgf.src.stats import range_envelope
from dgf.src.stats import spacetime

# 1. Spatio-temporal evaluation grid: 20 nodes observed over 100 time steps (n = 2000).
num_nodes = 20
num_times = 100
node_grid, time_grid = np.meshgrid(
    np.arange(num_nodes), np.arange(num_times), indexing="ij"
)
node_ids = node_grid.ravel()
times = time_grid.ravel()
index = spacetime.spacetime_items(node_ids, times)

# 2. Spatial network: line graph with 19 edges.
spatial_edges = np.column_stack(
    [np.arange(num_nodes - 1), np.arange(1, num_nodes)]
)

# 3. Declare model receptive field and data noise ranges:
# Model: 1 graph hop, 1 lookback step, 0 horizon steps.
# Data: 0 spatial noise hops, 0 temporal noise steps, gamma=0.1 tail budget.
# Yields: spatial_range = 2(1) + 0 = 2 hops, temporal_range = 1 + 0 + 0 = 1 step.
declaration = range_envelope.RangeDeclaration(
    k_hops=1,
    lookback=1,
    horizon=0,
    data_range_s=0,
    data_range_t=0,
    gamma=0.1,
)

# 4. Generate candidate partitions matching the declared dependence ranges.
candidates = range_envelope.range_candidates(
    index,
    spatial_edges,
    declaration,
    target_sizes=[2],
    window_multipliers=[1, 2],
)

# 5. Build certified range kappa function via the matrix-free Kronecker operator.
kappa_fn = range_envelope.range_kappa_fn(
    index, spatial_edges, declaration, method="kronecker"
)

# 6. Select the optimal partition maximizing size-adjusted effective sample size N_eff.
# Declaring m_item=3.5 (squared Gaussian residuals have relative second moment 3).
# ensures Cantelli issuance condition nu = 2000 / 14.79 = 135.2 > n_A = (3.5-1)*39 = 97.5.
choice = spacetime.choose_partition(
    candidates,
    kappa_fn=kappa_fn,
    m_item=3.5,
    is_complete_grid=index.is_complete_grid,
)

# 7. Evaluate validation residuals and compute the certified interval.
rng = np.random.default_rng(0)
residuals = rng.normal(loc=0.0, scale=0.5, size=index.n_items)

result = spacetime.spacetime_interval(
    residuals,
    index,
    choice,
    metric="mse",
    m_item=3.5,
)

# Output summary:
# winner: product_ts1_wl1, route: range, G: 2000, kappa: 14.7915
# interval: [0.1354, 1.6594], status: unrefuted
print(f"Winner partition: {choice.winner.name}")
print(f"Certification route: {choice.route}")
print(f"Clusters G: {choice.winner.G}, Certified kappa: {choice.winner_kappa:.4f}")
print(f"Interval: [{result.low:.4f}, {result.high:.4f}]")
print(f"Status: {result.status.value}")
```

The winner `product_ts1_wl1` is the singleton partition: each of the 2000
items is its own cluster.

## 4. The Two Declarations

Every dependent entry point works in two steps: you declare two bounds, and
the package tries to refute them.

-   **$\kappa$, default 1.0:** how much correlation between clusters to allow.
    -   $\kappa = 1$ asserts that clusters are uncorrelated. The $\kappa = 1$
        check can only test adjacent clusters: consecutive windows for
        temporal data, clusters joined by at least one edge for graph data.
    -   If you declare $\kappa = 1$, the $\kappa = 1$ check runs. If you
        declare $\kappa > 1$, the $\kappa > 1$ check runs instead (§8).
    -   Larger clusters make $\kappa = 1$ more plausible, because more of the
        correlation stays inside a cluster.
-   **$M_c$ (or the item-level $M$):** how heavy-tailed the cluster totals
    may be.
    -   The default is valid for nearly all real residuals, but it gives wide
        intervals.
    -   A smaller declared value gives a narrower interval, but only if it
        still holds. If you declare exactly the true value, the check refutes
        it about half the time, so declare it with a margin.
    -   Pass either `m_item` (item level; converted to $M_c$ from the actual
        cluster sizes as in §2) or `m` (cluster level), not both.

The interval is certified **under** these declarations. The checks (§7) can
contradict a declaration, but they can never confirm one.

### 4.1 Declaring M from archive data

The default $M$ is wide. For R² at the default, the interval is close to
uninformative below about 5000 clusters. In one simulation with true
R² = 0.8 and 6000 clusters, the median interval width, on the R² scale, was
between 0.83 and 0.89 at the default $M = 16$. With $M$ declared at its true
value 3, the median interval was about [0.69, 0.89].

To declare a tighter $M$ with a justification:

1.  Take residuals from **archive data**, such as an earlier model or an
    earlier period, that do not overlap the evaluated sample.
2.  Run `declare.estimate_m` on the archive summands (`|e|` for MAE, `e²`
    for MSE and RMSE), or `declare.estimate_m_r2` for R². Pass
    `cluster_ids` if the archive is correlated. The result reports
    `m_upper`, an upper bootstrap quantile of $M$, and tail diagnostics such
    as `top1pct_share`, the share of $\sum s^2$ held by the top 1% of items.
3.  Declare `m_upper`, or more if the diagnostics show that a few items
    dominate the tail.
4.  Save the choice with `declare.Declaration.from_estimate(...)` and
    `declare.save_declarations(path, {...})`. Later runs load it with
    `declare.load_declarations(path)` and pass `**declaration.kwargs()` to
    any entry point. This passes `m_item` for every metric, including R²;
    `kappa` only if the declaration sets one; and, for R², `m_labels` and
    `kappa_labels` if set. For `spacetime_interval`, the `m_item` must equal
    the one passed to `choose_partition`, and a `kappa` must be at least
    `choice.winner_kappa` (§10).

The saved record keeps the source, the method, the diagnostics and a
fingerprint of the archive, so that it can be reviewed and reused.

Because an earlier model or period may have heavier tails than the archive, add
a margin—for example `1.25 * m_upper`—whenever `top1pct_share` is large.

> [!IMPORTANT]
> The coverage bound holds for the **declared** $M$, not for the estimate.
> Bootstrap quantiles understate $M$ under heavy tails. Reusing a
> declaration on a new model or a new period is a judgement that you make
> and record.

### 4.2 Computing κ from the graph topology

Suppose you can bound how strongly the losses of two **adjacent** nodes
correlate, for example from earlier models. Call that bound `phi`.

*   `graph.topological_kappa(num_nodes, edges, cluster_ids, phi)` then
    returns a κ that is valid for the current partition (Lemma T′ in
    [`THEORY.md`](./THEORY.md)).
*   The κ equals 1 plus the largest eigenvalue of the matrix that holds `phi`
    for every edge between two different clusters and 0 elsewhere. The
    package computes a certified upper bound on that eigenvalue.
*   Hubs cost less than their degree suggests. A node with `k` edges to other
    clusters, whose neighbours have at most `d` such edges each, adds at most
    `phi·√(k·d)`, not `phi·k`.
*   Non-adjacent nodes are assumed uncorrelated. If that is not true, add the
    longer-range pairs to `edges`.
*   As for $M$, `phi` must come from outside the evaluated sample. For
    squared errors, `phi` bounds the correlation of the squared errors, not
    of the residuals.

## 5. Temporal Data

**What the interval means:** the average expected loss over the evaluated
items in the validation period. It equals the expected loss on future items
only if the process stays stationary.

> [!WARNING]
>
> Distribution drift is not covered. If the data-generating process shifts,
> trends, or changes regime between evaluation and production, coverage does
> not transfer to future items. The drift diagnostic (§7) can
> flag some drift within the validation period, but it cannot certify that
> the future matches the past.

**Choosing `num_windows` ($G$).** The window count trades correlation
resolution against sample size:

-   **Window length:** fewer, longer windows span more time, which makes
    $\kappa = 1$ more plausible.
-   **Minimum count:** the effective window count $\nu = G / \kappa$ must
    exceed $n_A = (M_c - 1)(1 - \alpha/2) / (\alpha/2)$. Otherwise the
    package refuses with `ASSUMPTION_REQUIRED` and reports the window count
    it needs. For example, MSE at the defaults with equal-size windows has
    $M_c = M = 16$, and at $\alpha = 0.05$ it needs more than $n_A = 585$
    effective windows.
-   **Uneven windows:** `count_h` measures how uneven the window item counts
    are. If it exceeds 1.1, the $\kappa$ check reports `UNTESTABLE`. The
    interval is still issued.
-   **Check limits:** the $\kappa = 1$ check needs at least 5 windows, and
    the $\kappa > 1$ check needs at least 30.

**Time buckets.** `temporal_interval_from_buckets(counts, totals, ...)`
takes per-bucket item counts and loss sums in time order, for data that is
stored only as aggregates.

*   Buckets are grouped into windows by the position of each bucket's middle
    item (Lemma W in [`THEORY.md`](./THEORY.md)). Each window then holds
    `m̄ ± b_max` items, where `m̄` is the mean window size and `b_max` is
    the largest bucket count.
*   Inputs with `m̄ < b_max` are rejected. Use finer buckets or fewer
    windows.
*   The κ = 1 check is a self-normalised test along the sequence of windows.
    It needs no guard on uneven counts, so trending or bursty traffic is
    fine.
    *   It needs at least 31 windows.
    *   Its measured false-refutation rate was 4.5–5.7% at a nominal 5%.
    *   Its power was 0.15 at 31 windows, 0.73 at 100 and 0.99 at 300,
        against AR(1) correlation with a correlation length of about one
        window.

**R².** Pass `metric="r2"` with `data=(y_true, y_pred)`. As for the other
metrics, `m` is the cluster-level $M_c$ for the squared errors, used
unchanged, and `m_item` is the item-level $M$, converted to $M_c$ from the
cluster sizes (§2, §9). `m_labels` and `kappa_labels` optionally declare the
label side separately.
The estimand is `1 − MSE/Var(y)`, where `Var(y)` counts both the label noise
and the spread of the per-item mean labels. At the default $M = 16$ and
$\alpha = 0.05$, R² needs more clusters than MSE:

*   an interval is issued from about 1200 effective clusters, against 585
    for MSE;
*   the upper endpoint stays at 1 until about 2700 effective clusters.

R² depends on three sequences: squared errors, squared label differences and
labels. Each is checked for κ at the checks' own 5% level. `kappa_check` is
refuted if any one of the three is refuted, so its false-refutation rate can
be up to 15%. The R² interval is Claim 7 in [`THEORY.md`](./THEORY.md).

**Pre-aggregated data (shards).** When the data are spread over machines,
build one shard per machine and merge them before computing the interval:

```python
from dgf.src.stats import cluster_sketch

# On each machine: residuals and window indices of the local items.
shard_a = cluster_sketch.PartialClusterShard.from_data(res_a, window_ids_a, "mse")
shard_b = cluster_sketch.PartialClusterShard.from_data(res_b, window_ids_b, "mse")

merged = shard_a.merge(shard_b)  # or shard_a + shard_b
result = temporal.temporal_interval_from_shard(merged, metric="mse")
```

*   A shard stores, per cluster, the cluster ID, the partial loss total and
    the item count. A cluster may be split over several shards; the merge
    adds the partial totals.
*   For temporal data, the cluster IDs must be the window indices 0 to
    G − 1 in time order.
*   For R², use `cluster_sketch.R2ClusterShard.from_data(y_true, y_pred,
    cluster_ids)`, which stores the totals of squared errors, labels and
    squared labels.
*   For graphs, pass the merged shard and `cluster_edges` to
    `graph.graph_interval_from_shard`.

## 6. Graph Data

**What the interval means:** the average expected loss over the evaluated
nodes. Nothing is claimed about nodes outside that set.

The graph entry point needs these inputs besides the losses:

1.  **`cluster_ids`**: one cluster label per node. Use the provided
    partitioner, `partitioners.partition_graph(num_nodes, edges,
    target_cluster_size)`. It ensures balanced sizes: no cluster is larger
    than $\lceil 1.5\,t \rceil$, and at most one cluster is smaller than
    $\lceil t/2 \rceil$, where $t$ is `target_cluster_size`.
2.  **`cluster_edges`**: the pairs of clusters joined by at least one edge.
    Compute it with `graph.cluster_adjacency(num_nodes, edges, cluster_ids)`.

Both must be computed **from the graph only**, never from losses, residuals
or labels. The interval and the checks assume this.

**Choosing `target_cluster_size`.** Larger clusters keep more of the
correlation along edges inside a cluster, which makes $\kappa = 1$ more
plausible. They also leave fewer clusters $G$, and the minimum count of §5
applies to $\nu = G / \kappa$. The simulations in
[`EVIDENCE.md`](./EVIDENCE.md) used 20–200 nodes per cluster on graphs with
20 000 nodes.

**The $\kappa = 1$ edge check** tests whether standardised cluster totals
correlate along cluster edges.

*   It needs at least 30 distinct cluster edges.
*   An **edge guard** reports `UNTESTABLE` when a few edges dominate the test
    statistic, measured by `edge_n_eff < 10` (§7). Graphs with hubs trigger
    it often.
*   `hub_ratio` is reported as a warning sign computed from the graph alone.
    Values near 1 mean that the test's normal approximation is poor.

## 7. Reading the Result

### Statuses and what the interval promises

Every result carries a `status` (`result.status`, or `result.interval.status` on
`GraphResult`) from the `Status` enum of `stats.independent`, and the dependent
entry points (`temporal`, `graph`, `spacetime`) also run a dependence check
(`result.kappa_check`, with outcome `NOT_REFUTED`, `REFUTED`, or `UNTESTABLE`
from `Refutation`, combined with `status` in `result.refuted`):

*   **`UNREFUTED`** (and `result.refuted == False` on dependent routes): an
    interval $[L, U]$ is returned and the checks do not contradict the declared
    $M$ (or $M_c$ and $\kappa$).
*   **`TAIL_UNRESOLVED`** (or `result.refuted == True` on dependent routes): an
    interval $[L, U]$ is still returned, but $\hat M$ exceeds the declared bound
    (or `kappa_check.outcome` is `REFUTED`). The data argue against the
    declaration.
*   **`ASSUMPTION_REQUIRED`**: no interval is returned (`low = high = nan`,
    `level = None`), because the effective sample size is below the minimum for
    $(M, \text{level})$ or the input contains non-finite values.

#### Coverage guarantee

**When your declared assumptions hold, the interval misses the true value
$\theta$ with probability at most $\alpha = 1 - \text{level}$.** Because a joint
event cannot be more probable than either of its parts, this also bounds the
joint rate of an unrefuted miss:

$$\Pr\bigl(\theta \notin [L, U] \;\wedge\; \text{status} = \texttt{UNREFUTED}\bigr) \;\le\; \Pr\bigl(\theta \notin [L, U]\bigr) \;\le\; \alpha$$

(and $\Pr(\theta \notin [L, U] \wedge \neg\texttt{refuted}) \le \alpha$ on
dependent routes; Claims 1, 1′, 3, 3′ in
[`THEORY_INDEPENDENT.md`](./THEORY_INDEPENDENT.md) and Claims 4, 5, 7 in
[`THEORY.md`](./THEORY.md)). Issuance depends only on the sample size, design,
and declarations—never on the losses—and the checks never alter $[L, U]$.

> [!WARNING]
> **The status is a verdict on the declaration, not a filter on the interval.**
> Do not discard `TAIL_UNRESOLVED` (or `refuted`) runs and read `level` as
> applying conditionally to `UNREFUTED` runs alone. Among `UNREFUTED` results,
> the miss rate can reach $\alpha / \Pr(\texttt{UNREFUTED})$ (or
> $\alpha / \Pr(\neg\texttt{refuted})$): near $\alpha$ when $M$ has headroom
> ($\Pr(\texttt{UNREFUTED}) \approx 1$), and up to $2\alpha$ when the true $M$
> equals the declared $M$ ($\Pr(\texttt{UNREFUTED}) \approx 1/2$).

#### Why $M$ must be declared, and what the defaults cover

No finite-sample interval can bound a mean without a declared tail bound $M$: a
law can agree with a Gaussian on $99.99\%$ of draws—so $n = 1000$ looks Gaussian
($\hat M \approx 3$, `UNREFUTED`) with probability $0.9999^{1000} \approx 90\%$—yet
place a $0.01\%$ spike far enough out to move the true mean above $[L, U]$
(Bahadur and Savage, 1956; Proposition REU1(2) in
[`THEORY_RANGE_ENVELOPE.md`](./THEORY_RANGE_ENVELOPE.md)). A wrong declared $M$
is a violated assumption; the checks catch many heavy tails as $n$ grows, but
cannot foresee an unseen spike.

*   **Default $M = 16$ (`mse`, `rmse`, `r2`):** for MSE and RMSE,
    $M = \mathbb{E}[e^4]/(\mathbb{E}[e^2])^2$, which equals $\mathrm{Kurt}(e)$
    when $\mathbb{E}[e] = 0$ (Gaussian: $3$; Laplace: $6$; Student-$t(5)$: $9$)
    and satisfies $M < \mathrm{Kurt}(e) + 2$ for biased errors, covering every
    error law with $\mathrm{Kurt}(e) \le 14$.
*   **Default $M = 4$ (`mae`, correlated `accuracy`):** for MAE,
    $M = \mathbb{E}[e^2]/(\mathbb{E}|e|)^2 \le M_{\text{MSE}}$ (zero-mean
    errors; Gaussian: $\pi/2 \approx 1.57$; Laplace: $2$; Student-$t(5)$:
    $3\pi^2/16 \approx 1.85$). For correlated
    accuracy, $M = 1/p$ covers $p \ge 0.25$ (independent accuracy needs no $M$).

The fields common to the temporal and graph result types:

| Field | Meaning |
| :--- | :--- |
| endpoints and status | Temporal: `result.low`, `result.high`, `result.status`. Graph: `result.interval.low`, `result.interval.high`, `result.interval.status`. |
| `kappa_check` | The $\kappa$ check: `outcome` (a `Refutation`) and its statistics. |
| `refuted` | `True` if either declaration was contradicted: `status` is `TAIL_UNRESOLVED` or `kappa_check.outcome` is `REFUTED`. |
| `message` | A plain-text explanation with remedies. |
| `m_declared`, `m_observed` | The declared $M_c$ and the observed relative second moment of the cluster totals. Temporal: on the result. Graph: under `result.interval`. |
| `r2_detail` | For R² only: the separate bounds and checks for the error side and the label side. |

Fields of the temporal result only:

| Field | Meaning |
| :--- | :--- |
| `drift` | A diagnostic that computes separate intervals on the first and second halves of the windows. Its `outcome` is `WARNING` when the two intervals are disjoint. |
| `count_h` | How uneven the window item counts are (§5). |
| `effective_n` | $\nu = G / \kappa$. |

Fields of the graph result only:

| Field | Meaning |
| :--- | :--- |
| `kappa_w_hat` | A point estimate of $\kappa$ from the correlation of cluster totals along cluster edges. Guidance only (§12). |
| `edge_n_eff` | The effective number of terms in the edge test statistic. Below 10, the edge guard reports `UNTESTABLE`. |
| `hub_ratio` | A measure of how much one direction dominates the cluster graph, between 0 and 1 (§6). |
| `batch_cut_fraction` | For $\kappa > 1$: the fraction of cluster edges that fall between two super-batches (§8). |

The space-time result adds further fields; §10 lists them.

### What to Do

-   **`refuted` is `True`:** the data contradict a declaration, so the
    interval is **not** certified. Declare a larger $\kappa$ or $M_c$, use
    larger clusters, or collect more data.
-   **`refuted` is `False`:** the data did not contradict the declarations.
    That does not establish that the declarations hold.
-   **`status` is `ASSUMPTION_REQUIRED`:** no interval was issued. Too few
    effective clusters is the usual cause. The message states how many
    clusters, or what level, would suffice.

## 8. What the Checks Cannot Detect

Each check has blind spots:

-   **Temporal $\kappa = 1$ check** (a one-sided test of the lag-1
    correlation of window totals, using the Fisher z-transform): it cannot
    detect correlation at lag 2 or longer, or non-linear dependence.
-   **Temporal $\kappa > 1$ check** (`check_kappa`: a lower confidence bound
    on $\kappa$ from the variance of 30 batch means of consecutive window
    totals): it is weak when the batches are shorter than the correlation
    length.
-   **Graph $\kappa > 1$ check** (`check_kappa_batches`: a lower confidence
    bound on $\kappa$ from the variance of about 30 super-batch totals, where
    a super-batch is a group of neighbouring clusters): it is weak when
    super-batches are shorter than the correlation length, and when most
    cluster edges fall between super-batches (`batch_cut_fraction` near 1).
    *   **Known issue:** `check_kappa_batches` can report "refuted" too
        often when batch variances differ strongly. In simulation with 20
        batches and a 100:1 variance ratio, the false-alarm rate was about
        0.19 instead of 0.05. This affects only the check, never the
        interval.
-   **Graph $\kappa = 1$ edge check:**
    *   It sees only adjacent cluster pairs. Correlation along longer paths
        is not tested directly.
    *   It is one-sided, so negative or oscillating correlation does not
        trigger it.
    *   Its false-refutation rate relies on a large-sample approximation,
        which is poor on hub-dominated graphs, so the rate there can exceed
        the nominal 5%. Claim 6 in [`THEORY.md`](./THEORY.md) describes when
        the approximation fails. The edge guard removes the worst cases but
        not all of them. In the coverage benchmark, hub graphs with
        uncorrelated clusters refuted at rates between 0.020 and 0.049
        ([`EVIDENCE.md`](./EVIDENCE.md)).
-   **None of the checks changes the interval.** They only flag. A refuted
    result still carries the endpoints computed under your declarations.

## 9. Independent Evaluation (`stats.independent`)

The `stats.independent` module gives finite-sample confidence intervals for MAE,
MSE, RMSE, accuracy and R² when the evaluation items are **independent
draws**, such as a test set sampled uniformly at random (Setup A in §3).

*   **Capabilities:** provides finite-sample confidence intervals for
    independent samples with streaming sketches (`IndependentSketch`, `R2Sketch`), JAX
    accumulators, distributed shard aggregation, and Clopper–Pearson intervals.
*   **Theory:** the theoretical derivation and finite-sample coverage bounds
    for the independent estimator are documented in
    [`THEORY_INDEPENDENT.md`](./THEORY_INDEPENDENT.md).

> [!TIP]
> If your data are correlated (for example adjacent graph nodes or
> sequential timestamps), `stats.independent` underestimates the uncertainty
> and produces intervals that are too narrow. Use `temporal`, `graph`, or
> `spacetime` instead; they cluster items and certify coverage against
> dependence between clusters.

### Minimal runnable example

```python
import numpy as np
from dgf.src.stats import independent

# 1. Validation residuals (prediction - label) from 1,000 independent test items.
rng = np.random.default_rng(0)
residuals = rng.normal(loc=0.0, scale=1.0, size=1000)

# 2. Compute finite-sample confidence interval for mean-squared error (MSE).
result = independent.confidence_interval(
    residuals,
    metric="mse",
    level=0.95,
)

print(
    f"MSE interval: [{result.low:.4f}, {result.high:.4f}], status:"
    f" {result.status.value}"
)
# Output: MSE interval: [0.5419, 1.4357], status: unrefuted
```

### Parameters and status

The parameters and outputs of `independent.confidence_interval` are:

*   **$\text{level}$ (`level`):** the nominal coverage probability in $(0, 1)$
    requested by the caller (default `0.95`). Whenever an interval is
    returned, the true population metric $\theta$ satisfies
    $\Pr(\theta \in [\text{low}, \text{high}]) \ge \text{level}$ under the
    declared tail bound.
*   **$\alpha$:** the allowed miss probability, $\alpha = 1 - \text{level}$
    (for example, $\alpha = 0.05$ when $\text{level} = 0.95$). The coverage
    bound splits this budget equally across tails ($\alpha / 2$ per
    side).
*   **$M$ (`m`):** the declared relative second-moment bound on the per-item
    loss summand $s_i \ge 0$:
    $$\mathbb{E}[s^2] \le M \cdot (\mathbb{E}[s])^2$$
    $M$ measures the tail weight of the loss summand (for zero-mean errors
    and MSE or RMSE, $M = \mathrm{Kurt}(e)$):
    -   If errors are light-tailed or sub-Gaussian, $M$ is small
        ($1 \le M \le 4$; for example $\pi/2 \approx 1.57$ for Gaussian MAE
        and $3$ for zero-mean Gaussian MSE).
    -   If errors have heavy tails (where rare outliers carry substantial
        loss), $M$ is larger ($10 \le M \le 50$).
    -   Declaring a larger $M$ accommodates heavier tails without sacrificing
        finite-sample validity, at the expense of a wider interval.
*   **$n_{\mathrm{eff}}$ (`effective_n`):** the effective sample size. When
    supplied, it replaces the strict independence assumption with a relaxed
    variance bound on the sample mean $\bar{s}$ via
    $\operatorname{Var}(\bar{s}) \le \frac{M - 1}{n_{\mathrm{eff}}} (\mathbb{E}[s])^2$.
*   **$\hat{M}$ (`m_observed`):** the empirical relative second moment
    computed from the sample:
    $$\hat{M} = \frac{n \sum_{i=1}^n s_i^2}{\left(\sum_{i=1}^n s_i\right)^2}$$
*   **`status` (`Result.status`):** `UNREFUTED`, `TAIL_UNRESOLVED`, or
    `ASSUMPTION_REQUIRED`; see
    [Statuses and what the interval promises](#statuses-and-what-the-interval-promises)
    in §7 for the meaning of each value and why you must not filter on
    `UNREFUTED` and read `level` conditionally.

### Metric defaults (`DEFAULT_M`)

When `m` is omitted, `independent.confidence_interval` uses these
conservative defaults from `independent/metrics.py`:

| Metric Name | Summand $s_i$ | Default $M$ (`DEFAULT_M`) | Description / Support |
| :--- | :--- | :---: | :--- |
| `"mae"` | $\lvert e_i \rvert$ | **4.0** | Mean absolute error |
| `"mse"` | $e_i^2$ | **16.0** | Mean squared error |
| `"rmse"` | $e_i^2$ | **16.0** | Root mean squared error (derived from MSE interval) |
| `"accuracy"` | $\mathbb{I}\{\text{correct}\}$ | **4.0** | Classification accuracy; exact Clopper–Pearson is used when $n_{\mathrm{eff}}$ is not specified |
| `"r2"` | $(e_i^2, b_i)$ | **16.0** | Coefficient of determination (paired tuple `data=(y_true, y_pred)`) |

`average_precision` is intentionally not supported, because no non-vacuous
distribution-free finite-sample bound exists below $n \approx 10^5$.

### How `m` differs for R² between `stats.independent` and the dependent modules

The parameter `m` means different things in the independent and the
dependent entry points when you evaluate $R^2$:

*   **In `stats.independent.confidence_interval` (the independent convention):**
    -   `m` is the **item-level** relative variance bound $M$ on squared
        errors $s_{a, i} = (y_i - \hat{y}_i)^2$ (default `16.0`).
    -   There is no separate `m_item` parameter.
*   **In the dependent entry points (`graph.graph_interval`,
    `temporal.temporal_interval`, `spacetime.spacetime_interval`):**
    -   `m` is the **cluster-level** bound $M_c$ on aggregated cluster totals
        $S_c$.
    -   `m_item` is the **item-level** bound $M$ on individual item losses
        (default `16.0` for MSE and $R^2$). When `m` is omitted, the
        cluster-level bound $M_c$ is derived from `m_item` with Lemma I′
        (`default_m_c`) from the cluster size distribution.

## 10. Space-Time Evaluation (`stats.spacetime`)

### What the interval means

The interval covers the average design-conditional expected loss over the
observed spatio-temporal items $\mathcal{O}$:
$\theta(D) = \frac{1}{n}\sum_{(i,t)\in\mathcal{O}} \mathbb{E}[s_{i,t} \mid D]$,
where $D$ is the design (§9.0 and §9.1.2 part 4 of
[`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md)).

*   On incomplete grids, nothing is claimed about missing items unless
    dropouts are unconfounded with losses.
*   Conditioning on the design $D$ requires that the observation mask
    $\mathcal{O}$ and the spatio-temporal graph edges are strictly exogenous
    to the evaluation losses (§9.1.2 part 4 and §9.5.1 of
    [`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md)).
*   The interval equals the expected loss at future time steps only if the
    process is stationary; the drift diagnostic is a partial check of that.

Spatio-temporal evaluation data (such as sensor networks, traffic grids, and
spatial time series) are correlated along both spatial graph edges and time.
Clustering nodes alone ignores temporal persistence; windowing time alone
ignores the spatial graph. The `stats.spacetime` module partitions observations
into space-time product blocks that account for both axes at once. The theory
and arguments are in [`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md).

The pipeline has four steps:

1.  Index the items with `spacetime_items(node_ids, times)`.
2.  Build candidate partitions with `candidate_partitions` (or
    `range_envelope.range_candidates`, §3).
3.  Choose one with `choose_partition`, which certifies κ for each candidate
    through one of the routes listed under "Route certification" below.
4.  Compute the interval with `spacetime_interval`.

### Consistency checks in `spacetime_interval`

`spacetime_interval` raises `ValueError` when its inputs disagree with the
partition choice:

*   **`m_item`:** if you omit `m_item`, it uses `choice.m_item`, the value
    passed to `choose_partition`. If you pass a different `m_item`, it
    raises.
*   **`kappa`:** if you omit `kappa`, it uses `choice.winner_kappa`. An
    explicit `kappa` smaller than `choice.winner_kappa` raises; overrides may
    only be more conservative.
*   **Data length:** the data must have `index.n_items` entries. For R², both
    `y_true` and `y_pred` are checked.
*   **Data form:** for `metric="r2"`, `data` must be the tuple
    `(y_true, y_pred)`; for every other metric, tuples are rejected.
*   **R² label declaration:** for `metric="r2"`, you must pass
    `kappa_labels`. The space-time κ bounds the dependence of the losses
    only; the label totals need their own declared variance inflation
    factor.

### Minimal runnable example

```python
import numpy as np
from dgf.src.stats import spacetime

# 1. Spatio-temporal evaluation grid: 20 nodes observed over 100 time steps (n = 2000).
num_nodes = 20
num_times = 100
node_grid, time_grid = np.meshgrid(
    np.arange(num_nodes), np.arange(num_times), indexing="ij"
)
node_ids = node_grid.ravel()
times = time_grid.ravel()
index = spacetime.spacetime_items(node_ids, times)

# 2. Spatial topology: a simple line graph (19 edges).
spatial_edges = np.column_stack(
    [np.arange(num_nodes - 1), np.arange(1, num_nodes)]
)

# 3. Candidate partitions: target community size 2, window length 1.
# Geometry: 20 nodes with target community size 2 yield G_s = 11 spatial communities
# under the size-constrained partitioner (9 pairs of nodes and 2 singleton nodes).
# 100 time steps with window length 1 yield G_t = 100 temporal windows.
# The product candidate groups items into G = 11 * 100 = 1100 space-time clusters.
candidates = spacetime.candidate_partitions(
    index,
    spatial_edges,
    target_sizes=[2],
    window_lengths=[1],
)

# 4. Spatio-temporal edges and partition selection.
edges, phi = spacetime.spacetime_edges(
    index,
    spatial_edges,
    max_lag=1,
    phi_spatial=0.0,
    phi_temporal=[0.0],
)
kappa_fn = spacetime.separable_kappa(kappa_s=1.0, kappa_t=1.0)

# m_item=16.0 is the documented default for MSE-type losses.
choice = spacetime.choose_partition(
    candidates,
    kappa_fn=kappa_fn,
    m_item=16.0,
    is_complete_grid=index.is_complete_grid,
    edges=edges,
)

# 5. Evaluate validation residuals.
rng = np.random.default_rng(0)
residuals = rng.normal(loc=0.0, scale=1.0, size=index.n_items)

# spacetime_interval uses the m_item recorded in choice (or checks consistency if passed).
result = spacetime.spacetime_interval(
    residuals,
    index,
    choice,
    metric="mse",
)

print(
    f"Winner: {choice.winner.name}, route: {choice.route}, G: {choice.winner.G}"
)
print(
    f"Interval: [{result.low:.4f}, {result.high:.4f}], status:"
    f" {result.status.value}"
)
# Expected output:
# Winner: product_ts2_wl1, route: separable, G: 1100
# Interval: [0.5665, 4.3000], status: unrefuted
```

### Incomplete grids and the topological fallback

In real sensor logs and traffic networks, dropouts, sensor failures, and
irregular reporting mean that observations do not form a complete Cartesian
grid: $n < N_{\mathrm{nodes}} \times T_{\mathrm{times}}$, indicated by
`index.is_complete_grid is False`.

On incomplete grids, the separable product $\kappa_s \kappa_t$ is unproved
and can fail. Missing items can shrink within-cluster sample counts much
faster than the total variance, which inflates the true variance inflation
factor (Propositions S1B and S1C, §9.1.2 of
[`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md)). The library therefore
behaves as follows:

-   If `choose_partition` is called on an incomplete grid with
    `separable_kappa` and `fallback_kappa_fn=None`, it raises `ValueError`.
-   When `fallback_kappa_fn=topological_kappa_fn(index.n_items, edges, phi)`
    is provided, `choose_partition` routes to `route="fallback_topological"`
    and certifies each candidate with Claim S2B.

**Why monotonicity holds only for topological $\kappa$.** Let
$\Phi^{\mathrm{cut}}$ be the matrix that holds the declared correlation bound
$\phi$ for each item edge between two different clusters and 0 elsewhere.
Topological $\kappa$ is computed from $1 + \lambda_{\max}(\Phi^{\mathrm{cut}})$.
When items are omitted, the observed cut matrix
$\Phi_{\mathcal{O}}^{\mathrm{cut}}$ is a principal submatrix of the full cut
matrix $\Phi^{\mathrm{cut}}$. By Perron–Frobenius monotonicity for
non-negative matrices, the largest eigenvalue of a principal submatrix cannot
exceed that of the original matrix
($\lambda_{\max}(\Phi_{\mathcal{O}}^{\mathrm{cut}}) \le \lambda_{\max}(\Phi^{\mathrm{cut}})$).
Topological bounds therefore need no mask-inflation factor
($\rho_{\mathrm{mask}} = 1$) on incomplete grids, provided that:

*   the observation mask $\mathcal{O}$ is exogenous to the losses (Lemma R in
    [`THEORY.md`](./THEORY.md); its condition on the mask is §9.1.2 part 4 of
    [`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md)); and
*   premises `(T1)` and `(T2)` hold on $\mathcal{O}$ (Claim S2B(4), §9.2.2
    of [`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md)). `(T1)`: for two items
    $i, j$ in different clusters, $|\operatorname{Cov}(s_i, s_j)| \le
    \phi_{ij} \sigma_i \sigma_j$, where $\sigma_i$ is the standard deviation
    of $s_i$. `(T2)`: $\sum_c \operatorname{Var}(S_c) \ge \sum_i
    \operatorname{Var}(s_i)$.

Separable $\kappa_s \kappa_t$ lacks this coverage bound because missing items break
the tensor-product denominator structure.

Here is a runnable incomplete-grid example, continuing the example above:

```python
# Create an incomplete grid by dropping 50 items (e.g. sensor outages).
observed_mask = np.ones(index.n_items, dtype=bool)
observed_mask[:50] = False

sub_node_ids = index.node_ids[observed_mask]
sub_times = index.times[observed_mask]
sub_residuals = residuals[observed_mask]

sub_index = spacetime.spacetime_items(sub_node_ids, sub_times)
assert not sub_index.is_complete_grid

sub_candidates = spacetime.candidate_partitions(
    sub_index, spatial_edges, target_sizes=[2], window_lengths=[1]
)
sub_edges, sub_phi = spacetime.spacetime_edges(
    sub_index,
    spatial_edges,
    max_lag=1,
    phi_spatial=0.0,
    phi_temporal=[0.0],
)

# Provide fallback_kappa_fn to handle the incomplete grid safely.
sub_fallback = spacetime.topological_kappa_fn(
    sub_index.n_items, sub_edges, sub_phi
)
sub_choice = spacetime.choose_partition(
    sub_candidates,
    kappa_fn=kappa_fn,
    m_item=16.0,
    is_complete_grid=sub_index.is_complete_grid,
    fallback_kappa_fn=sub_fallback,
    edges=sub_edges,
)

sub_result = spacetime.spacetime_interval(
    sub_residuals, sub_index, sub_choice, metric="mse", m_item=16.0
)
print(f"Fallback route: {sub_choice.route}, G: {sub_choice.winner.G}")
print(
    f"Interval: [{sub_result.low:.4f}, {sub_result.high:.4f}], status:"
    f" {sub_result.status.value}"
)
# Expected output:
# Fallback route: fallback_topological, G: 1050
# Interval: [0.5658, 4.4960], status: unrefuted
```

### Candidate partitions and partition choice

The following terms describe how candidates are built and scored:

*   **Item:** a single evaluated observation at a specific node and time step
    $(u, t)$.
*   **Community:** a spatial cluster of graph nodes $C_a \subset V_s$, where
    $V_s$ is the node set of the spatial graph.
*   **Window:** a contiguous temporal interval of time steps $W_w \subset T$,
    where $T$ is the set of time steps.
*   **Candidate partition:** a proposed grouping of items into clusters:
    spatial communities spanning all times ($G_t = 1$), temporal windows
    spanning all nodes ($G_s = 1$), or Cartesian space-time product blocks
    $C_a \times W_w$.
*   **$G, G_s, G_t$:** $G$ is the total number of non-empty space-time
    clusters; $G_s$ is the number of spatial communities; $G_t$ is the number
    of temporal windows. For product blocks, $G \le G_s \times G_t$ (with
    equality when all product blocks are non-empty).
*   **$M_{c,\mathrm{default}}$:** the default cluster-level relative
    second-moment bound derived from the item-level $m_{\mathrm{item}}$ and
    the cluster item counts $n_c$ with Lemma I′:
    $M_{c,\mathrm{default}} = \min\{r\,m_{\mathrm{item}},\ 1 + r(m_{\mathrm{item}} - 1) + \mathrm{CV}_n^2 + 2\,\mathrm{CV}_n\sqrt{r(m_{\mathrm{item}} - 1)}\}$,
    where $r = n_{\max}/\bar{m}$ (largest over mean cluster size) and
    $\mathrm{CV}_n^2 = G\sum_c n_c^2 / n^2 - 1$. It equals $m_{\mathrm{item}}$
    when all clusters have equal size.
*   **$r_{\mathrm{eff}}$:** the size-imbalance factor
    $r_{\mathrm{eff}} = \frac{M_{c,\mathrm{default}} - 1}{m_{\mathrm{item}} - 1} \ge 1$
    (or $n_{\max} / \bar{m}$ when $m_{\mathrm{item}} = 1$). Equal cluster
    sizes give $r_{\mathrm{eff}} = 1$.
*   **$\nu$ and $N_{\mathrm{eff}}$:** $\nu = G / \kappa$ is the unweighted
    effective cluster count.
    $N_{\mathrm{eff}} = \frac{G}{\kappa \cdot r_{\mathrm{eff}}} = \frac{\nu}{r_{\mathrm{eff}}}$
    is the size-adjusted effective sample size. Candidate selection maximizes
    $N_{\mathrm{eff}}$ to produce the narrowest valid interval.
*   **Issuance threshold ($n_A$):** an interval can be issued if and only if
    $\nu = G / \kappa > n_A$, where
    $n_A = (M_{c,\mathrm{default}} - 1)\frac{1 - \alpha/2}{\alpha/2}$. If
    $\nu \le n_A$, Cantelli's inequality (a one-sided Chebyshev inequality)
    does not yield a non-trivial bound and the result is
    `ASSUMPTION_REQUIRED`.
*   **$\phi_{\mathrm{spatial}}, \phi_{\mathrm{temporal}}, \phi_{\mathrm{cross}}$:**
    non-negative out-of-sample upper bounds on the correlation between the
    losses of adjacent spatio-temporal items along spatial graph edges,
    consecutive time steps, and cross space-time edges, respectively
    ([§4.2](#42-computing-κ-from-the-graph-topology)).
*   **$\mathrm{max\_lag}$ ($L$):** the maximum temporal lag over which
    temporal and cross spatio-temporal edges are formed in the strong product
    graph. (This $L$ is not the model lookback of §3.)
*   **Strong product graph ($G_s \boxtimes G_t$):** the graph on $(u, t)$
    pairs where two items are adjacent if they are spatial neighbours at the
    same time, the same node at adjacent times, or adjacent in both space and
    time. (Here $G_s$ and $G_t$ denote the spatial and temporal graphs, not
    the cluster counts.)
*   **Kronecker factorization:** the property that the joint covariance of
    space-time blocks factors into the Kronecker product of a spatial
    covariance matrix and a temporal covariance matrix
    ($\Sigma = \Sigma_s \otimes \Sigma_t$).
*   **Fisher-$z$:** the variance-stabilizing transformation
    $z = \operatorname{arctanh}(r)$ used in the temporal lag-1
    autocorrelation test.

**Design exogeneity (Lemma R and Proposition S5).** Partition choice in
`choose_partition` is a deterministic function of the graph topology, the
time steps, and the declarations; it never inspects evaluated losses,
residuals, or labels (Proposition S5, §9.5.2 of
[`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md)). By Lemma R (conditioning on
the design, argued in [`THEORY.md`](./THEORY.md) and applied in §9.5 of
[`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md)), conditioning on this design
preserves finite-sample coverage validity.

### Route certification

`choose_partition` certifies candidate partitions through one of four routes,
reported in `choice.route`:

| Route Name | Condition | Method | References |
| :--- | :--- | :--- | :--- |
| `"range"` | `kappa_fn` from `range_envelope.range_kappa_fn`, complete or incomplete grid | Range-envelope bound $1 + \lambda_{\max}(\text{cut}) + \gamma$ (§3) | Claim RE1 in [`THEORY_RANGE_ENVELOPE.md`](./THEORY_RANGE_ENVELOPE.md) |
| `"separable"` | Complete grid, `separable_kappa` supplied | Sets $\kappa = \kappa_s$ (if $G_t=1$), $\kappa_t$ (if $G_s=1$), or $\kappa_s \kappa_t$ (if $G_s>1, G_t>1$); runs joint check and (when $G_s>1, G_t>1$) marginal checks | Claim S1, §9.1 of [`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md) |
| `"topological"` | Complete or incomplete grid, `topological_kappa_fn` supplied | Computes certified $\kappa$ from cut eigenvalues | Lemma T′ in [`THEORY.md`](./THEORY.md); Claim S2B, §9.2.2 of [`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md) |
| `"fallback_topological"` | Incomplete grid, `separable_kappa` + `fallback_kappa_fn` supplied | Falls back to topological $\kappa$; skips separable marginal checks | Lemma T′ in [`THEORY.md`](./THEORY.md); Claim S2B(4), §9.2.2 of [`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md) |

**Which κ is used.** `choice.winner_kappa` is the inflation factor selected by
partition scoring, and `spacetime_interval` uses it by default. An explicit
`kappa=...` passed to `spacetime_interval` replaces it only if it is at least
`choice.winner_kappa`; a smaller value raises `ValueError`.
`result.kappa_declared` records the value used for interval construction.

### Marginal refutation checks and drift diagnostics

1.  **Marginal refutation checks under separability:** on the `"separable"`
    route with product blocks ($G_s > 1, G_t > 1$), Claim S1(3) implies that
    temporal window totals and spatial community totals must satisfy 1D
    variance-inflation bounds:
    -   `marginal_checks.temporal`: tests grand-mean-centred temporal window
        totals against the declared $\kappa_t$ with `check_uncorrelated` (if
        $\kappa_t = 1$) or `check_kappa` (if $\kappa_t > 1$).
    -   `marginal_checks.spatial`: tests per-window-centred spatial community
        totals against the declared $\kappa_s$ with
        `check_uncorrelated_edges` (if $\kappa_s = 1$) or
        `check_kappa_batches` (if $\kappa_s > 1$).
2.  **Split-sample drift diagnostic:** when $G_t \ge 4$, the evaluation
    horizon is split in half by window index ($G_t // 2$). Proposition D′ is
    argued for $G_t \ge 2$; `stats.spacetime` requires $G_t \ge 4$ so that each
    half has at least 2 windows. Separate sub-intervals are computed on the
    early and late halves at level $1 - \alpha/2$ (Proposition D′, §9.6 of
    [`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md)). If the two intervals are
    disjoint, `drift.outcome` reports `WARNING`.
    -   **Drift does not count toward `result.refuted`:** `result.refuted`
        checks only whether `result.interval.status` is `TAIL_UNRESOLVED` or
        `result.kappa_check.outcome` is `REFUTED`. A drift warning indicates
        non-stationarity across the evaluation window, but does not
        invalidate the observed-sample certificate.

### Choosing `phi`, `kappa_s`, and `kappa_t`

-   `phi` is an out-of-sample upper bound on the correlation between losses
    of adjacent spatio-temporal items
    ([§4.2](#42-computing-κ-from-the-graph-topology)).
-   `kappa_s` and `kappa_t` are variance-inflation declarations for spatial
    community totals and temporal window totals
    ([§4.1](#41-declaring-m-from-archive-data),
    [§4.2](#42-computing-κ-from-the-graph-topology)).
-   **Always declare them from archive data:** declare $\phi$, $\kappa_s$, and
    $\kappa_t$ from historical logs, earlier model checkpoints, or pilot
    datasets, never from the evaluation dataset itself.
-   **Role of `spacetime_edges` and `edges=`:** `spacetime_edges` constructs
    the item-level edges of the strong product graph on observed items and
    returns `(edges, phi)`. Passing `edges=edges` to `choose_partition` lets
    it compute `cluster_edges` for the winning candidate; passing `edges` and
    `phi` to `topological_kappa_fn` evaluates topological $\kappa$ across all
    candidate partitions.

### What to do (diagnostic table)

The following table maps each space-time diagnostic to its meaning and
remedy:

| Condition | Meaning | Remedy |
| :--- | :--- | :--- |
| Block $\kappa$ refuted (`kappa_check.outcome is REFUTED`) | Cross-cluster correlation exceeds declared $\kappa$ or $\phi$, or between-window mean drift shifts global-mean residuals (§9.4.1 of [`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md)) | Raise declared $\kappa$ or $\phi$, or use larger blocks (note: on the topological route, increasing window length $b \ge 2L$ does not reduce cut eigenvalues, by Proposition S3, §9.3 of [`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md)) |
| Temporal marginal refuted (`marginal_checks.temporal.outcome is REFUTED`) | Temporal window totals correlate more than declared $\kappa_t$ | Raise declared $\kappa_t$ or use longer time windows |
| Spatial marginal refuted (`marginal_checks.spatial.outcome is REFUTED`) | Community totals correlate more than declared $\kappa_s$ | Raise declared $\kappa_s$ or use larger spatial communities |
| Drift `WARNING` (`drift.outcome is WARNING`) | Early and late half intervals are disjoint; process non-stationary | Stationary reading unsupported; report interval as covering only observed period |
| Fallback route chosen (`route == "fallback_topological"`) | Incomplete grid cannot use separable $\kappa_s \kappa_t$ | Partition is certified under topological $\kappa$; typically wider |
| `status` is `ASSUMPTION_REQUIRED` | Too few effective clusters ($\nu = G / \kappa \le n_A$); note that for `metric="r2"`, $n_A$ is evaluated at $\alpha/4$ | Calibrate a tighter archive $m_{\mathrm{item}}$ or cluster-level $m$ ([§4.1](#41-declaring-m-from-archive-data)), use finer clusters if correlation allows, or collect more data |
| `status` is `TAIL_UNRESOLVED` | Empirical tail weight $\hat{M}_c > M_c$ | Raise declared $M$ or $M_c$, or collect more data |

> [!NOTE]
> Neither a drift `WARNING` nor a refuted marginal check (temporal or
> spatial) sets `result.refuted = True`. `result.refuted` is `True` if and
> only if `status` is `TAIL_UNRESOLVED` or the block `kappa_check.outcome` is
> `REFUTED`. Callers on the `"separable"` route must inspect
> `result.marginal_checks` directly.

### `SpaceTimeResult` field reference

`SpaceTimeResult` extends the graph result type (`GraphResult`) with
space-time metadata:

| Field | Type | Description |
| :--- | :--- | :--- |
| `low`, `high` | `float` | Lower and upper certified confidence interval endpoints. |
| `status` | `Status` | Interval issuance and $M_c$ tail check verdict (`UNREFUTED`, `TAIL_UNRESOLVED`, `ASSUMPTION_REQUIRED`). |
| `refuted` | `bool` | `True` if `status` is `TAIL_UNRESOLVED` or `kappa_check.outcome` is `REFUTED`. |
| `marginal_checks` | `MarginalChecks` or `None` | Marginal temporal and spatial refutation checks under separability. |
| `drift` | `DriftResult` or `None` | Split-sample temporal drift diagnostic outcome and half-horizon intervals. |
| `choice` | `Choice` | Partition selection diagnostics, candidate scoring table, route, and cluster edges. |
| `graph_result` | `GraphResult` | Underlying graph result object. |

## 11. How Useful It Is

**Temporal.** The relative full width depends on the sample size $n$
relative to the correlation length $L$:

| Regime ($n/L$) | Relative full width | Utility |
| :--- | :--- | :--- |
| $n/L \ge 20000$ | 0.14–0.32 | Tight and informative |
| $n/L \approx 2000$ | 0.44–1.6 | Usable; moderately wide |
| $n/L \le 200$ | 1–8, or refusal | Wide, or needs a larger sample |

These figures come from a synthetic AR(1) sweep across window counts and
correlation lengths.

**Graph.** On four real topologies (37 000–99 000 nodes, MSE, $\kappa = 1$),
the relative full width was 0.09–0.12. That result declared $M_c$ equal to
its true value, which is not possible in practice. With the default $M_c$,
intervals are wider; how much wider was not measured on real graphs. The
width also grows as the true variance inflation between clusters grows.

## 12. Estimates from the Same Data Are Guidance Only

Estimating $\kappa$ or $M_c$ from the evaluation data and plugging the
estimate into the bound does not give a certified interval. Use estimates
such as `kappa_w_hat` or `m_observed` only to choose conservative
declarations, and preferably compute them on data other than the evaluation
set.

## Not covered

The following are not supported. §3 lists experimental setups that are not
supported.

*   **Ranking metrics:** intervals for the area under the ROC curve (AUC) and
    average precision (AP).
*   **A self-normalized `check_kappa_batches`** that is robust to unequal
    batch variances (§8).
*   **A stronger graph $\kappa = 1$ check on hub-dominated graphs**, one that
    does not rely on a large-sample approximation.

## References

The real-graph topology benchmarks in [`EVIDENCE.md`](./EVIDENCE.md) use these
datasets:

- **GraphLand** (`hm-prices`, `avazu-ctr`, `city-roads-L`). Gleb Bazhenov,
  Oleg Platonov, and Liudmila Prokhorenkova. *GraphLand: Evaluating graph
  machine learning models on diverse industrial data.* In The Thirty-Ninth
  Annual Conference on Neural Information Processing Systems (Datasets and
  Benchmarks Track), 2025.
- **Open Graph Benchmark** (`ogbn-arxiv`). Weihua Hu, Matthias Fey, Marinka
  Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure
  Leskovec. *Open Graph Benchmark: Datasets for Machine Learning on Graphs.*
  In Advances in Neural Information Processing Systems 33 (NeurIPS), 2020.

§3 also cites:

- Yoshua Bengio and Yves Grandvalet. No Unbiased Estimator of the Variance
  of K-Fold Cross-Validation. *Journal of Machine Learning Research*,
  5:1089–1105, 2004.
- Claude Nadeau and Yoshua Bengio. Inference for the Generalization Error.
  *Machine Learning*, 52(3):239–281, 2003.

```bibtex
@inproceedings{bazhenov2025graphland,
  title={GraphLand: Evaluating graph machine learning models on diverse industrial data},
  author={Bazhenov, Gleb and Platonov, Oleg and Prokhorenkova, Liudmila},
  booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems (Datasets and Benchmarks Track)},
  year={2025}
}

@inproceedings{hu2020ogb,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

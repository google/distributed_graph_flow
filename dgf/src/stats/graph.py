# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Graph confidence intervals via cluster aggregation and adjacency refutation.

Provides the graph entry point `graph_interval` for graph-structured evaluation
data, and `graph_interval_from_shard` for pre-aggregated cluster totals. The caller
declares kappa (default 1.0, adjacent clusters uncorrelated) and optionally M_c
(default per-metric bound). The interval is certified under these declarations,
while statistical refutation tests check whether the cluster totals contradict them.

Estimand:
  The average expected loss over the evaluated nodes. Nothing is claimed about
  nodes outside that set. The interval is certified under the declared
  relative variance bound M_c and variance inflation factor kappa.

Refutation checks and null hypotheses:
  - For kappa = 1: A self-normalised edge test (`check_uncorrelated_edges`) tests
    whether standardized cluster residuals z_c = e_c / sqrt(n_c) have zero
    correlation along adjacent cluster pairs. The null hypothesis is that cluster
    errors are independent and mean-zero with arbitrary heteroscedastic variances.
  - For kappa > 1: A super-batch means lower confidence bound (`check_kappa_batches`)
    partitions the cluster graph into super-batches and tests whether variance
    inflation exceeds the declared kappa. Under positive adjacent-cluster dependence,
    super-batch variance estimation is biased low by edges cut between batches,
    so refutations are conservative to first order under balanced super-batches
    and Var(S_c) proportional to n_c (Proposition B).

Blind spots and limitations:
  1. Correlation between non-adjacent clusters: The edge test inspects only edges
     present in `cluster_edges`. Dependencies spanning paths longer than 1 edge
     are not directly tested by the edge test, though super-batch means capture
     larger-scale variance inflation.
  2. Negative or cancelling correlations: The test is one-sided (positive correlation
     only); negative or oscillating spatial correlations will not trigger refutation.
  3. Non-gating refutation: The refutation checks only flag potential model or
     dependence violations; they never alter the interval endpoints or status.
  4. Design requirement: `cluster_ids` and `cluster_edges` must be derived
     only from the graph (e.g. via the built-in partitioner and
     `cluster_adjacency`), never from the observed losses, residuals or
     labels. The interval and the checks are conditional on this design.
  5. Dense cluster graphs: when most cluster edges are cut between
     super-batches (`batch_cut_fraction` near 1), the kappa > 1 check sees
     little of the dependence and is weak. `kappa_w_hat` is reported next to
     it as a point diagnostic.
"""

import dataclasses
import math
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.stats

from dgf.src.stats import cluster_bound as _cluster_bound
from dgf.src.stats import cluster_sketch as _cluster_sketch
from dgf.src.stats import partitioners as _partitioners
from dgf.src.stats import refutation as _refutation
from dgf.src.stats.independent import interval as _interval
from dgf.src.stats.independent import metrics as _metrics

__all__ = [
    "GraphResult",
    "TopologicalKappa",
    "cluster_adjacency",
    "graph_interval",
    "graph_interval_from_shard",
    "topological_kappa",
]


# Refutation test confidence level, matching temporal._REFUTATION_LEVEL.
_REFUTATION_LEVEL: float = 0.95


@dataclasses.dataclass(frozen=True)
class GraphResult:
  """Confidence interval result for graph-structured evaluation data.

  Attributes:
    interval: Underlying ClusterResult with interval endpoints, status, and
      diagnostics. For R^2, interval.m_declared and interval.m_observed report
      the error side (M_c_A, m_hat_e2); the label side (M_c_q, m_hat_q) is in
      r2_detail (so status can be TAIL_UNRESOLVED while m_observed <=
      m_declared).
    kappa_declared: The declared variance inflation factor kappa (>= 1.0).
    kappa_check: Outcome and statistics of the kappa refutation check.
    kappa_w_hat: Empirical W-local variance inflation estimate (1 + 2 *
      (sum_{E} e_c e_d - B_e) / sum e_c^2, with B_e the null-mean
      correction; THEORY.md §9.6).
    num_cluster_edges: Number of unique undirected cluster adjacency edges.
    num_dropped_edges: Number of input edges dropped due to missing endpoints.
    hub_ratio: de Jong ratio λ_max(A)²/‖A‖_F² of the projected cluster adjacency
      A = ½PWP (design only; unit weights). Values near 1 mean one direction
      dominates the edge statistic and its normal approximation is poor; values
      near 0 are the CLT regime. It is a proxy: the CLT condition is on the
      variance-weighted D^{1/2} A D^{1/2}, which it matches only when
      per-cluster variances are roughly proportional to the counts.
    edge_n_eff: Effective number of edge terms (Σ|z_c z_d|)²/Σ(z_c z_d)² of the
      κ = 1 edge test (to leading order T ≤ √edge_n_eff); None when κ > 1.
    num_batches: Realized number of super-batches B (None when kappa = 1).
    batch_cut_fraction: Fraction of cluster edges cut between super-batches
      (None when kappa = 1).
    message: Human-readable summary of interval validity and refutation checks.
  """

  interval: _cluster_bound.ClusterResult
  kappa_declared: float
  kappa_check: _refutation.RefutationResult
  kappa_w_hat: float
  num_cluster_edges: int
  num_dropped_edges: int
  hub_ratio: float
  edge_n_eff: float | None
  num_batches: int | None
  batch_cut_fraction: float | None
  message: str
  r2_detail: _cluster_bound.R2Detail | None = None

  @property

  def refuted(self) -> bool:
    """True if the data contradict a declaration (M or kappa)."""
    return (
        self.interval.status is _interval.Status.TAIL_UNRESOLVED
        or self.kappa_check.outcome is _refutation.Refutation.REFUTED
    )


def cluster_adjacency(
    num_nodes: int,
    edges: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
  """Maps node edges to unique undirected cluster adjacency edges.

  Drops self-loop cluster pairs (edges internal to a cluster), canonicalizes
  each pair to (min, max), and deduplicates. Operates strictly on topology
  without referencing losses.

  Args:
    num_nodes: Total number of nodes in the graph.
    edges: Array of shape (E, 2) containing undirected edge endpoint node IDs.
    cluster_ids: Array of length num_nodes with cluster labels per node. Labels
      may be any integers.

  Returns:
    Array of shape (E_cluster, 2) with deduplicated undirected cluster ID pairs
    (c_min, c_max) representing adjacent clusters.

  Raises:
    ValueError: If array dimensions are invalid or edge endpoint IDs fall
      outside [0, num_nodes).
  """
  c_ids = np.asarray(cluster_ids)
  if c_ids.ndim != 1 or c_ids.size != num_nodes:
    raise ValueError(
        f"cluster_ids must be 1-D of length num_nodes ({num_nodes}), got shape"
        f" {c_ids.shape}."
    )

  ed = np.asarray(edges, dtype=np.int64)
  if ed.ndim != 2 or ed.shape[1] != 2:
    raise ValueError(f"edges must have shape (E, 2), got {ed.shape}.")

  if ed.shape[0] == 0:
    return np.zeros((0, 2), dtype=np.int64)

  if np.any(ed < 0) or np.any(ed >= num_nodes):
    raise ValueError(
        f"Edge endpoint node IDs must be in [0, {num_nodes}), got out-of-range"
        " ID."
    )

  u = ed[:, 0]
  v = ed[:, 1]
  c_u = c_ids[u]
  c_v = c_ids[v]

  cross = c_u != c_v
  if not np.any(cross):
    return np.zeros((0, 2), dtype=np.int64)

  c_u_cross = c_u[cross]
  c_v_cross = c_v[cross]

  # Map cluster IDs to dense 0..G-1 indices for efficient 1-D deduplication
  unique_clusters, inv = np.unique(c_ids, return_inverse=True)
  d_u = inv[u[cross]]
  d_v = inv[v[cross]]

  d_min = np.minimum(d_u, d_v)
  d_max = np.maximum(d_u, d_v)
  g_clusters = len(unique_clusters)

  pair_keys = np.unique(d_min * g_clusters + d_max)
  orig_u = unique_clusters[pair_keys // g_clusters]
  orig_v = unique_clusters[pair_keys % g_clusters]

  return np.column_stack([np.minimum(orig_u, orig_v), np.maximum(orig_u, orig_v)]).astype(
      np.int64
  )


def _compute_hub_ratio_impl(
    counts: Any,
    edges: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
) -> tuple[float, int]:
  """Computes de Jong ratio and iterations used for projected adjacency A = 1/2 P W P."""
  c = np.asarray(counts, dtype=np.float64)
  num_nodes = int(c.size)
  num_edges = len(edges)
  if num_edges == 0 or num_nodes <= 1:
    return 0.0, 0

  total_counts = float(np.sum(c))
  if total_counts <= 0.0:
    return 0.0, 0

  q = np.sqrt(c / total_counts)

  row = np.concatenate([edges[:, 0], edges[:, 1]])
  col = np.concatenate([edges[:, 1], edges[:, 0]])
  data = np.ones(len(row), dtype=np.float64)
  adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

  # Closed form for ||A||_F^2 = 1/4 * ||P W P||_F^2
  # ||P W P||_F^2 = ||W||_F^2 - 2 * ||W q||^2 + (q^T W q)^2, where ||W||_F^2 = 2 * |E|
  wq = adj.dot(q)
  q_wq = float(np.dot(q, wq))
  norm_wq_sq = float(np.dot(wq, wq))
  norm_pwp_f_sq = 2.0 * float(num_edges) - 2.0 * norm_wq_sq + (q_wq**2)
  norm_a_f_sq = 0.25 * norm_pwp_f_sq

  if norm_a_f_sq <= 0.0 or not math.isfinite(norm_a_f_sq):
    return 0.0, 0

  def apply_a(v: np.ndarray) -> np.ndarray:
    # A v = 1/2 P(W(P v)) where P x = x - q(q^T x)
    pv = v - q * float(np.dot(q, v))
    wpv = adj.dot(pv)
    return 0.5 * (wpv - q * float(np.dot(q, wpv)))

  # Start vector: standard normal projected with P and normalized
  rng = np.random.default_rng(seed)
  x0 = rng.standard_normal(num_nodes)
  x0_proj = x0 - q * float(np.dot(q, x0))
  norm_x0 = float(np.linalg.norm(x0_proj))
  if norm_x0 == 0.0:
    return 0.0, 0
  x = x0_proj / norm_x0

  mu_prev = -1.0
  mu = 0.0
  iters_used = 0
  for i in range(max_iter):
    iters_used = i + 1
    ax = apply_a(x)
    y = apply_a(ax)
    mu = float(np.dot(ax, ax))
    if abs(mu - mu_prev) < tol * max(abs(mu), 1e-300):
      break
    mu_prev = mu
    norm_y = float(np.linalg.norm(y))
    if norm_y == 0.0:
      return 0.0, iters_used
    x = y / norm_y

  lambda_max_sq = mu
  hub_ratio = lambda_max_sq / norm_a_f_sq
  return float(hub_ratio), iters_used


def _compute_hub_ratio(
    counts: Any,
    edges: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
) -> float:
  """Computes de Jong ratio lambda_max(A)^2 / ||A||_F^2 of projected adjacency A = 1/2 P W P."""
  ratio, _ = _compute_hub_ratio_impl(
      counts, edges, max_iter=max_iter, tol=tol, seed=seed
  )
  return ratio


def graph_interval_from_shard(
    shard: _cluster_sketch.PartialClusterShard | _cluster_sketch.R2ClusterShard,
    cluster_edges: np.ndarray,
    *,
    metric: str = "mse",
    level: float = 0.95,
    m: float | None = None,
    m_item: float | None = None,
    kappa: float | None = None,
    m_labels: float | None = None,
    kappa_labels: float | None = None,
    c_target: float | None = None,
    num_batches: int = 30,
) -> GraphResult:
  """Computes certified confidence interval and refutation checks from a cluster shard.

  Args:
    shard: PartialClusterShard or R2ClusterShard containing cluster IDs, cluster
      totals, and counts.
    cluster_edges: 2-D array of shape (E, 2) containing cluster ID pairs.
    metric: Evaluation metric name ('mse', 'mae', 'accuracy', 'r2', etc.).
    level: Confidence level in (0, 1) for the confidence interval. Default 0.95.
    m: Declared relative variance bound M_c for cluster totals, used unchanged.
      Mutually exclusive with m_item. If both m and m_item are None, uses sketch
      default.
    m_item: Item-level bound M (e.g. from declare.estimate_m); M_c is derived
      with Lemma I' from the actual cluster sizes. Mutually exclusive with m.
    kappa: Declared variance inflation factor kappa. If None, defaults to 1.0.
    m_labels: Item-level bound on labels for R2 metric. Converted with Lemma I'.
      If None, equals converted m_item if m_item given, equals unconverted m if
      m given, or default 16.0 converted.
    kappa_labels: Variance inflation factor on labels for R2 metric. If None,
      uses kappa.
    c_target: Cantelli target relative half-width factor in (0, 1). Default 0.2.
    num_batches: Number of super-batches for kappa > 1 testing. Default 30.

  Returns:
    GraphResult containing interval endpoints, kappa check outcome, diagnostics,
    and summary message.
  """
  if not (0.0 < level < 1.0):
    raise ValueError(f"level must be in (0, 1), got {level}.")
  if c_target is not None and not (0.0 < c_target < 1.0):
    raise ValueError(f"c_target must be in (0, 1), got {c_target}.")
  if m is not None and m_item is not None:
    raise ValueError("Cannot pass both 'm' and 'm_item'; specify only one.")
  if m is not None and not (math.isfinite(m) and m >= 1.0):
    raise ValueError(f"m must be finite and >= 1.0, got {m}.")
  if m_item is not None and not (math.isfinite(m_item) and m_item >= 1.0):
    raise ValueError(f"m_item must be finite and >= 1.0, got {m_item}.")
  if num_batches < 2:
    raise ValueError(f"num_batches must be >= 2, got {num_batches}.")

  resolved_kappa = 1.0 if kappa is None else float(kappa)
  if not (math.isfinite(resolved_kappa) and resolved_kappa >= 1.0):
    raise ValueError(f"kappa must be finite and >= 1.0, got {resolved_kappa}.")
  resolved_c_target = 0.2 if c_target is None else float(c_target)

  norm_metric = _metrics.normalize_metric(metric)
  if norm_metric == "r2":
    if not isinstance(shard, _cluster_sketch.R2ClusterShard):
      raise ValueError("For metric 'r2', shard must be an R2ClusterShard.")
    g = int(shard.cluster_ids.size)
    order = np.argsort(shard.cluster_ids)
    sorted_ids = shard.cluster_ids[order]

    ed = np.asarray(cluster_edges, dtype=np.int64)
    if ed.ndim != 2 or ed.shape[1] != 2:
      raise ValueError(f"cluster_edges must have shape (E, 2), got {ed.shape}.")

    if ed.shape[0] == 0:
      num_cluster_edges = 0
      num_dropped_edges = 0
      edges_for_check = np.zeros((0, 2), dtype=np.int64)
    else:
      u_orig = ed[:, 0]
      v_orig = ed[:, 1]
      pos_u = np.searchsorted(sorted_ids, u_orig)
      pos_v = np.searchsorted(sorted_ids, v_orig)
      valid_u = (pos_u < len(sorted_ids)) & (
          sorted_ids[np.clip(pos_u, 0, len(sorted_ids) - 1)] == u_orig
      )
      valid_v = (pos_v < len(sorted_ids)) & (
          sorted_ids[np.clip(pos_v, 0, len(sorted_ids) - 1)] == v_orig
      )
      kept_mask = valid_u & valid_v
      num_dropped_edges = int(np.count_nonzero(~kept_mask))

      if np.any(kept_mask):
        idx_u = order[pos_u[kept_mask]]
        idx_v = order[pos_v[kept_mask]]
        c1 = np.minimum(idx_u, idx_v)
        c2 = np.maximum(idx_u, idx_v)
        cross = c1 != c2
        if np.any(cross):
          pair_keys = np.unique(c1[cross] * g + c2[cross])
          edges_for_check = np.column_stack(
              [pair_keys // g, pair_keys % g]
          ).astype(np.int64)
        else:
          edges_for_check = np.zeros((0, 2), dtype=np.int64)
      else:
        edges_for_check = np.zeros((0, 2), dtype=np.int64)
      num_cluster_edges = len(edges_for_check)

    def run_kappa_check(res_vec, counts_vec, k_val):
      if k_val == 1.0:
        return _refutation.check_uncorrelated_edges(
            res_vec, counts_vec, edges_for_check, level=_REFUTATION_LEVEL
        )
      else:
        target_size = max(2, int(round(float(g) / float(num_batches))))
        batch_ids = _partitioners.partition_graph(
            g, edges_for_check, target_cluster_size=target_size, seed=0
        )
        return _refutation.check_kappa_batches(
            res_vec, counts_vec, batch_ids, k_val, level=_REFUTATION_LEVEL
        )

    low, high, level_out, status, r2_detail, k_msg, kappa_check = (
        _cluster_bound.compute_r2_interval(
            shard.e2_totals,
            shard.y_totals,
            shard.y2_totals,
            shard.cluster_counts,
            level=level,
            m=m,
            m_item=m_item,
            m_labels=m_labels,
            kappa=resolved_kappa,
            kappa_labels=kappa_labels,
            run_kappa_check_fn=run_kappa_check,
        )
    )

    c_up = (
        max(r2_detail.c_A, r2_detail.c_B_lower)
        if r2_detail is not None
        else float("nan")
    )

    if status is _interval.Status.ASSUMPTION_REQUIRED:
      is_saturated = _cluster_bound.is_mixing_saturated(g, resolved_kappa)
      if is_saturated:
        diag_enum = _cluster_bound.Diagnosis.SATURATED_GRAPH_MIXING
        diag_msg = (
            "Refused (saturated graph mixing):"
            f" kappa={resolved_kappa:.1f} limits effective_n to"
            f" {float(g)/resolved_kappa:.2f}."
        )
      else:
        diag_enum = _cluster_bound.Diagnosis.RESOLVABLE_SAMPLE_SIZE
        diag_msg = (
            "Refused (insufficient sample size):"
            f" effective_n={float(g)/resolved_kappa:.2f}."
        )
    elif status is _interval.Status.TAIL_UNRESOLVED:
      diag_enum = _cluster_bound.Diagnosis.TAIL_UNRESOLVED_REFUTED
      diag_msg = k_msg
    else:  # UNREFUTED
      is_finite_upper = r2_detail is not None and math.isfinite(r2_detail.U_B)
      if c_up <= resolved_c_target and is_finite_upper:
        diag_enum = _cluster_bound.Diagnosis.CERTIFIED_USEFUL
        diag_msg = (
            f"Certified: relative half-width factor c_up={c_up:.4f} <="
            f" c_target={resolved_c_target:.4f}."
        )
      else:
        diag_enum = _cluster_bound.Diagnosis.CERTIFIED_WIDE
        if r2_detail is not None and not is_finite_upper:
          diag_msg = (
              "Certified wide: upper variance bound U_B is infinite;"
              f" c_up={c_up:.4f}."
          )
        else:
          diag_msg = (
              f"Certified wide: c_up={c_up:.4f} >"
              f" c_target={resolved_c_target:.4f}."
          )

    full_message = (
        k_msg
        if status is _interval.Status.TAIL_UNRESOLVED
        else f"{diag_msg} {k_msg}".strip()
    )

    diag_payload = _cluster_bound.DiagnosticPayload(
        diagnosis=diag_enum,
        c_up=c_up,
        c_target=resolved_c_target,
        min_effective_n_exist=float("nan"),
        target_effective_n=float("nan"),
        max_supported_alpha=float("nan"),
        required_clusters_exist=None,
        required_items_exist=None,
        required_clusters_target=None,
        required_items_target=None,
        message=full_message,
    )
    c_res = _cluster_bound.ClusterResult(
        low=low,
        high=high,
        level=level_out,
        status=status,
        m_declared=(
            r2_detail.M_c_A
            if r2_detail is not None
            else (
                float(m)
                if m is not None
                else (float(m_item) if m_item is not None else 16.0)
            )
        ),
        m_observed=(
            r2_detail.m_hat_e2 if r2_detail is not None else float("nan")
        ),
        num_clusters=g,
        effective_n=float(g) / resolved_kappa,
        mean_cluster_size=float(shard.n_items) / float(g),
        diagnostic=diag_payload,
    )
    hub_ratio = _compute_hub_ratio(shard.cluster_counts, edges_for_check)

    return GraphResult(
        interval=c_res,
        kappa_declared=resolved_kappa,
        kappa_check=kappa_check,
        kappa_w_hat=float("nan"),
        num_cluster_edges=num_cluster_edges,
        num_dropped_edges=num_dropped_edges,
        hub_ratio=hub_ratio,
        edge_n_eff=getattr(kappa_check, "edge_n_eff", None),
        num_batches=None,
        batch_cut_fraction=None,
        message=full_message,
        r2_detail=r2_detail,
    )

  resolved_c_target = 0.2 if c_target is None else float(c_target)

  assert isinstance(shard, _cluster_sketch.PartialClusterShard)
  # 1. Main interval via cluster_interval on shard's sketch
  sk = shard.to_cluster_sketch(kappa_cluster=resolved_kappa)
  c_res = _cluster_bound.cluster_interval(
      sk,
      metric=metric,
      level=level,
      m=m,
      m_item=m_item,
      kappa_cluster=resolved_kappa,
      c_target=resolved_c_target,
  )

  # 2. Cluster index mapping
  g = int(shard.cluster_ids.size)
  order = np.argsort(shard.cluster_ids)
  sorted_ids = shard.cluster_ids[order]

  ed = np.asarray(cluster_edges, dtype=np.int64)
  if ed.ndim != 2 or ed.shape[1] != 2:
    raise ValueError(f"cluster_edges must have shape (E, 2), got {ed.shape}.")

  if ed.shape[0] == 0:
    num_cluster_edges = 0
    num_dropped_edges = 0
    edges_for_check = np.zeros((0, 2), dtype=np.int64)
  else:
    u_orig = ed[:, 0]
    v_orig = ed[:, 1]
    pos_u = np.searchsorted(sorted_ids, u_orig)
    pos_v = np.searchsorted(sorted_ids, v_orig)
    valid_u = (pos_u < len(sorted_ids)) & (
        sorted_ids[np.clip(pos_u, 0, len(sorted_ids) - 1)] == u_orig
    )
    valid_v = (pos_v < len(sorted_ids)) & (
        sorted_ids[np.clip(pos_v, 0, len(sorted_ids) - 1)] == v_orig
    )
    kept_mask = valid_u & valid_v
    num_dropped_edges = int(np.count_nonzero(~kept_mask))

    if np.any(kept_mask):
      idx_u = order[pos_u[kept_mask]]
      idx_v = order[pos_v[kept_mask]]
      c1 = np.minimum(idx_u, idx_v)
      c2 = np.maximum(idx_u, idx_v)
      cross = c1 != c2
      if np.any(cross):
        c1 = c1[cross]
        c2 = c2[cross]
        pair_keys = np.unique(c1 * g + c2)
        edges_for_check = np.column_stack([pair_keys // g, pair_keys % g]).astype(
            np.int64
        )
      else:
        edges_for_check = np.zeros((0, 2), dtype=np.int64)
    else:
      edges_for_check = np.zeros((0, 2), dtype=np.int64)

    num_cluster_edges = len(edges_for_check)

  # 3. Residuals
  totals = shard.cluster_totals
  counts = shard.cluster_counts
  n_total = float(np.sum(counts))
  theta_hat = float(np.sum(totals)) / n_total if n_total > 0 else 0.0
  e = totals - counts.astype(np.float64) * theta_hat

  # 4. Kappa check
  if resolved_kappa == 1.0:
    kappa_check = _refutation.check_uncorrelated_edges(
        e, counts, edges_for_check, level=_REFUTATION_LEVEL
    )
    realized_num_batches = None
    batch_cut_fraction = None
    edge_n_eff = kappa_check.edge_n_eff
    if kappa_check.outcome is _refutation.Refutation.UNTESTABLE:
      if kappa_check.reason is not None:
        k_msg = f"kappa check untestable ({kappa_check.reason})."
      else:
        k_msg = "kappa check untestable (fewer than 30 edges or non-finite)."
    elif kappa_check.outcome is _refutation.Refutation.REFUTED:
      k_msg = (
          "kappa=1 refuted: edge correlation of cluster residuals is significant"
          f" (T={kappa_check.statistic:.2f} > {kappa_check.threshold:.2f})."
      )
    else:
      k_msg = f"kappa=1 not refuted (T={kappa_check.statistic:.2f} <= {kappa_check.threshold:.2f})."
  else:
    target_size = max(2, int(round(float(g) / float(num_batches))))
    batch_ids = _partitioners.partition_graph(
        g, edges_for_check, target_cluster_size=target_size, seed=0
    )
    kappa_check = _refutation.check_kappa_batches(
        e, counts, batch_ids, resolved_kappa, level=_REFUTATION_LEVEL
    )
    realized_num_batches = len(np.unique(batch_ids))
    edge_n_eff = None
    if num_cluster_edges > 0:
      b_u = batch_ids[edges_for_check[:, 0]]
      b_v = batch_ids[edges_for_check[:, 1]]
      batch_cut_fraction = float(np.count_nonzero(b_u != b_v)) / float(
          num_cluster_edges
      )
    else:
      batch_cut_fraction = 0.0

    if kappa_check.outcome is _refutation.Refutation.UNTESTABLE:
      k_msg = "kappa check untestable (fewer than 10 super-batches or G < 2B)."
    elif kappa_check.outcome is _refutation.Refutation.REFUTED:
      k_msg = (
          f"kappa={resolved_kappa:g} refuted: super-batch lower bound"
          f" {kappa_check.statistic:.2f} > declared {kappa_check.threshold:.2f}."
      )
    else:
      k_msg = (
          f"kappa={resolved_kappa:g} not refuted: super-batch lower bound"
          f" {kappa_check.statistic:.2f} <= declared {kappa_check.threshold:.2f}."
      )

  # 5. Diagnostics
  sum_e2 = float(np.sum(e**2))
  if sum_e2 <= 0.0 or not math.isfinite(sum_e2):
    kappa_w_hat = float("nan")
  elif num_cluster_edges == 0:
    kappa_w_hat = 1.0
  else:
    # Centering bias correction in e-scale (as in check_uncorrelated_edges):
    # B_hat_e = sum_E [ n_c n_d Q / N^2 - (n_c e_d^2 + n_d e_c^2) / N ]
    # kappa_w_hat = 1 + 2 * (sum_E e_c e_d - B_hat_e) / Q
    u_idx = edges_for_check[:, 0]
    v_idx = edges_for_check[:, 1]
    n_u = counts[u_idx].astype(np.float64)
    n_v = counts[v_idx].astype(np.float64)
    e_u = e[u_idx]
    e_v = e[v_idx]
    n_total = float(np.sum(counts))
    b_term_e = (
        (n_u * n_v * sum_e2 / (n_total**2))
        - (n_u * (e_v**2) + n_v * (e_u**2)) / n_total
    )
    b_hat_e = float(np.sum(b_term_e))
    e_sum = float(np.sum(e_u * e_v))
    kappa_w_hat = float(1.0 + 2.0 * (e_sum - b_hat_e) / sum_e2)

  hub_ratio = _compute_hub_ratio(counts, edges_for_check)

  # 6. Message
  diag_msg = c_res.diagnostic.message if c_res.diagnostic is not None else ""
  full_message = f"{diag_msg} {k_msg}".strip()

  return GraphResult(
      interval=c_res,
      kappa_declared=resolved_kappa,
      kappa_check=kappa_check,
      kappa_w_hat=kappa_w_hat,
      num_cluster_edges=num_cluster_edges,
      num_dropped_edges=num_dropped_edges,
      hub_ratio=hub_ratio,
      edge_n_eff=edge_n_eff,
      num_batches=realized_num_batches,
      batch_cut_fraction=batch_cut_fraction,
      message=full_message,
  )


def graph_interval(
    data: Any,
    cluster_ids: Any,
    cluster_edges: np.ndarray,
    *,
    metric: str = "mse",
    m: float | None = None,
    m_item: float | None = None,
    m_labels: float | None = None,
    kappa_labels: float | None = None,
    **kwargs,
) -> GraphResult:
  """Computes certified confidence interval and refutation checks from raw data.

  Builds a PartialClusterShard or R2ClusterShard from per-item data and cluster
  assignments,
  then evaluates graph_interval_from_shard.
  """
  norm_metric = _metrics.normalize_metric(metric)
  if norm_metric == "r2":
    if not (isinstance(data, tuple) and len(data) == 2):
      raise ValueError(
          "For metric 'r2', raw data must be a tuple of (labels, predictions)."
      )
    shard = _cluster_sketch.R2ClusterShard.from_data(
        data[0], data[1], cluster_ids
    )
    return graph_interval_from_shard(
        shard,
        cluster_edges,
        metric=metric,
        m=m,
        m_item=m_item,
        m_labels=m_labels,
        kappa_labels=kappa_labels,
        **kwargs,
    )
  if isinstance(data, tuple):
    raise ValueError(
        f"Tuples are only accepted for metric='r2', got {metric!r}."
    )
  shard = _cluster_sketch.PartialClusterShard.from_data(
      data, cluster_ids, metric
  )
  return graph_interval_from_shard(
      shard, cluster_edges, metric=metric, m=m, m_item=m_item, **kwargs
  )


@dataclasses.dataclass(frozen=True)
class TopologicalKappa:
  """Result of topological kappa calculation (Lemma T').

  Attributes:
    kappa: The derived variance inflation factor 1 + lambda_upper.
    lambda_upper: Rigorous upper bound on lambda_max(Phi_cut).
    max_cut_row_sum: Maximum row sum of cut edge weights.
    num_cut_edges: Number of distinct undirected cut edges.
    num_items_on_cut: Number of distinct nodes with at least one cut edge.
  """

  kappa: float
  lambda_upper: float
  max_cut_row_sum: float
  num_cut_edges: int
  num_items_on_cut: int


def topological_kappa(
    num_nodes: int,
    edges: Any,
    cluster_ids: Any,
    phi: float | Any,
) -> TopologicalKappa:
  """Computes a certified variance inflation factor kappa from cut-edge correlations (Lemma T').

  This is the kappa that P1_c needs when:
  (T1) cross-cluster summand covariances satisfy |Cov(s_i, s_j)| <= phi_ij *
  sigma_i * sigma_j, and
  (T2) sum_c Var(S_c) >= sum_i Var(s_i).
  phi must be declared from data outside the evaluated sample.

  Args:
    num_nodes: Total number of nodes / items.
    edges: (E, 2) int array of item edges.
    cluster_ids: 1-D array of length num_nodes containing cluster assignments.
    phi: Float in [0, 1] or 1-D array of shape (E,) with per-edge bounds in [0,
      1].

  Returns:
    TopologicalKappa dataclass with kappa, lambda_upper, max_cut_row_sum,
    num_cut_edges, and num_items_on_cut.

  Raises:
    ValueError: If shapes are invalid, cluster_ids length != num_nodes,
      indices out of range, or phi is non-finite or not in [0, 1].
  """
  if not isinstance(num_nodes, (int, np.integer)) or num_nodes < 1:
    raise ValueError(f"num_nodes must be an integer >= 1, got {num_nodes}.")

  c_ids = np.asarray(cluster_ids)
  if c_ids.ndim != 1 or c_ids.shape[0] != num_nodes:
    raise ValueError(
        f"cluster_ids must be 1-D array of length {num_nodes}, got shape"
        f" {c_ids.shape}."
    )

  ed = np.asarray(edges, dtype=np.int64)
  if ed.ndim != 2 or ed.shape[1] != 2:
    raise ValueError(f"edges must have shape (E, 2), got {ed.shape}.")

  if ed.shape[0] > 0 and (np.any(ed < 0) or np.any(ed >= num_nodes)):
    raise ValueError(
        f"Edge node indices must be in [0, {num_nodes}), got out-of-range"
        " index."
    )

  if isinstance(phi, (int, float, np.floating, np.integer)):
    phi_f = float(phi)
    if not (math.isfinite(phi_f) and 0.0 <= phi_f <= 1.0):
      raise ValueError(f"phi must be finite and in [0, 1], got {phi_f}.")
    phi_arr = np.full(ed.shape[0], phi_f, dtype=np.float64)
  else:
    phi_arr = np.asarray(phi, dtype=np.float64)
    if phi_arr.ndim != 1 or phi_arr.shape[0] != ed.shape[0]:
      raise ValueError(
          f"phi array length {phi_arr.shape} does not match edges shape"
          f" {ed.shape}."
      )
    if not (
        np.all(np.isfinite(phi_arr))
        and np.all(phi_arr >= 0.0)
        and np.all(phi_arr <= 1.0)
    ):
      raise ValueError("All phi values must be finite and in [0, 1].")

  if ed.shape[0] == 0:
    return TopologicalKappa(
        kappa=1.0,
        lambda_upper=0.0,
        max_cut_row_sum=0.0,
        num_cut_edges=0,
        num_items_on_cut=0,
    )

  u = ed[:, 0]
  v = ed[:, 1]
  mask = (u != v) & (c_ids[u] != c_ids[v])
  if not np.any(mask):
    return TopologicalKappa(
        kappa=1.0,
        lambda_upper=0.0,
        max_cut_row_sum=0.0,
        num_cut_edges=0,
        num_items_on_cut=0,
    )

  u_cut = u[mask]
  v_cut = v[mask]
  p_cut = phi_arr[mask]

  c1 = np.minimum(u_cut, v_cut)
  c2 = np.maximum(u_cut, v_cut)
  pair_keys = c1 * np.int64(num_nodes) + c2
  unique_keys, inv = np.unique(pair_keys, return_inverse=True)
  max_phi = np.zeros(len(unique_keys), dtype=np.float64)
  np.maximum.at(max_phi, inv, p_cut)

  u_dedup = unique_keys // np.int64(num_nodes)
  v_dedup = unique_keys % np.int64(num_nodes)

  num_cut_edges = int(len(unique_keys))
  items_on_cut = np.unique(np.concatenate([u_dedup, v_dedup]))
  num_items_on_cut = int(len(items_on_cut))

  rows = np.concatenate([u_dedup, v_dedup])
  cols = np.concatenate([v_dedup, u_dedup])
  data = np.concatenate([max_phi, max_phi])

  phi_mat = sp.csr_matrix(
      (data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float64
  )

  row_sums = np.squeeze(np.asarray(phi_mat.sum(axis=1)))
  ub_rows = float(np.max(row_sums))
  if ub_rows <= 0.0:
    return TopologicalKappa(
        kappa=1.0,
        lambda_upper=0.0,
        max_cut_row_sum=0.0,
        num_cut_edges=num_cut_edges,
        num_items_on_cut=num_items_on_cut,
    )

  # 1. Run 200 power iterations from the all-ones vector to get v, normalised to max v = 1.
  # Shifted power iteration (Phi_cut + mu*I) breaks bipartite oscillation.
  # On bipartite graphs, unshifted power iteration oscillates between odd
  # and even steps, preventing convergence. Shifted power iteration ensures convergence.
  v = np.ones(num_nodes, dtype=np.float64)
  mu = ub_rows
  for _ in range(200):
    v_next = phi_mat.dot(v) + mu * v
    m_v = float(np.max(v_next))
    if m_v <= 0.0:
      break
    v = v_next / m_v

  m_v = float(np.max(v))
  if m_v > 0.0:
    v = v / m_v

  # 2. Set x = v + 1e-3
  # Note: For sparse graphs with small cut degrees, adding a fixed 1e-3 to v
  # perturbs the eigenvector for nodes with small components (e.g. 1/sqrt(k)), increasing
  # the Collatz-Wielandt quotient max_i (Phi_cut x)_i / x_i above 1e-6 relative tolerance.
  # Scaling the regularisation to min(1e-3, 1e-8 * max(v)) or evaluating on v directly
  # achieves the tight Collatz-Wielandt bound.
  eps = 1e-3
  x = v + eps

  # 3. Set ub_cw = max_i (Φ_cut x)_i / x_i
  phi_x = phi_mat.dot(x)
  ub_cw = float(np.max(phi_x / x))

  # Also compute bound on v directly (with safe zero handling) to ensure tightness
  v_pos = np.maximum(v, 1e-12)
  ub_cw_v = float(np.max(phi_mat.dot(v_pos) / v_pos))
  ub_cw_best = min(ub_cw, ub_cw_v)

  # 4. Set ub_rows = max_i Σ_j (Φ_cut)_ij

  # 5. Set lam_upper = min(ub_cw, ub_rows) * (1 + margin) where
  # margin = max(1e-9, 4 * (d_max + 2) * eps64).
  d_max = int(np.max(phi_mat.indptr[1:] - phi_mat.indptr[:-1])) if phi_mat.shape[0] > 0 else 0
  eps64 = float(np.finfo(np.float64).eps)
  margin = max(1e-9, 4.0 * (float(d_max) + 2.0) * eps64)
  lam_upper = float(min(ub_cw_best, ub_rows) * (1.0 + margin))
  kappa = float(1.0 + lam_upper)

  return TopologicalKappa(
      kappa=kappa,
      lambda_upper=lam_upper,
      max_cut_row_sum=float(ub_rows),
      num_cut_edges=num_cut_edges,
      num_items_on_cut=num_items_on_cut,
  )

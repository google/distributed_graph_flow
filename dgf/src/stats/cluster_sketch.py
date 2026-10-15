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

"""Sketch-level aggregation and distributed reduction over graph cluster totals.

Implements cluster-total sketching (`schema_version = 3`; see `DESIGN.md` §8):
- PartialClusterShard: pre-sketch distributed reducer for shards that split
  clusters before forming sketches (exact-summing S_{c,1} + S_{c,2} before
  squaring).
- ClusterSketch: cluster-total sketch maintaining G, sum(S_c), sum(S_c^2),
  top_32, kappa_cluster, and deterministic 64-bit cluster ID fingerprints.
- Merge safety: raises ValueError if attempting to merge ClusterSketches
  with overlapping cluster ID fingerprints.
"""

from collections.abc import Sequence
import dataclasses
from fractions import Fraction
import hashlib
import math
from typing import Any

import numpy as np

from dgf.src.stats.independent import metrics as _metrics
from dgf.src.stats.independent import sketch as _sketch

__all__ = [
    "ClusterSketch",
    "PartialClusterShard",
    "R2ClusterShard",
    "default_m_c_from_counts",
    "merge",
]


def _fingerprint_cluster_id(cid: Any) -> int:
  """Deterministic 64-bit cluster ID fingerprint using blake2b.

  Python's built-in hash() is salted per process and must never be used
  for cross-process sketch merging.

  Args:
    cid: Cluster label (int, string, or numpy scalar).

  Returns:
    Unsigned 64-bit integer fingerprint.
  """
  if isinstance(cid, (int, np.integer)):
    raw = f"int:{int(cid)}".encode("utf-8")
  else:
    raw = f"str:{str(cid)}".encode("utf-8")
  digest = hashlib.blake2b(raw, digest_size=8).digest()
  return int.from_bytes(digest, "little", signed=False)


@dataclasses.dataclass(frozen=True)
class PartialClusterShard:
  """Pre-sketch distributed reducer for shards where clusters may be split.

  When batches are sharded without partitioning by cluster ID, a cluster c
  may appear on multiple shards. To preserve algebraic exactness, cluster totals
  must be summed before squaring (because (a + b)^2 != a^2 + b^2).

  Attributes:
    cluster_ids: 1-D array of unique cluster IDs present in this shard.
    cluster_totals: 1-D float64 array of partial sums sum_{i in c} s_i per
      cluster.
    cluster_counts: 1-D int64 array of item counts per cluster.
  """

  cluster_ids: np.ndarray
  cluster_totals: np.ndarray
  cluster_counts: np.ndarray

  def __post_init__(self) -> None:
    c_ids = np.asarray(self.cluster_ids)
    c_totals = np.asarray(self.cluster_totals, dtype=np.float64)
    c_counts = np.asarray(self.cluster_counts, dtype=np.int64)

    if c_ids.ndim != 1 or c_totals.ndim != 1 or c_counts.ndim != 1:
      raise ValueError(
          f"All arrays must be 1-D; got shapes {c_ids.shape}, "
          f"{c_totals.shape}, {c_counts.shape}."
      )
    if not (c_ids.size == c_totals.size == c_counts.size):
      raise ValueError(
          f"Array lengths must match; got {c_ids.size}, {c_totals.size}, "
          f"{c_counts.size}."
      )
    if c_ids.size == 0:
      raise ValueError("PartialClusterShard cannot be empty.")
    if not np.all(np.isfinite(c_totals)):
      raise ValueError("cluster_totals must be finite.")
    if np.any(c_totals < 0.0):
      raise ValueError("cluster_totals must be non-negative.")
    if np.any(c_counts < 1):
      raise ValueError("cluster_counts must be strictly positive (>= 1).")

    # Rebind normalized arrays
    object.__setattr__(self, "cluster_ids", c_ids)
    object.__setattr__(self, "cluster_totals", c_totals)
    object.__setattr__(self, "cluster_counts", c_counts)

  @classmethod
  def from_summands(
      cls,
      summands: Sequence[float] | np.ndarray,
      cluster_ids: Sequence[Any] | np.ndarray,
  ) -> "PartialClusterShard":
    """Constructs a shard from raw per-item non-negative summands."""
    s = np.asarray(summands, dtype=np.float64)
    ids = np.asarray(cluster_ids)
    if s.ndim != 1:
      raise ValueError(f"summands must be 1-D, got shape {s.shape}.")
    if s.shape != ids.shape:
      raise ValueError(
          f"summands and cluster_ids must have equal length, got {s.shape} "
          f"and {ids.shape}."
      )
    if s.size == 0:
      raise ValueError("summands must be non-empty.")
    if not np.all(np.isfinite(s)):
      raise ValueError("summands must be finite.")
    if np.any(s < 0.0):
      raise ValueError("summands must be non-negative.")

    unique_ids, inverse = np.unique(ids, return_inverse=True)
    totals = np.bincount(inverse, weights=s).astype(np.float64)
    counts = np.bincount(inverse).astype(np.int64)

    return cls(
        cluster_ids=unique_ids,
        cluster_totals=totals,
        cluster_counts=counts,
    )

  @classmethod
  def from_data(
      cls,
      data: Sequence[float] | np.ndarray,
      cluster_ids: Sequence[Any] | np.ndarray,
      metric: str,
  ) -> "PartialClusterShard":
    """Extracts summands via metric rules and constructs a shard."""
    norm_metric = _metrics.normalize_metric(metric)
    if norm_metric == "r2":
      raise ValueError("cluster aggregation does not support r2 metric.")
    summands = _metrics.extract_summands(data, norm_metric)
    return cls.from_summands(summands, cluster_ids)

  def merge(self, other: "PartialClusterShard") -> "PartialClusterShard":
    """Merges with another shard, exact-summing split clusters."""
    if not isinstance(other, PartialClusterShard):
      return NotImplemented

    all_ids = np.concatenate([self.cluster_ids, other.cluster_ids])
    all_totals = np.concatenate([self.cluster_totals, other.cluster_totals])
    all_counts = np.concatenate([self.cluster_counts, other.cluster_counts])

    unique_ids, inverse = np.unique(all_ids, return_inverse=True)
    merged_totals = np.bincount(inverse, weights=all_totals).astype(np.float64)
    merged_counts = np.bincount(inverse, weights=all_counts).astype(np.int64)

    return PartialClusterShard(
        cluster_ids=unique_ids,
        cluster_totals=merged_totals,
        cluster_counts=merged_counts,
    )

  def __add__(self, other: "PartialClusterShard") -> "PartialClusterShard":
    return self.merge(other)

  def to_cluster_sketch(self, kappa_cluster: float = 1.0) -> "ClusterSketch":
    """Converts finalized cluster totals into a ClusterSketch."""
    n_items = int(np.sum(self.cluster_counts))
    num_clusters = int(self.cluster_totals.size)
    sum_s = float(np.sum(self.cluster_totals))
    sum_s2 = float(np.sum(self.cluster_totals**2))
    top_32 = tuple(
        sorted([float(x) for x in self.cluster_totals], reverse=True)[:32]
    )
    max_cluster_size = int(np.max(self.cluster_counts))
    sum_count_sq = int(np.sum(self.cluster_counts**2))
    fingerprints = frozenset(
        _fingerprint_cluster_id(cid) for cid in self.cluster_ids
    )

    return ClusterSketch(
        n_items=n_items,
        num_clusters=num_clusters,
        sum_s=sum_s,
        sum_s2=sum_s2,
        top_32=top_32,
        max_cluster_size=max_cluster_size,
        sum_count_sq=sum_count_sq,
        kappa_cluster=kappa_cluster,
        cluster_id_fingerprints=fingerprints,
    )


@dataclasses.dataclass(frozen=True)
class R2ClusterShard:
  """Cluster shard for R^2 evaluation data storing (E2_c, Y_c, Y2_c, n_c, ref).

  Attributes:
    cluster_ids: 1-D array of unique cluster identifiers.
    e2_totals: 1-D float array of sum_{i in c} (y_true - y_pred)^2.
    y_totals: 1-D float array of sum_{i in c} (y_true - ref).
    y2_totals: 1-D float array of sum_{i in c} (y_true - ref)^2.
    cluster_counts: 1-D int array of cluster sizes n_c.
    ref: Reference mean scalar used to centre y_totals and y2_totals.
  """

  cluster_ids: np.ndarray
  e2_totals: np.ndarray
  y_totals: np.ndarray
  y2_totals: np.ndarray
  cluster_counts: np.ndarray
  ref: float = 0.0

  def __post_init__(self):
    c_ids = np.asarray(self.cluster_ids)
    e2 = np.asarray(self.e2_totals, dtype=np.float64)
    y = np.asarray(self.y_totals, dtype=np.float64)
    y2 = np.asarray(self.y2_totals, dtype=np.float64)
    counts = np.asarray(self.cluster_counts, dtype=np.int64)

    if c_ids.ndim != 1:
      raise ValueError(f"cluster_ids must be 1-D, got shape {c_ids.shape}.")
    g = c_ids.shape[0]
    for arr, name in (
        (e2, "e2_totals"),
        (y, "y_totals"),
        (y2, "y2_totals"),
        (counts, "cluster_counts"),
    ):
      if arr.ndim != 1 or arr.shape[0] != g:
        raise ValueError(
            f"{name} must be 1-D with length {g}, got shape {arr.shape}."
        )
    if g > 0 and len(np.unique(c_ids)) != g:
      raise ValueError("cluster_ids must be unique within a single shard.")
    if np.any(counts < 1):
      raise ValueError("cluster_counts must be strictly positive (>= 1).")
    if np.any(e2 < 0.0):
      raise ValueError("e2_totals must be non-negative.")
    if np.any(y2 < 0.0):
      raise ValueError("y2_totals must be non-negative.")

    object.__setattr__(self, "cluster_ids", c_ids)
    object.__setattr__(self, "e2_totals", e2)
    object.__setattr__(self, "y_totals", y)
    object.__setattr__(self, "y2_totals", y2)
    object.__setattr__(self, "cluster_counts", counts)
    object.__setattr__(self, "ref", float(self.ref))

  @classmethod
  def from_data(
      cls,
      y_true: Sequence[float] | np.ndarray,
      y_pred: Sequence[float] | np.ndarray,
      cluster_ids: Sequence[Any] | np.ndarray,
      ref: float | None = None,
  ) -> "R2ClusterShard":
    """Constructs an R2ClusterShard from raw labels, predictions, and cluster assignments.

    Labels and predictions are shifted by reference mean `ref` (defaulting to
    float(np.mean(y_true))) before forming cluster totals to protect downstream
    variance calculations from catastrophic cancellation when |mean(y)| >> sd(y).
    y_totals and y2_totals are centred at self.ref.
    All downstream R^2 calculations (e2, res_y, Q_hat, v_hat) are shift-invariant.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ids = np.asarray(cluster_ids)
    if yt.ndim != 1 or yp.ndim != 1 or ids.ndim != 1:
      raise ValueError(
          "y_true, y_pred, and cluster_ids must all be 1-D arrays."
      )
    if not (yt.shape[0] == yp.shape[0] == ids.shape[0]):
      raise ValueError(
          f"Length mismatch: y_true {yt.shape}, y_pred {yp.shape}, cluster_ids"
          f" {ids.shape}."
      )
    if yt.size == 0:
      raise ValueError("Data must be non-empty.")

    ref_val = float(np.mean(yt)) if ref is None else float(ref)
    yt_centered = yt - ref_val
    yp_centered = yp - ref_val
    e = yp_centered - yt_centered
    e2 = e**2
    y2 = yt_centered**2

    unique_ids, inv = np.unique(ids, return_inverse=True)
    e2_tot = np.bincount(inv, weights=e2).astype(np.float64)
    y_tot = np.bincount(inv, weights=yt_centered).astype(np.float64)
    y2_tot = np.bincount(inv, weights=y2).astype(np.float64)
    counts = np.bincount(inv).astype(np.int64)

    return cls(
        cluster_ids=unique_ids,
        e2_totals=e2_tot,
        y_totals=y_tot,
        y2_totals=y2_tot,
        cluster_counts=counts,
        ref=ref_val,
    )

  def recenter(self, new_ref: float) -> "R2ClusterShard":
    """Re-centres cluster totals onto a new reference mean exactly."""
    new_r = float(new_ref)
    if new_r == self.ref:
      return self
    d = float(self.ref - new_r)
    n_c = self.cluster_counts.astype(np.float64)
    new_y = self.y_totals + n_c * d
    new_y2 = self.y2_totals + 2.0 * d * self.y_totals + n_c * (d**2)
    return R2ClusterShard(
        cluster_ids=self.cluster_ids,
        e2_totals=self.e2_totals,
        y_totals=new_y,
        y2_totals=new_y2,
        cluster_counts=self.cluster_counts,
        ref=new_r,
    )

  def merge(self, other: "R2ClusterShard") -> "R2ClusterShard":
    """Merges with another R2ClusterShard, exact-summing split clusters.

    If refs differ, re-centres the other shard onto self.ref exactly before adding.
    """
    if not isinstance(other, R2ClusterShard):
      return NotImplemented

    if other.ref != self.ref:
      d = float(other.ref - self.ref)
      n_c = other.cluster_counts.astype(np.float64)
      other_y = other.y_totals + n_c * d
      other_y2 = other.y2_totals + 2.0 * d * other.y_totals + n_c * (d**2)
    else:
      other_y = other.y_totals
      other_y2 = other.y2_totals

    all_ids = np.concatenate([self.cluster_ids, other.cluster_ids])
    all_e2 = np.concatenate([self.e2_totals, other.e2_totals])
    all_y = np.concatenate([self.y_totals, other_y])
    all_y2 = np.concatenate([self.y2_totals, other_y2])
    all_counts = np.concatenate([self.cluster_counts, other.cluster_counts])

    unique_ids, inv = np.unique(all_ids, return_inverse=True)
    merged_e2 = np.bincount(inv, weights=all_e2).astype(np.float64)
    merged_y = np.bincount(inv, weights=all_y).astype(np.float64)
    merged_y2 = np.bincount(inv, weights=all_y2).astype(np.float64)
    merged_counts = np.bincount(inv, weights=all_counts).astype(np.int64)

    return R2ClusterShard(
        cluster_ids=unique_ids,
        e2_totals=merged_e2,
        y_totals=merged_y,
        y2_totals=merged_y2,
        cluster_counts=merged_counts,
        ref=self.ref,
    )

  def __add__(self, other: "R2ClusterShard") -> "R2ClusterShard":
    return self.merge(other)

  @property
  def num_clusters(self) -> int:
    return int(self.cluster_ids.size)

  @property
  def n_items(self) -> int:
    return int(np.sum(self.cluster_counts))


def default_m_c_from_counts(
    m_item: float,
    n: int,
    g: int,
    n_max: int,
    s: int,
) -> float:
  """Computes default M_c bound on cluster totals from item-level bound m_item (Lemma I')."""
  if n == 0 or g == 0:
    return float(m_item)

  ratio1 = Fraction(n_max * g, n)
  ratio2 = Fraction(g * s - n * n, n * n)

  if ratio1 == 1 and ratio2 == 0:
    return float(m_item)

  r = float(ratio1)
  cv2 = float(ratio2)
  m = float(m_item)

  branch_a = r * m
  diff = max(0.0, m - 1.0)
  cv = math.sqrt(max(0.0, cv2))
  branch_b = 1.0 + r * diff + cv2 + 2.0 * cv * math.sqrt(max(0.0, r * diff))

  return float(min(branch_a, branch_b))


@dataclasses.dataclass(frozen=True)
class ClusterSketch:

  """Cluster-total sketch for batch-means bounds (schema_version = 3).

  Attributes:
    n_items: Total items/nodes across all clusters.
    num_clusters: Number of clusters G.
    sum_s: Sum of cluster totals sum_{c=1}^G S_c.
    sum_s2: Sum of squared cluster totals sum_{c=1}^G S_c^2.
    top_32: Up to 32 largest cluster totals, sorted descending.
    max_cluster_size: Maximum items in any single cluster max_c n_c.
    sum_count_sq: Sum of squared cluster item counts sum_{c=1}^G n_c^2.
    kappa_cluster: Declared variance inflation factor κ of the cluster totals:
      Var(Σ S_c) ≤ κ Σ Var(S_c) (≥ 1.0). A correlation row-sum bound is a
      sufficient condition.
    cluster_id_fingerprints: Set of 64-bit deterministic cluster ID
      fingerprints.
    schema_version: Must be 3.
  """

  n_items: int
  num_clusters: int
  sum_s: float
  sum_s2: float
  top_32: tuple[float, ...]
  max_cluster_size: int
  sum_count_sq: int
  kappa_cluster: float = 1.0
  cluster_id_fingerprints: frozenset[int] = frozenset()
  schema_version: int = 3

  def __post_init__(self) -> None:
    if self.schema_version != 3:
      raise ValueError(
          f"schema_version must be 3, got {self.schema_version}."
      )
    if not isinstance(self.max_cluster_size, int) or isinstance(
        self.max_cluster_size, bool
    ):
      raise ValueError(
          "max_cluster_size must be an int, got "
          f"{type(self.max_cluster_size).__name__}."
      )
    if not isinstance(self.sum_count_sq, int) or isinstance(
        self.sum_count_sq, bool
    ):
      raise ValueError(
          f"sum_count_sq must be an int, got {type(self.sum_count_sq).__name__}."
      )
    if self.n_items < 1:
      raise ValueError(f"n_items must be >= 1, got {self.n_items}.")
    if self.num_clusters < 1:
      raise ValueError(f"num_clusters must be >= 1, got {self.num_clusters}.")
    if self.n_items < self.num_clusters:
      raise ValueError(
          f"n_items ({self.n_items}) cannot be less than num_clusters "
          f"({self.num_clusters})."
      )
    if not (1 <= self.max_cluster_size <= self.n_items):
      raise ValueError(
          f"max_cluster_size must be between 1 and n_items ({self.n_items}), "
          f"got {self.max_cluster_size}."
      )
    if self.max_cluster_size * self.num_clusters < self.n_items:
      raise ValueError(
          f"max_cluster_size ({self.max_cluster_size}) * num_clusters "
          f"({self.num_clusters}) must be >= n_items ({self.n_items})."
      )
    min_sum_count_sq = math.ceil((self.n_items**2) / self.num_clusters)
    if self.sum_count_sq < min_sum_count_sq:
      raise ValueError(
          f"sum_count_sq ({self.sum_count_sq}) must be >= ceil(n_items^2 / "
          f"num_clusters) ({min_sum_count_sq})."
      )
    if self.sum_count_sq > self.max_cluster_size * self.n_items:
      raise ValueError(
          f"sum_count_sq ({self.sum_count_sq}) must be <= max_cluster_size * "
          f"n_items ({self.max_cluster_size * self.n_items})."
      )
    if not (math.isfinite(self.kappa_cluster) and self.kappa_cluster >= 1.0):
      raise ValueError(
          f"kappa_cluster must be finite and >= 1.0, got {self.kappa_cluster}."
      )
    if not math.isfinite(self.sum_s) or self.sum_s < 0.0:
      raise ValueError(
          f"sum_s must be finite and non-negative, got {self.sum_s}."
      )
    if not math.isfinite(self.sum_s2) or self.sum_s2 < 0.0:
      raise ValueError(
          f"sum_s2 must be finite and non-negative, got {self.sum_s2}."
      )

  @property
  def effective_n(self) -> float:
    """Effective cluster sample size nu = G / kappa_cluster."""
    return float(self.num_clusters) / self.kappa_cluster

  @property
  def mean_cluster_size(self) -> float:
    """Average cluster size n / G."""
    return float(self.n_items) / float(self.num_clusters)

  def default_m_c(self, m_item: float) -> float:
    """Computes default M_c bound on cluster totals from item-level bound m_item.

    Both branches are valid upper bounds on the uncentred cluster second-moment
    ratio under the pooled item premise. The first branch r * M is tight when the
    size-n_max clusters can carry all the mean (k * n_max >= n / M). The second
    branch is smaller only for M near 1. The same value serves as the width
    constant and the tail-status threshold.
    """
    return default_m_c_from_counts(
        m_item,
        self.n_items,
        self.num_clusters,
        self.max_cluster_size,
        self.sum_count_sq,
    )

  def to_independent_sketch(self) -> _sketch.IndependentSketch:
    """Converts to an IndependentSketch on cluster totals with effective_n = G / kappa_cluster."""
    eff_n = self.effective_n if self.effective_n >= 1.0 else 1.0
    return _sketch.IndependentSketch(
        n=self.num_clusters,
        sum_s=self.sum_s,
        sum_s2=self.sum_s2,
        top_32=self.top_32,
        effective_n=eff_n,
        schema_version=1,
    )

  @classmethod
  def from_summands(
      cls,
      summands: Sequence[float] | np.ndarray,
      cluster_ids: Sequence[Any] | np.ndarray,
      kappa_cluster: float = 1.0,
  ) -> "ClusterSketch":
    """Constructs a ClusterSketch from per-item summands and cluster labels."""
    return PartialClusterShard.from_summands(summands, cluster_ids).to_cluster_sketch(
        kappa_cluster=kappa_cluster
    )

  @classmethod
  def from_data(
      cls,
      data: Sequence[float] | np.ndarray,
      cluster_ids: Sequence[Any] | np.ndarray,
      metric: str,
      kappa_cluster: float = 1.0,
  ) -> "ClusterSketch":
    """Constructs a ClusterSketch from raw residuals and cluster labels."""
    return PartialClusterShard.from_data(data, cluster_ids, metric).to_cluster_sketch(
        kappa_cluster=kappa_cluster
    )

  def merge(self, other: "ClusterSketch") -> "ClusterSketch":
    """Merges two cluster-disjoint ClusterSketch instances.

    Valid when the two shards are mutually uncorrelated; for correlated
    shards, pass kappa_cluster explicitly to cluster_interval.

    Raises ValueError if the two sketches share any cluster IDs.
    """
    if not isinstance(other, ClusterSketch):
      return NotImplemented
    if self.schema_version != 3 or other.schema_version != 3:
      raise ValueError(
          "Can only merge schema_version=3 ClusterSketch instances; got "
          f"{self.schema_version} and {other.schema_version}."
      )

    if (
        self.cluster_id_fingerprints
        and other.cluster_id_fingerprints
        and bool(self.cluster_id_fingerprints & other.cluster_id_fingerprints)
    ):
      raise ValueError(
          "Cannot merge ClusterSketch instances with overlapping cluster IDs. "
          "Merging would incorrectly square partial sums separately. Merge "
          "PartialClusterShard instances before converting to sketch."
      )

    merged_kappa = max(self.kappa_cluster, other.kappa_cluster)

    merged_top_32 = tuple(
        sorted(list(self.top_32) + list(other.top_32), reverse=True)[:32]
    )

    return ClusterSketch(
        n_items=self.n_items + other.n_items,
        num_clusters=self.num_clusters + other.num_clusters,
        sum_s=self.sum_s + other.sum_s,
        sum_s2=self.sum_s2 + other.sum_s2,
        top_32=merged_top_32,
        max_cluster_size=max(self.max_cluster_size, other.max_cluster_size),
        sum_count_sq=self.sum_count_sq + other.sum_count_sq,
        kappa_cluster=merged_kappa,
        cluster_id_fingerprints=self.cluster_id_fingerprints
        | other.cluster_id_fingerprints,
        schema_version=3,
    )

  def __add__(self, other: "ClusterSketch") -> "ClusterSketch":
    return self.merge(other)


def merge(a: ClusterSketch, b: ClusterSketch) -> ClusterSketch:
  """Disjoint-cluster sketch merge helper."""
  return a.merge(b)

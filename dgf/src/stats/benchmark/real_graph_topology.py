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

"""Graph topology extraction helpers for DGF benchmark tasks."""

from typing import Any
import dgf
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph


def fetch_g2_graph(
    task_short: str,
    cache_dir: str | None = None,
) -> tuple[dgf.data.InMemoryGraph, dgf.data.GraphSchema]:
  """Fetches a G2 benchmark graph and schema via public dgf.io loaders."""
  effective_cache = cache_dir if cache_dir else None
  if task_short in ("hm-prices", "avazu-ctr", "city-roads-L"):
    return dgf.io.fetch_graphland_graph(
        task_short,
        cache_dir=effective_cache,
        mask_name="RL",
    )
  if task_short == "ogbn-arxiv":
    return dgf.io.fetch_ogb_graph(
        "arxiv",
        cache_dir=effective_cache,
    )
  raise ValueError(f"Unknown G2 task: {task_short}")


def extract_test_node_indices(
    graph: dgf.data.InMemoryGraph,
    target_nodeset: str = "nodes",
) -> np.ndarray:
  """Extracts test node indices from graph's #split feature."""
  split = graph.node_sets[target_nodeset].features.get("#split")
  if split is None:
    raise ValueError(f"No #split feature found in node set '{target_nodeset}'")
  test_idxs = np.where(split == b"test")[0]
  if len(test_idxs) == 0:
    test_idxs = np.where(split == "test")[0]
  return test_idxs.astype(np.int64)


def extract_task_edges(
    graph: dgf.data.InMemoryGraph,
    schema: dgf.data.GraphSchema,
    target_nodeset: str = "nodes",
) -> np.ndarray:
  """Extracts unique undirected edges on the target nodeset, removing self-loops and duplicates."""
  candidate_edges = []
  edge_sets_found = []
  for name, es in graph.edge_sets.items():
    es_schema = schema.edge_sets.get(name)
    src = (
        es_schema.source
        if es_schema is not None
        else getattr(es, "source", None)
    )
    dst = (
        es_schema.target
        if es_schema is not None
        else getattr(es, "target", None)
    )
    edge_sets_found.append((name, src, dst))
    if src == target_nodeset and dst == target_nodeset:
      candidate_edges.append(es.adjacency)

  if len(candidate_edges) != 1:
    raise ValueError(
        "Could not identify unique edge set with "
        f"endpoints on target nodeset '{target_nodeset}'. "
        f"Found edge sets: {edge_sets_found}"
    )

  adj = candidate_edges[0]
  if adj.shape[0] != 2 and adj.shape[1] == 2:
    adj = adj.T

  mask = adj[0] != adj[1]
  u = adj[0][mask]
  v = adj[1][mask]
  src = np.minimum(u, v)
  dst = np.maximum(u, v)
  edges_arr = np.stack([src, dst], axis=1)
  if len(edges_arr) == 0:
    return np.zeros((0, 2), dtype=np.int64)
  unique_edges = np.unique(edges_arr, axis=0)
  return unique_edges


def build_test_subgraph(
    unique_edges: np.ndarray,
    test_node_idxs: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
  """Builds test-node induced subgraph and computes graph statistics."""
  n_test = len(test_node_idxs)
  if len(unique_edges) == 0 or n_test == 0:
    local_edges = np.zeros((0, 2), dtype=np.int64)
  else:
    max_node_id = max(int(np.max(unique_edges)), int(np.max(test_node_idxs)))
    mapping = np.full(max_node_id + 1, -1, dtype=np.int64)
    mapping[test_node_idxs] = np.arange(n_test, dtype=np.int64)

    u = unique_edges[:, 0]
    v = unique_edges[:, 1]
    lu = mapping[u]
    lv = mapping[v]
    mask = (lu >= 0) & (lv >= 0)
    if np.any(mask):
      lu_sub = lu[mask]
      lv_sub = lv[mask]
      sub_min = np.minimum(lu_sub, lv_sub)
      sub_max = np.maximum(lu_sub, lv_sub)
      local_edges = np.unique(np.stack([sub_min, sub_max], axis=1), axis=0)
    else:
      local_edges = np.zeros((0, 2), dtype=np.int64)

  m_test = len(local_edges)
  deg = np.zeros(n_test, dtype=np.int64)
  if m_test > 0:
    np.add.at(deg, local_edges[:, 0], 1)
    np.add.at(deg, local_edges[:, 1], 1)

  frac_with_neighbor = float(np.mean(deg >= 1)) if n_test > 0 else 0.0

  if m_test > 0 and n_test > 0:
    row = np.concatenate([local_edges[:, 0], local_edges[:, 1]])
    col = np.concatenate([local_edges[:, 1], local_edges[:, 0]])
    data_vals = np.ones(len(row), dtype=np.int32)
    adj_sparse = sp.csr_matrix((data_vals, (row, col)), shape=(n_test, n_test))
    num_components, labels = csgraph.connected_components(
        adj_sparse, directed=False
    )
    _, counts = np.unique(labels, return_counts=True)
    largest_component_size = int(np.max(counts)) if len(counts) > 0 else 0
  else:
    num_components = n_test
    largest_component_size = 1 if n_test > 0 else 0

  stats = {
      "node_count": n_test,
      "edge_count": m_test,
      "fraction_with_neighbor": frac_with_neighbor,
      "num_components": int(num_components),
      "largest_component_size": largest_component_size,
  }
  return local_edges, stats

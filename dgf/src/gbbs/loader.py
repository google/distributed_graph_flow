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

"""GBBS graph construction API.

Example creating a GBBS handle from a projection of an InMemoryGraph object:

```
grpah, schema = dgf.io.read_graph("/path/to/graph")
  dgf.gbbs.loader.validate_projection(
      schema, "node_set", "edge_set", weight_key="weight"
  )
handle = dgf.gbbs.loader.gbbs_graph_handle_from_graph(
    graph, "node_set", "edge_set", weight_key="weight"
)
...
```

Example of materializing the projection directly from disk without loading all
heterogeneous pieces and features into memory.

```
# This will only load the necessary graph pieces into memory. Unfortunately
# there is currently one graph copy in memory on the journey to the GBBS
# container.
handle = dgf.gbbs.loader.read_gbbs_projection("/path/to/graph", "node_set",
"edge_set", weight_key="weight")
```
"""

import atexit
import os
import time
from typing import Optional

from absl import logging
from dgf.src.api import io as dgf_io
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.gbbs import _gbbs_ext
from dgf.src.io import graph_constants
from dgf.src.io import schema as schema_io_lib
import numpy as np

# Expose the opaque handle directly.
GbbsGraphHandle = _gbbs_ext.GbbsGraphHandle
EdgeSchema = schema_lib.EdgeSchema
GraphSchema = schema_lib.GraphSchema

# Explicitly register a graceful shutdown handler so `parlay::scheduler` doesn't
# race with Python tearing down its background tracking processes.
atexit.register(_gbbs_ext.shutdown_parlay)


def set_num_parlay_workers(num_workers: int) -> None:
  """Sets the number of worker threads for Parlay / GBBS operations.

  Parlay uses a fork-join work-stealing thread pool to execute parallel graph
  algorithms and loader operations. By default, when the module is imported,
  the worker count is automatically initialized to the machine's hardware
  concurrency (`os.cpu_count()`), or overridden by the `PARLAY_NUM_THREADS`
  environment variable if present.

  Calling this function is an optional user-exposed knob to customize or
  throttle thread concurrency. Typically, this value is set once per program
  before executing graph operations.

  Args:
    num_workers: The desired number of worker threads.
  """
  _gbbs_ext.set_num_parlay_workers(num_workers)


def num_parlay_workers() -> int:
  """Returns the number of active worker threads in Parlay / GBBS.

  By default, this reflects the machine's hardware concurrency (or the
  `PARLAY_NUM_THREADS` environment variable) unless overridden via
  `set_num_parlay_workers`.
  """
  return _gbbs_ext.num_parlay_workers()


def create_gbbs_graph_handle(
    num_nodes: int,
    adjacency: np.ndarray,
    weights: Optional[np.ndarray] = None,
    symmetric: bool = True,
) -> GbbsGraphHandle:
  """Creates a GBBS graph from COO adjacency arrays and optional weights.

  This is the primary low-level entry point for constructing a GBBS graph
  handle from raw numpy arrays.

  Args:
    num_nodes: Total number of nodes in the graph.
    adjacency: int64 numpy array of shape ``[2, num_edges]``.  Row 0 contains
      source node IDs, row 1 contains target node IDs.
    weights: Optional float32 numpy array of shape ``[num_edges]``.  When
      ``None``, all edges receive a default weight of 1.0.
    symmetric: If ``True`` (default), builds an undirected/symmetric GBBS graph.
      If ``False``, builds a directed/asymmetric graph.

  Returns:
    An opaque ``GbbsGraphHandle`` backed by C++ memory.

  Raises:
    ValueError: If *adjacency* does not have the expected shape or dtype.
  """
  adjacency = np.ascontiguousarray(adjacency, dtype=np.int64)
  if adjacency.ndim != 2 or adjacency.shape[0] != 2:
    raise ValueError(
        f"adjacency must have shape [2, num_edges], got {adjacency.shape}"
    )

  if weights is not None:
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    if weights.ndim != 1 or weights.shape[0] != adjacency.shape[1]:
      raise ValueError(
          f"weights must have shape [{adjacency.shape[1]}], got {weights.shape}"
      )

  return _gbbs_ext.create_gbbs_graph_handle(
      num_nodes=num_nodes,
      adjacency=adjacency,
      weights=weights,
      symmetric=symmetric,
  )


def gbbs_graph_handle_from_graph(
    graph: in_memory_graph_lib.InMemoryGraph,
    node_set_name: str,
    edge_set_name: str,
    weight_key: Optional[str] = None,
    symmetric: bool = True,
) -> GbbsGraphHandle:
  """Converts an ``InMemoryGraph`` into a ``GbbsGraphHandle``.

  Extracts a homogeneous subgraph (one node set, one edge set) from the
  given ``InMemoryGraph`` and constructs a GBBS graph handle from its
  COO adjacency arrays.

  Args:
    graph: An ``InMemoryGraph``.
    node_set_name: Name of the node set to use.
    edge_set_name: Name of the edge set to use.
    weight_key: Optional feature key on the edge set to use as edge weights.
    symmetric: If ``True`` (default), builds an undirected/symmetric GBBS graph.
      If ``False``, builds a directed/asymmetric graph.

  Returns:
    An opaque ``GbbsGraphHandle``.

  Raises:
    ValueError: If the named node/edge set is missing or the weight key is
      invalid.
  """
  if node_set_name not in graph.node_sets:
    raise ValueError(
        f"Node set {node_set_name!r} not found in graph. "
        f"Available: {list(graph.node_sets.keys())}"
    )
  if edge_set_name not in graph.edge_sets:
    raise ValueError(
        f"Edge set {edge_set_name!r} not found in graph. "
        f"Available: {list(graph.edge_sets.keys())}"
    )

  node_set = graph.node_sets[node_set_name]
  edge_set = graph.edge_sets[edge_set_name]

  num_nodes = node_set.num_nodes
  if num_nodes is None:
    raise ValueError(
        f"Node set {node_set_name!r} has num_nodes=None; "
        "cannot determine graph size."
    )

  adjacency = edge_set.adjacency  # shape [2, num_edges]

  weights = None
  if weight_key is not None:
    if weight_key not in edge_set.features:
      raise ValueError(
          f"Weight key {weight_key!r} not found in edge set "
          f"{edge_set_name!r} features: {list(edge_set.features.keys())}"
      )
    weights = edge_set.features[weight_key]

  return create_gbbs_graph_handle(
      num_nodes=num_nodes,
      adjacency=adjacency,
      weights=weights,
      symmetric=symmetric,
  )


def validate_projection(
    schema: GraphSchema,
    node_set_name: str,
    edge_set_name: str,
    weight_key: Optional[str] = None,
) -> None:
  """Validates that a node/edge set can be projected onto a GBBS graph.

  Args:
    schema: The graph schema.
    node_set_name: The name of the node set.
    edge_set_name: The name of the edge set.
    weight_key: The optional edge feature key to use as weights.

  Returns:
    None

  Raises:
    ValueError: If the homogeneous projection is not valid.
  """
  if node_set_name not in schema.node_sets:
    raise ValueError(
        f"Node set {node_set_name!r} not found in schema. "
        f"Available: {list(schema.node_sets.keys())}"
    )
  if edge_set_name not in schema.edge_sets:
    raise ValueError(
        f"Edge set {edge_set_name!r} not found in schema. "
        f"Available: {list(schema.edge_sets.keys())}"
    )

  edge_schema = schema.edge_sets[edge_set_name]
  if edge_schema.source != node_set_name or edge_schema.target != node_set_name:
    raise ValueError(
        f"Edge set {edge_set_name!r} is not homogeneous; "
        f"source={edge_schema.source!r}, target={edge_schema.target!r}, "
        f"expected source=target={node_set_name!r}"
    )

  if weight_key is not None:
    if weight_key not in edge_schema.features:
      raise ValueError(
          f"Weight key {weight_key!r} not found in edge set "
          f"{edge_set_name!r} features: {list(edge_schema.features.keys())}"
      )

    weight_feature_schema = edge_schema.features[weight_key]
    if not weight_feature_schema.format.is_numerical():
      raise ValueError(
          f"Weight key {weight_key!r} has non-numeric feature type "
          f"{weight_feature_schema.format} in edge set "
          f"{edge_set_name!r}"
      )


def create_override_schema(
    schema: GraphSchema,
    node_set_name: str,
    edge_set_name: str,
    weight_key: Optional[str] = None,
) -> GraphSchema:
  """Creates an override schema for loading a projection as a GBBS graph."""
  node_schema = schema.node_sets[node_set_name]
  edge_schema = schema.edge_sets[edge_set_name]

  edge_features = {}
  if weight_key is not None:
    edge_features[weight_key] = edge_schema.features[weight_key]

  return GraphSchema(
      node_sets={node_set_name: node_schema},
      edge_sets={
          edge_set_name: EdgeSchema(
              source=edge_schema.source,
              target=edge_schema.target,
              features=edge_features,
          )
      },
  )


def read_gbbs_projection(
    path: str,
    node_set_name: str,
    edge_set_name: str,
    weight_key: Optional[str] = None,
    symmetric: bool = True,
    verbose: bool = False,
    remove_dangling_edges: bool = False,
):
  """Read a GBBS graph (projection) from a materialized DGF graph.

  Args:
    path: Path to the materialized DGF graph.
    node_set_name: Name of the node set to use.
    edge_set_name: Name of the edge set to use.
    weight_key: Optional feature key on the edge set to use as weights.
    symmetric: If ``True`` (default), builds an undirected/symmetric GBBS graph.
      If ``False``, builds a directed/asymmetric graph.
    verbose: If ``True``, print verbose output.
    remove_dangling_edges: Passed to the dgf `read_graph` function.

  Returns:
    A ``GbbsGraphHandle``.

  Raises:
    ValueError: If invalid projection specified.
  """
  orig_schema = schema_io_lib.read_schema(
      os.path.join(path, graph_constants.FILENAME_SCHEMA)
  )

  validate_projection(
      orig_schema, node_set_name, edge_set_name, weight_key=weight_key
  )

  override_schema = create_override_schema(
      orig_schema, node_set_name, edge_set_name, weight_key=weight_key
  )

  proj_graph, _ = dgf_io.read_graph(
      path,
      override_schema=override_schema,
      verbose=verbose,
      remove_dangling_edges=remove_dangling_edges,
  )

  start_time = time.time()
  gbbs_handle = gbbs_graph_handle_from_graph(
      proj_graph,
      node_set_name,
      edge_set_name,
      weight_key,
      symmetric,
  )
  end = time.time()

  if verbose:
    logging.info(
        f"GBBS graph created (post loading) in {end - start_time:.4f} seconds"
    )

  return gbbs_handle

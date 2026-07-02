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

"""Homogenize heterogeneous graph into homogeneous ones."""

from collections import defaultdict
import dataclasses
from typing import Any, Dict, Optional, Protocol, Tuple, Type, Union
from dgf.src.data import in_memory_graph
from dgf.src.data import jax_in_memory_graph as jax_in_memory_graph_lib
from dgf.src.data import schema as schema_lib
import jax.numpy as jnp
import numpy as np

Features = Union[in_memory_graph.Features, jax_in_memory_graph_lib.Features]
Graph = Union[
    in_memory_graph.InMemoryGraph, jax_in_memory_graph_lib.JaxInMemoryGraph
]
NodeSet = Union[
    in_memory_graph.InMemoryNodeSet, jax_in_memory_graph_lib.JaxInMemoryNodeSet
]
EdgeSet = Union[
    in_memory_graph.InMemoryEdgeSet, jax_in_memory_graph_lib.JaxInMemoryEdgeSet
]
GraphType = Union[
    Type[in_memory_graph.InMemoryGraph],
    Type[jax_in_memory_graph_lib.JaxInMemoryGraph],
]
NodeSetType = Union[
    Type[in_memory_graph.InMemoryNodeSet],
    Type[jax_in_memory_graph_lib.JaxInMemoryNodeSet],
]
EdgeSetType = Union[
    Type[in_memory_graph.InMemoryEdgeSet],
    Type[jax_in_memory_graph_lib.JaxInMemoryEdgeSet],
]


# Specific numpy or jax functions.
# TODO(gbm): Make something that can be reused like SD.
@dataclasses.dataclass
class Engine:
  graph_cls: GraphType
  nodeset_cls: NodeSetType
  edgeset_cls: EdgeSetType
  engine: Any


def get_engine(graph: Graph) -> Engine:
  if isinstance(graph, in_memory_graph.InMemoryGraph):
    return Engine(
        graph_cls=in_memory_graph.InMemoryGraph,
        nodeset_cls=in_memory_graph.InMemoryNodeSet,
        edgeset_cls=in_memory_graph.InMemoryEdgeSet,
        engine=np,
    )
  elif isinstance(graph, jax_in_memory_graph_lib.JaxInMemoryGraph):
    return Engine(
        graph_cls=jax_in_memory_graph_lib.JaxInMemoryGraph,
        nodeset_cls=jax_in_memory_graph_lib.JaxInMemoryNodeSet,
        edgeset_cls=jax_in_memory_graph_lib.JaxInMemoryEdgeSet,
        engine=jnp,
    )
  else:
    raise TypeError(f"Unsupported graph type: {type(graph)}")


class FeatureSetProcessor(Protocol):
  """Processes a feature set, returning a flattened feature array."""

  def __call__(
      self,
      src_features: Features,
      src_feature_schema: schema_lib.FeatureSetSchema,
      num_rows: int,
  ) -> Tuple[Features, schema_lib.FeatureSetSchema]:
    ...


def apply_feature(
    graph: Graph,
    schema: schema_lib.GraphSchema,
    process_nodesets: Optional[Dict[str, FeatureSetProcessor]] = None,
    process_edgesets: Optional[Dict[str, FeatureSetProcessor]] = None,
) -> Tuple[Graph, schema_lib.GraphSchema]:
  """Applies feature processors to the node and edge sets of a graph.

  This function does not check the validity of the output e.g. the returned
  schema matches the returned value. For more safety, call
  `dgf.validate.validate_graph` on the output.

  The FeatureSetProcessor should not modify their input values i.e. they have to
  be functionnal.

  Usage Example:

  ```python
  # Create a simple FeatureSetProcessor for the "n1" nodeset.
  def process_n1(
        values: Dict[str, np.ndarray],
        schemas: Dict[str, ydf.data.FeatureSchema],
        num_nodes: int) -> Tuple[
            Dict[str, np.ndarray],
            Dict[str, ydf.data.FeatureSchema]
            ]:

    new_values = {
        "f2" : values["f1"] * 3,
        "f3" : np.cast(values["f1"], np.int)
    }
    new_schemas = {
      "f2": schemas["f1"], # "f2" has the same schema as "f1"
      "f3": dgf.data.FeatureSchema(format=dgf.data.FeatureFormat.INTEGER_64)
    }
    return new_values, new_schemas

  # Read a graph.
  graph, schema = ydf.io.read_graph(...)

  # Process the features of the "n1" nodesets and remove any other nodesets.
  Don't modify the features of the edgesets.
  new_graph, new_schema = ydf.transform.apply_feature(
      graph, schema, {"n1":process_n1})

  # Validate the result (make sure your processor works).
  ydf.validate.validate_graph(new_graph, new_schema)
  ```

  Args:
    graph: The input `InMemoryGraph`.
    schema: The `GraphSchema` corresponding to the input graph.
    process_nodesets: An optional dictionary mapping node set names to
      `FeatureSetProcessor` instances. Non specified nodesets are removed.
    process_edgesets: An optional dictionary mapping edge set names to
      `FeatureSetProcessor` instances. Non specified edgeset are removed.

  Returns:
    A tuple containing:
      - A new `ydf.data.InMemoryGraph` with the transformed features.
      - A new `ydf.data.GraphSchema` reflecting the schema of the transformed
      features.
  """

  engine = get_engine(graph)

  if process_nodesets is None and process_edgesets is None:
    raise ValueError(
        "At least one of 'process_nodesets' or 'process_edgesets' must be"
        " provided."
    )

  if process_nodesets is not None:
    new_nodeset_values = {}
    new_nodeset_schemas = {}
    for nodeset_name, src_nodeset_schema in schema.node_sets.items():
      src_nodeset = graph.node_sets[nodeset_name]
      processor = process_nodesets.get(nodeset_name)
      if processor is None:
        continue
      assert src_nodeset.num_nodes is not None
      new_values, new_schemas = processor(
          src_nodeset.features,
          src_nodeset_schema.features,
          src_nodeset.num_nodes,
      )
      new_nodeset_values[nodeset_name] = engine.nodeset_cls(
          features=new_values,  # pyrefly: ignore[bad-argument-type]
          num_nodes=src_nodeset.num_nodes,
      )
      new_nodeset_schemas[nodeset_name] = schema_lib.NodeSchema(
          features=new_schemas
      )
  else:
    new_nodeset_values = graph.node_sets
    new_nodeset_schemas = schema.node_sets

  if process_edgesets is not None:
    new_edgeset_values = {}
    new_edgeset_schemas = {}
    for edgeset_name, src_edgeset_schema in schema.edge_sets.items():
      src_edgeset = graph.edge_sets[edgeset_name]
      processor = process_edgesets.get(edgeset_name)
      if processor is None:
        continue
      new_values, new_schemas = processor(
          src_edgeset.features,
          src_edgeset_schema.features,
          src_edgeset.num_edges(),
      )
      new_edgeset_values[edgeset_name] = engine.edgeset_cls(
          features=new_values,  # pyrefly: ignore[bad-argument-type]
          adjacency=src_edgeset.adjacency,  # pyrefly: ignore[bad-argument-type]
      )
      new_edgeset_schemas[edgeset_name] = schema_lib.EdgeSchema(
          features=new_schemas,
          source=src_edgeset_schema.source,
          target=src_edgeset_schema.target,
      )

  else:
    new_edgeset_values = graph.edge_sets
    new_edgeset_schemas = schema.edge_sets

  return (
      engine.graph_cls(
          node_sets=new_nodeset_values,  # pyrefly: ignore[bad-argument-type]
          edge_sets=new_edgeset_values,  # pyrefly: ignore[bad-argument-type]
      ),
      schema_lib.GraphSchema(
          node_sets=new_nodeset_schemas,
          edge_sets=new_edgeset_schemas,
      ),
  )


def homogenize(
    graph: Graph,
    schema: schema_lib.GraphSchema,
    homogenized_nodeset_name: str = "nodes",
    homogenized_edgeset_name: str = "edges",
) -> Tuple[Graph, schema_lib.GraphSchema, Dict[str, int]]:
  """Homogenizes a heterogeneous graph into a homogeneous one.

  All nodesets on the input graph must have the same feature schemas. Similarly,
  the all the edgesets of the input graph must have the same feature schemas. If
  this is not the case (e.g., different nodesets have different features), this
  alignements can be done with "ydf.transform.apply_feature`.

  This function can be applied both on InMemoryGraph (NumPy-based) and
  JaxInMemoryGraph (JAX-based). This method is jittable.

  Usage example:

  ```python
  # Read a graph.
  graph, schema = ydf.io.read_graph(...)
  new_graph, new_schema = ydf.transform.homogenize(graph, schema)
  ```

  Args:
    graph: The input `InMemoryGraph`.
    schema: The `GraphSchema` corresponding to the input graph.
    homogenized_nodeset_name: The name of the new homogenized node set.
    homogenized_edgeset_name: The name of the new homogenized edge set.

  Returns:
    A tuple containing:
      - A new `InMemoryGraph` representing the homogenized graph.
      - A new `GraphSchema` for the homogenized graph.
      - A mapping from original nodeset names to the starting index of their
      nodes within the homogenized graph.
  """
  homogenizer = Homogenizer(
      schema=schema,
      homogenized_nodeset_name=homogenized_nodeset_name,
      homogenized_edgeset_name=homogenized_edgeset_name,
  )
  graph, node_offets = homogenizer(graph)
  return graph, homogenizer.output_schema(), node_offets


class Homogenizer:
  """Utility to Homogenizes a heterogeneous graph into a homogeneous one.

  Unlike the `homogenize` function, this class enables the separation of schema
  computation from the actual value transformation.

  See "homogenize" function documentation for details.
  """

  def __init__(
      self,
      schema: schema_lib.GraphSchema,
      homogenized_nodeset_name: str = "nodes",
      homogenized_edgeset_name: str = "edges",
  ):
    self._homogenized_nodeset_name = homogenized_nodeset_name
    self._homogenized_edgeset_name = homogenized_edgeset_name
    self._schema = schema

    # Check that all nodeset schemas are identical.
    if not schema.node_sets:
      raise ValueError(
          "The graph must contain at least one node set to be homogenized."
      )

    it = iter(schema.node_sets.items())
    _, first_nodeset_schema = next(it)
    for nodeset_name, nodeset_schema in it:
      if first_nodeset_schema.features != nodeset_schema.features:
        raise ValueError(
            "All node sets must have the same FeatureSetSchema to be"
            " homogenized. Found differing schemas between"
            f" '{list(schema.node_sets.keys())[0]}' and '{nodeset_name}'."
        )

    # Check that all edgeset schemas are identical.
    if schema.edge_sets:
      it = iter(schema.edge_sets.items())
      _, first_edgeset_schema = next(it)
      for edgeset_name, edgeset_schema in it:
        if first_edgeset_schema.features != edgeset_schema.features:
          raise ValueError(
              "All edge sets must have the same feature schema to be"
              " homogenized. Found differing schemas between"
              f" '{list(schema.edge_sets.keys())[0]}' and '{edgeset_name}'."
          )
    else:
      first_edgeset_schema = None

    new_edgeset_schemas = {}
    if first_edgeset_schema is not None:
      new_edgeset_schemas[homogenized_edgeset_name] = schema_lib.EdgeSchema(
          source=homogenized_nodeset_name,
          target=homogenized_nodeset_name,
          features=first_edgeset_schema.features,
      )

    self._output_schema = schema_lib.GraphSchema(
        node_sets={
            homogenized_nodeset_name: schema_lib.NodeSchema(
                features=first_nodeset_schema.features,
            )
        },
        edge_sets=new_edgeset_schemas,
    )

  def output_schema(self) -> schema_lib.GraphSchema:
    """The homogeneous schema."""
    return self._output_schema

  def __call__(self, graph: Graph) -> Tuple[Graph, Dict[str, int]]:
    """Homogenizes the provided graph based on the precomputed schema.

    Args:
      graph: The input `InMemoryGraph` or `JaxInMemoryGraph` to be homogenized.

    Returns:
      A tuple containing:
        - A new `Graph` instance representing the homogenized graph.
        - A dictionary mapping original node set names to their starting index
          in the concatenated homogenized node set.
    """
    engine = get_engine(graph)

    # Convert nodeset
    new_nodeset_feature_list = defaultdict(list)
    dst_num_nodes = 0
    nodeset_offsets = {}
    for nodeset_name, nodeset_schema in self._schema.node_sets.items():
      nodeset_value = graph.node_sets[nodeset_name]
      nodeset_offsets[nodeset_name] = dst_num_nodes
      assert nodeset_value.num_nodes is not None
      dst_num_nodes += nodeset_value.num_nodes

      for feature_name, feature_schema in nodeset_schema.features.items():
        del feature_schema
        feature_value = nodeset_value.features[feature_name]
        new_nodeset_feature_list[feature_name].append(feature_value)

    # Convert edgeset
    new_adjacency_list = []
    new_edgeset_feature_list = defaultdict(list)
    for edgeset_name, edgeset_schema in self._schema.edge_sets.items():
      edgeset_value = graph.edge_sets[edgeset_name]
      offset = nodeset_offsets[edgeset_schema.source]
      target_offset = nodeset_offsets[edgeset_schema.target]
      adjacency = edgeset_value.adjacency + engine.engine.array(
          [[offset], [target_offset]]
      )
      new_adjacency_list.append(adjacency)
      for feature_name, feature_schema in edgeset_schema.features.items():
        del feature_schema
        feature_value = edgeset_value.features[feature_name]
        new_edgeset_feature_list[feature_name].append(feature_value)

    new_nodeset_featureset = {
        name: engine.engine.concatenate(items, axis=0)
        for name, items in new_nodeset_feature_list.items()
    }
    new_edgeset_featureset = {
        name: engine.engine.concatenate(items, axis=0)
        for name, items in new_edgeset_feature_list.items()
    }
    new_adjacency = engine.engine.concatenate(new_adjacency_list, axis=1)

    new_edgeset_values = {}
    if self._schema.edge_sets:
      new_edgeset_values[self._homogenized_edgeset_name] = engine.edgeset_cls(
          adjacency=new_adjacency,
          features=new_edgeset_featureset,
      )
    return (
        engine.graph_cls(
            node_sets={  # pyrefly: ignore[bad-argument-type]
                self._homogenized_nodeset_name: engine.nodeset_cls(
                    num_nodes=dst_num_nodes,
                    features=new_nodeset_featureset,
                )
            },
            edge_sets=new_edgeset_values,
        ),
        nodeset_offsets,
    )

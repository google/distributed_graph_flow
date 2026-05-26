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

"""Generates synthetic graph samples based on a sampling plan."""

import dataclasses
import random
from typing import Dict, Set, Tuple, Union
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.io import tf_graph_sample
from dgf.src.sampling import config as config_lib
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np
import tqdm


@dataclasses.dataclass
class SyntheticFeatureConfig:
  """Configuration for generating synthetic graph samples.

  Attributes:
    max_integer_value: The maximum absolute value for randomly generated integer
      features.
    max_float_value: The maximum absolute value for randomly generated float
      features.
    min_variable_length_dimention: The minimum size for dimensions marked as
      variable length (None in schema.shape).
    max_variable_length_dimention: The maximum size for dimensions marked as
      variable length (None in schema.shape).
    timeseries_length: The length of generated time series features.
    proba_empty_time_series: The probability of generating an empty time series
      for features with a TIMESERIES semantic and variable length.
  """

  max_integer_value: int = 10
  max_float_value: float = 5.0
  min_variable_length_dimention: int = 10
  max_variable_length_dimention: int = 10
  timeseries_length: int = 20
  proba_empty_time_series: float = 0.1


@dataclasses.dataclass
class SyntheticGraphSampleConfig(SyntheticFeatureConfig):
  """Configuration for generating synthetic graph samples.

  Attributes:
    proba_cycle: The probability of adding a cycle when growing the graph.
  """

  proba_cycle: float = 0.1


@dataclasses.dataclass
class SyntheticGraphConfig(SyntheticFeatureConfig):
  """Configuration for generating synthetic graph."""

  num_nodes: int = 100
  num_edges: int = 100


def generate_synthetic_graph_sample(
    schema: schema_lib.GraphSchema,
    plan: Union[config_lib.SimpleSamplingConfig, config_lib.SamplingPlan],
    config: SyntheticGraphSampleConfig = SyntheticGraphSampleConfig(),
) -> in_memory_graph.InMemoryGraph:
  """Generates a single synthetic graph sample based on a sampling plan.

  Args:
    schema: The graph schema defining node and edge sets.
    plan: The sampling plan (either SimpleSamplingConfig or SamplingPlan)
      describing how to grow the graph.
    config: Configuration for the synthetic graph generation process.

  Returns:
    An InMemoryGraph instance representing the generated synthetic graph.
  """
  if isinstance(plan, config_lib.SimpleSamplingConfig):
    plan = config_lib.simple_sampling_config_to_sampling_plan(plan, schema)

  # Maps each edgeset name to a list of (source, target) node index tuples.
  edgesets: Dict[str, Set[Tuple[int, int]]] = {
      edgeset: set() for edgeset in schema.edge_sets
  }
  # Number of nodes in each nodeset.
  nodesets: Dict[str, int] = {nodeset: 0 for nodeset in schema.node_sets}

  # Grow the edgesets.
  def grow_sample(src_node_idx: int, plan_node: config_lib.PlanNode):
    for child in plan_node.children:

      for _ in range(child.hop_width):
        num_target_nodes = nodesets[child.node.nodeset]
        reuse_node = (
            num_target_nodes > 0 and random.random() < config.proba_cycle
        )
        if reuse_node:
          # Reuse an existing node.
          target_node_idx = random.randint(0, num_target_nodes - 1)
        else:
          # Create a new node
          target_node_idx = num_target_nodes
          nodesets[child.node.nodeset] = target_node_idx + 1

        # add edge
        if child.reversed:
          edge = (target_node_idx, src_node_idx)
        else:
          edge = (src_node_idx, target_node_idx)
        edgesets[child.edgeset].add(edge)

        grow_sample(target_node_idx, child.node)

  # Add a node.
  nodesets[plan.root.nodeset] = 1

  # Start grow.
  grow_sample(0, plan.root)

  # Generate the feature values
  return in_memory_graph.InMemoryGraph(
      node_sets={
          nodeset_name: in_memory_graph.InMemoryNodeSet(
              num_nodes=num_nodes,
              features=gen_featureset_values(
                  num_items=num_nodes,
                  schema=schema.node_sets[nodeset_name].features,
                  config=config,
              ),
          )
          for nodeset_name, num_nodes in nodesets.items()
      },
      edge_sets={
          edgeset_name: in_memory_graph.InMemoryEdgeSet(
              adjacency=np.array(list(edges), dtype=np.int64).T,
              features=gen_featureset_values(
                  num_items=len(edges),
                  schema=schema.edge_sets[edgeset_name].features,
                  config=config,
              ),
          )
          for edgeset_name, edges in edgesets.items()
      },
  )


def generate_synthetic_graph(
    schema: schema_lib.GraphSchema,
    config: SyntheticGraphConfig = SyntheticGraphConfig(),
) -> in_memory_graph.InMemoryGraph:
  """Generates a synthetic graph based on a schema and configuration.

  Args:
    schema: The graph schema defining node and edge sets.
    config: Configuration for the synthetic graph generation, including the
      number of nodes and edges.

  Returns:
    An InMemoryGraph instance representing the generated synthetic graph.
  """
  # TODO(gbm): Create some labels.

  node_sets = {
      nodeset_name: in_memory_graph.InMemoryNodeSet(
          num_nodes=config.num_nodes,
          features=gen_featureset_values(
              num_items=config.num_nodes,
              schema=nodeset_schema.features,
              config=config,
          ),
      )
      for nodeset_name, nodeset_schema in schema.node_sets.items()
  }

  def gen_adjacency(num_edges) -> np.ndarray:
    """Generates a random adjacency matrix of shape [2, num_edges]."""
    return np.random.randint(
        0, config.num_nodes, size=(2, num_edges), dtype=np.int64
    )

  edge_sets = {
      edgeset_name: in_memory_graph.InMemoryEdgeSet(
          adjacency=gen_adjacency(config.num_edges),
          features=gen_featureset_values(
              num_items=config.num_edges,
              schema=edgeset_schema.features,
              config=config,
          ),
      )
      for edgeset_name, edgeset_schema in schema.edge_sets.items()
  }

  graph = in_memory_graph.InMemoryGraph(
      node_sets=node_sets,
      edge_sets=edge_sets,
  )
  in_memory_graph_validate_lib.validate_graph(graph, schema)
  return graph


def gen_featureset_values(
    num_items: int,
    schema: schema_lib.FeatureSetSchema,
    config: SyntheticFeatureConfig,
) -> Dict[str, np.ndarray]:
  return {
      feature_name: gen_feature_values(num_items, feature_schema, config)
      for feature_name, feature_schema in schema.items()
  }


def gen_feature_values(
    num_items: int,
    schema: schema_lib.FeatureSchema,
    config: SyntheticFeatureConfig,
) -> np.ndarray:
  """Generates synthetic feature values based on the schema.

  Args:
    num_items: The number of nodes for which to generate features.
    schema: The FeatureSchema defining the feature format, semantic, and shape.
    config: Configuration for the synthetic graph generation process.

  Returns:
    A numpy array containing the generated feature values.
  """

  is_variable_length = schema.shape is not None and any(
      d is None for d in schema.shape
  )
  dtype = feature_format_lib.FEATURE_FORMAT_TO_NP_DTYPE[schema.format]
  all_features = []
  for item_idx in range(num_items):
    feature_shape = ()
    if schema.shape is not None:
      concrete_shape = []
      for dim in schema.shape:
        if dim is None:
          if schema.semantic == schema_lib.FeatureSemantic.TIMESERIES:
            if random.random() < config.proba_empty_time_series:
              dimention = 0
            else:
              dimention = config.timeseries_length
          else:
            # Generate a random length for variable dimensions.
            dimention = random.randint(
                config.min_variable_length_dimention,
                config.max_variable_length_dimention + 1,
            )
          concrete_shape.append(dimention)
        else:
          concrete_shape.append(dim)
      feature_shape = tuple(concrete_shape)

    # Generate values for a single feature instance with the determined shape
    if schema.format.is_integer():
      if schema.semantic == schema_lib.FeatureSemantic.PRIMARY_ID:
        values = np.array(item_idx)
      elif (
          schema.semantic == schema_lib.FeatureSemantic.CATEGORICAL
          and schema.num_categorical_values is not None
      ):
        values = np.random.randint(
            0, schema.num_categorical_values, size=feature_shape, dtype=dtype
        )
      else:
        values = np.random.randint(
            -config.max_integer_value,
            config.max_integer_value + 1,
            size=feature_shape,
            dtype=dtype,
        )
    elif schema.format.is_float():
      values = np.random.uniform(
          -config.max_float_value,
          config.max_float_value,
          size=feature_shape,
      ).astype(dtype)
      if (
          schema.semantic == schema_lib.FeatureSemantic.EMBEDDING
          and feature_shape
      ):
        # Normalize embeddings along the last dimension.
        norm = np.linalg.norm(values, axis=-1, keepdims=True).astype(dtype)
        # Avoid division by zero.
        norm[norm == 0.0] = 1.0
        values = values / norm
    elif schema.format == schema_lib.FeatureFormat.BYTES:
      if schema.semantic == schema_lib.FeatureSemantic.PRIMARY_ID:
        values = np.array(f"ID_{item_idx}".encode("utf-8"), dtype=dtype)
      else:
        num_unique_values = schema.num_categorical_values or 100
        # Generate a pool of unique byte strings and sample from it.
        pool = [f"CAT_{i}".encode("utf-8") for i in range(num_unique_values)]
        total_elements = np.prod(feature_shape) if feature_shape else 1
        bytes_list = random.choices(pool, k=total_elements)
        values = np.array(bytes_list, dtype=dtype).reshape(feature_shape)
    elif schema.format == schema_lib.FeatureFormat.BOOL:
      values = np.random.randint(0, 2, size=feature_shape, dtype=dtype)
    else:
      raise ValueError(f"Unsupported feature format: {schema.format}")

    all_features.append(values)

  if is_variable_length:
    # Note: Prevents numpy to merge those arrays.
    ret = np.empty(len(all_features), dtype=np.object_)
    ret[:] = all_features
    return ret
  else:
    return np.array(all_features)


def write_synthetic_graph_sample_as_tfgnn_graphs(
    schema: schema_lib.GraphSchema,
    plan: Union[config_lib.SimpleSamplingConfig, config_lib.SamplingPlan],
    path: str,
    num_samples: int,
    config: SyntheticGraphSampleConfig = SyntheticGraphSampleConfig(),
    num_shards: int = 10,
    verbose: bool = True,
    validate: bool = False,
) -> None:
  """Generates and writes synthetic graph samples as TF-GNN graphs.

  Args:
    schema: The graph schema defining node and edge sets.
    plan: The sampling plan (either SimpleSamplingConfig or SamplingPlan)
      describing how to grow the graph.
    path: The base path where the TF-GNN graphs will be written.
    num_samples: The number of synthetic graph samples to generate.
    config: Configuration for the synthetic graph generation process.
    num_shards: The number of shards to use when writing the TF-GNN graphs.
    verbose: If true, print verbose output.
    validate: If true, validate the generated graphs against the schema.
  """

  def generator():
    samples_iter = range(num_samples)
    if verbose:
      samples_iter = tqdm.tqdm(samples_iter, total=num_samples)
    for _ in samples_iter:
      sample = generate_synthetic_graph_sample(
          schema=schema, plan=plan, config=config
      )
      if validate:
        in_memory_graph_validate_lib.validate_graph(sample, schema)
      yield sample

  tf_graph_sample.write_tfgnn_graphs(
      graphs=generator(), path=path, num_shards=num_shards, schema=schema
  )

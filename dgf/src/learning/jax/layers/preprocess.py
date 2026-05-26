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

"""Jax layers for preprocessing graph features before core message passing.

This module provides layers for embedding and transforming graph features,
preparing them for input into graph neural networks.

The main layers are:
  - EmbedFeatureSet: Embeds a set of features (e.g., for a node set) into a
    single dense embedding vector. It handles categorical and pre-embedded
    features.
  - EmbedGraph: Applies EmbedFeatureSet to all node sets within a graph,
    producing a graph with embedded node features.
  - EmbedAndHomogenizeGraph: Embeds features for all node sets and then
    homogenizes the graph structure. This means all node sets are merged into a
    single node set, and all edge sets are merged into a single edge set. This
    is useful for models that expect a homogeneous graph input.


All the layers follow the 3 steps:
  - A config dataclass e.g. config = EmbedFeatureSetConfig(...)
  - A layer class created with layer = config.make()
  - The application of a layer e.g. layer(x)
  - The output schema of the layer e.g. config.output_schema
"""

import dataclasses
from typing import List, Optional
import dataclasses_json
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers.registry import registry as layer_registry
from dgf.src.transform import homogenize as homogenize_lib
import flax.linen as nn
import jax.numpy as jnp

JaxBaseConfig = common.JaxBaseConfig


@dataclasses.dataclass
class EmbedFeatureSetConfig:
  """Configuration for the EmbedFeatureSet layer.

  Attributes:
    categorical_feature_embedding_dim: The dimension of the embedding for
      categorical features.
  """

  categorical_feature_embedding_dim: int = 64

  def make(
      self, schema: schema_lib.FeatureSetSchema, name: Optional[str] = None
  ) -> "EmbedFeatureSet":
    return EmbedFeatureSet(config=self, schema=schema, name=name)

  def output_schema(
      self, schema: schema_lib.FeatureSetSchema
  ) -> Optional[schema_lib.FeatureSchema]:
    num_output_dims = 0
    for _, feature_schema in schema.items():
      if feature_schema.semantic == schema_lib.FeatureSemantic.EMBEDDING:
        shape = feature_schema.shape
        num_output_dims += (
            feature_schema.shape[0]
            if shape is not None and shape is not tuple()
            else 1
        )
      elif feature_schema.semantic == schema_lib.FeatureSemantic.CATEGORICAL:
        num_output_dims += self.categorical_feature_embedding_dim

    if num_output_dims == 0:
      return None
    return schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.EMBEDDING,
        shape=(num_output_dims,),
    )


class EmbedFeatureSet(nn.Module):
  """Computes a fixed sized dense embedding for a set of feature values.

  This module takes a dictionary of features and converts them into a single
  concatenated dense embedding. Categorical features are embedded using
  `nn.Embed`, while pre-embedded features are used directly.

  Attributes:
    config: The configuration for this layer.
    schema: A `schema_lib.FeatureSetSchema` object defining the expected
      features and their semantic types.

  Usage example:

  ```python

  # Define a schema
  feature_schema = {
      "embedding_feature": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.EMBEDDING,
          shape=(16,)
      ),
      "categorical_feature": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.CATEGORICAL,
          num_categorical_values=10,
      ),
  }

  # Instantiate the module
  embedder = EmbedFeatureSetConfig().make(schema=feature_schema)

  # Example input
  input = {
      "embedding_feature": jnp.ones((1, 16), dtype=jnp.float32),
      "categorical_feature": jnp.array([3], dtype=jnp.int64),
  }

  # Initialize and apply the module
  variables = embedder.init(jax.random.PRNGKey(0), features, training=False)
  output = embedder.apply(variables, input, training=False)
  ```
  """

  config: EmbedFeatureSetConfig
  schema: schema_lib.FeatureSetSchema

  @nn.compact
  def __call__(
      self,
      features: jax_in_memory_graph.Features,
      training: bool,
  ) -> Optional[jnp.ndarray]:

    embedding_list = []
    for feature_name in sorted(self.schema.keys()):
      feature_schema = self.schema[feature_name]
      raw_value = features[feature_name]
      if feature_schema.semantic == schema_lib.FeatureSemantic.EMBEDDING:
        if raw_value.dtype != jnp.float32:
          raise TypeError(
              f"Feature {feature_name!r} with EMBEDDING semantic must have"
              f" dtype jnp.float32, but got {raw_value.dtype}."
          )
        if raw_value.ndim == 1:
          raw_value = jnp.expand_dims(raw_value, axis=1)
        # Directly add the value
        embedding_list.append(raw_value)
      elif feature_schema.semantic == schema_lib.FeatureSemantic.CATEGORICAL:
        if raw_value.dtype not in [jnp.int64, jnp.int32]:
          raise TypeError(
              f"Feature {feature_name!r} with CATEGORICAL semantic must have"
              f" dtype jnp.int64 or jnp.int32, but got {raw_value.dtype}."
          )
        if raw_value.ndim != 1:
          raise ValueError(
              f"Feature {feature_name!r} with CATEGORICAL semantic must have"
              f" ndim == 1, but got {raw_value.ndim}."
          )
        # Create an embedding table
        embedding = nn.Embed(
            num_embeddings=feature_schema.num_categorical_values,
            features=self.config.categorical_feature_embedding_dim,
            name=f"embed_{feature_name}",
        )
        embedding_list.append(embedding(raw_value))
      else:
        raise NotImplementedError(
            f"Unsupported feature semantic {feature_schema!r} for feature"
            f" {feature_name!r}"
        )

    # Concatenate the fixed size embeddings.
    if not embedding_list:
      return None
    else:
      return jnp.concatenate(embedding_list, axis=1)


@layer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class EmbedGraphConfig(common.ArchitectureProvider):
  """Configuration for "EmbedGraph".

  Attributes:
    feature_embedder: The configuration for the feature embedder layer.
  """

  feature_embedder: EmbedFeatureSetConfig = dataclasses.field(
      default_factory=EmbedFeatureSetConfig
  )

  def make(
      self, schema: schema_lib.GraphSchema, name: Optional[str] = None
  ) -> "EmbedGraph":
    return EmbedGraph(config=self, schema=schema, name=name)

  def architecture(self) -> str:
    return f"EmbedGraph(cat-embedding={self.feature_embedder.categorical_feature_embedding_dim})"

  def output_schema(
      self, schema: schema_lib.GraphSchema
  ) -> schema_lib.GraphSchema:
    return schema_lib.GraphSchema(
        node_sets={
            nodeset_name: schema_lib.NodeSchema(
                features={
                    "embedding": (
                        self.feature_embedder.output_schema(
                            nodeset_schema.features
                        )
                        or schema_lib.FeatureSchema(
                            format=schema_lib.FeatureFormat.FLOAT_32,
                            semantic=schema_lib.FeatureSemantic.EMBEDDING,
                            shape=(1,),
                        )
                    )
                }
            )
            for nodeset_name, nodeset_schema in schema.node_sets.items()
        },
        edge_sets=schema.edge_sets,
    )


class EmbedGraph(nn.Module):
  """Compute a fixed sized dense embedding for all the features in a graph.

  Unlike "EmbedAndHomogenizeGraph", "EmbedGraph" does not homogenize the nodes.

  Attributes:
    config: The configuration for this layer.
    schema: A `schema_lib.GraphSchema` object defining the expected graph
      structure and features.
  """

  config: EmbedGraphConfig
  schema: schema_lib.GraphSchema

  @nn.compact
  def __call__(
      self,
      graph: jax_in_memory_graph.JaxInMemoryGraph,
      training: bool,
  ) -> jax_in_memory_graph.JaxInMemoryGraph:

    new_nodesets = {}
    for nodeset_name, nodeset_schema in self.schema.node_sets.items():
      feature_embedder = self.config.feature_embedder.make(
          nodeset_schema.features
      )
      embedding = feature_embedder(
          graph.node_sets[nodeset_name].features, training=training
      )
      num_nodes = graph.node_sets[nodeset_name].num_nodes
      if embedding is None:
        # Empty feature set
        embedding = jnp.ones((num_nodes, 1), dtype=jnp.float32)
      new_nodesets[nodeset_name] = jax_in_memory_graph.JaxInMemoryNodeSet(
          num_nodes=num_nodes,
          features={"embedding": embedding},
      )
    return jax_in_memory_graph.JaxInMemoryGraph(
        node_sets=new_nodesets,
        edge_sets=graph.edge_sets,
    )


@dataclasses.dataclass
class EmbedAndHomogenizeGraphConfig:
  """Config for EmbedAndHomogenizeGraph.

  See "EmbedAndHomogenizeGraph" class for details.

  Attributes:
    target_nodeset: The name of the nodeset that contains the seed nodes.
    node_embedding_dim: The dimension of the node embeddings in the output
      homogeneous graph.
    node_type_dim: The dimension of the node type embeddings.
    categorical_feature_embedding_dim: The dimension of the embedding for
      categorical features.
    ignore_target_nodeset_features: A set of feature names to ignore in the
      target nodeset.
    node_embedding: Module applied to each nodeset to convert the feature value
      + type encoding into a consistent shaped embedding across all the
      nodesets.
  """

  target_nodeset: str
  node_embedding_dim: int = 64
  node_type_dim: int = 16
  categorical_feature_embedding_dim: int = 64
  ignore_target_nodeset_features: List[str] = dataclasses.field(
      default_factory=list
  )
  node_embedding: Optional[common.BuildableModule] = layer_registry.field(
      default=None
  )

  def __post_init__(self):
    if self.node_embedding is None:
      self.node_embedding = standard.ingest_feature(
          dims=self.node_embedding_dim
      )

  def make(
      self, schema: schema_lib.GraphSchema, name: Optional[str] = None
  ) -> "EmbedAndHomogenizeGraph":
    return EmbedAndHomogenizeGraph(config=self, schema=schema, name=name)


class EmbedAndHomogenizeGraph(nn.Module):
  """Convert a heterogeneous graph into a homogeneous one.

  This module takes a heterogeneous graph and transforms it into a single
  homogeneous graph structure suitable for models that operate on homogeneous
  graphs.

  Operations:
    - For each nodeset, all its features are first processed into a dense
      embedding using `EmbedFeatureSet`.
    - The combined embedding is then projected to a fixed size
      `node_embedding_dim` using a dense layer.
    - A learned, nodeset-specific type encoding is added to the
      embedding.
    - All nodesets are merged into a single homogeneous nodeset.
    - All edgesets are merged into a single homogeneous edgeset.
    - The input `seed_node_idxs`, which are indices within the `target_nodeset`,
      are mapped to their corresponding indices in the output homogeneous
      nodeset.

  Usage example:

  ```python
    class MyModel(nn.Module):

    body: gnn_lib.MPNN
    head: classification_lib.ClassificationHead
    config: CoreModelConfig

    @nn.compact
    def __call__(self, batch: Batch, training: bool):
      graph, seed_node_idxs = batch
      homogenize_layer = jax_layer.EmbedAndHomogenizeGraph(
        schema=self.schema,
        target_nodeset=self.config.target_nodeset,
        node_embedding_dim=self.config.node_embedding_dim,
      )
      homo_graph, homoe_nodeset_offsets = homogenize_layer(
          graph, seed_node_idxs, training=training
      )
      homo_schema = homogenize_layer.output_schema
      assert homo_schema is not None

      # Example of application: Convert the graph to a SD graph.
      sd_graph = sparse_deferred_lib.jax_graph_to_sparse_deferred_struct(
        homo_graph, homo_schema)
  ```
  """

  config: EmbedAndHomogenizeGraphConfig
  schema: schema_lib.GraphSchema

  output_schema: int = dataclasses.field(init=False)

  def __post_init__(self):

    projected_nodeset_schemas = {}
    for nodeset_name, nodeset_schema in self.schema.node_sets.items():
      projected_nodeset_schemas[nodeset_name] = schema_lib.NodeSchema(
          features={
              "initial_state": schema_lib.FeatureSchema(
                  format=schema_lib.FeatureFormat.FLOAT_32,
                  semantic=schema_lib.FeatureSemantic.EMBEDDING,
                  shape=(self.config.node_embedding_dim,),
              )
          }
      )
    projected_edgeset_schemas = {}
    for edgeset_name, edgeset_schema in self.schema.edge_sets.items():
      # TODO(gbm) Add support for edgeset features.
      projected_edgeset_schemas[edgeset_name] = schema_lib.EdgeSchema(
          source=edgeset_schema.source,
          target=edgeset_schema.target,
          features={},
      )
    projected_schema = schema_lib.GraphSchema(
        node_sets=projected_nodeset_schemas,
        edge_sets=projected_edgeset_schemas,
    )

    self._homogenizer = homogenize_lib.Homogenizer(projected_schema)
    self.output_schema = self._homogenizer.output_schema()
    super().__post_init__()

  @nn.compact
  def __call__(
      self,
      graph: jax_in_memory_graph.JaxInMemoryGraph,
      seed_node_idxs: jnp.ndarray,
      training: bool,
  ):

    nodeset_type_embs = nn.Embed(
        num_embeddings=len(self.schema.node_sets),
        features=self.config.node_type_dim,
        name="node_type_embedding",
    )

    process_nodesets = {}
    for nodeset_idx, nodeset_name in enumerate(
        sorted(self.schema.node_sets.keys())
    ):
      nodeset_schema = self.schema.node_sets[nodeset_name]
      nodeset_value = graph.node_sets[nodeset_name]
      assert nodeset_value.num_nodes is not None

      # Remove ignored features
      featureset_schema = nodeset_schema.features
      if nodeset_name == self.config.target_nodeset:
        # Filter ignored features.
        featureset_schema = {
            k: v
            for k, v in featureset_schema.items()
            if k not in self.config.ignore_target_nodeset_features
        }

      # Project all the feature values into a dense embedding.
      feature_embedder = EmbedFeatureSetConfig(
          categorical_feature_embedding_dim=self.config.categorical_feature_embedding_dim
      ).make(
          schema=featureset_schema,
          name=f"nodeset_feature_embedder_{nodeset_name}",
      )
      feature_embedding = feature_embedder(
          nodeset_value.features, training=training
      )

      # Get the nodeset type encoding embedding.
      nodeset_types = jnp.tile(nodeset_idx, (nodeset_value.num_nodes,))
      nodeset_type_encoding = nodeset_type_embs(nodeset_types)

      if feature_embedding is None:
        # The node has no features, so its initial state is simply the nodeset
        # type embedding.
        node_embedding = nodeset_type_encoding
      else:
        # Concatenate the type and feature embedding.
        node_embedding = jnp.concatenate(
            [feature_embedding, nodeset_type_encoding],
            axis=-1,
        )

      assert self.config.node_embedding is not None
      node_embedding = self.config.node_embedding.make()(node_embedding)

      if node_embedding.shape[-1] != self.config.node_embedding_dim:
        raise ValueError(
            "The output dimension of the node embedding module must be equal"
            f" to node_embedding_dim, but got {node_embedding.shape[-1]} and"
            f" {self.config.node_embedding_dim} respectively."
        )

      process_nodesets[nodeset_name] = jax_in_memory_graph.JaxInMemoryNodeSet(
          features={"initial_state": node_embedding},
          num_nodes=nodeset_value.num_nodes,
      )

    # TODO(gbm): Add support for edgeset features.
    processed_graph = jax_in_memory_graph.JaxInMemoryGraph(
        node_sets=process_nodesets,
        edge_sets=graph.edge_sets,
    )

    homo_graph, homo_nodeset_offsets = self._homogenizer(processed_graph)
    homo_seed_node_idxs = (
        seed_node_idxs + homo_nodeset_offsets[self.config.target_nodeset]
    )
    return homo_graph, homo_seed_node_idxs

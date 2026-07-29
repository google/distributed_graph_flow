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
from typing import List, Optional, Tuple, Union
import dataclasses_json
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers.registry import registry as layer_registry
from dgf.src.transform import homogenize as homogenize_lib
from dgf.src.util import temporal as temporal_util
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

  def output_schema(self, schema: schema_lib.FeatureSetSchema) -> Union[
      Optional[schema_lib.FeatureSchema],
      Tuple[
          Optional[schema_lib.FeatureSchema],
          schema_lib.FeatureSchema,
          Optional[schema_lib.FeatureSchema],
      ],
  ]:
    num_static_dims = 0
    num_timeseries_dims = 0
    has_timeseries = any(
        f.is_timeseries and f.semantic != schema_lib.FeatureSemantic.MASK
        for f in schema.values()
    )

    # Check that all timeseries features have the same sequence length.
    ts_seq_lens = {
        name: f.shape[0]
        for name, f in schema.items()
        if f.is_timeseries and f.shape and f.shape[0] is not None
    }
    unique_seq_lens = set(ts_seq_lens.values())
    if len(unique_seq_lens) > 1:
      raise ValueError(
          "All timeseries features in a feature set must have matching sequence"
          f" lengths in schema, but got conflicting lengths: {ts_seq_lens}."
      )

    ts_seq_len = next(iter(unique_seq_lens), None)
    for feature_schema in schema.values():
      if feature_schema.semantic == schema_lib.FeatureSemantic.MASK:
        continue
      elif feature_schema.semantic == schema_lib.FeatureSemantic.EMBEDDING:
        shape = feature_schema.shape
        if feature_schema.is_timeseries:
          dim = (
              shape[-1]
              if shape is not None and shape is not tuple()
              else 1
          )
          num_timeseries_dims += dim
        else:
          dim = (
              shape[0]
              if shape is not None and shape is not tuple()
              else 1
          )
          num_static_dims += dim
      elif feature_schema.semantic == schema_lib.FeatureSemantic.CATEGORICAL:
        if feature_schema.is_timeseries:
          num_timeseries_dims += self.categorical_feature_embedding_dim
        else:
          num_static_dims += self.categorical_feature_embedding_dim

    static_schema = (
        schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(num_static_dims,),
        )
        if num_static_dims > 0
        else None
    )

    timeseries_schema = (
        schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(ts_seq_len, num_timeseries_dims),
            is_timeseries=True,
        )
        if num_timeseries_dims > 0
        else None
    )

    timeseries_mask_schema = (
        schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BOOL,
            semantic=schema_lib.FeatureSemantic.MASK,
            shape=(ts_seq_len, num_timeseries_dims),
            is_timeseries=True,
        )
        if num_timeseries_dims > 0
        else None
    )

    if has_timeseries:
      if timeseries_schema is None:
        raise ValueError(
            "Schema indicated timeseries features, but no timeseries schema"
            " was generated."
        )
      return (static_schema, timeseries_schema, timeseries_mask_schema)
    return static_schema


class EmbedFeatureSet(nn.Module):
  """Computes a fixed sized dense embedding for a set of feature values.

  This module takes a dictionary of features and converts them into
  concatenated dense embeddings. Static features are embedded into a 2D tensor
  `(batch_size, static_embed_dim)`, and timeseries features are embedded into a
  3D tensor `(batch_size, sequence_length, ts_embed_dim)`.

  Attributes:
    config: The configuration for this layer.
    schema: A `schema_lib.FeatureSetSchema` object defining the expected
      features and their semantic types.

  Usage example:

  ```python

  # Define a schema with static and timeseries features
  feature_schema = {
      "embedding_feature": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.EMBEDDING,
          shape=(16,),
      ),
      "categorical_feature": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.CATEGORICAL,
          num_categorical_values=10,
      ),
      "timeseries_categorical_feature": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.CATEGORICAL,
          num_categorical_values=20,
          shape=(4,),
          is_timeseries=True,
      ),
      "timeseries_embedding_feature": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.EMBEDDING,
          shape=(4, 8),
          is_timeseries=True,
      ),
  }

  # Instantiate the module
  embedder = EmbedFeatureSetConfig().make(schema=feature_schema)

  # Example input
  input_data = {
      "embedding_feature": jnp.ones((1, 16), dtype=jnp.float32),
      "categorical_feature": jnp.array([3], dtype=jnp.int64),
      "timeseries_categorical_feature": jnp.array(
          [[1, 2, 3, 4]], dtype=jnp.int64
      ),
      "timeseries_embedding_feature": jnp.ones((1, 4, 8), dtype=jnp.float32),
  }

  # Initialize and apply the module
  variables = embedder.init(jax.random.PRNGKey(0), input_data, training=False)
  static_output, ts_output = embedder.apply(
      variables, input_data, training=False
  )
  ```
  """

  config: EmbedFeatureSetConfig
  schema: schema_lib.FeatureSetSchema

  @nn.compact
  def __call__(
      self,
      features: jax_in_memory_graph.Features,
      training: bool,
  ) -> Union[
      Optional[jnp.ndarray],
      Tuple[
          Optional[jnp.ndarray],
          jnp.ndarray,
          Optional[jnp.ndarray],
      ],
  ]:

    static_embedding_list = []
    timeseries_embedding_list = []
    timeseries_mask_list = []
    has_timeseries = any(
        f.is_timeseries and f.semantic != schema_lib.FeatureSemantic.MASK
        for f in self.schema.values()
    )

    mask_features = {
        name: features[name]
        for name, sch in self.schema.items()
        if sch.semantic == schema_lib.FeatureSemantic.MASK and name in features
    }

    for feature_name in sorted(self.schema.keys()):
      feature_schema = self.schema[feature_name]
      raw_value = features[feature_name]

      if feature_schema.semantic == schema_lib.FeatureSemantic.MASK:
        continue

      elif feature_schema.semantic == schema_lib.FeatureSemantic.EMBEDDING:
        if raw_value.dtype != jnp.float32:
          raise TypeError(
              f"Feature {feature_name!r} with EMBEDDING semantic must have"
              f" dtype jnp.float32, but got {raw_value.dtype}."
          )
        if feature_schema.is_timeseries:
          if raw_value.ndim == 2:
            raw_value = jnp.expand_dims(raw_value, axis=-1)
          if raw_value.ndim != 3:
            raise ValueError(
                f"Feature {feature_name!r} with EMBEDDING semantic and"
                f" is_timeseries=True must have ndim == 3 (or 2), but got"
                f" {raw_value.ndim}."
            )

          mask_name = temporal_util.get_mask_feature_name(
              feature_name, self.schema
          )
          if mask_name and mask_name in mask_features:
            mask_val = mask_features[mask_name]
            if mask_val.ndim == 2:
              mask_val = jnp.expand_dims(mask_val, axis=-1)
            raw_value = raw_value * mask_val.astype(raw_value.dtype)
            mask_channel = jnp.broadcast_to(
                mask_val.astype(jnp.bool_), raw_value.shape
            )
          else:
            mask_channel = jnp.ones(raw_value.shape, dtype=jnp.bool_)

          timeseries_embedding_list.append(raw_value)
          timeseries_mask_list.append(mask_channel)
        else:
          if raw_value.ndim == 1:
            raw_value = jnp.expand_dims(raw_value, axis=1)
          if raw_value.ndim != 2:
            raise ValueError(
                f"Feature {feature_name!r} with EMBEDDING semantic must have"
                f" ndim == 2 (or 1), but got {raw_value.ndim}."
            )
          static_embedding_list.append(raw_value)

      elif feature_schema.semantic == schema_lib.FeatureSemantic.CATEGORICAL:
        if raw_value.dtype not in [jnp.int64, jnp.int32]:
          raise TypeError(
              f"Feature {feature_name!r} with CATEGORICAL semantic must have"
              f" dtype jnp.int64 or jnp.int32, but got {raw_value.dtype}."
          )
        embedding = nn.Embed(
            num_embeddings=feature_schema.num_categorical_values,  # pyrefly: ignore[bad-argument-type]
            features=self.config.categorical_feature_embedding_dim,
            name=f"embed_{feature_name}",
        )
        if feature_schema.is_timeseries:
          if raw_value.ndim != 2:
            raise ValueError(
                f"Feature {feature_name!r} with CATEGORICAL semantic and"
                f" is_timeseries=True must have ndim == 2, but got"
                f" {raw_value.ndim}."
            )
          emb = embedding(raw_value)
          mask_name = temporal_util.get_mask_feature_name(
              feature_name, self.schema
          )
          if mask_name and mask_name in mask_features:
            mask_val = mask_features[mask_name]
            if mask_val.ndim == 2:
              mask_val = jnp.expand_dims(mask_val, axis=-1)
            emb = emb * mask_val.astype(emb.dtype)
            mask_channel = jnp.broadcast_to(
                mask_val.astype(jnp.bool_), emb.shape
            )
          else:
            mask_channel = jnp.ones(emb.shape, dtype=jnp.bool_)

          timeseries_embedding_list.append(emb)
          timeseries_mask_list.append(mask_channel)
        else:
          if raw_value.ndim != 1:
            raise ValueError(
                f"Feature {feature_name!r} with CATEGORICAL semantic must have"
                f" ndim == 1, but got {raw_value.ndim}."
            )
          static_embedding_list.append(embedding(raw_value))

      else:
        raise NotImplementedError(
            f"Unsupported feature semantic {feature_schema!r} for feature"
            f" {feature_name!r}"
        )

    static_output = (
        jnp.concatenate(static_embedding_list, axis=1)
        if static_embedding_list
        else None
    )
    timeseries_output = (
        jnp.concatenate(timeseries_embedding_list, axis=-1)
        if timeseries_embedding_list
        else None
    )
    timeseries_mask_output = (
        jnp.concatenate(timeseries_mask_list, axis=-1)
        if timeseries_mask_list
        else None
    )

    if has_timeseries:
      if timeseries_output is None:
        raise ValueError(
            "Schema indicated timeseries features, but no timeseries embedding"
            " was generated."
        )
      return (static_output, timeseries_output, timeseries_mask_output)
    else:
      return static_output


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
    node_sets = {}
    for nodeset_name, nodeset_schema in schema.node_sets.items():
      out_schema = self.feature_embedder.output_schema(nodeset_schema.features)
      if isinstance(out_schema, tuple):
        static_s = out_schema[0]
        ts_s = out_schema[1] if len(out_schema) > 1 else None
        ts_mask_s = out_schema[2] if len(out_schema) > 2 else None
        features = {}
        if static_s is not None:
          features["embedding"] = static_s
        else:
          features["embedding"] = schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.FLOAT_32,
              semantic=schema_lib.FeatureSemantic.EMBEDDING,
              shape=(1,),
          )
        if ts_s is not None:
          features["timeseries_embedding"] = ts_s
        if ts_mask_s is not None:
          features["timeseries_mask"] = ts_mask_s
      else:
        features = {
            "embedding": (
                out_schema
                or schema_lib.FeatureSchema(
                    format=schema_lib.FeatureFormat.FLOAT_32,
                    semantic=schema_lib.FeatureSemantic.EMBEDDING,
                    shape=(1,),
                )
            )
        }
      node_sets[nodeset_name] = schema_lib.NodeSchema(features=features)
    return schema_lib.GraphSchema(
        node_sets=node_sets,
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
      if isinstance(embedding, tuple):
        static_emb = embedding[0]
        ts_emb = embedding[1] if len(embedding) > 1 else None
        ts_mask = embedding[2] if len(embedding) > 2 else None
        node_features = {}
        if static_emb is not None:
          node_features["embedding"] = static_emb
        else:
          node_features["embedding"] = jnp.ones(
              (num_nodes, 1), dtype=jnp.float32
          )
        if ts_emb is not None:
          node_features["timeseries_embedding"] = ts_emb
        if ts_mask is not None:
          node_features["timeseries_mask"] = ts_mask
      else:
        if embedding is None:
          embedding = jnp.ones((num_nodes, 1), dtype=jnp.float32)
        node_features = {"embedding": embedding}
      new_nodesets[nodeset_name] = jax_in_memory_graph.JaxInMemoryNodeSet(
          num_nodes=num_nodes,
          features=node_features,
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
      embedding using `EmbedFeatureSet` (note: timeseries sequence features are
      ignored/dropped; only static features are homogenized).
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
    self.output_schema = self._homogenizer.output_schema()  # pyrefly: ignore[bad-assignment]
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

      if isinstance(feature_embedding, tuple):
        feature_embedding = feature_embedding[0]

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

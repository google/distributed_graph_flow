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
  - EmbedFeatureSet: Embeds a set of static features for a node or edge set into
    a single dense embedding vector. It handles categorical and pre-embedded
    features while ignoring timeseries features.
  - EmbedFeatureGroups: Embeds static features and timeseries sequence groups,
    producing named embedding and mask feature arrays alongside their schemas.
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

import collections
import dataclasses
from typing import List, Optional
import dataclasses_json
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax import common
from dgf.src.learning.jax.layers import standard
from dgf.src.learning.jax.layers.registry import registry as layer_registry
from dgf.src.transform import homogenize as homogenize_lib
from dgf.src.util import log
import flax.linen as nn
import jax.numpy as jnp

JaxBaseConfig = common.JaxBaseConfig


def _is_non_mask_timeseries(
    feature_schema: schema_lib.FeatureSchema,
) -> bool:
  """Returns True if the feature is a timeseries sequence (excluding masks)."""
  return (
      feature_schema.is_timeseries
      and feature_schema.semantic != schema_lib.FeatureSemantic.MASK
  )


def _has_defined_timeseries_dimension(
    shape: Optional[schema_lib.Shape],
) -> bool:
  """Returns True if shape is non-empty and has a defined sequence length (shape[0] is not None)."""
  return shape is not None and len(shape) > 0 and shape[0] is not None


@dataclasses.dataclass
class EmbedFeatureGroupsConfig:
  """Configuration for the EmbedFeatureGroups layer.

  Attributes:
    categorical_feature_embedding_dim: The dimension of the embedding for
      categorical features.
  """

  categorical_feature_embedding_dim: int = 64

  def make(
      self, schema: schema_lib.FeatureSetSchema, name: Optional[str] = None
  ) -> "EmbedFeatureGroups":
    return EmbedFeatureGroups(config=self, schema=schema, name=name)

  def output_schema(
      self, schema: schema_lib.FeatureSetSchema
  ) -> schema_lib.FeatureSetSchema:
    """Computes output schemas for static features and timeseries groups."""
    output_schemas = {}

    # 1. Static features
    num_static_dims = 0
    for feature_name, feature_schema in schema.items():
      if feature_schema.is_timeseries:
        continue
      if feature_schema.semantic == schema_lib.FeatureSemantic.EMBEDDING:
        shape = feature_schema.shape
        if shape is not None and len(shape) > 0 and shape[0] is not None:
          num_static_dims += shape[0]
        else:
          num_static_dims += 1
      elif feature_schema.semantic == schema_lib.FeatureSemantic.CATEGORICAL:
        num_static_dims += self.categorical_feature_embedding_dim
      else:
        log.warning(
            f"Feature {feature_name!r} with semantic {feature_schema.semantic}"
            " is unexpected/unsupported for embedding and will be ignored in"
            " output schema calculation."
        )

    if num_static_dims > 0:
      output_schemas["embedding"] = schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.EMBEDDING,
          shape=(num_static_dims,),
      )

    # 2. Timeseries features grouped by group
    ts_groups = collections.defaultdict(list)
    for feature_name, feature_schema in schema.items():
      if _is_non_mask_timeseries(feature_schema):
        grp = (
            feature_schema.group
            if feature_schema.group is not None
            else feature_name
        )
        ts_groups[grp].append((feature_name, feature_schema))

    for grp_name in sorted(ts_groups.keys()):
      group_features = ts_groups[grp_name]
      seq_len = None
      total_dim = 0

      # Features in the sequence group may have different channel dimensions
      # (e.g. [T, 2] and [T, 1] or univariate [T]). They are merged by
      # concatenating along the last axis, yielding total dimension [T, sum(D)].
      for feature_name, feature_schema in group_features:
        shape = feature_schema.shape
        # Timeseries features must have a non-empty shape with a defined
        # sequence length at shape[0].
        if not _has_defined_timeseries_dimension(shape):
          raise ValueError(
              f"Timeseries feature '{feature_name}' in sequence group"
              f" '{grp_name}' must have a defined sequence length (shape[0]"
              f" cannot be None), but got shape={shape}."
          )
        if len(shape) > 2:
          raise ValueError(
              f"Timeseries feature '{feature_name}' in sequence group"
              f" '{grp_name}' has unsupported shape={shape} with {len(shape)}"
              " dimensions. Only 1D (sequence length only, e.g. [T]) or 2D"
              " (sequence length and feature dimension, e.g. [T, D]) shapes are"
              " supported."
          )

        # Validate matching sequence length across all features in the group.
        if seq_len is None:
          seq_len = shape[0]
        elif shape[0] != seq_len:
          raise ValueError(
              f"All timeseries features in sequence group '{grp_name}' must"
              " have matching sequence lengths, but got conflicting lengths:"
              f" feature '{feature_name}' has length {shape[0]}, expected"
              f" {seq_len}."
          )

        if feature_schema.semantic == schema_lib.FeatureSemantic.EMBEDDING:
          if len(shape) == 2:
            if shape[1] is None or shape[1] <= 0:
              raise ValueError(
                  f"Timeseries feature '{feature_name}' in sequence group"
                  f" '{grp_name}' has invalid feature dimension shape[1]="
                  f"{shape[1]}. Must be a positive integer."
              )
            total_dim += shape[1]
          else:
            total_dim += 1
        elif feature_schema.semantic == schema_lib.FeatureSemantic.CATEGORICAL:
          if len(shape) > 1:
            raise ValueError(
                f"Categorical timeseries feature '{feature_name}' in sequence"
                f" group '{grp_name}' must be 1D with shape (sequence_length,),"
                f" but got shape={shape}."
            )
          total_dim += self.categorical_feature_embedding_dim
        else:
          log.warning(
              f"Timeseries feature {feature_name!r} in sequence group"
              f" {grp_name!r} with semantic {feature_schema.semantic} is"
              " unexpected/unsupported for embedding and will be ignored in"
              " output schema calculation."
          )

      if total_dim > 0:
        output_schemas[f"embedding_{grp_name}"] = schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(seq_len, total_dim),
            is_timeseries=True,
            group=grp_name,
        )
        output_schemas[f"mask_{grp_name}"] = schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BOOL,
            semantic=schema_lib.FeatureSemantic.MASK,
            shape=(seq_len,),
            is_timeseries=True,
            group=grp_name,
        )

    return output_schemas


class EmbedFeatureGroups(nn.Module):
  """Computes embeddings for static features and timeseries sequence groups.

  For static features:
    Embeds categorical features and direct numeric embeddings into a single
    2D dense embedding `embedding` of shape `(batch_size, static_embed_dim)`.

  For each timeseries group `g`:
    1. Embeds categorical and numeric features, concatenating along axis=-1
       into `embedding_{g}` of shape `(batch_size, seq_len, total_embed_dim)`.
       All features in the group must share the same sequence length `T`.
       Features may have different channel dimensions (e.g., shape `(T, 2)` and
       `(T, 1)` or univariate `(T,)`), which are concatenated along axis=-1
       to produce `(batch_size, T, total_embed_dim)`.
    2. Extracts or generates a boolean mask `mask_{g}` of shape
       `(batch_size, seq_len)`.
    3. Applies masking to zero-out invalid timesteps in `embedding_{g}`.

  Attributes:
    config: The configuration for this layer.
    schema: A `schema_lib.FeatureSetSchema` object defining the expected
      features and their semantic types.

  Usage example:

  ```python
  # Define a schema with static features, timeseries features, and masks
  feature_schema = {
      "static_cat": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.CATEGORICAL,
          num_categorical_values=10,
      ),
      "static_embed": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.EMBEDDING,
          shape=(16,),
      ),
      "events_cat": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_32,
          semantic=schema_lib.FeatureSemantic.CATEGORICAL,
          num_categorical_values=50,
          shape=(8,),
          is_timeseries=True,
          group="events",
      ),
      "events_embed": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.EMBEDDING,
          shape=(8, 4),
          is_timeseries=True,
          group="events",
      ),
      "events_mask": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BOOL,
          semantic=schema_lib.FeatureSemantic.MASK,
          shape=(8,),
          is_timeseries=True,
          group="events",
      ),
  }

  # Instantiate the module via config
  config = EmbedFeatureGroupsConfig(categorical_feature_embedding_dim=32)
  embedder = config.make(schema=feature_schema)

  # Example input data for batch_size=2, seq_len=8
  input_data = {
      "static_cat": jnp.array([1, 4], dtype=jnp.int64),
      "static_embed": jnp.ones((2, 16), dtype=jnp.float32),
      "events_cat": jnp.ones((2, 8), dtype=jnp.int32),
      "events_embed": jnp.ones((2, 8, 4), dtype=jnp.float32),
      "events_mask": jnp.ones((2, 8), dtype=jnp.bool_),
  }

  # Initialize and apply the module
  variables = embedder.init(jax.random.PRNGKey(0), input_data, training=False)
  outputs = embedder.apply(variables, input_data, training=False)

  # outputs contains:
  # - "embedding": shape (2, 32 + 16) = (2, 48)
  # - "embedding_events": shape (2, 8, 32 + 4) = (2, 8, 36)
  # - "mask_events": shape (2, 8)
  ```
  """

  config: EmbedFeatureGroupsConfig
  schema: schema_lib.FeatureSetSchema

  @nn.compact
  def __call__(
      self,
      features: jax_in_memory_graph.Features,
      training: bool,
  ) -> jax_in_memory_graph.Features:
    outputs: jax_in_memory_graph.Features = {}

    # 1. Process static features
    static_embedding_list = []
    for feature_name in sorted(self.schema.keys()):
      feature_schema = self.schema[feature_name]
      if (
          feature_schema.is_timeseries
          or feature_schema.semantic == schema_lib.FeatureSemantic.MASK
      ):
        continue
      raw_value = features[feature_name]
      if feature_schema.semantic == schema_lib.FeatureSemantic.EMBEDDING:
        if raw_value.dtype != jnp.float32:
          raise TypeError(
              f"Feature {feature_name!r} with EMBEDDING semantic must have"
              f" dtype jnp.float32, but got {raw_value.dtype}."
          )
        if raw_value.ndim == 1:
          raw_value = jnp.expand_dims(raw_value, axis=1)
        if raw_value.ndim != 2:
          raise ValueError(
              f"Feature {feature_name!r} with EMBEDDING semantic must have"
              f" ndim 1 or 2, but got {raw_value.ndim}."
          )
        static_embedding_list.append(raw_value)
      elif feature_schema.semantic == schema_lib.FeatureSemantic.CATEGORICAL:
        if feature_schema.num_categorical_values is None:
          raise ValueError(
              f"Feature {feature_name!r} with CATEGORICAL semantic must have"
              " num_categorical_values specified in its schema."
          )
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
        embedding = nn.Embed(
            num_embeddings=feature_schema.num_categorical_values,
            features=self.config.categorical_feature_embedding_dim,
            name=f"embed_{feature_name}",
        )
        static_embedding_list.append(embedding(raw_value))
      else:
        log.warning(
            f"Feature {feature_name!r} with semantic"
            f" {feature_schema.semantic} is unexpected/unsupported for"
            " embedding and will be ignored."
        )

    if static_embedding_list:
      outputs["embedding"] = jnp.concatenate(static_embedding_list, axis=1)

    # 2. Process timeseries features per sequence group
    ts_groups = collections.defaultdict(list)
    mask_features = {}
    for feature_name, feature_schema in self.schema.items():
      grp = (
          feature_schema.group
          if feature_schema.group is not None
          else feature_name
      )
      if feature_schema.semantic == schema_lib.FeatureSemantic.MASK:
        mask_features[grp] = feature_name
      elif _is_non_mask_timeseries(feature_schema):
        ts_groups[grp].append(feature_name)

    for grp_name in sorted(ts_groups.keys()):
      feature_names = sorted(ts_groups[grp_name])
      group_embedding_list = []

      for feature_name in feature_names:
        feature_schema = self.schema[feature_name]
        raw_value = features[feature_name]

        if feature_schema.semantic == schema_lib.FeatureSemantic.EMBEDDING:
          if raw_value.dtype != jnp.float32:
            raise TypeError(
                f"Feature {feature_name!r} with EMBEDDING semantic must have"
                f" dtype jnp.float32, but got {raw_value.dtype}."
            )
          if raw_value.ndim == 2:
            raw_value = jnp.expand_dims(raw_value, axis=-1)
          if raw_value.ndim != 3:
            raise ValueError(
                f"Feature {feature_name!r} with EMBEDDING semantic and"
                f" is_timeseries=True must have ndim 2 or 3, but got"
                f" {raw_value.ndim}."
            )
          group_embedding_list.append(raw_value)

        elif feature_schema.semantic == schema_lib.FeatureSemantic.CATEGORICAL:
          if feature_schema.num_categorical_values is None:
            raise ValueError(
                f"Feature {feature_name!r} with CATEGORICAL semantic and"
                " is_timeseries=True must have num_categorical_values specified"
                " in its schema."
            )
          if raw_value.dtype not in [jnp.int64, jnp.int32]:
            raise TypeError(
                f"Feature {feature_name!r} with CATEGORICAL semantic must have"
                f" dtype jnp.int64 or jnp.int32, but got {raw_value.dtype}."
            )
          if raw_value.ndim != 2:
            raise ValueError(
                f"Feature {feature_name!r} with CATEGORICAL semantic and"
                f" is_timeseries=True must have ndim == 2, but got"
                f" {raw_value.ndim}."
            )
          embedding = nn.Embed(
              num_embeddings=feature_schema.num_categorical_values,
              features=self.config.categorical_feature_embedding_dim,
              name=f"embed_{feature_name}",
          )
          group_embedding_list.append(embedding(raw_value))
        else:
          log.warning(
              f"Timeseries feature {feature_name!r} in sequence group"
              f" {grp_name!r} with semantic {feature_schema.semantic} is"
              " unexpected/unsupported for embedding and will be ignored."
          )

      if group_embedding_list:
        ts_embedding = jnp.concatenate(group_embedding_list, axis=-1)
        # Find or generate mask for this group
        mask_name = mask_features.get(grp_name)

        if mask_name is not None and mask_name in features:
          mask = features[mask_name]
          if mask.dtype != jnp.bool_:
            mask = mask.astype(jnp.bool_)
        else:
          mask = jnp.ones(ts_embedding.shape[:2], dtype=jnp.bool_)

        # Zero-out masked timesteps in the embedding
        ts_embedding = jnp.where(
            jnp.expand_dims(mask, axis=-1), ts_embedding, 0.0
        )

        outputs[f"embedding_{grp_name}"] = ts_embedding
        outputs[f"mask_{grp_name}"] = mask

    return outputs


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
    for feature_name, feature_schema in schema.items():
      if feature_schema.is_timeseries:
        log.warning(
            f"Timeseries feature {feature_name!r} is ignored in output schema"
            " calculation for EmbedFeatureSet."
        )
    static_schema = {
        feature_name: feature_schema
        for feature_name, feature_schema in schema.items()
        if not feature_schema.is_timeseries
        and feature_schema.semantic != schema_lib.FeatureSemantic.MASK
    }
    groups_config = EmbedFeatureGroupsConfig(
        categorical_feature_embedding_dim=self.categorical_feature_embedding_dim
    )
    return groups_config.output_schema(static_schema).get("embedding", None)


class EmbedFeatureSet(nn.Module):
  """Computes a fixed sized dense embedding for static feature values.

  This module delegates to `EmbedFeatureGroups` to embed static features into
  a single concatenated dense embedding of shape `(batch_size, static_dim)`,
  while ignoring any timeseries sequence features and masks.

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
          shape=(16,),
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
  variables = embedder.init(jax.random.PRNGKey(0), input, training=False)
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
    for feature_name, feature_schema in self.schema.items():
      if feature_schema.is_timeseries:
        log.warning(
            f"Timeseries feature {feature_name!r} is ignored in"
            " EmbedFeatureSet."
        )
    static_schema = {
        feature_name: feature_schema
        for feature_name, feature_schema in self.schema.items()
        if not feature_schema.is_timeseries
        and feature_schema.semantic != schema_lib.FeatureSemantic.MASK
    }
    groups_embedder = EmbedFeatureGroupsConfig(
        categorical_feature_embedding_dim=self.config.categorical_feature_embedding_dim
    ).make(schema=static_schema)
    outputs = groups_embedder(features, training=training)
    return outputs.get("embedding", None)


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
      embedding using `EmbedFeatureSet` (note: timeseries sequence features are
      ignored; only static features are homogenized).
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
            feature_name: feature_schema
            for feature_name, feature_schema in featureset_schema.items()
            if feature_name not in self.config.ignore_target_nodeset_features
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

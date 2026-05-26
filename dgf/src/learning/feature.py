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

"""A set of utilities to feed normalized features into a GNN model."""

from typing import Dict, Sequence, Tuple
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from flax import linen as nn
import jax
import jax.numpy as jnp


class EmbedNodesetFeaturesModule(nn.Module):
  """A FLAX module to transform a set of features into a fixed-size embedding.

  This module can be used before the message passing stage.

  This method is best applied on normalized features value computed with
  `dgf.transform.auto_normalize`. It only supports and applies the following
  transformations depending on the semantic+format of the feature:
    - semantic=EMBEDDING format=float32 => Identity
    - semantic=CATEGORICAL format=int64 => Create embedding table

  Usage example:
  ```python
  # Your GNN model
  class GNNModel(nn.Module):
    schema: dgf.data.GraphSchema
    @nn.compact
    def __call__(
        self,
        graph: dgf.data.JaxInMemoryGraph,
        training: bool,
        seed_node_idxs: jax.Array,
    ) -> jax.Array:
      # Ingest the features
      node_embeddings =
      EmbedNodesetFeaturesModule(
            schema=schema,ignore_features=[("target_nodeset","label")])(
                graph,training=training)

      # Message passing
      for _ in range(num_rounds):
        node_embeddings = message_passing(node_embeddings, ...)

      # Result
      return node_embeddings["target_nodeset"]
  ```

  If using Sparse Deferred (or another JAX GNN library), you can combine this
  method with `dgf.convert.sparse_deferred_struct_to_graph`.

  Attributes:
    schema: The graph schema.
    ignore_features: A sequence of `(nodeset_name, feature_name)` tuples
      specifying features to ignore during ingestion.
    categorical_feature_embedding_dim: The embedding dimension used for
      categorical features.
  """

  schema: schema_lib.GraphSchema
  ignore_features: Sequence[Tuple[str, str]] = ()
  categorical_feature_embedding_dim: int = 64

  @nn.compact
  def __call__(
      self, graph: jax_in_memory_graph.JaxInMemoryGraph, training: bool
  ) -> Dict[str, jax.Array]:

    # The embedding value for all the nodesets
    node_embeddings: Dict[str, jax.Array] = {}

    # Compute a fixed size embedding for all the nodesets.
    for nodeset_name, nodeset_schema in self.schema.node_sets.items():
      embedding_values = []
      for feature_name, feature_schema in nodeset_schema.features.items():
        if (nodeset_name, feature_name) in self.ignore_features:
          continue
        raw_value = graph.node_sets[nodeset_name].features[feature_name]
        processed_value = self._process_single_feature(
            nodeset_name, feature_name, feature_schema, raw_value
        )
        embedding_values.append(processed_value)

      # Concatenate the fixed size embeddings.
      if not embedding_values:
        raise ValueError(
            f"No features found for nodeset '{nodeset_name}' to embed."
            " Ensure the schema is correct or adjust `ignore_features`."
        )

      node_embeddings[nodeset_name] = jnp.concatenate(embedding_values, axis=1)

    return node_embeddings

  def _process_single_feature(
      self,
      nodeset_name: str,
      feature_name: str,
      feature_schema: schema_lib.FeatureSchema,
      raw_value: jax.Array,
  ) -> jax.Array:
    """Processes a single feature based on its semantic."""
    if feature_schema.semantic == schema_lib.FeatureSemantic.EMBEDDING:
      if feature_schema.format != schema_lib.FeatureFormat.FLOAT_32:
        raise ValueError(
            f"Embedding feature '{feature_name}' in nodeset"
            f" '{nodeset_name}' has unexpected dtype {feature_schema.format}."
            " Embedding is expected to be of dtype float32."
        )
      if raw_value.ndim == 1:
        raw_value = jnp.expand_dims(raw_value, axis=1)
      return raw_value
    elif feature_schema.semantic == schema_lib.FeatureSemantic.CATEGORICAL:
      if feature_schema.format != schema_lib.FeatureFormat.INTEGER_64:
        raise ValueError(
            f"Categorical feature '{feature_name}' in nodeset"
            f" '{nodeset_name}' has unexpected dtype {feature_schema.format}"
            ". Categorical is expected to be of dtype int64."
        )
      if raw_value.ndim == 1:
        raw_value = jnp.expand_dims(raw_value, axis=1)
      # Create an embedding table.
      embedding = nn.Embed(
          num_embeddings=feature_schema.num_categorical_values,
          features=self.categorical_feature_embedding_dim,
          name=f"embed_{nodeset_name}_{feature_name}",
      )
      processed_value = embedding(raw_value)
      processed_value = processed_value.reshape(processed_value.shape[0], -1)
      return processed_value
    else:
      raise NotImplementedError(
          f"Unsupported feature semantic {feature_schema.semantic!r} for"
          f" feature '{feature_name}' in nodeset '{nodeset_name}'"
      )

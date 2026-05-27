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

"""Normalization of feature values for GNN models."""

import abc
import copy
import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple
import dataclasses_json
from dgf.src.data import in_memory_graph
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
from dgf.src.data import tf_in_memory_graph
from dgf.src.util import dataclass_registry
from dgf.src.util import log
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

normalizer_registry = dataclass_registry.create_registry("normalizers")


@dataclasses.dataclass
class AbstractFeatureNormalizer(abc.ABC):
  """Abstract base class for normalizing model input features.

  Usage example:

  ```python
  input_schema: statistics_lib.FeatureSetSchema = ...
  input_stats: statistics_lib.FeatureStatistics = ...

  # Create a normalizer (assume `MyNormalizer` derives
  # `AbstractFeatureNormalizer`).
  normalizer = MyNormalizer("f1", stats.nodesets["n1"].features["f1"])
  # Or, if the normalizer has a factory constructor.
  normalizer = MyNormalizer.create(...)

  # Normalize feature stored as numpy array
  raw_value = np.array(...)
  normalized_value = normalizer.normalize_numpy(raw_value)

  # Get the schema of the normalizer output.
  normalized_schema = normalizer.output_schema()
  ```

  You can apply normalizers (i.e., classes that derive from
  AbstractFeatureNormalizer) manually, or you can call `auto_normalize` to
  automatically apply the feature normalizers on a graph.

  Attributes:
    input_feature: The name of the input feature this normalizer operates on.
    type: The name of the normalizer class.
  """

  input_feature: str
  type: str

  @abc.abstractmethod
  def output_schema(self) -> schema_lib.FeatureSetSchema:
    """Returns the schema of the normalized feature."""
    pass

  @abc.abstractmethod
  def normalize_numpy(self, value: np.ndarray) -> Dict[str, np.ndarray]:
    """Applies the normalization to a numpy array of feature values."""
    pass

  @abc.abstractmethod
  def normalize_tensorflow(self, value: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Applies the normalization to a dictionary of TensorFlow tensors."""
    pass

  def tensorflow_resources(self) -> List[tf.Tensor]:
    """Returns the list of TensorFlow resources used by `normalize_tensorflow`.

    Access to resources is necessary to serialize a `tf.Module`. If the
    normalization operation uses resources, `normalize_tensorflow` is expected
    to call this method. Calling this method might initialize the TensorFlow
    resources if they have not been created already.

    Returns:
      A list of TensorFlow resource tensors.
    """
    return []


@normalizer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class IdentityNormalizer(AbstractFeatureNormalizer):
  """A normalizer that simply pass a feature without changing it."""

  input_schema: schema_lib.FeatureSchema
  type: str = dataclasses.field(default="IdentityNormalizer", init=False)

  def output_schema(self) -> schema_lib.FeatureSetSchema:
    return {self.input_feature: self.input_schema}

  def normalize_numpy(self, value: np.ndarray) -> Dict[str, np.ndarray]:
    return {self.input_feature: value}

  def normalize_tensorflow(self, value: tf.Tensor) -> Dict[str, tf.Tensor]:
    return {self.input_feature: value}


@normalizer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class DictionaryIndexNormalizer(AbstractFeatureNormalizer):
  """Normalizes features by mapping dictionary keys to their integer indices.

  This normalizer is suitable for categorical features where a dictionary of
  unique values and their assigned indices is available in the statistics.

  If the feature stats indexing is a dense [0, num_items) indexing (which should
  be the case), the output will be in [0, num_items+1).

  This operation is the same as the "compression mapping" in GraphAI.
  """

  # TODO(gbm): Renam to LookupTable.
  # TODO(gbm): Check that the indexing is in a dense [0, num_items).

  dictionary_map: Dict[str, int]
  out_of_vocab_value: int
  output_shape: Tuple[Optional[int], ...]
  output_feature_name: str
  type: str = dataclasses.field(default="DictionaryIndexNormalizer", init=False)
  tf_table: Any = dataclasses.field(
      default=None,
      init=False,
      metadata=dataclasses_json.config(exclude=dataclasses_json.Exclude.ALWAYS),
  )

  @classmethod
  def create(
      cls,
      feature_name: str,
      input_schema: schema_lib.FeatureSchema,
      input_stats: statistics_lib.FeatureStatistics,
  ) -> "DictionaryIndexNormalizer":
    if input_schema.format != schema_lib.FeatureFormat.BYTES:
      raise ValueError(
          f"Feature '{feature_name}' has format '{input_schema.format}', but "
          "DictionaryIndexNormalizer only supports BYTES features."
      )
    if not input_stats.dictionary:
      raise ValueError(
          f"Feature '{feature_name}' does not have a dictionary in its"
          " statistics."
      )
    return DictionaryIndexNormalizer(
        input_feature=feature_name,
        dictionary_map={
            key: item.index for key, item in input_stats.dictionary.items()
        },
        out_of_vocab_value=len(input_stats.dictionary),
        output_shape=input_schema.shape or (),
        output_feature_name=f"{feature_name}_INDEX",
    )

  def output_schema(self) -> schema_lib.FeatureSetSchema:
    return {
        self.output_feature_name: schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            shape=self.output_shape,
            num_categorical_values=len(self.dictionary_map) + 1,
        )
    }

  def normalize_numpy(self, value: np.ndarray) -> Dict[str, np.ndarray]:
    # TODO(gbm): Would sorting the key + np.searchsorted be much faster?
    def _lookup(v):
      return self.dictionary_map.get(
          v.decode("utf-8", "surrogateescape"),
          self.out_of_vocab_value,
      )

    vectorized_lookup = np.vectorize(_lookup)
    # TODO(gbm): Parametrize to int32.
    return {self.output_feature_name: vectorized_lookup(value)}

  def tensorflow_resources(self) -> List[tf.Tensor]:
    if self.tf_table is None:
      keys = list(self.dictionary_map.keys())
      values = list(self.dictionary_map.values())
      initializer = tf.lookup.KeyValueTensorInitializer(
          keys=keys,
          values=values,
          key_dtype=tf.string,
          value_dtype=tf.int64,
      )
      self.tf_table = tf.lookup.StaticHashTable(
          initializer,
          default_value=tf.constant(self.out_of_vocab_value, dtype=tf.int64),
      )
    return [self.tf_table]

  def normalize_tensorflow(self, value: tf.Tensor) -> Dict[str, tf.Tensor]:
    self.tensorflow_resources()
    return {self.output_feature_name: self.tf_table.lookup(value)}


@normalizer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class SoftQuantileNormalizer(AbstractFeatureNormalizer):
  """Normalizes a numerical feature by replacing it with its soft quantile -0.5.

  Soft quantile are different from regular quantiles are follow:
    - When falling in between two quantiles, the value is linearly interpolated.
    - When falling outside of the quantile ranges (e.g., smaller that the first
    quantile, or larget that the last quantile), the value is extrapolated.

  This normalizer is suitable for numerical features where the quantiles are
  available in the statistics. The output is a float between -0.5-eps and
  0.5+eps, where eps is the extrapolation (generally ~ 1/num quantiles).
  """

  output_feature_name: str
  output_shape: Tuple[Optional[int], ...]
  quantiles: np.ndarray = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda x: x.tolist(),
          decoder=lambda x: np.asarray(x, dtype=np.float32),
      )
  )
  type: str = dataclasses.field(default="SoftQuantileNormalizer", init=False)

  @classmethod
  def create(
      cls,
      feature_name: str,
      input_schema: schema_lib.FeatureSchema,
      input_stats: statistics_lib.FeatureStatistics,
  ) -> "SoftQuantileNormalizer":
    if (
        not input_schema.format.is_integer()
        and not input_schema.format.is_float()
    ):
      raise ValueError(
          f"Feature '{feature_name}' has format '{input_schema.format}', but "
          "SoftQuantileNormalizer only supports INTEGER or FLOAT features."
      )
    if not input_stats.quantiles:
      raise ValueError(
          f"Feature '{feature_name}' does not have quantiles in its statistics."
      )
    if len(input_stats.quantiles) < 2:
      raise ValueError(
          f"Feature '{feature_name}' has less than 2 quantiles in its"
          " statistics, cannot perform soft quantile normalization."
      )

    quantiles = np.unique(np.array(input_stats.quantiles, dtype=np.float32))
    if len(quantiles) == 1:
      quantiles = np.append(quantiles, quantiles[0] + 1)

    return SoftQuantileNormalizer(
        input_feature=feature_name,
        quantiles=quantiles,
        output_shape=input_schema.shape or (),
        output_feature_name=f"{feature_name}_SOFT_QUANTILE",
    )

  def output_schema(self) -> schema_lib.FeatureSetSchema:
    return {
        self.output_feature_name: schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=self.output_shape,
        )
    }

  def normalize_numpy(self, value: np.ndarray) -> Dict[str, np.ndarray]:
    # TODO(gbm): Add support for multi-dim features.

    value = value.astype(np.float32)
    quantiles = self.quantiles
    num_buckets = len(quantiles) - 1

    # Handle NaN values
    nan_mask = np.isnan(value)

    # Find the index of the quantile bucket for each value.
    # indices will range from 0 to len(quantiles)-2.
    # `value` is expected to be in
    # [quantiles[bucket_idx], quantiles[bucket_idx+1]).
    bucket_idx = np.searchsorted(quantiles, value, side="right") - 1
    bucket_idx = np.clip(bucket_idx, 0, len(quantiles) - 2)

    # The `smooth_bucket_idx` similar to `bucket_idx`, but with the decimal part
    # indicating where in the bucket we are located. The interger part of
    # `smooth_bucket_idx` is always equal to `bucket_idx`.
    #
    # Example:
    #     quantiles = [0.0, 3.0, 5.0, 102.0, 202.5, 303.0]
    #     value = [4.0]
    #     # We have
    #     bucket_idx = 1 # since 4 is in [3, 5)
    #     smooth_bucket_idx = 1.5 # since 4 is in the middle of [3, 5).
    lower = quantiles[bucket_idx]
    upper = quantiles[bucket_idx + 1]
    smooth_bucket_idx = bucket_idx.astype(np.float32) + (value - lower) / (
        upper - lower
    )

    # Note: This is not really a quantile.
    soft_quantile = smooth_bucket_idx / num_buckets
    soft_quantile[nan_mask] = 0
    return {self.output_feature_name: soft_quantile - 0.5}

  def normalize_tensorflow(self, value: tf.Tensor) -> Dict[str, tf.Tensor]:
    value = tf.cast(value, tf.float32)
    quantiles = tf.constant(self.quantiles, dtype=tf.float32)
    num_buckets = len(self.quantiles) - 1

    bucket_idx = tf.searchsorted(quantiles, value, side="right") - 1
    bucket_idx = tf.clip_by_value(bucket_idx, 0, len(self.quantiles) - 2)

    lower = tf.gather(quantiles, bucket_idx)
    upper = tf.gather(quantiles, bucket_idx + 1)

    smooth_bucket_idx = tf.cast(bucket_idx, tf.float32) + (value - lower) / (
        upper - lower
    )

    soft_quantile = smooth_bucket_idx / num_buckets - 0.5
    return {self.output_feature_name: soft_quantile}


@normalizer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class HashStringNormalizer(AbstractFeatureNormalizer):

  num_buckets: int
  output_shape: Tuple[Optional[int], ...]
  output_feature_name: str
  type: str = dataclasses.field(default="HashStringNormalizer", init=False)

  @classmethod
  def create(
      cls,
      feature_name: str,
      input_schema: schema_lib.FeatureSchema,
      num_buckets: int,
  ) -> "DictionaryIndexNormalizer":
    if input_schema.format != schema_lib.FeatureFormat.BYTES:
      raise ValueError(
          f"Feature '{feature_name}' has format '{input_schema.format}', but "
          "HashStringNormalizer only supports BYTES features."
      )
    return HashStringNormalizer(
        input_feature=feature_name,
        num_buckets=num_buckets,
        output_shape=input_schema.shape or (),
        output_feature_name=f"{feature_name}_HASH",
    )

  def output_schema(self) -> schema_lib.FeatureSetSchema:
    return {
        self.output_feature_name: schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            shape=self.output_shape,
            num_categorical_values=self.num_buckets,
        )
    }

  def normalize_numpy(self, value: np.ndarray) -> Dict[str, np.ndarray]:
    tensor_value = tf.constant(value)
    hashed_tensor = tf.strings.to_hash_bucket_fast(
        tensor_value, self.num_buckets
    )
    return {self.output_feature_name: hashed_tensor.numpy()}

  def normalize_tensorflow(self, value: tf.Tensor) -> Dict[str, tf.Tensor]:
    return {
        self.output_feature_name: tf.strings.to_hash_bucket_fast(
            value, self.num_buckets
        )
    }


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class AutoNormalizeConfig:
  """Configuration for automatic feature normalization for GNNs.

  Attributes:
    categorical_bytes_to_index: If True, categorical features with BYTES format
      will be converted to integer indices using `DictionaryIndexNormalizer`.
    numerical_soft_quantile: If True, numerical features (INTEGER or FLOAT) will
      be normalized using `SoftQuantileNormalizer`.
    keep_raw_features: A list of feature names that should bypass all
      normalization and be included in the output graph as-is. This is useful
      for preserving features like unique identifiers or other metadata not
      intended for model input.
    ignore_features_without_stats: Whether to ignore features that are not in
      `keep_raw_features` and do not have associated statistics. If `False`
      (default), an error is raised. If `True`, such features are skipped.
    consume_primary_keys: If True, features marked as PRIMARY_ID will be
      normalized. If false, primary key features are skipped.
    primary_keys_num_hash_buckets: Number of hash buckets to use when
      normalizing primary key features.
  """

  categorical_bytes_to_index: bool = True
  numerical_soft_quantile: bool = True
  keep_raw_features: Set[str] = dataclasses.field(
      default_factory=lambda: set([])
  )
  ignore_features_without_stats: bool = False
  consume_primary_keys: bool = False
  primary_keys_num_hash_buckets: int = 256


def auto_normalize(
    schema: schema_lib.GraphSchema,
    stats: statistics_lib.GraphFeatureStatistics,
    config: AutoNormalizeConfig = AutoNormalizeConfig(),
) -> "GraphNormalizer":
  """Create a generally good GraphNormalizer from feature statistics.

  Usage example:

  ```python
  # Schema of the graph
  input_schema = dgf.data.GraphSchema(...)
  # Statistics about the graph features
  input_stats = dgf.data.GraphFeatureStatistics(...)
  # Actual graph input data
  input_graph = dgf.data.InMemoryGraph(...)

  # Instantiate normalizer
  normalizer = auto_normalize(input_schema, input_stats)

  # Normalize graph features
  output_graph = normalizer.normalize_numpy(input_graph)

  # Get the schema of the normalized graph
  output_schema = normalizer.output_schema()
  ```

  Applies a set of generally useful transformations to numeric and categorical
  features based on provided statistics. For example, numeric features can be
  z-score normalized, and categorical features can be converted to embeddings.

  If an input feature cannot be consumed (e.g., not yet implemented) a warning
  is printed.

  This methods instantiates and calls AbstractFeatureNormalizers on each
  features according to the input semantic+format+stats. For more control, you
  can manually instantiate and call the AbstractFeatureNormalizers.

  Args:
    schema: Schema of the graph features.
    stats: Precomputed statistics of the graph features.
    config: Configuration for the automatic normalization process.

  Returns:
    A GraphNormalizer instance.
  """

  edgesets = {}
  for edgeset_name, edgeset_schema in schema.edge_sets.items():
    # TODO(gbm): Normalize edgeset feature values.
    edgesets[edgeset_name] = EdgeSetNormalizerConfig(
        source=edgeset_schema.source,
        target=edgeset_schema.target,
        normalizers=[],
    )

  # Normalizers for node set features.
  nodesets = {}
  for nodeset_name, nodeset_schema in schema.node_sets.items():
    nodeset_normalizers = []
    nodeset_stats = stats.node_sets[nodeset_name]
    for feature_name, feature_schema in nodeset_schema.features.items():
      if feature_name in config.keep_raw_features:
        # Simply pass the values, no questions asked.
        nodeset_normalizers.append(
            IdentityNormalizer(
                input_feature=feature_name,
                input_schema=feature_schema,
            )
        )
        continue

      if feature_name not in nodeset_stats.features:
        if config.ignore_features_without_stats:
          continue
        raise ValueError(
            f"Feature '{feature_name}' not found in statistics for node set"
            f" '{nodeset_name}'. Make sure the stats are computed, or add"
            " this feature name to the `keep_raw_features` argument."
        )

      feature_stats = nodeset_stats.features[feature_name]
      feature_has_normalized = False

      if feature_schema.semantic == schema_lib.FeatureSemantic.UNKNOWN:
        log.warning(
            f"Feature '{feature_name}' in node set '{nodeset_name}'"
            " has an UNKNOWN semantic and will not be normalized."
        )

      if feature_schema.semantic == schema_lib.FeatureSemantic.PRIMARY_ID:
        if config.consume_primary_keys:
          if feature_schema.format == schema_lib.FeatureFormat.BYTES:
            nodeset_normalizers.append(
                HashStringNormalizer.create(
                    feature_name,
                    feature_schema,
                    config.primary_keys_num_hash_buckets,
                )
            )
            feature_has_normalized = True
          else:
            log.warning(
                "No normalizer compatible with primary key %s", feature_schema
            )

      # Categorical bytes dictionary
      if (
          config.categorical_bytes_to_index
          and feature_schema.semantic == schema_lib.FeatureSemantic.CATEGORICAL
          and feature_schema.format == schema_lib.FeatureFormat.BYTES
      ):
        nodeset_normalizers.append(
            DictionaryIndexNormalizer.create(
                feature_name, feature_schema, feature_stats
            )
        )
        feature_has_normalized = True

      # Categorical integer
      if (
          feature_schema.semantic == schema_lib.FeatureSemantic.CATEGORICAL
          and feature_schema.format.is_integer()
      ):
        feature_schema = copy.deepcopy(feature_schema)
        if feature_schema.num_categorical_values is None:
          feature_schema.num_categorical_values = (
              round(feature_stats.maximum) + 1
          )
        nodeset_normalizers.append(
            IdentityNormalizer(feature_name, input_schema=feature_schema)
        )
        feature_has_normalized = True

      # Numerical
      if (
          config.numerical_soft_quantile
          and feature_schema.semantic == schema_lib.FeatureSemantic.NUMERICAL
      ):
        nodeset_normalizers.append(
            SoftQuantileNormalizer.create(
                feature_name, feature_schema, feature_stats
            )
        )
        feature_has_normalized = True

      # Embedding
      if feature_schema.semantic == schema_lib.FeatureSemantic.EMBEDDING:
        nodeset_normalizers.append(
            IdentityNormalizer(feature_name, input_schema=feature_schema)
        )
        feature_has_normalized = True

      if not feature_has_normalized:
        log.warning(
            f"No normalizer created for node set '{nodeset_name}',"
            f" feature '{feature_name}'."
        )
    nodesets[nodeset_name] = NodeSetNormalizerConfig(nodeset_normalizers)
  return GraphNormalizer(
      config=GraphNormalizerConfig(nodesets=nodesets, edgesets=edgesets),
  )


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class NodeSetNormalizerConfig:
  """Raw information of a NodeSetNormalizer for easy serialization."""

  normalizers: List[AbstractFeatureNormalizer] = (
      normalizer_registry.field_list()
  )


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class EdgeSetNormalizerConfig:
  """Raw information of a EdgeSetNormalizer for easy serialization."""

  source: str
  target: str
  normalizers: List[AbstractFeatureNormalizer] = (
      normalizer_registry.field_list()
  )


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class GraphNormalizerConfig:
  """Raw information of a GraphNormalizer for easy serialization."""

  nodesets: Dict[str, NodeSetNormalizerConfig]
  edgesets: Dict[str, EdgeSetNormalizerConfig]

  def make(self) -> "GraphNormalizer":
    return GraphNormalizer(config=self)

  def nice_print(self, return_output: bool = False) -> Optional[str]:
    """Generates a human-readable string representation of the normalizer.

    Args:
      return_output: If true, returns the output text instead of printing it.

    Returns:
      A string containing the human-readable representation of the normalizer.
    """
    lines = ["Graph Normalizer:\n"]

    # Node Sets
    lines.append("Node Sets:")
    if not self.nodesets:
      lines.append("  (No node sets)")
    else:
      for nodeset_name in sorted(self.nodesets.keys()):
        nodeset_config = self.nodesets[nodeset_name]
        lines.append(f"  {nodeset_name}:")
        if not nodeset_config.normalizers:
          lines.append("    (No normalizers)")
        else:
          sorted_normalizers = sorted(
              nodeset_config.normalizers, key=lambda x: x.input_feature
          )
          for normalizer in sorted_normalizers:
            lines.append(f"    - {normalizer.input_feature}: {normalizer.type}")
        lines.append("")

    # Edge Sets
    lines.append("Edge Sets:")
    if not self.edgesets:
      lines.append("  (No edge sets)")
    else:
      for edgeset_name in sorted(self.edgesets.keys()):
        edgeset_config = self.edgesets[edgeset_name]
        lines.append(
            f"  {edgeset_name}: (Source: {edgeset_config.source}, Target:"
            f" {edgeset_config.target})"
        )
        if not edgeset_config.normalizers:
          lines.append("    (No normalizers)")
        else:
          sorted_normalizers = sorted(
              edgeset_config.normalizers, key=lambda x: x.input_feature
          )
          for normalizer in sorted_normalizers:
            lines.append(f"    - {normalizer.input_feature}: {normalizer.type}")
        lines.append("")

    text_content = "\n".join(lines)
    if return_output:
      return text_content
    else:
      print(text_content)
      return None


@dataclasses.dataclass
class GraphNormalizer:
  """Applies a collection of individual AbstractFeatureNormalizer on a graph.

  A graph normalier prepares features values before they can be consumed by a
  core GNN model.
  """

  config: GraphNormalizerConfig

  def output_schema(self) -> schema_lib.GraphSchema:
    """Returns the schema of the graph after normalization.

    This schema reflects the changes in feature formats, semantics, and shapes
    resulting from the applied normalizations.
    """

    edge_sets = {}
    for edgeset_name, edgeset in self.config.edgesets.items():
      features = {}
      for normalizer in edgeset.normalizers:
        features.update(normalizer.output_schema())
      edge_sets[edgeset_name] = schema_lib.EdgeSchema(
          source=edgeset.source, target=edgeset.target, features=features
      )

    node_sets = {}
    for nodeset_name, nodeset in self.config.nodesets.items():
      features = {}
      for normalizer in nodeset.normalizers:
        features.update(normalizer.output_schema())
      node_sets[nodeset_name] = schema_lib.NodeSchema(features=features)

    return schema_lib.GraphSchema(node_sets=node_sets, edge_sets=edge_sets)

  def get_normalized_feature_names(
      self,
      nodeset_name: str,
      original_feature_name: str,
  ) -> List[str]:
    """Gets the normalized feature names derived from a given input feature.

    An original feature can be transformed into one or more new features by
    different normalizers. This method returns the names of all such generated
    features.

    Args:
      nodeset_name: The name of the node set containing the feature.
      original_feature_name: The name of the input feature before normalization.

    Returns:
      A list of feature names in the normalized graph that originated from
      `original_feature_name`.
    """
    return [
        output_feature
        for normalizer in self.config.nodesets[nodeset_name].normalizers
        for output_feature in normalizer.output_schema()
        if normalizer.input_feature == original_feature_name
    ]

  def normalize_numpy(
      self, graph: in_memory_graph.InMemoryGraph
  ) -> in_memory_graph.InMemoryGraph:
    """Normalizes the features of the input graph.

    Args:
      graph: The input `InMemoryGraph` with raw feature values.

    Returns:
      A new `InMemoryGraph` with normalized feature values.
    """

    # Normalize edgeset features
    dst_graph_edge_sets = {}
    for edgeset_name, edgeset in self.config.edgesets.items():
      input_edgeset = graph.edge_sets[edgeset_name]
      output_features = {}
      for normalizer in edgeset.normalizers:
        input_feature_value = input_edgeset.features[normalizer.input_feature]
        output_features.update(normalizer.normalize_numpy(input_feature_value))
      dst_graph_edge_sets[edgeset_name] = in_memory_graph.InMemoryEdgeSet(
          adjacency=input_edgeset.adjacency,
          features=output_features,
      )

    # Normalize nodeset features
    dst_graph_node_sets = {}
    for nodeset_name, nodeset in self.config.nodesets.items():
      input_nodeset = graph.node_sets[nodeset_name]
      output_features = {}
      for normalizer in nodeset.normalizers:
        if normalizer.input_feature not in input_nodeset.features:
          continue
        input_feature_value = input_nodeset.features[normalizer.input_feature]
        output_features.update(normalizer.normalize_numpy(input_feature_value))
      dst_graph_node_sets[nodeset_name] = in_memory_graph.InMemoryNodeSet(
          features=output_features,
          num_nodes=input_nodeset.num_nodes,
      )

    dst_graph = in_memory_graph.InMemoryGraph(
        node_sets=dst_graph_node_sets, edge_sets=dst_graph_edge_sets
    )
    return dst_graph

  def tensorflow_resources(self) -> List[tf.Tensor]:
    """Returns all the tf resources of all the operations."""
    resources = []
    for edgeset in self.config.edgesets.values():
      for normalizer in edgeset.normalizers:
        resources.extend(normalizer.tensorflow_resources())
    for nodeset in self.config.nodesets.values():
      for normalizer in nodeset.normalizers:
        resources.extend(normalizer.tensorflow_resources())
    return resources

  def normalize_tensorflow(
      self, graph: tf_in_memory_graph.TFInMemoryGraph
  ) -> tf_in_memory_graph.TFInMemoryGraph:
    """Normalizes the features of the input graph using tensorflow.

    Args:
      graph: The input `InMemoryGraph` with raw feature values.

    Returns:
      A new `InMemoryGraph` with normalized feature values.
    """

    # Normalize edgeset features
    dst_graph_edge_sets = {}
    for edgeset_name, edgeset in self.config.edgesets.items():
      input_edgeset = graph.edge_sets[edgeset_name]
      output_features = {}
      for normalizer in edgeset.normalizers:
        input_feature_value = input_edgeset.features[normalizer.input_feature]
        output_features.update(
            normalizer.normalize_tensorflow(input_feature_value)
        )
      dst_graph_edge_sets[edgeset_name] = tf_in_memory_graph.TFInMemoryEdgeSet(
          adjacency=input_edgeset.adjacency,
          features=output_features,
      )

    # Normalize nodeset features
    dst_graph_node_sets = {}
    for nodeset_name, nodeset in self.config.nodesets.items():
      input_nodeset = graph.node_sets[nodeset_name]
      output_features = {}
      for normalizer in nodeset.normalizers:
        if normalizer.input_feature not in input_nodeset.features:
          continue
        input_feature_value = input_nodeset.features[normalizer.input_feature]
        output_features.update(
            normalizer.normalize_tensorflow(input_feature_value)
        )
      dst_graph_node_sets[nodeset_name] = tf_in_memory_graph.TFInMemoryNodeSet(
          features=output_features,
          num_nodes=input_nodeset.num_nodes,
      )

    return tf_in_memory_graph.TFInMemoryGraph(
        node_sets=dst_graph_node_sets, edge_sets=dst_graph_edge_sets
    )

  def normalize_numpy_to_jax(
      self,
      graph: in_memory_graph.InMemoryGraph,
      include_adjacencies: bool = True,
  ) -> jax_in_memory_graph.JaxInMemoryGraph:
    """Normalizes graph features using numpy and returns jax arrays.

    This function is equivalent, but consumes less memory than, calling
    "normalize" + "graph_to_jax_graph".

    Args:
      graph: The input `InMemoryGraph` with raw feature values.
      include_adjacencies: If True, the adjacency information from the input
        graph will be included in the output `JaxInMemoryGraph`. Otherwise, the
        output graph's edge sets will have `None` for adjacency.

    Returns:
      A new `JaxInMemoryGraph` with normalized feature values.
    """

    def asarray(x):
      return jnp.asarray(x)

    # Normalize edgeset features
    dst_graph_edge_sets = {}
    for edgeset_name, edgeset in self.config.edgesets.items():
      input_edgeset = graph.edge_sets[edgeset_name]
      output_features = {}
      for normalizer in edgeset.normalizers:
        input_feature_value = input_edgeset.features[normalizer.input_feature]
        output_features.update({
            k: asarray(v)
            for k, v in normalizer.normalize_numpy(input_feature_value).items
        })
      dst_graph_edge_sets[edgeset_name] = (
          jax_in_memory_graph.JaxInMemoryEdgeSet(
              adjacency=(
                  asarray(input_edgeset.adjacency)
                  if include_adjacencies
                  else None
              ),
              features=output_features,
          )
      )

    # Normalize nodeset features
    dst_graph_node_sets = {}
    for nodeset_name, nodeset in self.config.nodesets.items():
      input_nodeset = graph.node_sets[nodeset_name]
      output_features = {}
      for normalizer in nodeset.normalizers:
        if normalizer.input_feature not in input_nodeset.features:
          continue
        input_feature_value = input_nodeset.features[normalizer.input_feature]
        output_features.update({
            k: asarray(v)
            for k, v in normalizer.normalize_numpy(input_feature_value).items()
        })
      dst_graph_node_sets[nodeset_name] = (
          jax_in_memory_graph.JaxInMemoryNodeSet(
              features=output_features,
              num_nodes=input_nodeset.num_nodes,
          )
      )

    return jax_in_memory_graph.JaxInMemoryGraph(
        node_sets=dst_graph_node_sets, edge_sets=dst_graph_edge_sets
    )

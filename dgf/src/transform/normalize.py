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

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Set
import copy
import dataclasses
import enum
import inspect
import math
from typing import Any, Literal

import dataclasses_json
from dgf.src.data import in_memory_graph
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
from dgf.src.data import tf_in_memory_graph
from dgf.src.transform import normalize_base
from dgf.src.transform import normalize_text
from dgf.src.util import log
from dgf.src.util.weak_dep.weak_dep_tensorflow import tf
import jax.numpy as jnp
import numpy as np

normalizer_registry = normalize_base.normalizer_registry
AbstractFeatureNormalizer = normalize_base.AbstractFeatureNormalizer
TextEmbeddingNormalizer = normalize_text.TextEmbeddingNormalizer


class NumericalMissingValuePolicy(enum.StrEnum):
  """What `auto_normalize` does when float features have NaN/±inf values.

  Checked once on the feature statistics, when the normalizer is created.
  """

  # Missing values are an error (no fallback).
  ERROR = "error"
  # TODO(dtl): Add other policies, like FILL, INDICATOR, and FILL_LEARNED.


@normalizer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class IdentityNormalizer(AbstractFeatureNormalizer):
  """A normalizer that simply pass a feature without changing it."""

  input_schema: schema_lib.FeatureSchema
  type: str = dataclasses.field(default="IdentityNormalizer", init=False)

  def output_schema(self) -> schema_lib.FeatureSetSchema:
    return {self.input_feature: self.input_schema}

  def normalize_numpy(self, value: np.ndarray) -> dict[str, np.ndarray]:
    return {self.input_feature: value}

  def normalize_tensorflow(self, value: tf.Tensor) -> dict[str, tf.Tensor]:
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

  dictionary_map: dict[str, int]
  out_of_vocab_value: int
  output_shape: tuple[int | None, ...]
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

    if input_stats.count == 0:
      log.warning(
          f"No value observed for feature {feature_name}. Creating a empty"
          " DictionaryIndexNormalizer."
      )
      dictionary_map = {"SENTINEL": 0}
    else:
      if not input_stats.dictionary:
        raise ValueError(
            f"Feature '{feature_name}' does not have a dictionary in its"
            " statistics."
        )
      dictionary_map = {
          key: item.index for key, item in input_stats.dictionary.items()
      }
    return DictionaryIndexNormalizer(
        input_feature=feature_name,
        dictionary_map=dictionary_map,
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

  def normalize_numpy(self, value: np.ndarray) -> dict[str, np.ndarray]:
    # TODO(gbm): Would sorting the key + np.searchsorted be much faster?
    def _lookup(v):
      return self.dictionary_map.get(
          v.decode("utf-8", "surrogateescape"),
          self.out_of_vocab_value,
      )

    vectorized_lookup = np.vectorize(_lookup, otypes=[np.int64])
    # TODO(gbm): Parametrize to int32.
    return {self.output_feature_name: vectorized_lookup(value)}

  def tensorflow_resources(self) -> list[tf.Tensor]:
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

  def normalize_tensorflow(self, value: tf.Tensor) -> dict[str, tf.Tensor]:
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

  Note:
    For timeseries features, this normalizer will perform some amount of
    lookahead, i.e. the normalization depends on future feature statistics and
    leaks some of that information. This can lead to worse-than-expected
    performance on the validation and test set.
  """

  output_feature_name: str
  output_shape: tuple[int | None, ...]
  quantiles: np.ndarray = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda x: x.tolist(),
          decoder=lambda x: np.asarray(x, dtype=np.float32),
      )
  )
  is_timeseries: bool = False
  group: str | None = None
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

    # A variable-length feature (e.g., a timeseries) can be present on
    # entities (count > 0) without any value, e.g. when all the values are
    # after the seed timestamps. In this case, no min/max or quantiles exist.
    no_observed_value = input_stats.count == 0 or (
        not input_stats.quantiles and not math.isfinite(input_stats.minimum)
    )
    if no_observed_value:
      log.warning(
          f"No value observed for feature {feature_name}. Creating a empty"
          " SoftQuantileNormalizer."
      )
      quantiles = np.asarray([0, 1], dtype=np.float32)
    else:

      if not input_stats.quantiles:
        raise ValueError(
            f"Feature '{feature_name}' does not have quantiles in its"
            " statistics."
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
        is_timeseries=input_schema.is_timeseries,
        group=input_schema.group,
    )

  def output_schema(self) -> schema_lib.FeatureSetSchema:
    return {
        self.output_feature_name: schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=self.output_shape,
            is_timeseries=self.is_timeseries,
            group=self.group,
        )
    }

  def normalize_numpy(self, value: np.ndarray) -> dict[str, np.ndarray]:
    # TODO(gbm): Add support for multi-dim features.

    value = value.astype(np.float32)
    # Non-finite values (NaN, +-inf) pass through unchanged, also in
    # `normalize_tensorflow`. Serving does not raise by default; training
    # rejects missing values earlier, at the statistics.
    quantiles = self.quantiles
    num_buckets = len(quantiles) - 1

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
    delta = np.maximum(upper - lower, 1e-12)
    adjusted = (value - lower) / delta
    smooth_bucket_idx = bucket_idx.astype(np.float32) + adjusted

    # Note: This is not really a quantile, so it's fine to adjust it a little:
    #  * Center the output to [-0.5, 0.5] so that the median is 0.
    #  * Clip to [-100, 100] to avoid overflows when squaring downstream.
    #  * We pass through not only NaNs, but also ±inf (instead of clipping).
    soft_quantile = np.clip(smooth_bucket_idx / num_buckets - 0.5, -100, 100)
    soft_quantile = np.where(np.isfinite(value), soft_quantile, value)
    return {self.output_feature_name: soft_quantile}

  def normalize_tensorflow(self, value: tf.Tensor) -> dict[str, tf.Tensor]:
    value = tf.cast(value, tf.float32)
    quantiles = tf.constant(self.quantiles, dtype=tf.float32)
    num_buckets = len(self.quantiles) - 1

    orig_shape = tf.shape(value)
    flat_value = tf.reshape(value, [-1])
    bucket_idx = tf.searchsorted(quantiles, flat_value, side="right") - 1
    bucket_idx = tf.clip_by_value(bucket_idx, 0, len(self.quantiles) - 2)
    bucket_idx = tf.reshape(bucket_idx, orig_shape)

    lower = tf.gather(quantiles, bucket_idx)
    upper = tf.gather(quantiles, bucket_idx + 1)
    delta = tf.maximum(upper - lower, 1e-12)
    adjusted = (value - lower) / delta
    smooth_bucket_idx = tf.cast(bucket_idx, tf.float32) + adjusted

    # See comment in normalize_numpy for details.
    soft_quantile = smooth_bucket_idx / num_buckets - 0.5
    soft_quantile = tf.clip_by_value(soft_quantile, -100, 100)
    soft_quantile = tf.where(tf.math.is_finite(value), soft_quantile, value)
    return {self.output_feature_name: soft_quantile}


@normalizer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class HashStringNormalizer(AbstractFeatureNormalizer):
  """Normalizes a BYTES feature by hashing it to a bucket index."""

  num_buckets: int
  output_shape: tuple[int | None, ...]
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
    return HashStringNormalizer(  # pyrefly: ignore[bad-return]
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

  def normalize_numpy(self, value: np.ndarray) -> dict[str, np.ndarray]:
    tensor_value = tf.constant(value)
    hashed_tensor = tf.strings.to_hash_bucket_fast(
        tensor_value, self.num_buckets
    )
    return {self.output_feature_name: hashed_tensor.numpy()}

  def normalize_tensorflow(self, value: tf.Tensor) -> dict[str, tf.Tensor]:
    return {
        self.output_feature_name: tf.strings.to_hash_bucket_fast(
            value, self.num_buckets
        )
    }


@normalizer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class SinusoidTimedeltaNormalizer(AbstractFeatureNormalizer):
  """Normalizes a time delta feature by applying sinusoidal embeddings.

  The input is expected to be a time delta (semantic=TIMEDELTA).
  This normalizer projects the scalar time delta into a multi-dimensional
  embedding using sine and cosine functions of different frequencies.
  """

  output_feature_name: str
  output_shape: tuple[int | None, ...]
  embedding_dim: int = 32
  min_period: float = 2.0
  max_period: float = 31536000.0  # one year in seconds
  is_timeseries: bool = False
  group: str | None = None
  frequencies: np.ndarray = dataclasses.field(
      default_factory=lambda: np.array([], dtype=np.float32),
      metadata=dataclasses_json.config(
          encoder=lambda x: x.tolist(),
          decoder=lambda x: np.asarray(x, dtype=np.float32),
      ),
  )
  type: str = dataclasses.field(
      default="SinusoidTimedeltaNormalizer", init=False
  )

  def __post_init__(self):
    if self.frequencies.size == 0:
      half_dim = self.embedding_dim // 2
      if half_dim == 1:
        periods = np.array([self.max_period], dtype=np.float32)
      else:
        periods = np.exp(
            np.linspace(
                np.log(self.min_period),
                np.log(self.max_period),
                half_dim,
                dtype=np.float32,
            )
        )
      self.frequencies = (2 * np.pi) / periods

  @classmethod
  def create(
      cls,
      feature_name: str,
      input_schema: schema_lib.FeatureSchema,
      embedding_dim: int = 32,
  ) -> "SinusoidTimedeltaNormalizer":
    if embedding_dim <= 0 or embedding_dim % 2 != 0:
      raise ValueError(
          f"embedding_dim must be a positive even integer, got {embedding_dim}."
      )

    if input_schema.semantic != schema_lib.FeatureSemantic.TIMEDELTA:
      raise ValueError(
          f"Feature '{feature_name}' has semantic '{input_schema.semantic}',"
          " but SinusoidTimedeltaNormalizer only supports TIMEDELTA features."
      )

    if not input_schema.is_static_shape():
      raise ValueError(
          "SinusoidTimedeltaNormalizer requires fixed-length feature tensors,"
          f" but feature '{feature_name}' has a dynamic shape"
          f" ({input_schema.shape}). Please run PadAndCapTimeseries"
          " first."
      )

    input_shape = input_schema.shape or ()
    # Output shape appends the embedding dimension
    output_shape = tuple(list(input_shape) + [embedding_dim])

    return SinusoidTimedeltaNormalizer(
        input_feature=feature_name,
        embedding_dim=embedding_dim,
        output_shape=output_shape,
        output_feature_name=f"{feature_name}_SINUSOID",
        is_timeseries=input_schema.is_timeseries,
        group=input_schema.group,
    )

  def output_schema(self) -> schema_lib.FeatureSetSchema:
    return {
        self.output_feature_name: schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=self.output_shape,
            is_timeseries=self.is_timeseries,
            group=self.group,
        )
    }

  def normalize_numpy(self, value: np.ndarray) -> dict[str, np.ndarray]:
    if value.dtype == np.object_:
      raise ValueError(
          "SinusoidTimedeltaNormalizer requires fixed-length feature tensors,"
          f" but feature '{self.input_feature}' is a variable-length object"
          " array. Please run PadAndCapTimeseries first."
      )

    value = value.astype(np.float32)
    arg = np.expand_dims(value, -1) * self.frequencies
    sin_emb = np.sin(arg)
    cos_emb = np.cos(arg)
    emb = np.concatenate([sin_emb, cos_emb], axis=-1)
    return {self.output_feature_name: emb}

  def normalize_tensorflow(self, value: tf.Tensor) -> dict[str, tf.Tensor]:
    value = tf.cast(value, tf.float32)
    freqs = tf.constant(self.frequencies, dtype=tf.float32)

    arg = tf.expand_dims(value, -1) * freqs
    sin_emb = tf.sin(arg)
    cos_emb = tf.cos(arg)
    emb = tf.concat([sin_emb, cos_emb], axis=-1)
    return {self.output_feature_name: emb}


_output_timeseries_group = normalize_base.output_timeseries_group


@normalizer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class TimedeltaNormalizer(AbstractFeatureNormalizer):
  """Normalizes a TIMESTAMP feature into a TIMEDELTA feature relative to seed timestamps.

  Subtracts the timestamp value from the node's seed timestamp:
  delta = seed_timestamp - t_i
  """

  input_schema: schema_lib.FeatureSchema
  output_feature_name: str
  type: str = dataclasses.field(default="TimedeltaNormalizer", init=False)

  @classmethod
  def create(
      cls,
      feature_name: str,
      input_schema: schema_lib.FeatureSchema,
  ) -> "TimedeltaNormalizer":
    if input_schema.semantic != schema_lib.FeatureSemantic.TIMESTAMP:
      raise ValueError(
          f"Feature '{feature_name}' has semantic '{input_schema.semantic}',"
          " but TimedeltaNormalizer only supports TIMESTAMP features."
      )

    if not input_schema.is_static_shape():
      raise ValueError(
          "TimedeltaNormalizer requires fixed-length feature tensors,"
          f" but feature '{feature_name}' has a dynamic shape"
          f" ({input_schema.shape}). Please run padding first."
      )

    return TimedeltaNormalizer(
        input_feature=feature_name,
        input_schema=input_schema,
        output_feature_name=f"{feature_name}_seed_delta",
    )

  def output_schema(self) -> schema_lib.FeatureSetSchema:
    ts_group = _output_timeseries_group(self.input_feature, self.input_schema)
    return {
        self.output_feature_name: schema_lib.FeatureSchema(
            format=self.input_schema.format,
            semantic=schema_lib.FeatureSemantic.TIMEDELTA,
            shape=self.input_schema.shape,
            is_timeseries=self.input_schema.is_timeseries,
            group=ts_group,
        )
    }

  def normalize_numpy(
      self,
      value: np.ndarray,
      seed_timestamps: np.ndarray | None = None,
  ) -> dict[str, np.ndarray]:
    assert seed_timestamps is not None, (
        "seed_timestamps must be provided to normalize timestamp feature"
        f" '{self.input_feature}'."
    )

    assert value.dtype != np.object_, (
        "TimedeltaNormalizer requires fixed-length feature tensors,"
        f" but feature '{self.input_feature}' is a variable-length object"
        " array. Please run padding first."
    )

    seed_arr = seed_timestamps
    if value.ndim > 1:
      seed_arr = seed_arr.reshape(seed_arr.shape + (1,) * (value.ndim - 1))

    deltas = seed_arr - value
    return {self.output_feature_name: deltas}

  def normalize_tensorflow(
      self,
      value: tf.Tensor,
      seed_timestamps: tf.Tensor | None = None,
  ) -> dict[str, tf.Tensor]:
    assert seed_timestamps is not None, (
        "seed_timestamps must be provided to normalize timestamp feature"
        f" '{self.input_feature}'."
    )

    assert value.shape.rank is not None, (
        "TimedeltaNormalizer requires fixed-length feature tensors,"
        f" but feature '{self.input_feature}' has unknown rank."
        " Please run padding first."
    )

    seed_tensor = tf.cast(seed_timestamps, value.dtype)
    if value.shape.rank > 1:
      seed_tensor = tf.reshape(
          seed_tensor, (-1,) + (1,) * (value.shape.rank - 1)
      )

    deltas = tf.subtract(seed_tensor, value)
    return {self.output_feature_name: deltas}


class CalendarFeature(str, enum.Enum):
  """Supported calendar features to extract from timestamps."""

  SECOND = "second"
  MINUTE = "minute"
  HOUR = "hour"
  DAY_OF_WEEK = "day_of_week"
  DAY_OF_MONTH = "day_of_month"
  MONTH = "month"
  YEAR = "year"


# Analytically known support of each supported calendar component, as a
# half-open `[minimum, maximum)` interval. Calendar values are derived inside
# the normalizer and therefore have no entry in the graph feature statistics,
# but their expected distribution is uniform and fully known a priori, so they
# can be rescaled from the range alone.
#
# `CalendarFeature.YEAR` is the one member that cannot be supported this way: a
# year has no a-priori range, so there is nothing to rescale from.
_CALENDAR_FEATURE_RANGES: dict[CalendarFeature, tuple[float, float]] = {
    CalendarFeature.SECOND: (0.0, 60.0),
    CalendarFeature.MINUTE: (0.0, 60.0),
    CalendarFeature.HOUR: (0.0, 24.0),
    CalendarFeature.DAY_OF_WEEK: (0.0, 7.0),
    CalendarFeature.DAY_OF_MONTH: (1.0, 32.0),
    CalendarFeature.MONTH: (1.0, 13.0),
}

# `SECOND` and `MINUTE` are supported but not enabled by default
# TODO(simonmeierhans): Automatically select features based on observed
# timestamps.
DEFAULT_CALENDAR_FEATURES: tuple[CalendarFeature, ...] = (
    CalendarFeature.HOUR,
    CalendarFeature.DAY_OF_WEEK,
    CalendarFeature.DAY_OF_MONTH,
    CalendarFeature.MONTH,
)

# Shifts the unix epoch to 0000-03-01, i.e. the number of days between
# 0000-03-01 and 1970-01-01. Starting the year in March moves the leap day to
# the *end* of the year, which is what lets `_month_from_days_numpy` compute the
# month with a single closed-form expression instead of a leap-year table. This
# is intrinsic to the algorithm and is required for all dates, not just dates
# before the unix epoch.
_DAYS_FROM_CIVIL_EPOCH = 719468

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 60 * 60
_SECONDS_PER_DAY = 24 * _SECONDS_PER_HOUR

# Unix epoch day 0 (1970-01-01) was a Thursday. Adding 3 shifts the week so that
# Monday maps to 0.
_EPOCH_WEEKDAY_OFFSET = 3


@normalizer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class CalendarNormalizer(AbstractFeatureNormalizer):
  """Extracts normalized UTC calendar components from a TIMESTAMP feature.

  The input is expected to be a unix timestamp in seconds (semantic=TIMESTAMP).
  One output feature is emitted per requested calendar component, named
  `{input_feature}_{component}_CALENDAR`. Each component is linearly rescaled
  from its analytically known range (e.g. [0, 24) for the hour of the day) to
  [-0.5, 0.5) and emitted with semantic EMBEDDING, so it can be fed to a model
  directly.

  Usage example:

  ```python
    normalizer = CalendarNormalizer.create(
        "created_at",
        schema,
        (CalendarFeature.HOUR, CalendarFeature.DAY_OF_WEEK),
    )
    # {"created_at_hour_CALENDAR": ..., "created_at_day_of_week_CALENDAR": ...}
    features = normalizer.normalize_numpy(timestamps)
  ```

  The rescaling uses the known support of each component instead of observed
  statistics: the components are derived inside this normalizer, so they have
  no entry in the graph feature statistics, and their distribution is uniform
  and fully known a priori.

  Note:
    The rescaling is monotonic and therefore destroys cyclicity: hour 23 and
    hour 0 are adjacent in time but are mapped to opposite ends of the output
    range. Apply `SinusoidTimedeltaNormalizer` to the raw timestamp instead if
    a cyclic encoding is required.

  Attributes:
    calendar_features: The calendar components to extract.
    input_schema: The schema of the input TIMESTAMP feature.
  """

  calendar_features: tuple[CalendarFeature, ...]
  input_schema: schema_lib.FeatureSchema
  type: str = dataclasses.field(default="CalendarNormalizer", init=False)

  def __post_init__(self):
    self.calendar_features = tuple(
        CalendarFeature(calendar_feature)
        for calendar_feature in self.calendar_features
    )

    if self.input_schema.semantic != schema_lib.FeatureSemantic.TIMESTAMP:
      raise ValueError(
          f"Feature '{self.input_feature}' has semantic"
          f" '{self.input_schema.semantic}', but CalendarNormalizer only"
          " supports TIMESTAMP features."
      )

    if not self.input_schema.is_static_shape():
      raise ValueError(
          "CalendarNormalizer requires fixed-length feature tensors, but"
          f" feature '{self.input_feature}' has a dynamic shape"
          f" ({self.input_schema.shape}). Please run padding first."
      )

    if not self.calendar_features:
      raise ValueError(
          f"No calendar feature requested for feature '{self.input_feature}'."
      )

    for calendar_feature in self.calendar_features:
      if calendar_feature not in _CALENDAR_FEATURE_RANGES:
        raise ValueError(
            f"Calendar feature '{calendar_feature.value}' is not supported by"
            " CalendarNormalizer. Supported features:"
            f" {[f.value for f in _CALENDAR_FEATURE_RANGES]}."
        )

  @classmethod
  def create(
      cls,
      feature_name: str,
      input_schema: schema_lib.FeatureSchema,
      calendar_features: tuple[CalendarFeature, ...],
  ) -> "CalendarNormalizer":
    return CalendarNormalizer(
        input_feature=feature_name,
        calendar_features=tuple(calendar_features),
        input_schema=input_schema,
    )

  def output_feature_name(self, calendar_feature: CalendarFeature) -> str:
    """Returns the name of the output feature for one calendar component."""
    return f"{self.input_feature}_{calendar_feature.value}_CALENDAR"

  def output_schema(self) -> schema_lib.FeatureSetSchema:
    calendar_group = _output_timeseries_group(
        self.input_feature, self.input_schema
    )
    return {
        self.output_feature_name(calendar_feature): schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=self.input_schema.shape or (),
            is_timeseries=self.input_schema.is_timeseries,
            group=calendar_group,
        )
        for calendar_feature in self.calendar_features
    }

  def normalize_numpy(self, value: np.ndarray) -> dict[str, np.ndarray]:
    if value.dtype == np.object_:
      raise ValueError(
          "CalendarNormalizer requires fixed-length feature tensors, but"
          f" feature '{self.input_feature}' is a variable-length object array."
          " Please run padding first."
      )

    timestamps = value.astype(np.int64)
    # Only computed if a day-based component is requested.
    days = None

    output_features = {}
    for calendar_feature in self.calendar_features:
      if calendar_feature == CalendarFeature.SECOND:
        component = timestamps % _SECONDS_PER_MINUTE
      elif calendar_feature == CalendarFeature.MINUTE:
        component = (timestamps // _SECONDS_PER_MINUTE) % 60
      elif calendar_feature == CalendarFeature.HOUR:
        component = (timestamps // _SECONDS_PER_HOUR) % 24
      elif calendar_feature == CalendarFeature.DAY_OF_WEEK:
        if days is None:
          days = timestamps // _SECONDS_PER_DAY
        component = (days + _EPOCH_WEEKDAY_OFFSET) % 7
      elif calendar_feature == CalendarFeature.DAY_OF_MONTH:
        if days is None:
          days = timestamps // _SECONDS_PER_DAY
        component = _day_of_month_from_days_numpy(days)
      elif calendar_feature == CalendarFeature.MONTH:
        if days is None:
          days = timestamps // _SECONDS_PER_DAY
        component = _month_from_days_numpy(days)
      else:
        raise ValueError(
            f"Unsupported calendar feature '{calendar_feature.value}'."
        )

      minimum, maximum = _CALENDAR_FEATURE_RANGES[calendar_feature]
      output_features[self.output_feature_name(calendar_feature)] = (
          (component - minimum) / (maximum - minimum) - 0.5
      ).astype(np.float32)

    return output_features

  def normalize_tensorflow(self, value: tf.Tensor) -> dict[str, tf.Tensor]:
    timestamps = tf.cast(value, tf.int64)
    # Only computed if a day-based component is requested.
    days = None

    output_features = {}
    for calendar_feature in self.calendar_features:
      if calendar_feature == CalendarFeature.SECOND:
        component = tf.math.floormod(timestamps, _SECONDS_PER_MINUTE)
      elif calendar_feature == CalendarFeature.MINUTE:
        component = tf.math.floormod(
            tf.math.floordiv(timestamps, _SECONDS_PER_MINUTE), 60
        )
      elif calendar_feature == CalendarFeature.HOUR:
        component = tf.math.floormod(
            tf.math.floordiv(timestamps, _SECONDS_PER_HOUR), 24
        )
      elif calendar_feature == CalendarFeature.DAY_OF_WEEK:
        if days is None:
          days = tf.math.floordiv(timestamps, _SECONDS_PER_DAY)
        component = tf.math.floormod(days + _EPOCH_WEEKDAY_OFFSET, 7)
      elif calendar_feature == CalendarFeature.DAY_OF_MONTH:
        if days is None:
          days = tf.math.floordiv(timestamps, _SECONDS_PER_DAY)
        component = _day_of_month_from_days_tensorflow(days)
      elif calendar_feature == CalendarFeature.MONTH:
        if days is None:
          days = tf.math.floordiv(timestamps, _SECONDS_PER_DAY)
        component = _month_from_days_tensorflow(days)
      else:
        raise ValueError(
            f"Unsupported calendar feature '{calendar_feature.value}'."
        )

      minimum, maximum = _CALENDAR_FEATURE_RANGES[calendar_feature]
      output_features[self.output_feature_name(calendar_feature)] = (
          tf.cast(component, tf.float32) - minimum
      ) / (maximum - minimum) - 0.5

    return output_features


def _month_from_days_numpy(days: np.ndarray) -> np.ndarray:
  """Returns the 1-based civil month for days since the unix epoch.

  Implements Howard Hinnant's `civil_from_days` algorithm using integer
  arithmetic only.

  Args:
    days: Days since 1970-01-01, as integers. May be negative.

  Returns:
    The civil month, in [1, 12].
  """
  shifted_days = days + _DAYS_FROM_CIVIL_EPOCH
  era = shifted_days // 146097
  day_of_era = shifted_days - era * 146097
  year_of_era = (
      day_of_era
      - day_of_era // 1460
      + day_of_era // 36524
      - day_of_era // 146096
  ) // 365
  day_of_year = day_of_era - (
      365 * year_of_era + year_of_era // 4 - year_of_era // 100
  )
  # `shifted_month` is 0 for March and 11 for February of the following year.
  shifted_month = (5 * day_of_year + 2) // 153
  return shifted_month + 3 - 12 * (shifted_month >= 10)


def _month_from_days_tensorflow(days: tf.Tensor) -> tf.Tensor:
  """TensorFlow equivalent of `_month_from_days_numpy`."""
  shifted_days = days + _DAYS_FROM_CIVIL_EPOCH
  era = tf.math.floordiv(shifted_days, 146097)
  day_of_era = shifted_days - era * 146097
  year_of_era = tf.math.floordiv(
      day_of_era
      - tf.math.floordiv(day_of_era, 1460)
      + tf.math.floordiv(day_of_era, 36524)
      - tf.math.floordiv(day_of_era, 146096),
      365,
  )
  day_of_year = day_of_era - (
      365 * year_of_era
      + tf.math.floordiv(year_of_era, 4)
      - tf.math.floordiv(year_of_era, 100)
  )
  shifted_month = tf.math.floordiv(5 * day_of_year + 2, 153)
  return (
      shifted_month + 3 - 12 * tf.cast(shifted_month >= 10, shifted_month.dtype)
  )


def _day_of_month_from_days_numpy(days: np.ndarray) -> np.ndarray:
  """Returns the 1-based civil day of the month for days since the unix epoch.

  Implements Howard Hinnant's `civil_from_days` algorithm using integer
  arithmetic only.

  Args:
    days: Days since 1970-01-01, as integers. May be negative.

  Returns:
    The civil day of the month, in [1, 31].
  """
  shifted_days = days + _DAYS_FROM_CIVIL_EPOCH
  era = shifted_days // 146097
  day_of_era = shifted_days - era * 146097
  year_of_era = (
      day_of_era
      - day_of_era // 1460
      + day_of_era // 36524
      - day_of_era // 146096
  ) // 365
  day_of_year = day_of_era - (
      365 * year_of_era + year_of_era // 4 - year_of_era // 100
  )
  shifted_month = (5 * day_of_year + 2) // 153
  return day_of_year - (153 * shifted_month + 2) // 5 + 1


def _day_of_month_from_days_tensorflow(days: tf.Tensor) -> tf.Tensor:
  """TensorFlow equivalent of `_day_of_month_from_days_numpy`."""
  shifted_days = days + _DAYS_FROM_CIVIL_EPOCH
  era = tf.math.floordiv(shifted_days, 146097)
  day_of_era = shifted_days - era * 146097
  year_of_era = tf.math.floordiv(
      day_of_era
      - tf.math.floordiv(day_of_era, 1460)
      + tf.math.floordiv(day_of_era, 36524)
      - tf.math.floordiv(day_of_era, 146096),
      365,
  )
  day_of_year = day_of_era - (
      365 * year_of_era
      + tf.math.floordiv(year_of_era, 4)
      - tf.math.floordiv(year_of_era, 100)
  )
  shifted_month = tf.math.floordiv(5 * day_of_year + 2, 153)
  return day_of_year - tf.math.floordiv(153 * shifted_month + 2, 5) + 1


# Default cap on the one-hot width inferred by `OneHotNormalizer.create`. At
# 1024 categories the dense float32 output is 4 KiB per entity (and per
# timestep), which is already larger than typical node feature vectors; wider
# vocabularies (e.g. hashed IDs) are better served by an index plus a learned
# embedding (`DictionaryIndexNormalizer`). The cap only guards against
# accidental blow-ups and can be raised or disabled by the caller.
DEFAULT_MAX_ONE_HOT_CATEGORIES = 1024


@normalizer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class OneHotNormalizer(AbstractFeatureNormalizer):
  """One-hot encodes integer or boolean categorical features into float32 vectors.

  Categories in `[0, num_categories)` are mapped to unit basis vectors, while
  out-of-vocabulary or out-of-range indices encode as all zeros. With
  `drop_if_binary=True`, a feature with `num_categories == 2` is instead
  committed to a single indicator column (`value == 1`); a value outside
  `[0, 2)` then raises instead of silently looking like category `0`. This is
  a deliberate exception to serving never raising: it is opt-in, and a third
  category would otherwise be encoded wrongly.

  Usage example:

  ```python
  normalizer = dgf.transform.OneHotNormalizer.create(
      "category_idx",
      dgf.data.FeatureSchema(
          format=dgf.data.FeatureFormat.INTEGER_64,
          semantic=dgf.data.FeatureSemantic.CATEGORICAL,
          shape=(),
          num_categorical_values=3,
      ),
  )
  outputs = normalizer.normalize_numpy(np.array([0, 2, 3], dtype=np.int64))
  ```

  Attributes:
    num_categories: Number of known categories in `[0, num_categories)`.
    output_shape: Static output feature shape (excluding the leading entity
      dimension), ending in `1` for dropped binary features or `num_categories`.
    output_feature_name: Name of the emitted float32 embedding feature.
    drop_if_binary: If True, a feature with `num_categories == 2` emits a single
      `(..., 1)` column (`index == 1`) and values outside `[0, 2)` raise. If
      False (default), it emits a `(..., 2)` vector and unseen values encode as
      zeros like any other out-of-range index.
    is_timeseries: Whether the feature is a timeseries.
    group: Feature group of the output. Defaults to `input_feature` for a
      timeseries without a group.
  """

  num_categories: int
  output_shape: tuple[int | None, ...]
  output_feature_name: str
  drop_if_binary: bool = False
  is_timeseries: bool = False
  group: str | None = None
  type: str = dataclasses.field(default="OneHotNormalizer", init=False)

  def __post_init__(self):
    # The output is renamed, so a timeseries without a group would be grouped
    # by the output name, apart from the other outputs of the same input. Done
    # here rather than in `create` so direct construction and JSON get it too.
    if self.is_timeseries and self.group is None:
      self.group = self.input_feature
    if self.num_categories < 1:
      raise ValueError(
          "OneHotNormalizer requires num_categories >= 1, got"
          f" {self.num_categories}."
      )
    expected_dim = (
        1
        if (self.drop_if_binary and self.num_categories == 2)
        else self.num_categories
    )
    if not self.output_shape or self.output_shape[-1] != expected_dim:
      raise ValueError(
          "OneHotNormalizer expected trailing output_shape dimension"
          f" {expected_dim}, got {self.output_shape}."
      )

  @classmethod
  def create(
      cls,
      feature_name: str,
      input_schema: schema_lib.FeatureSchema,
      input_stats: statistics_lib.FeatureStatistics | None = None,
      *,
      num_categories: int | None = None,
      drop_if_binary: bool = False,
      max_one_hot_categories: int | None = DEFAULT_MAX_ONE_HOT_CATEGORIES,
  ) -> "OneHotNormalizer":
    """Creates a `OneHotNormalizer` from a feature schema and optional stats.

    Args:
      feature_name: Name of the categorical input feature.
      input_schema: Schema of the input feature (must be INTEGER or BOOL with a
        static shape).
      input_stats: Optional feature statistics used to infer `num_categories`
        from `round(input_stats.maximum) + 1` when neither `num_categories` nor
        `input_schema.num_categorical_values` is set.
      num_categories: Explicit category count overriding schema/stats inference.
      drop_if_binary: If True, a feature with `num_categories == 2` emits a
        single indicator column (`value == 1`) and values outside `[0, 2)`
        raise. If False (default), it emits a 2-element vector and unseen values
        encode as zeros.
      max_one_hot_categories: Maximum allowed category count before raising a
        `ValueError` to guard against accidentally creating very wide dense
        arrays (e.g., from hashed integer IDs). Pass `None` to disable.

    Returns:
      A configured `OneHotNormalizer`.
    """
    if (
        not input_schema.format.is_integer()
        and input_schema.format != schema_lib.FeatureFormat.BOOL
    ):
      raise ValueError(
          f"Feature '{feature_name}' has format '{input_schema.format}', but"
          " OneHotNormalizer only supports INTEGER or BOOL features."
      )
    if not input_schema.is_static_shape():
      raise ValueError(
          f"Feature '{feature_name}' has dynamic shape {input_schema.shape};"
          " OneHotNormalizer requires a static shape."
      )
    if max_one_hot_categories is not None and max_one_hot_categories < 1:
      raise ValueError(
          "OneHotNormalizer requires max_one_hot_categories >= 1 or None, got"
          f" {max_one_hot_categories}."
      )

    inferred_from_stats = False
    if num_categories is not None:
      resolved_categories = num_categories
    elif input_schema.num_categorical_values is not None:
      resolved_categories = input_schema.num_categorical_values
    elif input_schema.format == schema_lib.FeatureFormat.BOOL:
      resolved_categories = 2
    elif input_stats is not None:
      if input_stats.count == 0 or not math.isfinite(input_stats.maximum):
        raise ValueError(
            f"No finite maximum observed for feature '{feature_name}' in"
            " input_stats (count == 0 or non-finite maximum); cannot infer"
            " num_categories for OneHotNormalizer. Provide explicit"
            " `num_categories` or set `input_schema.num_categorical_values`."
        )
      resolved_categories = round(input_stats.maximum) + 1
      if resolved_categories < 1:
        raise ValueError(
            f"Feature '{feature_name}' has no non-negative value in"
            f" input_stats (maximum={input_stats.maximum}), e.g. only -1"
            " sentinels; cannot infer num_categories for OneHotNormalizer."
            " Drop the feature or pass `num_categories`."
        )
      inferred_from_stats = True
    else:
      raise ValueError(
          f"Cannot infer num_categories for feature '{feature_name}'; provide"
          " num_categories, input_schema.num_categorical_values, or"
          " input_stats."
      )

    if resolved_categories < 1:
      raise ValueError(
          f"Feature '{feature_name}' resolved to invalid num_categories="
          f"{resolved_categories}; expected >= 1."
      )
    if (
        max_one_hot_categories is not None
        and resolved_categories > max_one_hot_categories
    ):
      # Only a count inferred from the max of raw integer IDs can be fixed by
      # indexing them densely.
      index_hint = (
          " set `input_schema.num_categorical_values`, map sparse/hashed IDs"
          " with `DictionaryIndexNormalizer`,"
          if inferred_from_stats
          else ""
      )
      raise ValueError(
          f"Feature '{feature_name}' resolved to {resolved_categories}"
          " categories, which exceeds max_one_hot_categories="
          f"{max_one_hot_categories}. To avoid allocating a very wide dense"
          f" array per entity,{index_hint} disable `categorical_one_hot` to"
          " use learned embeddings, or increase `max_one_hot_categories`."
      )

    is_binary = drop_if_binary and resolved_categories == 2
    if is_binary and input_stats is not None and input_stats.count > 0:
      if math.isfinite(input_stats.minimum) and input_stats.minimum < 0:
        raise ValueError(
            f"Feature '{feature_name}' has minimum={input_stats.minimum} < 0"
            " in input_stats (e.g. negative sentinels), which is incompatible"
            " with drop_if_binary=True (values must be in [0, 2)). Set"
            " drop_if_binary=False to encode out-of-range values as zeros."
        )
      if math.isfinite(input_stats.maximum) and input_stats.maximum >= 2:
        raise ValueError(
            f"Feature '{feature_name}' has maximum={input_stats.maximum} >= 2"
            " in input_stats, which is incompatible with drop_if_binary=True"
            " (values must be in [0, 2)). Set drop_if_binary=False to encode"
            " out-of-range values as zeros."
        )

    out_dim = 1 if is_binary else resolved_categories
    output_shape = (*(input_schema.shape or ()), out_dim)
    return OneHotNormalizer(
        input_feature=feature_name,
        num_categories=resolved_categories,
        output_shape=output_shape,
        output_feature_name=f"{feature_name}_ONE_HOT",
        drop_if_binary=drop_if_binary,
        is_timeseries=input_schema.is_timeseries,
        group=input_schema.group,
    )

  def output_schema(self) -> schema_lib.FeatureSetSchema:
    return {
        self.output_feature_name: schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=self.output_shape,
            is_timeseries=self.is_timeseries,
            group=self.group,
        )
    }

  def _binary_range_message(self) -> str:
    return (
        f"Feature '{self.input_feature}' is encoded as a single binary column"
        " (drop_if_binary=True), so values must be in [0, 2)."
    )

  def normalize_numpy(self, value: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.asarray(value, dtype=np.int64)
    if self.drop_if_binary and self.num_categories == 2:
      out_of_range = (idx < 0) | (idx >= 2)
      if out_of_range.any():
        raise ValueError(
            f"{self._binary_range_message()} Got"
            f" {np.unique(idx[out_of_range]).tolist()}."
        )
      encoded = (idx[..., None] == 1).astype(np.float32)
    else:
      categories = np.arange(self.num_categories, dtype=np.int64)
      encoded = (idx[..., None] == categories).astype(np.float32)
    return {self.output_feature_name: encoded}

  def normalize_tensorflow(self, value: tf.Tensor) -> dict[str, tf.Tensor]:
    idx = tf.cast(value, tf.int64)
    if self.drop_if_binary and self.num_categories == 2:
      in_range = tf.reduce_all((idx >= 0) & (idx < 2))
      check = tf.debugging.Assert(
          in_range, [self._binary_range_message(), idx]
      )
      with tf.control_dependencies([check]):
        encoded = tf.cast(
            tf.equal(tf.expand_dims(idx, axis=-1), 1), tf.float32
        )
    else:
      encoded = tf.one_hot(idx, depth=self.num_categories, dtype=tf.float32)
    return {self.output_feature_name: encoded}


def _validate_fixed_numeric_schema(
    feature_name: str,
    input_schema: schema_lib.FeatureSchema,
    normalizer_name: str,
) -> tuple[int, ...] | None:
  """Validates a static-shape numeric schema and returns narrowed `int` dims."""
  if (
      not input_schema.format.is_integer()
      and not input_schema.format.is_float()
  ):
    raise ValueError(
        f"Feature '{feature_name}' has format '{input_schema.format}', but"
        f" {normalizer_name} only supports INTEGER or FLOAT features."
    )
  # The output is float32, which cannot be an index, ID, timestamp, or mask.
  if input_schema.semantic not in (
      schema_lib.FeatureSemantic.UNKNOWN,
      schema_lib.FeatureSemantic.NUMERICAL,
      schema_lib.FeatureSemantic.EMBEDDING,
      schema_lib.FeatureSemantic.TIMEDELTA,
      schema_lib.FeatureSemantic.TIMESERIES,
  ):
    raise ValueError(
        f"{normalizer_name} outputs continuous float32 values, so feature"
        f" '{feature_name}' with semantic {input_schema.semantic.value} cannot"
        " be normalized by it. Use OneHotNormalizer or"
        " DictionaryIndexNormalizer for categorical features, or keep the"
        " feature raw."
    )
  if not input_schema.is_static_shape():
    raise ValueError(
        f"Feature '{feature_name}' has dynamic shape {input_schema.shape};"
        f" {normalizer_name} requires a static shape."
    )
  if input_schema.shape is None:
    return None
  return tuple(int(dim) for dim in input_schema.shape if dim is not None)


def _validate_fit_values(
    feature_name: str,
    schema_shape: tuple[int, ...] | None,
    values: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
  """Returns `values` as float64 with NaN for non-finite, and the item shape."""
  arr = np.asarray(values)
  if arr.ndim < 1 or not (
      np.issubdtype(arr.dtype, np.integer)
      or np.issubdtype(arr.dtype, np.floating)
  ):
    raise ValueError(
        f"Feature '{feature_name}' must be a dense integer or float array with"
        f" at least 1 dimension, got dtype={arr.dtype}, shape={arr.shape}."
    )
  item_shape = tuple(int(dim) for dim in arr.shape[1:])
  output_shape = schema_shape if schema_shape is not None else item_shape
  if item_shape != output_shape:
    raise ValueError(
        f"Feature '{feature_name}' values have item shape {item_shape},"
        f" expected {output_shape}."
    )
  vals = arr.astype(np.float64)
  return np.where(np.isfinite(vals), vals, np.nan), output_shape


def _check_item_shape(
    shape: Any,
    expected: tuple[int | None, ...],
    normalizer: AbstractFeatureNormalizer,
) -> None:
  """Raises if the per-entity shape of a value does not match `expected`.

  Normalizers with per-channel parameters need this: NumPy would otherwise
  broadcast a mismatched value silently, e.g. a `(n, 1)` value against a fill
  of shape `(4,)` into `(n, 4)`. Unknown (None) dimensions match anything.

  Args:
    shape: Shape of the value, including the leading entity dimension: a
      NumPy shape tuple or a `tf.TensorShape`.
    expected: Expected per-entity shape.
    normalizer: The calling normalizer, used in the error message.
  """
  item_shape = tf.TensorShape(shape)[1:]
  if not item_shape.is_compatible_with(tf.TensorShape(expected)):
    raise ValueError(
        f"{type(normalizer).__name__} for feature"
        f" '{normalizer.input_feature}' expects values of per-entity shape"
        f" {tuple(expected)}, got {tuple(item_shape.as_list())}. Check that the"
        " feature matches the schema the normalizer was created with."
    )


@normalizer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class ImputeNanNormalizer(AbstractFeatureNormalizer):
  """Replaces non-finite (NaN and ±inf) entries in numeric features.

  The fill comes from `FeatureStatistics` (`create`, one scalar) or from
  training values (`from_values`, per channel).

  Usage example:

  ```python
  normalizer = dgf.transform.ImputeNanNormalizer.from_values(
      "score",
      dgf.data.FeatureSchema(
          format=dgf.data.FeatureFormat.FLOAT_32,
          semantic=dgf.data.FeatureSemantic.NUMERICAL,
          shape=(2,),
      ),
      train_values,
      strategy="median",
  )
  outputs = normalizer.normalize_numpy(test_values)
  ```

  Attributes:
    output_feature_name: Name of the imputed float32 output feature.
    output_shape: Static shape of each entity's feature value.
    fill_values: Finite float32 array of shape `output_shape` substituted where
      inputs are NaN or ±inf.
    output_semantic: Semantic of the output feature, taken from the input.
    is_timeseries: Whether the feature is a timeseries.
    group: Feature group of the output. Defaults to `input_feature` for a
      timeseries without a group.
  """

  output_feature_name: str
  output_shape: tuple[int, ...]
  fill_values: np.ndarray = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda x: np.asarray(x, dtype=np.float32).tolist(),
          decoder=lambda x: np.asarray(x, dtype=np.float32),
      )
  )
  output_semantic: schema_lib.FeatureSemantic = (
      schema_lib.FeatureSemantic.NUMERICAL
  )
  is_timeseries: bool = False
  group: str | None = None
  type: str = dataclasses.field(default="ImputeNanNormalizer", init=False)

  def __post_init__(self):
    # See `OneHotNormalizer.__post_init__` for the timeseries group fallback.
    if self.is_timeseries and self.group is None:
      self.group = self.input_feature
    self.fill_values = np.asarray(self.fill_values, dtype=np.float32)
    if self.fill_values.shape != tuple(self.output_shape):
      raise ValueError(
          f"ImputeNanNormalizer for feature '{self.input_feature}' expects"
          f" fill_values of shape {tuple(self.output_shape)}, got"
          f" {self.fill_values.shape}."
      )
    if not np.isfinite(self.fill_values).all():
      raise ValueError(
          f"ImputeNanNormalizer for feature '{self.input_feature}' requires"
          " finite fill_values."
      )

  @classmethod
  def create(
      cls,
      feature_name: str,
      input_schema: schema_lib.FeatureSchema,
      input_stats: statistics_lib.FeatureStatistics,
      *,
      strategy: Literal[
          "median", "mean", "zero", "median_with_fallbacks"
      ] = "median",
  ) -> "ImputeNanNormalizer":
    """Creates an `ImputeNanNormalizer` from `FeatureStatistics`.

    Args:
      feature_name: Name of the numeric input feature.
      input_schema: Schema of the input feature (static-shape INTEGER or FLOAT).
      input_stats: Feature statistics providing median quantiles or mean.
      strategy: Fill strategy:
        - `median`: Uses `np.median(input_stats.quantiles)` (mean of the two
          middle quantiles for even lengths). Raises `ValueError` if quantiles
          are missing or non-finite.
        - `mean`: Uses `input_stats.mean`. Raises `ValueError` if no finite
          mean was observed.
        - `zero`: Fills with `0.0`.
        - `median_with_fallbacks`: Uses the quantile median when available and
          finite, otherwise falls back to a finite `input_stats.mean`, and
          finally to `0.0` (with a warning) if neither is available.

    Returns:
      A configured `ImputeNanNormalizer`.
    """
    schema_shape = _validate_fixed_numeric_schema(
        feature_name, input_schema, "ImputeNanNormalizer"
    )
    output_shape: tuple[int, ...] = schema_shape or ()

    def _finite_quantile_median() -> float | None:
      if not input_stats.quantiles:
        return None
      quantiles = np.asarray(input_stats.quantiles, dtype=np.float64)
      # Any non-finite quantile means the statistics are unusable, even if
      # the middle element happens to be finite (e.g. `[1, 2, inf]`).
      if not np.isfinite(quantiles).all():
        return None
      return float(np.median(quantiles))

    if strategy == "zero":
      scalar_fill = 0.0
    elif strategy == "mean":
      if input_stats.count == 0 or not math.isfinite(input_stats.mean):
        raise ValueError(
            f"No finite mean observed for feature '{feature_name}' in"
            " input_stats; cannot create ImputeNanNormalizer with"
            " strategy='mean'."
        )
      scalar_fill = float(input_stats.mean)
    elif strategy == "median":
      med = _finite_quantile_median()
      if med is None:
        raise ValueError(
            f"No finite quantiles observed for feature '{feature_name}' in"
            " input_stats; cannot create ImputeNanNormalizer with"
            " strategy='median'. Use strategy='median_with_fallbacks' to fall"
            " back to mean or 0.0."
        )
      scalar_fill = med
    elif strategy == "median_with_fallbacks":
      med = _finite_quantile_median()
      if med is not None:
        scalar_fill = med
      elif input_stats.count > 0 and math.isfinite(input_stats.mean):
        scalar_fill = float(input_stats.mean)
      else:
        log.warning(
            f"No finite statistics observed for feature {feature_name}."
            " Defaulting ImputeNanNormalizer fill value to 0.0."
        )
        scalar_fill = 0.0
    else:
      raise ValueError(
          f"Unsupported ImputeNanNormalizer.create strategy '{strategy}'."
      )

    fill_values = np.full(output_shape, scalar_fill, dtype=np.float32)
    return ImputeNanNormalizer(
        input_feature=feature_name,
        output_feature_name=f"{feature_name}_IMPUTED",
        output_shape=output_shape,
        fill_values=fill_values,
        output_semantic=input_schema.semantic,
        is_timeseries=input_schema.is_timeseries,
        group=input_schema.group,
    )

  @classmethod
  def from_values(
      cls,
      feature_name: str,
      input_schema: schema_lib.FeatureSchema,
      values: np.ndarray,
      *,
      strategy: Literal["median", "mean", "zero"] = "median",
  ) -> "ImputeNanNormalizer":
    """Fits per-column non-finite replacement statistics from training values.

    Args:
      feature_name: Name of the numeric input feature.
      input_schema: Schema of the input feature (static-shape INTEGER or FLOAT).
      values: Array of shape `(num_entities, *feature_shape)` used to fit
        per-column fill values. NaN and ±inf are ignored. For pre-padded
        timeseries, numeric padding counts as a value, so pad with NaN to
        exclude it.
      strategy: Imputation strategy (`median`, `mean`, or `zero`; defaults to
        `median`).

    Returns:
      A fitted `ImputeNanNormalizer`.
    """
    if strategy not in ("median", "mean", "zero"):
      raise ValueError(
          f"Unsupported ImputeNanNormalizer strategy '{strategy}'; expected"
          " 'median', 'mean', or 'zero'."
      )
    schema_shape = _validate_fixed_numeric_schema(
        feature_name, input_schema, "ImputeNanNormalizer"
    )
    vals, output_shape = _validate_fit_values(
        feature_name, schema_shape, values
    )
    num_cols = int(np.prod(output_shape, dtype=np.int64))
    flat_f64 = vals.reshape(vals.shape[0], num_cols)

    if num_cols == 0 or strategy == "zero":
      flat_fill = np.zeros(num_cols, dtype=np.float64)
    else:
      if flat_f64.shape[0] == 0 or np.isnan(flat_f64).all(axis=0).any():
        raise ValueError(
            f"Feature '{feature_name}' has a column that is entirely non-finite"
            f" (NaN/±inf) and cannot be imputed with strategy='{strategy}'."
        )
      if strategy == "median":
        flat_fill = np.nanmedian(flat_f64, axis=0)
      else:
        flat_fill = np.nanmean(flat_f64, axis=0)

    return ImputeNanNormalizer(
        input_feature=feature_name,
        output_feature_name=f"{feature_name}_IMPUTED",
        output_shape=output_shape,
        fill_values=flat_fill.reshape(output_shape),
        output_semantic=input_schema.semantic,
        is_timeseries=input_schema.is_timeseries,
        group=input_schema.group,
    )

  def output_schema(self) -> schema_lib.FeatureSetSchema:
    return {
        self.output_feature_name: schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=self.output_semantic,
            shape=self.output_shape,
            is_timeseries=self.is_timeseries,
            group=self.group,
        )
    }

  def normalize_numpy(self, value: np.ndarray) -> dict[str, np.ndarray]:
    # Cast first, as in `normalize_tensorflow`: a float64 value beyond the
    # float32 range becomes ±inf and is filled, so the output stays finite.
    with np.errstate(over="ignore"):
      val = np.asarray(value).astype(np.float32)
    _check_item_shape(val.shape, self.output_shape, self)
    return {
        self.output_feature_name: np.where(
            np.isfinite(val), val, self.fill_values
        )
    }

  def normalize_tensorflow(self, value: tf.Tensor) -> dict[str, tf.Tensor]:
    val = tf.cast(value, tf.float32)
    _check_item_shape(val.shape, self.output_shape, self)
    fill = tf.broadcast_to(
        tf.constant(self.fill_values, dtype=tf.float32), tf.shape(val)
    )
    return {self.output_feature_name: tf.where(tf.math.is_finite(val), val, fill)}


def _z_score_scale(std: Any, feature_name: str) -> np.ndarray:
  """Validates `std` and maps 0 (a constant channel) to 1."""
  std_arr = np.asarray(std, dtype=np.float64)
  if (~np.isfinite(std_arr) | (std_arr < 0.0)).any():
    raise ValueError(
        f"ZScoreNormalizer for feature '{feature_name}' requires finite,"
        " non-negative std values."
    )
  return np.where(std_arr == 0.0, 1.0, std_arr)


@normalizer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class ZScoreNormalizer(AbstractFeatureNormalizer):
  """Standardizes numeric or embedding features via `(value - mean) / std`.

  `mean` and `std` are float64 and `value - mean` runs in float64, so inputs
  with a large offset (e.g. `1e8 + small signal`) keep their signal. A constant
  channel gets `std = 1`, so it is centered only. A channel that varies only by
  rounding noise is not constant, so a slightly different serving value can
  map to a large output. NaN and ±inf inputs pass through.

  Usage example:

  ```python
  normalizer = dgf.transform.ZScoreNormalizer.from_values(
      "feat",
      dgf.data.FeatureSchema(
          format=dgf.data.FeatureFormat.FLOAT_32,
          semantic=dgf.data.FeatureSemantic.EMBEDDING,
          shape=(16,),
      ),
      train_embeddings,
  )
  outputs = normalizer.normalize_numpy(test_embeddings)
  ```

  Attributes:
    output_feature_name: Name of the standardized float32 output feature.
    output_shape: Static shape of each entity's feature value.
    mean: Finite float64 per-element mean of shape `output_shape`.
    std: Positive, finite float64 per-element scale of shape `output_shape`.
    output_semantic: Semantic of the output feature, taken from the input.
    is_timeseries: Whether the feature is a timeseries.
    group: Feature group of the output. Defaults to `input_feature` for a
      timeseries without a group.
  """

  output_feature_name: str
  output_shape: tuple[int, ...]
  mean: np.ndarray = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda x: np.asarray(x, dtype=np.float64).tolist(),
          decoder=lambda x: np.asarray(x, dtype=np.float64),
      )
  )
  std: np.ndarray = dataclasses.field(
      metadata=dataclasses_json.config(
          encoder=lambda x: np.asarray(x, dtype=np.float64).tolist(),
          decoder=lambda x: np.asarray(x, dtype=np.float64),
      )
  )
  output_semantic: schema_lib.FeatureSemantic = (
      schema_lib.FeatureSemantic.NUMERICAL
  )
  is_timeseries: bool = False
  group: str | None = None
  type: str = dataclasses.field(default="ZScoreNormalizer", init=False)

  def __post_init__(self):
    # See `OneHotNormalizer.__post_init__` for the timeseries group fallback.
    if self.is_timeseries and self.group is None:
      self.group = self.input_feature
    self.mean = np.asarray(self.mean, dtype=np.float64)
    self.std = np.asarray(self.std, dtype=np.float64)
    for name, param in (("mean", self.mean), ("std", self.std)):
      if param.shape != tuple(self.output_shape):
        raise ValueError(
            f"ZScoreNormalizer for feature '{self.input_feature}' expects"
            f" {name} of shape {tuple(self.output_shape)}, got {param.shape}."
        )
    if not np.isfinite(self.mean).all():
      raise ValueError(
          f"ZScoreNormalizer for feature '{self.input_feature}' requires"
          " finite mean values."
      )
    self.std = _z_score_scale(self.std, self.input_feature)

  @classmethod
  def from_values(
      cls,
      feature_name: str,
      input_schema: schema_lib.FeatureSchema,
      values: np.ndarray,
  ) -> "ZScoreNormalizer":
    """Fits per-element mean and standard deviation across the leading axis.

    Args:
      feature_name: Name of the numeric or embedding input feature.
      input_schema: Schema of the input feature (static-shape INTEGER or FLOAT).
      values: Array of shape `(num_entities, *feature_shape)`. NaN and ±inf are
        ignored. For pre-padded timeseries, numeric padding counts as a value,
        so pad with NaN to exclude it.

    Returns:
      A fitted `ZScoreNormalizer`.
    """
    schema_shape = _validate_fixed_numeric_schema(
        feature_name, input_schema, "ZScoreNormalizer"
    )
    vals, output_shape = _validate_fit_values(
        feature_name, schema_shape, values
    )
    arr = np.asarray(values)
    # float64 merges distinct integers beyond 2**53, e.g. nanosecond times.
    # Compared on both sides, as `np.abs` overflows on the int64 minimum.
    if np.issubdtype(arr.dtype, np.integer) and (
        (arr > 2**53) | (arr < -(2**53))
    ).any():
      raise ValueError(
          f"Feature '{feature_name}' has integer values beyond 2**53, which"
          " float64 cannot represent exactly; subtract an offset first."
      )
    num_cols = int(np.prod(output_shape, dtype=np.int64))
    if num_cols == 0:
      mean = np.zeros(output_shape, dtype=np.float64)
      std = np.ones(output_shape, dtype=np.float64)
    else:
      if vals.shape[0] == 0 or np.isnan(vals).all(axis=0).any():
        raise ValueError(
            f"Feature '{feature_name}' has no values or a channel that is"
            " entirely non-finite (NaN/±inf); cannot fit ZScoreNormalizer."
        )
      lo = np.nanmin(vals, axis=0)
      hi = np.nanmax(vals, axis=0)
      constant = hi == lo
      # Shifting by `lo` keeps the sum small, so the mean of large values is
      # not rounded. A constant channel gets std 0, which `_z_score_scale`
      # maps to 1, so it normalizes to exactly 0.
      mean = lo + np.nanmean(vals - lo, axis=0)
      std = np.where(constant, 0.0, np.nanstd(vals - lo, axis=0))
      if ((std == 0.0) & ~constant).any() or not np.isfinite(std).all():
        raise ValueError(
            f"Feature '{feature_name}' has a spread whose standard deviation"
            " underflows or overflows float64; rescale the feature first."
        )

    return ZScoreNormalizer(
        input_feature=feature_name,
        output_feature_name=f"{feature_name}_Z_SCORE",
        output_shape=output_shape,
        mean=mean,
        std=std,
        output_semantic=input_schema.semantic,
        is_timeseries=input_schema.is_timeseries,
        group=input_schema.group,
    )

  def output_schema(self) -> schema_lib.FeatureSetSchema:
    return {
        self.output_feature_name: schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=self.output_semantic,
            shape=self.output_shape,
            is_timeseries=self.is_timeseries,
            group=self.group,
        )
    }

  def normalize_numpy(self, value: np.ndarray) -> dict[str, np.ndarray]:
    # `value - mean` runs in float64: a float32 subtraction would cancel the
    # signal of inputs with a large offset (float32 ulp at 1e8 is 8).
    val = np.asarray(value, dtype=np.float64)
    _check_item_shape(val.shape, self.output_shape, self)
    return {
        self.output_feature_name: ((val - self.mean) / self.std).astype(
            np.float32
        )
    }

  def normalize_tensorflow(self, value: tf.Tensor) -> dict[str, tf.Tensor]:
    val = tf.cast(value, tf.float64)
    _check_item_shape(val.shape, self.output_shape, self)
    mean = tf.constant(self.mean, dtype=tf.float64)
    std = tf.constant(self.std, dtype=tf.float64)
    return {self.output_feature_name: tf.cast((val - mean) / std, tf.float32)}


@normalizer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class SequentialNormalizer(AbstractFeatureNormalizer):
  """Applies multiple AbstractFeatureNormalizers in sequence.

  Executes normalizer stages sequentially and emits only the final stage output.
  """

  stages: list[AbstractFeatureNormalizer] = normalizer_registry.field_list()
  type: str = dataclasses.field(default="SequentialNormalizer", init=False)
  _stage_kwargs: list[frozenset[str]] = dataclasses.field(
      default_factory=list,
      init=False,
      metadata=dataclasses_json.config(exclude=dataclasses_json.Exclude.ALWAYS),
  )

  def __post_init__(self):
    _validate_stages(self.input_feature, self.stages)
    self._stage_kwargs: list[frozenset[str]] = [
        _accepted_kwargs(s) for s in self.stages
    ]

  @classmethod
  def create(
      cls,
      stages: Sequence[AbstractFeatureNormalizer],
  ) -> "SequentialNormalizer":
    if not stages:
      raise ValueError("SequentialNormalizer requires at least one stage.")
    return SequentialNormalizer(
        input_feature=stages[0].input_feature,
        stages=list(stages),
    )

  def output_schema(self) -> schema_lib.FeatureSetSchema:
    return self.stages[-1].output_schema()

  def _normalize(
      self,
      stage_fn_getter: Callable[[AbstractFeatureNormalizer], Any],
      value: Any,
      **kwargs: Any,
  ) -> dict[str, Any]:
    current_features: dict[str, Any] = {self.input_feature: value}
    stage_output: dict[str, Any] = {}
    for stage, accepted_kwargs in zip(self.stages, self._stage_kwargs):
      assert stage.input_feature in current_features, (
          f"Stage '{stage.type}' expects input feature '{stage.input_feature}',"
          f" but available features are {list(current_features.keys())}."
      )
      stage_input = current_features[stage.input_feature]
      stage_output = _filter_and_call(
          stage_fn_getter(stage), stage_input, accepted_kwargs, kwargs
      )
      current_features.update(stage_output)
    return stage_output

  def normalize_numpy(
      self,
      value: np.ndarray,
      **kwargs: Any,
  ) -> dict[str, np.ndarray]:
    return self._normalize(lambda s: s.normalize_numpy, value, **kwargs)

  def normalize_tensorflow(
      self,
      value: tf.Tensor,
      **kwargs: Any,
  ) -> dict[str, tf.Tensor]:
    return self._normalize(lambda s: s.normalize_tensorflow, value, **kwargs)

  def tensorflow_resources(self) -> list[tf.Tensor]:
    resources = []
    for stage in self.stages:
      resources.extend(stage.tensorflow_resources())
    return resources


def _validate_stages(
    input_feature: str,
    stages: Sequence[AbstractFeatureNormalizer],
) -> None:
  """Validates that stages form a valid, connected sequential pipeline."""
  if not stages:
    raise ValueError("SequentialNormalizer requires at least one stage.")
  if input_feature != stages[0].input_feature:
    raise ValueError(
        f"SequentialNormalizer input_feature '{input_feature}' does not"
        f" match first stage input_feature '{stages[0].input_feature}'."
    )
  available = {input_feature}
  for stage in stages:
    if stage.input_feature not in available:
      raise ValueError(
          f"Stage '{stage.type}' expects input feature"
          f" '{stage.input_feature}', but available features up to this stage"
          f" are {sorted(available)}."
      )
    available.update(stage.output_schema().keys())


def _accepted_kwargs(stage: AbstractFeatureNormalizer) -> frozenset[str]:
  """Returns the set of accepted keyword argument names."""
  if isinstance(stage, SequentialNormalizer):
    return frozenset().union(*stage._stage_kwargs)
  np_params = list(inspect.signature(stage.normalize_numpy).parameters)[1:]
  tf_params = list(inspect.signature(stage.normalize_tensorflow).parameters)[1:]
  if set(np_params) != set(tf_params):
    raise ValueError(
        f"Stage '{stage.type}' has mismatched kwargs between normalize_numpy "
        f"({np_params}) and normalize_tensorflow ({tf_params})."
    )
  return frozenset(np_params)


def _filter_and_call(
    method: Any,
    value: Any,
    accepted_kwargs: Set[str],
    kwargs: dict[str, Any],
) -> Any:
  if not accepted_kwargs:
    return method(value)
  filtered = {k: kwargs[k] for k in accepted_kwargs if k in kwargs}
  return method(value, **filtered)


def _create_timestamp_normalizer(
    feature_name: str,
    feature_schema: schema_lib.FeatureSchema,
    embedding_dim: int = 32,
) -> SequentialNormalizer:
  """Creates a sequential normalizer for TIMESTAMP features."""
  timedelta_norm = TimedeltaNormalizer.create(feature_name, feature_schema)
  delta_schema = timedelta_norm.output_schema()[
      timedelta_norm.output_feature_name
  ]
  sinusoid_norm = SinusoidTimedeltaNormalizer.create(
      timedelta_norm.output_feature_name,
      delta_schema,
      embedding_dim=embedding_dim,
  )
  return SequentialNormalizer.create([timedelta_norm, sinusoid_norm])


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class AutoNormalizeConfig:
  """Configuration for automatic feature normalization for GNNs.

  Attributes:
    categorical_bytes_to_index: If True, categorical features with BYTES format
      will be converted to integer indices using `DictionaryIndexNormalizer`.
    categorical_one_hot: If True, categorical features (BYTES when
      `categorical_bytes_to_index` is True, INTEGER, or BOOL) are one-hot
      encoded into float32 embeddings using `OneHotNormalizer`. INTEGER
      categorical features without `num_categorical_values` require non-empty
      finite statistics (empty BYTES categorical features emit a 1-column
      all-zero vector).
    categorical_one_hot_drop_if_binary: If True and `categorical_one_hot` is
      True, categorical features with exactly two categories emit a single
      indicator column and raise on an unseen value. If False (default), they
      emit a 2-element vector and unseen values encode as zeros.
    max_one_hot_categories: Maximum category count allowed by `OneHotNormalizer`
      when `categorical_one_hot` is True (defaults to
      `DEFAULT_MAX_ONE_HOT_CATEGORIES`). Pass `None` to disable.
    numerical_soft_quantile: If True, numerical features (INTEGER or FLOAT) will
      be normalized using `SoftQuantileNormalizer`.
    keep_raw_features: A set of `(nodeset_name, feature_name)` tuples that
      should bypass all normalization and be included in the output graph as-is.
      This is useful for preserving features like unique identifiers or labels
      not intended for model input.
    ignore_features_without_stats: Whether to ignore features that are not in
      `keep_raw_features` and do not have associated statistics. If `False`
      (default), an error is raised. If `True`, such features are skipped.
    consume_primary_keys: If True, features marked as PRIMARY_ID will be
      normalized. If false, primary key features are skipped.
    primary_keys_num_hash_buckets: Number of hash buckets to use when
      normalizing primary key features.
    timestamp_normalize: If True, features with TIMESTAMP semantic will be
      normalized into a sinusoidal embedding relative to seed timestamps using a
      sequential `TimedeltaNormalizer` -> `SinusoidTimedeltaNormalizer`.
    timedelta_normalize: If True, features with TIMEDELTA semantic will be
      normalized using `SinusoidTimedeltaNormalizer`.
    timedelta_embedding_dim: Embedding dimension to use when normalizing
      TIMESTAMP or TIMEDELTA features with sinusoidal embeddings.
    has_seed_timestamps: Whether seed timestamps will be provided at runtime for
      timestamp normalization. If False, timestamp features will not be
      normalized using relative timedelta embeddings.
    calendar_normalize: If True, features with TIMESTAMP semantic additionally
      produce one rescaled feature per entry in `calendar_features`, using a
      `CalendarNormalizer`. This is independent of `timestamp_normalize`:
      calendar features encode the absolute wall-clock position of a timestamp,
      whereas `timestamp_normalize` encodes its recency relative to the seed.
    calendar_features: The calendar components to extract when
      `calendar_normalize` is True.
    text_normalize: If True, features with TEXT semantic and BYTES format will
      be normalized into dense embeddings using `TextEmbeddingNormalizer`.
    text_embedder: Name of a built-in text embedder, a model handle/path, or a
      `TextEmbedderConfig` used when `text_normalize` is True.
    numerical_missing_value_policy: What to do when the statistics of a float
      feature report NaN or ±inf values. Checked once, when the normalizer is
      created; values at normalization time are not checked.
  """

  categorical_bytes_to_index: bool = True
  categorical_one_hot: bool = False
  categorical_one_hot_drop_if_binary: bool = False
  max_one_hot_categories: int | None = DEFAULT_MAX_ONE_HOT_CATEGORIES
  numerical_soft_quantile: bool = True
  keep_raw_features: set[tuple[str, str]] = dataclasses.field(
      default_factory=set,
      metadata=dataclasses_json.config(
          encoder=lambda s: [list(x) for x in s],
          decoder=lambda l: set(tuple(x) for x in l),
      ),
  )
  ignore_features_without_stats: bool = False
  consume_primary_keys: bool = False
  primary_keys_num_hash_buckets: int = 256
  # TODO(simonmeierhans): Enable by default after updating callers to pass
  # timestamps to the normalizer.
  timestamp_normalize: bool = False
  timedelta_normalize: bool = True
  timedelta_embedding_dim: int = 32
  has_seed_timestamps: bool = False
  # TODO(simonmeierhans): Consider enabling by default after benchmarking.
  calendar_normalize: bool = False
  calendar_features: tuple[CalendarFeature, ...] = DEFAULT_CALENDAR_FEATURES
  text_normalize: bool = True
  text_embedder: str | normalize_text.TextEmbedderConfig = "nnlm-en-dim128"
  numerical_missing_value_policy: NumericalMissingValuePolicy = (
      dataclasses.field(
          default=NumericalMissingValuePolicy.ERROR,
          metadata=dataclasses_json.config(
              decoder=NumericalMissingValuePolicy
          ),
      )
  )


def _check_numerical_missing_values(
    nodeset_name: str,
    feature_name: str,
    feature_stats: statistics_lib.FeatureStatistics,
    policy: NumericalMissingValuePolicy,
) -> None:
  """Applies `policy` to the NaN/±inf counts of a numerical float feature."""
  if policy == NumericalMissingValuePolicy.ERROR:
    num_missing = (
        feature_stats.num_nan
        + feature_stats.num_posinf
        + feature_stats.num_neginf
    )
    if num_missing:
      raise ValueError(
          f"Feature '{feature_name}' in node set '{nodeset_name}' has"
          f" {feature_stats.num_nan} NaN, {feature_stats.num_posinf} +inf and"
          f" {feature_stats.num_neginf} -inf values in its statistics, which"
          " conflicts with numerical_missing_value_policy='error'. Please"
          " clean or impute these values."
      )
  else:
    raise ValueError(f"Unsupported numerical missing value policy: {policy!r}")


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

  # Validate keep_raw_features.
  for entry in config.keep_raw_features:
    if (
        not isinstance(entry, (tuple, list))
        or len(entry) != 2
        or not isinstance(entry[0], str)
        or not isinstance(entry[1], str)
    ):
      raise ValueError(
          f"Invalid entry in `keep_raw_features`: {entry!r}. Expected a"
          " (nodeset_or_edgeset_name, feature_name) tuple of strings."
      )
    set_name, feature_name = entry
    set_schema = schema.node_sets.get(set_name) or schema.edge_sets.get(
        set_name
    )
    if set_schema is None:
      raise ValueError(
          f"Set '{set_name}' from `keep_raw_features` not found in graph node"
          " sets or edge sets."
      )
    if feature_name not in set_schema.features:
      raise ValueError(
          f"Feature '{feature_name}' from `keep_raw_features` not found in set"
          f" '{set_name}'."
      )

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
      # Only checks the statistics for NaN/±inf here; the normalizers are
      # created below. This runs first so that raw (kept) features, e.g. a
      # regression label, are checked too.
      if (
          feature_schema.semantic == schema_lib.FeatureSemantic.NUMERICAL
          and feature_schema.format.is_float()
          and feature_name in nodeset_stats.features
      ):
        _check_numerical_missing_values(
            nodeset_name,
            feature_name,
            nodeset_stats.features[feature_name],
            config.numerical_missing_value_policy,
        )
      if (
          (nodeset_name, feature_name) in config.keep_raw_features
          # Mask features should be preserved.
          or feature_schema.semantic == schema_lib.FeatureSemantic.MASK
      ):
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
        dict_norm = DictionaryIndexNormalizer.create(
            feature_name, feature_schema, feature_stats
        )
        if config.categorical_one_hot:
          # `DictionaryIndexNormalizer` drops the timeseries flag and group
          # from its index schema; restore them from the original feature so
          # the one-hot output keeps its (implicit) sequence group.
          idx_schema = dataclasses.replace(
              dict_norm.output_schema()[dict_norm.output_feature_name],
              is_timeseries=feature_schema.is_timeseries,
              group=_output_timeseries_group(feature_name, feature_schema),
          )
          num_categories = len(dict_norm.dictionary_map)
          # Without any observed value, `DictionaryIndexNormalizer` uses a
          # placeholder dictionary whose out-of-vocabulary index (0) collides
          # with the placeholder entry. Move it outside `[0, num_categories)`
          # (together with the placeholder entry itself) so that every value
          # one-hot encodes as zeros, as documented for unseen values.
          if dict_norm.out_of_vocab_value < num_categories:
            dict_norm = dataclasses.replace(
                dict_norm,
                dictionary_map=dict.fromkeys(
                    dict_norm.dictionary_map, num_categories
                ),
                out_of_vocab_value=num_categories,
            )
          one_hot_norm = OneHotNormalizer.create(
              dict_norm.output_feature_name,
              idx_schema,
              num_categories=num_categories,
              drop_if_binary=config.categorical_one_hot_drop_if_binary,
              max_one_hot_categories=config.max_one_hot_categories,
          )
          nodeset_normalizers.append(
              SequentialNormalizer.create([dict_norm, one_hot_norm])
          )
        else:
          nodeset_normalizers.append(dict_norm)
        feature_has_normalized = True

      # Text bytes embedding
      if (
          config.text_normalize
          and feature_schema.semantic == schema_lib.FeatureSemantic.TEXT
          and feature_schema.format == schema_lib.FeatureFormat.BYTES
      ):
        nodeset_normalizers.append(
            TextEmbeddingNormalizer.create(
                feature_name,
                feature_schema,
                embedder=config.text_embedder,
            )
        )
        feature_has_normalized = True

      # Categorical integer (or boolean when one-hot encoding)
      if feature_schema.semantic == schema_lib.FeatureSemantic.CATEGORICAL and (
          feature_schema.format.is_integer()
          or (
              config.categorical_one_hot
              and feature_schema.format == schema_lib.FeatureFormat.BOOL
          )
      ):
        if config.categorical_one_hot:
          nodeset_normalizers.append(
              OneHotNormalizer.create(
                  feature_name,
                  feature_schema,
                  feature_stats,
                  drop_if_binary=config.categorical_one_hot_drop_if_binary,
                  max_one_hot_categories=config.max_one_hot_categories,
              )
          )
        else:
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

      # Timedelta
      if (
          config.timedelta_normalize
          and feature_schema.semantic == schema_lib.FeatureSemantic.TIMEDELTA
      ):
        nodeset_normalizers.append(
            SinusoidTimedeltaNormalizer.create(
                feature_name,
                feature_schema,
                embedding_dim=config.timedelta_embedding_dim,
            )
        )
        feature_has_normalized = True

      # Timestamp
      if (
          config.timestamp_normalize
          and feature_schema.semantic == schema_lib.FeatureSemantic.TIMESTAMP
      ):
        if not feature_schema.is_static_shape():
          log.warning(
              f"Feature '{feature_name}' in node set '{nodeset_name}' is a"
              " TIMESTAMP with dynamic shape and cannot be normalized with"
              " sinusoidal timestamp embeddings. Please run padding first."
          )
        elif not config.has_seed_timestamps:
          log.warning(
              f"Feature '{feature_name}' in node set '{nodeset_name}' is a "
              "TIMESTAMP but cannot be autonormalized because no seed "
              "timestamps are available."
          )
        else:
          nodeset_normalizers.append(
              _create_timestamp_normalizer(
                  feature_name,
                  feature_schema,
                  embedding_dim=config.timedelta_embedding_dim,
              )
          )
          feature_has_normalized = True

      # Calendar
      if (
          config.calendar_normalize
          and feature_schema.semantic == schema_lib.FeatureSemantic.TIMESTAMP
      ):
        nodeset_normalizers.append(
            CalendarNormalizer.create(
                feature_name, feature_schema, config.calendar_features
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

  normalizers: list[AbstractFeatureNormalizer] = (
      normalizer_registry.field_list()
  )


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class EdgeSetNormalizerConfig:
  """Raw information of a EdgeSetNormalizer for easy serialization."""

  source: str
  target: str
  normalizers: list[AbstractFeatureNormalizer] = (
      normalizer_registry.field_list()
  )


def _normalizer_summary(normalizer: AbstractFeatureNormalizer) -> str:
  """Returns a one-line human-readable summary of a feature normalizer."""
  summary = f"{normalizer.input_feature}: {normalizer.type}"
  if isinstance(normalizer, TextEmbeddingNormalizer):
    summary += (
        f" ({normalizer.embedder_handle}, dim={normalizer.embedding_dim})"
    )
  return summary


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class GraphNormalizerConfig:
  """Raw information of a GraphNormalizer for easy serialization."""

  nodesets: dict[str, NodeSetNormalizerConfig]
  edgesets: dict[str, EdgeSetNormalizerConfig]

  def make(self) -> "GraphNormalizer":
    return GraphNormalizer(config=self)

  def nice_print(self, return_output: bool = False) -> str | None:
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
            lines.append(f"    - {_normalizer_summary(normalizer)}")
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
            lines.append(f"    - {_normalizer_summary(normalizer)}")
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

  A graph normalizer prepares feature values before they can be consumed by a
  core GNN model.
  """

  config: GraphNormalizerConfig
  _nodeset_kwargs: dict[str, list[Set[str]]] = dataclasses.field(
      default_factory=dict, init=False
  )
  _edgeset_kwargs: dict[str, list[Set[str]]] = dataclasses.field(
      default_factory=dict, init=False
  )
  _all_accepted_kwargs: frozenset[str] = dataclasses.field(
      default_factory=frozenset, init=False
  )

  def __post_init__(self):
    self._nodeset_kwargs: dict[str, list[Set[str]]] = {
        name: [_accepted_kwargs(s) for s in cfg.normalizers]
        for name, cfg in self.config.nodesets.items()
    }
    self._edgeset_kwargs: dict[str, list[Set[str]]] = {
        name: [_accepted_kwargs(s) for s in cfg.normalizers]
        for name, cfg in self.config.edgesets.items()
    }
    all_accepted_kwargs = set()
    for accepted_kwargs_list in self._nodeset_kwargs.values():
      all_accepted_kwargs.update(*accepted_kwargs_list)
    for accepted_kwargs_list in self._edgeset_kwargs.values():
      all_accepted_kwargs.update(*accepted_kwargs_list)
    self._all_accepted_kwargs: frozenset[str] = frozenset(all_accepted_kwargs)

  @property
  def accepted_kwargs(self) -> frozenset[str]:
    """Returns the set of accepted keyword arguments for this normalizer."""
    return self._all_accepted_kwargs

  def _validate_kwargs(self, kwargs: dict[str, Any]) -> None:
    # If no keyword arguments are provided, we return early.
    if not kwargs:
      return
    unexpected_kwargs = set(kwargs) - self._all_accepted_kwargs
    if unexpected_kwargs:
      raise ValueError(
          "Unexpected keyword arguments for GraphNormalizer: "
          f"{sorted(unexpected_kwargs)}. "
          f"Accepted arguments: {sorted(self._all_accepted_kwargs)}."
      )
    for kwarg_name, entity_dict in kwargs.items():
      if not isinstance(entity_dict, Mapping):
        raise ValueError(
            f"Keyword argument '{kwarg_name}' to GraphNormalizer must be a dict"
            " mapping entity names (nodeset and edgeset names) to"
            f" arrays/tensors, got {type(entity_dict).__name__}."
        )

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
  ) -> list[str]:
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
      self,
      graph: in_memory_graph.InMemoryGraph,
      **kwargs: Any,
  ) -> in_memory_graph.InMemoryGraph:
    """Normalizes the features of the input graph.

    Args:
      graph: The input `InMemoryGraph` with raw feature values.
      **kwargs: Additional keyword arguments forwarded to feature normalizers
        (e.g., `seed_timestamps`). Each keyword argument must be a `Dict[str,
        np.ndarray]` keyed by entity (node set or edge set) name.

    Returns:
      A new `InMemoryGraph` with normalized feature values.
    """
    # Raises an error if a keyword argument is not accepted by any normalizer.
    self._validate_kwargs(kwargs)

    # Normalize edgeset features
    dst_graph_edge_sets = {}
    for edgeset_name, edgeset in self.config.edgesets.items():
      input_edgeset = graph.edge_sets[edgeset_name]
      output_features = {}
      entity_kwargs = {
          kwarg_name: entity_dict[edgeset_name]
          for kwarg_name, entity_dict in kwargs.items()
          if edgeset_name in entity_dict
      }
      for normalizer, accepted_kwargs in zip(
          edgeset.normalizers, self._edgeset_kwargs[edgeset_name]
      ):
        input_feature_value = input_edgeset.features[normalizer.input_feature]
        output_features.update(
            _filter_and_call(
                normalizer.normalize_numpy,
                input_feature_value,
                accepted_kwargs,
                entity_kwargs,
            )
        )
      dst_graph_edge_sets[edgeset_name] = in_memory_graph.InMemoryEdgeSet(
          adjacency=input_edgeset.adjacency,
          features=output_features,
      )

    # Normalize nodeset features
    dst_graph_node_sets = {}
    for nodeset_name, nodeset in self.config.nodesets.items():
      input_nodeset = graph.node_sets[nodeset_name]
      output_features = {}
      entity_kwargs = {
          kwarg_name: entity_dict[nodeset_name]
          for kwarg_name, entity_dict in kwargs.items()
          if nodeset_name in entity_dict
      }
      for normalizer, accepted_kwargs in zip(
          nodeset.normalizers, self._nodeset_kwargs[nodeset_name]
      ):
        if normalizer.input_feature not in input_nodeset.features:
          continue
        input_feature_value = input_nodeset.features[normalizer.input_feature]
        output_features.update(
            _filter_and_call(
                normalizer.normalize_numpy,
                input_feature_value,
                accepted_kwargs,
                entity_kwargs,
            )
        )
      dst_graph_node_sets[nodeset_name] = in_memory_graph.InMemoryNodeSet(
          features=output_features,
          num_nodes=input_nodeset.num_nodes,
      )

    dst_graph = in_memory_graph.InMemoryGraph(
        node_sets=dst_graph_node_sets, edge_sets=dst_graph_edge_sets
    )
    return dst_graph

  def tensorflow_resources(self) -> list[tf.Tensor]:
    """Returns all the tf resources of all the operations."""
    set_configs = [
        *self.config.edgesets.values(),
        *self.config.nodesets.values(),
    ]
    resources = {}
    for set_config in set_configs:
      for normalizer in set_config.normalizers:
        for res in normalizer.tensorflow_resources():
          resources.setdefault(id(res), res)
    return list(resources.values())

  def normalize_tensorflow(
      self,
      graph: tf_in_memory_graph.TFInMemoryGraph,
      **kwargs: Any,
  ) -> tf_in_memory_graph.TFInMemoryGraph:
    """Normalizes the features of the input graph using tensorflow.

    Args:
      graph: The input `InMemoryGraph` with raw feature values.
      **kwargs: Additional keyword arguments forwarded to feature normalizers
        (e.g., `seed_timestamps`). Each keyword argument must be a `Dict[str,
        tf.Tensor]` keyed by entity (node set or edge set) name.

    Returns:
      A new `InMemoryGraph` with normalized feature values.
    """
    self._validate_kwargs(kwargs)

    # Normalize edgeset features
    dst_graph_edge_sets = {}
    for edgeset_name, edgeset in self.config.edgesets.items():
      input_edgeset = graph.edge_sets[edgeset_name]
      output_features = {}
      entity_kwargs = {
          kwarg_name: entity_dict[edgeset_name]
          for kwarg_name, entity_dict in kwargs.items()
          if edgeset_name in entity_dict
      }
      for normalizer, accepted_kwargs in zip(
          edgeset.normalizers, self._edgeset_kwargs[edgeset_name]
      ):
        input_feature_value = input_edgeset.features[normalizer.input_feature]
        output_features.update(
            _filter_and_call(
                normalizer.normalize_tensorflow,
                input_feature_value,
                accepted_kwargs,
                entity_kwargs,
            )
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
      entity_kwargs = {
          kwarg_name: entity_dict[nodeset_name]
          for kwarg_name, entity_dict in kwargs.items()
          if nodeset_name in entity_dict
      }
      for normalizer, accepted_kwargs in zip(
          nodeset.normalizers, self._nodeset_kwargs[nodeset_name]
      ):
        if normalizer.input_feature not in input_nodeset.features:
          continue
        input_feature_value = input_nodeset.features[normalizer.input_feature]
        output_features.update(
            _filter_and_call(
                normalizer.normalize_tensorflow,
                input_feature_value,
                accepted_kwargs,
                entity_kwargs,
            )
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
      **kwargs: Any,
  ) -> jax_in_memory_graph.JaxInMemoryGraph:
    """Normalizes graph features using numpy and returns jax arrays.

    This function is equivalent, but consumes less memory than, calling
    "normalize" + "graph_to_jax_graph".

    Args:
      graph: The input `InMemoryGraph` with raw feature values.
      include_adjacencies: If True, the adjacency information from the input
        graph will be included in the output `JaxInMemoryGraph`. Otherwise, the
        output graph's edge sets will have `None` for adjacency.
      **kwargs: Additional keyword arguments forwarded to feature normalizers
        (e.g., `seed_timestamps`). Each keyword argument must be a `Dict[str,
        np.ndarray]` keyed by entity (node set or edge set) name.

    Returns:
      A new `JaxInMemoryGraph` with normalized feature values.
    """
    self._validate_kwargs(kwargs)

    def asarray(x):
      return jnp.asarray(x)

    # Normalize edgeset features
    dst_graph_edge_sets = {}
    for edgeset_name, edgeset in self.config.edgesets.items():
      input_edgeset = graph.edge_sets[edgeset_name]
      output_features = {}
      entity_kwargs = {
          kwarg_name: entity_dict[edgeset_name]
          for kwarg_name, entity_dict in kwargs.items()
          if edgeset_name in entity_dict
      }
      for normalizer, accepted_kwargs in zip(
          edgeset.normalizers, self._edgeset_kwargs[edgeset_name]
      ):
        input_feature_value = input_edgeset.features[normalizer.input_feature]
        output_features.update({
            k: asarray(v)
            for k, v in _filter_and_call(
                normalizer.normalize_numpy,
                input_feature_value,
                accepted_kwargs,
                entity_kwargs,
            ).items()
        })
      dst_graph_edge_sets[edgeset_name] = (
          jax_in_memory_graph.JaxInMemoryEdgeSet(
              adjacency=(
                  asarray(input_edgeset.adjacency)  # pyrefly: ignore[bad-argument-type]
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
      entity_kwargs = {
          kwarg_name: entity_dict[nodeset_name]
          for kwarg_name, entity_dict in kwargs.items()
          if nodeset_name in entity_dict
      }
      for normalizer, accepted_kwargs in zip(
          nodeset.normalizers, self._nodeset_kwargs[nodeset_name]
      ):
        if normalizer.input_feature not in input_nodeset.features:
          continue
        input_feature_value = input_nodeset.features[normalizer.input_feature]
        output_features.update({
            k: asarray(v)
            for k, v in _filter_and_call(
                normalizer.normalize_numpy,
                input_feature_value,
                accepted_kwargs,
                entity_kwargs,
            ).items()
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

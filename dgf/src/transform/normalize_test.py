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

import math

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
from dgf.src.io import tf as tf_io
from dgf.src.transform import normalize as normalize_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np
import tensorflow as tf


def _reference_calendar_feature(
    timestamps: np.ndarray, calendar_feature: normalize_lib.CalendarFeature
) -> np.ndarray:
  """Computes a calendar component independently of `normalize`.

  Uses numpy's datetime64 calendar for day-of-month and month instead of the
  closed-form integer arithmetic under test, so the two disagree if either is wrong.
  """
  if calendar_feature == normalize_lib.CalendarFeature.SECOND:
    return (timestamps % 60).astype(np.float32)
  if calendar_feature == normalize_lib.CalendarFeature.MINUTE:
    return ((timestamps // 60) % 60).astype(np.float32)
  if calendar_feature == normalize_lib.CalendarFeature.HOUR:
    return ((timestamps // 3600) % 24).astype(np.float32)
  if calendar_feature == normalize_lib.CalendarFeature.DAY_OF_WEEK:
    return (((timestamps // 86400) + 3) % 7).astype(np.float32)
  if calendar_feature == normalize_lib.CalendarFeature.DAY_OF_MONTH:
    dt = timestamps.astype("datetime64[s]")
    return (
        dt.astype("datetime64[D]").astype(int)
        - dt.astype("datetime64[M]").astype("datetime64[D]").astype(int)
        + 1
    ).astype(np.float32)
  if calendar_feature == normalize_lib.CalendarFeature.MONTH:
    months = timestamps.astype("datetime64[s]").astype("datetime64[M]")
    return (months.astype(int) % 12 + 1).astype(np.float32)
  raise ValueError(f"Unsupported calendar feature: '{calendar_feature}'.")


class DictionaryIndexNormalizerTest(absltest.TestCase):

  def test_basic(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
    )
    input_stats = statistics_lib.FeatureStatistics(
        count=3,
        minimum=0,
        maximum=0,
        dictionary={
            "red": statistics_lib.DictionaryItem(index=0, count=2),
            "green": statistics_lib.DictionaryItem(index=1, count=1),
        },
        quantiles=[],
    )
    normalizer = normalize_lib.DictionaryIndexNormalizer.create(
        "test_feature", input_schema, input_stats
    )

    input_values = np.array([b"red", b"blue", b"green", b"red"])
    output_features = normalizer.normalize_numpy(input_values)

    expected_output_features = {"test_feature_INDEX": np.array([0, 2, 1, 0])}
    test_util.assert_are_equal(self, output_features, expected_output_features)

    tf_output_features = normalizer.normalize_tensorflow(
        tf.constant(input_values)
    )
    expected_tf_output_features = {
        k: tf.constant(v, dtype=tf.int64)
        for k, v in expected_output_features.items()
    }
    tf_output_features_np = {
        k: v.numpy() for k, v in tf_output_features.items()
    }
    expected_tf_output_features_np = {
        k: v.numpy() for k, v in expected_tf_output_features.items()
    }
    test_util.assert_are_equal(
        self, tf_output_features_np, expected_tf_output_features_np
    )

    output_schema = normalizer.output_schema()
    expected_output_schema = {
        "test_feature_INDEX": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            shape=(),
            num_categorical_values=3,  # red, green, OOV
        )
    }
    self.assertEqual(output_schema, expected_output_schema)

  def test_invalid_format(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
    )
    input_stats = statistics_lib.FeatureStatistics(
        count=3, minimum=0, maximum=10, dictionary={}, quantiles=[]
    )
    with self.assertRaisesRegex(ValueError, "only supports BYTES features"):
      normalize_lib.DictionaryIndexNormalizer.create(
          "test_feature", input_schema, input_stats
      )

  def test_missing_dictionary(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
    )
    input_stats = statistics_lib.FeatureStatistics(
        count=3, minimum=0, maximum=0, dictionary={}, quantiles=[]
    )
    with self.assertRaisesRegex(ValueError, "does not have a dictionary"):
      normalize_lib.DictionaryIndexNormalizer.create(
          "test_feature", input_schema, input_stats
      )

  def test_timeseries(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        is_timeseries=True,
        group="ts_group",
        shape=(2,),
    )
    input_stats = statistics_lib.FeatureStatistics(
        count=3,
        dictionary={
            "red": statistics_lib.DictionaryItem(index=0, count=2),
            "green": statistics_lib.DictionaryItem(index=1, count=1),
        },
    )
    normalizer = normalize_lib.DictionaryIndexNormalizer.create(
        "test_feature", input_schema, input_stats
    )

    output_schema = normalizer.output_schema()["test_feature_INDEX"]
    self.assertFalse(output_schema.is_timeseries)
    self.assertIsNone(output_schema.group)
    self.assertEqual(output_schema.shape, (2,))

    input_values = np.array([[b"red", b"blue"], [b"green", b"red"]])
    output_features = normalizer.normalize_numpy(input_values)
    expected_output_features = {
        "test_feature_INDEX": np.array([[0, 2], [1, 0]], dtype=np.int64)
    }
    test_util.assert_are_equal(self, output_features, expected_output_features)


class SoftQuantileNormalizerTest(parameterized.TestCase):

  def test_no_observed_value(self):
    # The feature is present (count > 0) but has no value, e.g., a timeseries
    # whose values are all after the seed timestamps.
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(None,),
        is_timeseries=True,
    )
    input_stats = statistics_lib.FeatureStatistics(count=8, quantiles=[])
    normalizer = normalize_lib.SoftQuantileNormalizer.create(
        "test_feature", input_schema, input_stats
    )
    np.testing.assert_array_equal(normalizer.quantiles, [0.0, 1.0])

  def test_missing_quantiles_raises(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
    )
    input_stats = statistics_lib.FeatureStatistics(
        count=8, minimum=-1.0, maximum=4.0, quantiles=[]
    )
    with self.assertRaisesRegex(ValueError, "does not have quantiles"):
      normalize_lib.SoftQuantileNormalizer.create(
          "test_feature", input_schema, input_stats
      )

  def test_basic(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
    )
    input_stats = statistics_lib.FeatureStatistics(
        count=8,
        minimum=-1.0,
        maximum=4.0,
        dictionary={},
        quantiles=[0.0, 2.0, 2.5, 3.0],
    )
    normalizer = normalize_lib.SoftQuantileNormalizer.create(
        "test_feature", input_schema, input_stats
    )

    input_values = np.array(
        [-1.0, 0.0, 1.0, 2.0, 2.25, 2.5, 3.0, 4.0], dtype=np.float32
    )
    output_features = normalizer.normalize_numpy(input_values)

    expected_output_features = {
        "test_feature_SOFT_QUANTILE": np.array(
            [
                -0.5 / 3.0 - 0.5,  # Below quantiles[0]
                0.0 - 0.5,  # At quantiles[0]
                0.5 / 3.0 - 0.5,  # Between 0.0 and 2.0: (0 + (1.0-0.0)/2.0) / 3
                1.0 / 3.0 - 0.5,  # At quantiles[1]: (1 + (2.0-2.0)/0.5) / 3
                1.5 / 3.0
                - 0.5,  # Between 2.0 and 2.5: (1 + (2.25-2.0)/0.5) / 3
                2.0 / 3.0 - 0.5,  # At quantiles[2]: (2 + (2.5-2.5)/0.5) / 3
                1.0 - 0.5,  # At quantiles[3]
                5.0 / 3.0 - 0.5,  # Above quantiles[3]
            ],
            dtype=np.float32,
        )
    }
    test_util.assert_are_equal(
        self, output_features, expected_output_features, abs_tol=1e-6
    )

    tf_output_features = normalizer.normalize_tensorflow(
        tf.constant(input_values)
    )
    expected_tf_output_features = {
        k: tf.constant(v) for k, v in expected_output_features.items()
    }
    tf_output_features_np = {
        k: v.numpy() for k, v in tf_output_features.items()
    }
    expected_tf_output_features_np = {
        k: v.numpy() for k, v in expected_tf_output_features.items()
    }
    test_util.assert_are_equal(
        self,
        tf_output_features_np,
        expected_tf_output_features_np,
        abs_tol=1e-6,
    )

    output_schema = normalizer.output_schema()
    expected_output_schema = {
        "test_feature_SOFT_QUANTILE": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(),
        )
    }
    self.assertEqual(output_schema, expected_output_schema)

  def test_invalid_format(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
    )
    input_stats = statistics_lib.FeatureStatistics(
        count=3, minimum=0, maximum=10, dictionary={}, quantiles=[0, 5, 10]
    )
    with self.assertRaisesRegex(
        ValueError,
        "SoftQuantileNormalizer only supports INTEGER or FLOAT features",
    ):
      normalize_lib.SoftQuantileNormalizer.create(
          "test_feature", input_schema, input_stats
      )

  def test_missing_quantiles(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
    )
    input_stats = statistics_lib.FeatureStatistics(
        count=3, minimum=0, maximum=10, dictionary={}, quantiles=[]
    )
    with self.assertRaisesRegex(ValueError, "does not have quantiles"):
      normalize_lib.SoftQuantileNormalizer.create(
          "test_feature", input_schema, input_stats
      )

  def test_too_few_quantiles(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
    )
    input_stats = statistics_lib.FeatureStatistics(
        count=3, minimum=0, maximum=10, dictionary={}, quantiles=[5.0]
    )
    with self.assertRaisesRegex(ValueError, "has less than 2 quantiles"):
      normalize_lib.SoftQuantileNormalizer.create(
          "test_feature", input_schema, input_stats
      )

  def test_non_finite_input_passes_through(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
    )
    input_stats = statistics_lib.FeatureStatistics(
        count=3,
        minimum=0,
        maximum=10,
        dictionary={},
        quantiles=[0.0, 5.0, 10.0],
    )
    normalizer = normalize_lib.SoftQuantileNormalizer.create(
        "test_feature", input_schema, input_stats
    )
    input_values = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    # Non-finite values pass through unchanged.
    expected = np.array([-0.4, np.nan, np.inf, -np.inf], dtype=np.float32)

    np_output = normalizer.normalize_numpy(input_values)
    tf_output = normalizer.normalize_tensorflow(tf.constant(input_values))

    np.testing.assert_allclose(
        np_output["test_feature_SOFT_QUANTILE"], expected, atol=1e-6
    )
    np.testing.assert_allclose(
        tf_output["test_feature_SOFT_QUANTILE"].numpy(), expected, atol=1e-6
    )

  def test_timeseries_feature_2d(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        is_timeseries=True,
        group="ts_group",
        shape=(3,),
    )
    input_stats = statistics_lib.FeatureStatistics(
        count=12,
        minimum=0.0,
        maximum=4.0,
        quantiles=[0.0, 2.0, 2.5, 3.0],
    )
    normalizer = normalize_lib.SoftQuantileNormalizer.create(
        "test_ts_feature", input_schema, input_stats
    )

    output_schema = normalizer.output_schema()
    expected_output_schema = {
        "test_ts_feature_SOFT_QUANTILE": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(3,),
            is_timeseries=True,
            group="ts_group",
        )
    }
    self.assertEqual(output_schema, expected_output_schema)

    # 2 entities, sequence length 3
    input_values = np.array(
        [[0.0, 2.0, 3.0], [1.0, 2.25, 4.0]], dtype=np.float32
    )
    output_features = normalizer.normalize_numpy(input_values)

    expected_output_features = {
        "test_ts_feature_SOFT_QUANTILE": np.array(
            [
                [0.0 - 0.5, 1.0 / 3.0 - 0.5, 1.0 - 0.5],
                [0.5 / 3.0 - 0.5, 1.5 / 3.0 - 0.5, 5.0 / 3.0 - 0.5],
            ],
            dtype=np.float32,
        )
    }
    test_util.assert_are_equal(
        self, output_features, expected_output_features, abs_tol=1e-6
    )

    tf_output_features = normalizer.normalize_tensorflow(
        tf.constant(input_values)
    )
    test_util.assert_are_equal(
        self,
        {k: v.numpy() for k, v in tf_output_features.items()},
        expected_output_features,
        abs_tol=1e-6,
    )

  def test_outliers_clamp_to_100(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
    )
    input_stats = statistics_lib.FeatureStatistics(
        count=10,
        minimum=0.0,
        maximum=3.0,
        quantiles=[0.0, 1.0, 2.0, 3.0],
    )
    normalizer = normalize_lib.SoftQuantileNormalizer.create(
        "test_feature", input_schema, input_stats
    )
    input_values = np.array(
        [-1e30, -1e6, -300.0, 303.0, 1e6, 1e30], dtype=np.float32
    )
    expected = {
        "test_feature_SOFT_QUANTILE": np.array(
            [-100.0, -100.0, -100.0, 100.0, 100.0, 100.0], dtype=np.float32
        )
    }
    np_out = normalizer.normalize_numpy(input_values)
    tf_out = {
        k: v.numpy()
        for k, v in normalizer.normalize_tensorflow(
            tf.constant(input_values)
        ).items()
    }
    test_util.assert_are_equal(self, np_out, expected, abs_tol=0.0)
    test_util.assert_are_equal(self, tf_out, expected, abs_tol=0.0)

  @parameterized.named_parameters(
      (
          "outer_duplicate_buckets",
          [1.0, 1.0, 2.0, 2.0],
          [0.0, 1.0, 1.5, 2.0, 3.0],
      ),
      (
          "all_equal_quantiles",
          [5.0, 5.0, 5.0],
          [4.0, 5.0, 6.0],
      ),
      (
          "sub_1e12_bucket_width",
          [0.0, 1e-20, 1.0, 1.0 + 1e-20],
          [-1.0, 0.0, 0.5, 1.0, 2.0],
      ),
  )
  def test_duplicate_and_tiny_width_quantiles_are_finite(
      self, quantiles, values
  ):
    normalizer = normalize_lib.SoftQuantileNormalizer(
        input_feature="test_feature",
        output_feature_name="test_feature_SOFT_QUANTILE",
        output_shape=(),
        quantiles=np.asarray(quantiles, dtype=np.float32),
    )
    input_values = np.asarray(values, dtype=np.float32)
    np_out = normalizer.normalize_numpy(input_values)[
        "test_feature_SOFT_QUANTILE"
    ]
    tf_out = (
        normalizer.normalize_tensorflow(tf.constant(input_values))[
            "test_feature_SOFT_QUANTILE"
        ].numpy()
    )
    self.assertEqual(np_out.dtype, np.float32)
    self.assertEqual(tf_out.dtype, np.float32)
    self.assertTrue(np.all(np.isfinite(np_out)))
    self.assertTrue(np.all(np.isfinite(tf_out)))
    self.assertTrue(np.all(np.abs(np_out) <= 100.0))
    np.testing.assert_allclose(np_out, tf_out, rtol=1e-6, atol=1e-6)

  def test_clean_data_bitwise_parity(self):
    quantiles = np.linspace(-10.0, 50.0, num=101, dtype=np.float32)
    normalizer = normalize_lib.SoftQuantileNormalizer(
        input_feature="test_feature",
        output_feature_name="test_feature_SOFT_QUANTILE",
        output_shape=(),
        quantiles=quantiles,
    )
    rng = np.random.default_rng(0)
    # In-range values and moderate extrapolations well inside [-100, 100].
    input_values = np.concatenate([
        quantiles,
        rng.uniform(-10.0, 50.0, size=200).astype(np.float32),
        np.array([-15.0, -10.5, 50.5, 60.0], dtype=np.float32),
    ])

    # Unmodified pre-CL3 reference formula.
    num_buckets = len(quantiles) - 1
    bucket_idx = np.searchsorted(quantiles, input_values, side="right") - 1
    bucket_idx = np.clip(bucket_idx, 0, len(quantiles) - 2)
    lower = quantiles[bucket_idx]
    upper = quantiles[bucket_idx + 1]
    legacy_np = (
        bucket_idx.astype(np.float32) + (input_values - lower) / (upper - lower)
    ) / num_buckets - 0.5

    tf_val = tf.constant(input_values, dtype=tf.float32)
    tf_q = tf.constant(quantiles, dtype=tf.float32)
    tf_idx = tf.clip_by_value(
        tf.searchsorted(tf_q, tf_val, side="right") - 1, 0, len(quantiles) - 2
    )
    tf_lower = tf.gather(tf_q, tf_idx)
    tf_upper = tf.gather(tf_q, tf_idx + 1)
    legacy_tf = (
        (
            tf.cast(tf_idx, tf.float32)
            + (tf_val - tf_lower) / (tf_upper - tf_lower)
        )
        / num_buckets
        - 0.5
    ).numpy()

    np_out = normalizer.normalize_numpy(input_values)[
        "test_feature_SOFT_QUANTILE"
    ]
    tf_out = (
        normalizer.normalize_tensorflow(tf_val)["test_feature_SOFT_QUANTILE"]
        .numpy()
    )
    np.testing.assert_array_equal(np_out, legacy_np)
    np.testing.assert_array_equal(tf_out, legacy_tf)


class HashStringNormalizerTest(absltest.TestCase):

  def test_basic(self):
    num_buckets = 100
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.BYTES,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
    )
    normalizer = normalize_lib.HashStringNormalizer.create(
        "test_feature", input_schema, num_buckets=num_buckets
    )

    input_values = np.array([b"red", b"blue", b"green", b"red"])
    output_features = normalizer.normalize_numpy(input_values)

    self.assertIn("test_feature_HASH", output_features)
    hashed_values = output_features["test_feature_HASH"]
    self.assertEqual(hashed_values.dtype, np.int64)
    self.assertTrue(np.all(hashed_values >= 0))
    self.assertTrue(np.all(hashed_values < num_buckets))
    # Check for no collisions.
    self.assertEqual(hashed_values[0], hashed_values[3])
    self.assertNotEqual(hashed_values[0], hashed_values[1])
    self.assertNotEqual(hashed_values[0], hashed_values[2])
    self.assertNotEqual(hashed_values[1], hashed_values[2])

    tf_output_features = normalizer.normalize_tensorflow(
        tf.constant(input_values)
    )
    expected_tf_output_features = {
        k: tf.constant(v) for k, v in output_features.items()
    }
    # Farmhash returns matching values.
    tf_output_features_np = {
        k: v.numpy() for k, v in tf_output_features.items()
    }
    expected_tf_output_features_np = {
        k: v.numpy() for k, v in expected_tf_output_features.items()
    }
    test_util.assert_are_equal(
        self, tf_output_features_np, expected_tf_output_features_np
    )

    output_schema = normalizer.output_schema()
    expected_output_schema = {
        "test_feature_HASH": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
            shape=(),
            num_categorical_values=num_buckets,
        )
    }
    self.assertEqual(output_schema, expected_output_schema)

  def test_invalid_format(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
    )
    with self.assertRaisesRegex(ValueError, "only supports BYTES features"):
      normalize_lib.HashStringNormalizer.create(
          "test_feature", input_schema, num_buckets=10
      )


class SinusoidTimedeltaNormalizerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="scalar",
          schema_kwargs=dict(shape=()),
          embedding_dim=4,
          expected_shape=(4,),
          expected_group=None,
          expected_is_ts=False,
      ),
      dict(
          testcase_name="timeseries_creation_time",
          schema_kwargs=dict(
              is_timeseries=True, is_creation_time=True, shape=(2,)
          ),
          embedding_dim=4,
          expected_shape=(2, 4),
          expected_group=None,
          expected_is_ts=True,
      ),
      dict(
          testcase_name="timeseries_group",
          schema_kwargs=dict(
              is_timeseries=True,
              is_creation_time=False,
              group="custom_group",
              shape=(2,),
          ),
          embedding_dim=4,
          expected_shape=(2, 4),
          expected_group="custom_group",
          expected_is_ts=True,
      ),
  )
  def test_output_schema(
      self,
      schema_kwargs,
      embedding_dim,
      expected_shape,
      expected_group,
      expected_is_ts,
  ):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.TIMEDELTA,
        **schema_kwargs,
    )
    normalizer = normalize_lib.SinusoidTimedeltaNormalizer.create(
        "timestamp_feature", schema, embedding_dim=embedding_dim
    )
    out_schema = normalizer.output_schema()["timestamp_feature_SINUSOID"]
    self.assertEqual(out_schema.semantic, schema_lib.FeatureSemantic.EMBEDDING)
    self.assertEqual(out_schema.format, schema_lib.FeatureFormat.FLOAT_32)
    self.assertEqual(out_schema.is_timeseries, expected_is_ts)
    self.assertEqual(out_schema.shape, expected_shape)
    self.assertEqual(out_schema.group, expected_group)

  @parameterized.product(
      embedding_dim=[2, 4],
      use_tf=[False, True],
      is_timeseries=[False, True],
  )
  def test_normalization(self, embedding_dim, use_tf, is_timeseries):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.TIMEDELTA,
        is_timeseries=is_timeseries,
        shape=(2,) if is_timeseries else (),
    )
    normalizer = normalize_lib.SinusoidTimedeltaNormalizer.create(
        "timestamp_feature", schema, embedding_dim=embedding_dim
    )

    if is_timeseries:
      input_values = [[0.0, 1.0], [2.0, 0.0], [1.0, 2.0]]
    else:
      input_values = [0.0, 1.0, 2.0]

    if use_tf:
      res = normalizer.normalize_tensorflow(
          tf.constant(input_values, dtype=tf.float32)
      )
      out = {k: v.numpy() for k, v in res.items()}
    else:
      out = normalizer.normalize_numpy(np.array(input_values, dtype=np.float32))

    if embedding_dim == 2:
      freqs = np.array([2 * np.pi / 31536000.0], dtype=np.float32)
    elif embedding_dim == 4:
      freqs = np.array([np.pi, 2 * np.pi / 31536000.0], dtype=np.float32)
    else:
      raise ValueError(f"Unsupported embedding_dim in test: {embedding_dim}")

    vals = np.array(input_values, dtype=np.float32)[..., None] * freqs
    expected = np.concatenate([np.sin(vals), np.cos(vals)], axis=-1)

    expected_out = {"timestamp_feature_SINUSOID": expected}
    test_util.assert_are_equal(self, out, expected_out, abs_tol=1e-5)

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_semantic",
          schema_kwargs=dict(semantic=schema_lib.FeatureSemantic.NUMERICAL),
          expected_regex="only supports TIMEDELTA features",
      ),
      dict(
          testcase_name="dynamic_shape",
          schema_kwargs=dict(
              semantic=schema_lib.FeatureSemantic.TIMEDELTA, shape=(None,)
          ),
          expected_regex="requires fixed-length feature tensors",
      ),
      dict(
          testcase_name="odd_embedding_dim",
          schema_kwargs=dict(semantic=schema_lib.FeatureSemantic.TIMEDELTA),
          embedding_dim=5,
          expected_regex="embedding_dim must be a positive even integer",
      ),
      dict(
          testcase_name="zero_embedding_dim",
          schema_kwargs=dict(semantic=schema_lib.FeatureSemantic.TIMEDELTA),
          embedding_dim=0,
          expected_regex="embedding_dim must be a positive even integer",
      ),
  )
  def test_create_raises(self, schema_kwargs, expected_regex, embedding_dim=4):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32, **schema_kwargs
    )
    with self.assertRaisesRegex(ValueError, expected_regex):
      normalize_lib.SinusoidTimedeltaNormalizer.create(
          "timestamp_feature", schema, embedding_dim=embedding_dim
      )

  def test_normalize_numpy_object_array_raises(self):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.TIMEDELTA,
        shape=(2,),
    )
    normalizer = normalize_lib.SinusoidTimedeltaNormalizer.create(
        "timestamp_feature", schema, embedding_dim=4
    )
    input_np = np.array([np.array([0.0, 1.0]), np.array([2.0])], dtype=object)
    with self.assertRaisesRegex(
        ValueError, "requires fixed-length feature tensors"
    ):
      normalizer.normalize_numpy(input_np)


class CalendarNormalizerTest(parameterized.TestCase):

  _ALL_FEATURES = tuple(normalize_lib._CALENDAR_FEATURE_RANGES)

  def _timestamp_schema(self, **schema_kwargs):
    schema_kwargs.setdefault("shape", ())
    return schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        **schema_kwargs,
    )

  def _rescale(self, raw_value, calendar_feature):
    """Maps a raw calendar component onto [-0.5, 0.5)."""
    minimum, maximum = normalize_lib._CALENDAR_FEATURE_RANGES[calendar_feature]
    return (np.asarray(raw_value, dtype=np.float64) - minimum) / (
        maximum - minimum
    ) - 0.5

  def _timestamp_corpus(self, num_random: int = 10000) -> np.ndarray:
    edge_cases = np.array(
        [
            -315619200,  # 1960-01-01 00:00:00, 10 years ago (leap year).
            -310608000,  # 1960-02-28 00:00:00, day before leap day.
            -310521600,  # 1960-02-29 00:00:00, leap day.
            -310435201,  # 1960-02-29 23:59:59, last second of leap day.
            -310435200,  # 1960-03-01 00:00:00, day after pre-epoch leap day.
            -283996801,  # 1960-12-31 23:59:59, end of leap year.
            -86401,  # 1969-12-30 23:59:59, pre-epoch.
            0,  # 1970-01-01 00:00:00.
            1,
            5097600,  # 1970-03-01, the algorithm's March-based year start.
            15638400,  # 1970-07-01, a month-formula rounding boundary.
            28857600,  # 1970-12-01, the other rounding boundary.
            68169600,  # 1972-02-29, a leap day.
            1700000000,  # 2023-11-14 22:13:20.
        ],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed=0)
    random_timestamps = rng.integers(
        low=-631152000, high=2524608000, size=num_random, dtype=np.int64
    )
    return np.concatenate([edge_cases, random_timestamps])

  @parameterized.named_parameters(
      dict(
          testcase_name="scalar",
          schema_kwargs=dict(shape=()),
          expected_shape=(),
          expected_group=None,
          expected_is_ts=False,
      ),
      dict(
          testcase_name="timeseries_creation_time",
          schema_kwargs=dict(
              is_timeseries=True, is_creation_time=True, shape=(2,)
          ),
          expected_shape=(2,),
          expected_group="timestamp_feature",
          expected_is_ts=True,
      ),
      dict(
          testcase_name="timeseries_group",
          schema_kwargs=dict(
              is_timeseries=True,
              is_creation_time=False,
              group="custom_group",
              shape=(2,),
          ),
          expected_shape=(2,),
          expected_group="custom_group",
          expected_is_ts=True,
      ),
  )
  def test_output_schema(
      self, schema_kwargs, expected_shape, expected_group, expected_is_ts
  ):
    normalizer = normalize_lib.CalendarNormalizer.create(
        "timestamp_feature",
        self._timestamp_schema(**schema_kwargs),
        (normalize_lib.CalendarFeature.HOUR,),
    )
    out_schema = normalizer.output_schema()["timestamp_feature_hour_CALENDAR"]
    self.assertEqual(out_schema.semantic, schema_lib.FeatureSemantic.EMBEDDING)
    self.assertEqual(out_schema.format, schema_lib.FeatureFormat.FLOAT_32)
    self.assertEqual(out_schema.shape, expected_shape)
    self.assertEqual(out_schema.group, expected_group)
    self.assertEqual(out_schema.is_timeseries, expected_is_ts)

  def test_output_schema_emits_one_feature_per_component(self):
    normalizer = normalize_lib.CalendarNormalizer.create(
        "timestamp_feature", self._timestamp_schema(), self._ALL_FEATURES
    )
    expected_names = [
        "timestamp_feature_second_CALENDAR",
        "timestamp_feature_minute_CALENDAR",
        "timestamp_feature_hour_CALENDAR",
        "timestamp_feature_day_of_week_CALENDAR",
        "timestamp_feature_day_of_month_CALENDAR",
        "timestamp_feature_month_CALENDAR",
    ]
    self.assertEqual(list(normalizer.output_schema()), expected_names)
    self.assertEqual(
        [
            normalizer.output_feature_name(calendar_feature)
            for calendar_feature in self._ALL_FEATURES
        ],
        expected_names,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="wrong_semantic",
          schema_kwargs=dict(semantic=schema_lib.FeatureSemantic.NUMERICAL),
          calendar_features=(normalize_lib.CalendarFeature.HOUR,),
          expected_regex="only supports TIMESTAMP features",
      ),
      dict(
          testcase_name="dynamic_shape",
          schema_kwargs=dict(shape=(None,)),
          calendar_features=(normalize_lib.CalendarFeature.HOUR,),
          expected_regex="requires fixed-length feature tensors",
      ),
      dict(
          testcase_name="unsupported_year",
          schema_kwargs=dict(),
          calendar_features=(
              normalize_lib.CalendarFeature.HOUR,
              normalize_lib.CalendarFeature.YEAR,
          ),
          expected_regex="is not supported by CalendarNormalizer",
      ),
      dict(
          testcase_name="no_features",
          schema_kwargs=dict(),
          calendar_features=(),
          expected_regex="No calendar feature requested",
      ),
  )
  def test_invalid_configuration_raises(
      self, schema_kwargs, calendar_features, expected_regex
  ):
    schema_kwargs.setdefault(
        "semantic", schema_lib.FeatureSemantic.TIMESTAMP
    )
    schema_kwargs.setdefault("shape", ())
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64, **schema_kwargs
    )
    with self.assertRaisesRegex(ValueError, expected_regex):
      normalize_lib.CalendarNormalizer.create(
          "timestamp_feature", schema, calendar_features
      )

    # The constructor must reject the same inputs: deserializing a config
    # bypasses `create` entirely.
    with self.assertRaisesRegex(ValueError, expected_regex):
      normalize_lib.CalendarNormalizer(
          input_feature="timestamp_feature",
          calendar_features=calendar_features,
          input_schema=schema,
      )

  @parameterized.named_parameters(
      # (timestamp, second, minute, hour, day_of_week, day_of_month, month)
      # 1960-01-01 00:00:00 UTC, a Friday. -10 years in the past (leap year).
      ("pre_epoch_leap_year_start", -315619200, 0.0, 0.0, 0.0, 4.0, 1.0, 1.0),
      # 1960-02-28 00:00:00 UTC, a Sunday. Day before pre-epoch leap day.
      (
          "pre_epoch_day_before_leap_day",
          -310608000,
          0.0,
          0.0,
          0.0,
          6.0,
          28.0,
          2.0,
      ),
      # 1960-02-29 00:00:00 UTC, a Monday. Exercises pre-epoch leap day.
      ("pre_epoch_leap_day", -310521600, 0.0, 0.0, 0.0, 0.0, 29.0, 2.0),
      # 1960-02-29 23:59:59 UTC, a Monday.
      (
          "pre_epoch_last_second_of_leap_day",
          -310435201,
          59.0,
          59.0,
          23.0,
          0.0,
          29.0,
          2.0,
      ),
      # 1960-03-01 00:00:00 UTC, a Tuesday. Day after pre-epoch leap day.
      (
          "pre_epoch_day_after_leap_day",
          -310435200,
          0.0,
          0.0,
          0.0,
          1.0,
          1.0,
          3.0,
      ),
      # 1960-12-31 23:59:59 UTC, a Saturday.
      (
          "pre_epoch_leap_year_end",
          -283996801,
          59.0,
          59.0,
          23.0,
          5.0,
          31.0,
          12.0,
      ),
      # 1970-01-01 00:00:00 UTC, a Thursday.
      ("epoch", 0, 0.0, 0.0, 0.0, 3.0, 1.0, 1.0),
      ("epoch_plus_one_hour", 3600, 0.0, 0.0, 1.0, 3.0, 1.0, 1.0),
      ("epoch_plus_ninety_seconds", 90, 30.0, 1.0, 0.0, 3.0, 1.0, 1.0),
      ("last_second_of_first_day", 86399, 59.0, 59.0, 23.0, 3.0, 1.0, 1.0),
      # 1970-01-02, a Friday.
      ("second_day", 86400, 0.0, 0.0, 0.0, 4.0, 2.0, 1.0),
      # 1969-12-31 23:59:59 UTC, a Wednesday. Exercises negative timestamps.
      ("before_epoch", -1, 59.0, 59.0, 23.0, 2.0, 31.0, 12.0),
      # 1970-03-01, a Sunday.
      ("first_of_march", 5097600, 0.0, 0.0, 0.0, 6.0, 1.0, 3.0),
      # 1972-02-29, a Tuesday. Exercises the leap day.
      ("leap_day", 68169600, 0.0, 0.0, 0.0, 1.0, 29.0, 2.0),
      # 1970-07-01, a Wednesday, and 1970-12-01, a Tuesday. These are the only
      # two days of the year on which the month formula's rounding constant
      # changes the answer, so they are the cases that pin it down.
      ("first_of_july", 15638400, 0.0, 0.0, 0.0, 2.0, 1.0, 7.0),
      ("first_of_december", 28857600, 0.0, 0.0, 0.0, 1.0, 1.0, 12.0),
  )
  def test_normalize_numpy_known_values(
      self,
      timestamp,
      expected_second,
      expected_minute,
      expected_hour,
      expected_day_of_week,
      expected_day_of_month,
      expected_month,
  ):
    expected_raw = {
        normalize_lib.CalendarFeature.SECOND: expected_second,
        normalize_lib.CalendarFeature.MINUTE: expected_minute,
        normalize_lib.CalendarFeature.HOUR: expected_hour,
        normalize_lib.CalendarFeature.DAY_OF_WEEK: expected_day_of_week,
        normalize_lib.CalendarFeature.DAY_OF_MONTH: expected_day_of_month,
        normalize_lib.CalendarFeature.MONTH: expected_month,
    }
    normalizer = normalize_lib.CalendarNormalizer.create(
        "timestamp_feature", self._timestamp_schema(), self._ALL_FEATURES
    )
    got = normalizer.normalize_numpy(np.array([timestamp], dtype=np.int64))

    for calendar_feature, raw_value in expected_raw.items():
      np.testing.assert_allclose(
          got[normalizer.output_feature_name(calendar_feature)],
          np.array(
              [self._rescale(raw_value, calendar_feature)], dtype=np.float32
          ),
          atol=1e-6,
          err_msg=f"Mismatch for {calendar_feature}.",
      )

  def test_matches_reference_implementation(self):
    """Pins the closed-form arithmetic against an independent implementation."""
    timestamps = self._timestamp_corpus()
    normalizer = normalize_lib.CalendarNormalizer.create(
        "timestamp_feature", self._timestamp_schema(), self._ALL_FEATURES
    )
    got = normalizer.normalize_numpy(timestamps)

    for calendar_feature in self._ALL_FEATURES:
      reference = _reference_calendar_feature(timestamps, calendar_feature)
      value = got[normalizer.output_feature_name(calendar_feature)]
      np.testing.assert_allclose(
          value,
          self._rescale(reference, calendar_feature).astype(np.float32),
          atol=1e-6,
          err_msg=f"Mismatch for {calendar_feature}.",
      )
      # Rescaling maps every component onto [-0.5, 0.5).
      self.assertGreaterEqual(value.min(), -0.5)
      self.assertLess(value.max(), 0.5)

  @parameterized.named_parameters(
      ("scalar", False),
      ("timeseries", True),
  )
  def test_numpy_tensorflow_parity(self, is_timeseries):
    schema = self._timestamp_schema(
        is_timeseries=is_timeseries, shape=(3,) if is_timeseries else ()
    )
    normalizer = normalize_lib.CalendarNormalizer.create(
        "timestamp_feature", schema, self._ALL_FEATURES
    )
    timestamps = self._timestamp_corpus()
    if is_timeseries:
      timestamps = timestamps.reshape(-1, 3)

    numpy_output = normalizer.normalize_numpy(timestamps)
    tensorflow_output = normalizer.normalize_tensorflow(
        tf.constant(timestamps)
    )
    self.assertEqual(
        list(numpy_output), list(normalizer.output_schema())
    )
    for output_name, numpy_value in numpy_output.items():
      np.testing.assert_allclose(
          numpy_value,
          tensorflow_output[output_name].numpy(),
          atol=1e-6,
          err_msg=f"Mismatch for {output_name}.",
      )
      self.assertEqual(numpy_value.shape, timestamps.shape)

  @parameterized.named_parameters(
      ("second", normalize_lib.CalendarFeature.SECOND),
      ("minute", normalize_lib.CalendarFeature.MINUTE),
      ("hour", normalize_lib.CalendarFeature.HOUR),
      ("day_of_week", normalize_lib.CalendarFeature.DAY_OF_WEEK),
      ("day_of_month", normalize_lib.CalendarFeature.DAY_OF_MONTH),
      ("month", normalize_lib.CalendarFeature.MONTH),
  )
  def test_single_component_matches_all_components(self, calendar_feature):
    timestamps = np.array(
        [-310521600, -86401, 0, 1, 5097600, 68169600, 1700000000],
        dtype=np.int64,
    )
    schema = self._timestamp_schema()
    expected = normalize_lib.CalendarNormalizer.create(
        "timestamp_feature", schema, self._ALL_FEATURES
    ).normalize_numpy(timestamps)

    normalizer = normalize_lib.CalendarNormalizer.create(
        "timestamp_feature", schema, (calendar_feature,)
    )
    output_name = normalizer.output_feature_name(calendar_feature)

    numpy_output = normalizer.normalize_numpy(timestamps)
    self.assertEqual(list(numpy_output), [output_name])
    np.testing.assert_array_equal(
        numpy_output[output_name], expected[output_name]
    )

    tensorflow_output = normalizer.normalize_tensorflow(tf.constant(timestamps))
    self.assertEqual(list(tensorflow_output), [output_name])
    np.testing.assert_allclose(
        tensorflow_output[output_name].numpy(),
        expected[output_name],
        atol=1e-6,
    )

  def test_normalize_numpy_object_array_raises(self):
    normalizer = normalize_lib.CalendarNormalizer.create(
        "timestamp_feature",
        self._timestamp_schema(is_timeseries=True, shape=(2,)),
        (normalize_lib.CalendarFeature.HOUR,),
    )
    input_np = np.array([np.array([0, 1]), np.array([2])], dtype=object)
    with self.assertRaisesRegex(
        ValueError, "requires fixed-length feature tensors"
    ):
      normalizer.normalize_numpy(input_np)

  def test_accepts_no_kwargs(self):
    """Unlike the timestamp chain, calendar needs no seed timestamps."""
    normalizer = normalize_lib.CalendarNormalizer.create(
        "timestamp_feature", self._timestamp_schema(), self._ALL_FEATURES
    )
    self.assertEmpty(normalize_lib._accepted_kwargs(normalizer))  # pylint: disable=protected-access

  def test_deserializing_invalid_config_raises(self):
    """A config with no components must not silently emit zero features."""
    normalizer = normalize_lib.CalendarNormalizer.create(
        "timestamp_feature", self._timestamp_schema(), self._ALL_FEATURES
    )
    serialized = normalizer.to_dict()  # pyrefly: ignore[missing-attribute]
    serialized["calendar_features"] = []

    with self.assertRaisesRegex(ValueError, "No calendar feature requested"):
      normalize_lib.CalendarNormalizer.from_dict(serialized)  # pyrefly: ignore[missing-attribute]


class AutoNormalierTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.input_stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "n1": statistics_lib.FeatureSetStatistics(
                features={
                    "f1": statistics_lib.FeatureStatistics(
                        count=2,
                        minimum=0,
                        maximum=0,
                        dictionary={
                            "red": statistics_lib.DictionaryItem(
                                index=0, count=1
                            ),
                            "green": statistics_lib.DictionaryItem(
                                index=1, count=1
                            ),
                        },
                        quantiles=[],
                    ),
                    "f2": statistics_lib.FeatureStatistics(
                        count=2,
                        minimum=0,
                        maximum=4,
                        dictionary={},
                        quantiles=[0.0, 2.0, 2.5, 3.0],
                    ),
                }
            ),
            "n2": statistics_lib.FeatureSetStatistics(
                features={
                    "f3": statistics_lib.FeatureStatistics(
                        count=2,
                        minimum=0,
                        maximum=4,
                        dictionary={},
                        quantiles=[0.0, 3.0, 4.0, 6.0],
                    ),
                }
            ),
        }
    )
    self.input_schema = gen_test_graph.generate_schema(
        False, False, semantic=True, variable_length=False
    )
    self.input_graph = gen_test_graph.generate_in_memory_graph(False, False)

    del self.input_schema.node_sets["n2"].features["f4"]

  def test_normalize_graph(self):
    normalizer = normalize_lib.auto_normalize(
        self.input_schema, self.input_stats
    )

    output_graph = normalizer.normalize_numpy(self.input_graph)
    output_schema = normalizer.output_schema()

    expected_output_schema = schema_lib.GraphSchema(
        node_sets={
            "n2": schema_lib.NodeSchema(
                features={
                    "f3_SOFT_QUANTILE": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.EMBEDDING,
                        shape=(),
                    )
                }
            ),
            "n1": schema_lib.NodeSchema(
                features={
                    "f2": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.EMBEDDING,
                        shape=(2,),
                    ),
                    "f1_INDEX": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                        num_categorical_values=3,
                        shape=(1,),
                    ),
                }
            ),
        },
        edge_sets={
            "e2": schema_lib.EdgeSchema(source="n1", target="n2", features={}),
            "e1": schema_lib.EdgeSchema(source="n1", target="n1", features={}),
        },
    )
    expected_ouptut_graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n2": in_memory_graph_lib.InMemoryNodeSet(
                features={
                    "f3_SOFT_QUANTILE": np.array(
                        [2 / 3 - 0.5, 5 / 6 - 0.5], dtype=np.float32
                    )
                },
                num_nodes=2,
            ),
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                features={
                    "f2": np.array([[0.0, 1.0], [2.0, 3.0]]),
                    "f1_INDEX": np.array([[2], [0]]),
                },
                num_nodes=2,
            ),
        },
        edge_sets={
            "e2": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0, 0], [0, 1]]), features={}
            ),
            "e1": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0, 0], [0, 1]]), features={}
            ),
        },
    )

    test_util.assert_are_equal(self, output_schema, expected_output_schema)
    test_util.assert_are_equal(
        self, output_graph, expected_ouptut_graph, abs_tol=1e-6
    )

    test_util.assert_are_equal(
        self,
        normalizer.get_normalized_feature_names("n1", "f1"),
        ["f1_INDEX"],
    )
    test_util.assert_are_equal(
        self,
        normalizer.get_normalized_feature_names("n1", "f2"),
        ["f2"],
    )
    test_util.assert_are_equal(
        self,
        normalizer.get_normalized_feature_names("n2", "f3"),
        ["f3_SOFT_QUANTILE"],
    )

  def test_normalize_tf_graph(self):
    normalizer = normalize_lib.auto_normalize(
        self.input_schema, self.input_stats
    )
    np_graph = gen_test_graph.generate_in_memory_graph(
        False, False, variable_length=False
    )
    tf_graph = tf_io.graph_to_tf_graph(np_graph)
    tf_output_graph = normalizer.normalize_tensorflow(tf_graph)
    np_expected_ouptut_graph = normalizer.normalize_numpy(np_graph)
    tf_expected_ouptut_graph = tf_io.graph_to_tf_graph(np_expected_ouptut_graph)
    test_util.assert_are_equal(
        self, tf_output_graph, tf_expected_ouptut_graph, abs_tol=1e-6
    )

  def test_with_ids(self):
    input_stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "n1": statistics_lib.FeatureSetStatistics(
                features={
                    "f1": statistics_lib.FeatureStatistics(
                        count=2,
                        minimum=0,
                        maximum=0,
                        dictionary={
                            "red": statistics_lib.DictionaryItem(
                                index=0, count=1
                            ),
                            "green": statistics_lib.DictionaryItem(
                                index=1, count=1
                            ),
                        },
                        quantiles=[],
                    ),
                    "f2": statistics_lib.FeatureStatistics(
                        count=0,
                        minimum=0,
                        maximum=4,
                        dictionary={},
                        quantiles=[0.0, 2.0, 2.5, 3.0],
                    ),
                }
            ),
            "n2": statistics_lib.FeatureSetStatistics(
                features={
                    "f3": statistics_lib.FeatureStatistics(
                        count=0,
                        minimum=0,
                        maximum=4,
                        dictionary={},
                        quantiles=[0.0, 3.0, 4.0, 6.0],
                    ),
                }
            ),
        }
    )
    input_schema = gen_test_graph.generate_schema(
        False, False, semantic=True, variable_length=False
    )
    del input_schema.node_sets["n2"].features["f4"]
    normalizer = normalize_lib.auto_normalize(
        input_schema,
        input_stats,
        normalize_lib.AutoNormalizeConfig(keep_raw_features={("n1", "f1")}),
    )
    output_schema = normalizer.output_schema()
    expected_output_schema = schema_lib.GraphSchema(
        node_sets={
            "n2": schema_lib.NodeSchema(
                features={
                    "f3_SOFT_QUANTILE": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.EMBEDDING,
                        shape=(),
                    )
                }
            ),
            "n1": schema_lib.NodeSchema(
                features={
                    "f2": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.EMBEDDING,
                        shape=(2,),
                    ),
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                        shape=(1,),
                    ),
                }
            ),
        },
        edge_sets={
            "e2": schema_lib.EdgeSchema(source="n1", target="n2", features={}),
            "e1": schema_lib.EdgeSchema(source="n1", target="n1", features={}),
        },
    )
    test_util.assert_are_equal(self, output_schema, expected_output_schema)

  def _add_shared_numerical_feature(self):
    """Adds a numerical "f3" to "n1", so "f3" exists in both node sets."""
    self.input_schema.node_sets["n1"].features["f3"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    self.input_stats.node_sets["n1"].features["f3"] = (
        statistics_lib.FeatureStatistics(
            count=2,
            minimum=0,
            maximum=4,
            dictionary={},
            quantiles=[0.0, 3.0, 4.0, 6.0],
        )
    )

  def test_keep_raw_features_scoped_to_nodeset(self):
    """A (nodeset_name, feature_name) tuple only applies to that node set."""
    self._add_shared_numerical_feature()

    normalizer = normalize_lib.auto_normalize(
        self.input_schema,
        self.input_stats,
        normalize_lib.AutoNormalizeConfig(keep_raw_features={("n1", "f3")}),
    )

    output_schema = normalizer.output_schema()
    test_util.assert_are_equal(
        self,
        normalizer.get_normalized_feature_names("n1", "f3"),
        ["f3"],
    )
    test_util.assert_are_equal(
        self,
        output_schema.node_sets["n1"].features["f3"].semantic,
        schema_lib.FeatureSemantic.NUMERICAL,
    )
    # The feature with the same name in the other node set is still normalized.
    test_util.assert_are_equal(
        self,
        normalizer.get_normalized_feature_names("n2", "f3"),
        ["f3_SOFT_QUANTILE"],
    )

  def test_keep_raw_features_bare_string_raises(self):
    """Passing a bare string instead of a tuple raises ValueError."""
    with self.assertRaisesRegex(ValueError, "Expected a .* tuple"):
      normalize_lib.auto_normalize(
          self.input_schema,
          self.input_stats,
          normalize_lib.AutoNormalizeConfig(keep_raw_features={"f1"}),  # pyrefly: ignore[bad-argument-type]
      )

  def test_keep_raw_features_scoped_to_edgeset(self):
    """An (edgeset_name, feature_name) tuple is accepted for edge sets."""
    self.input_schema.edge_sets["e1"].features["weight"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )

    normalizer = normalize_lib.auto_normalize(
        self.input_schema,
        self.input_stats,
        normalize_lib.AutoNormalizeConfig(keep_raw_features={("e1", "weight")}),
    )
    self.assertIsNotNone(normalizer)

  def test_keep_raw_features_invalid_set_raises(self):
    """A tuple with a non-existent set raises ValueError."""
    with self.assertRaisesRegex(ValueError, "Set 'unknown_set'.*not found"):
      normalize_lib.auto_normalize(
          self.input_schema,
          self.input_stats,
          normalize_lib.AutoNormalizeConfig(
              keep_raw_features={("unknown_set", "f1")}
          ),
      )

  def test_keep_raw_features_invalid_feature_raises(self):
    """A tuple with a non-existent feature in a valid set raises ValueError."""
    with self.assertRaisesRegex(ValueError, "Feature 'unknown_feature'.*not found"):
      normalize_lib.auto_normalize(
          self.input_schema,
          self.input_stats,
          normalize_lib.AutoNormalizeConfig(
              keep_raw_features={("n1", "unknown_feature")}
          ),
      )

  def test_keep_raw_features_json_roundtrip(self):
    """AutoNormalizeConfig with tuples correctly serializes and deserializes."""
    config = normalize_lib.AutoNormalizeConfig(
        keep_raw_features={("n1", "f3"), ("n2", "f3")}
    )
    json_str = config.to_json()  # pyrefly: ignore[missing-attribute]
    loaded = normalize_lib.AutoNormalizeConfig.from_json(json_str)  # pyrefly: ignore[missing-attribute]
    self.assertIn(("n1", "f3"), loaded.keep_raw_features)
    self.assertIn(("n2", "f3"), loaded.keep_raw_features)

  def test_serialize(self):
    normalizer = normalize_lib.auto_normalize(
        self.input_schema, self.input_stats
    )
    loaded_config = normalize_lib.GraphNormalizerConfig.from_json(  # pyrefly: ignore[missing-attribute]
        normalizer.config.to_json()  # pyrefly: ignore[missing-attribute]
    )
    loaded_normalizer = loaded_config.make()

    test_util.assert_are_equal(
        self, normalizer.config, loaded_normalizer.config, abs_tol=1e-6
    )
    test_util.assert_are_equal(
        self,
        normalizer.normalize_numpy(self.input_graph),
        loaded_normalizer.normalize_numpy(self.input_graph),
        abs_tol=1e-6,
    )
    test_util.assert_are_equal(
        self, normalizer.output_schema(), loaded_normalizer.output_schema()
    )

  def test_nice_print(self):
    normalizer = normalize_lib.auto_normalize(
        self.input_schema, self.input_stats
    )
    output = normalizer.config.nice_print(return_output=True)
    expected_output = """Graph Normalizer:

Node Sets:
  n1:
    - f1: DictionaryIndexNormalizer
    - f2: IdentityNormalizer

  n2:
    - f3: SoftQuantileNormalizer

Edge Sets:
  e1: (Source: n1, Target: n1)
    (No normalizers)

  e2: (Source: n1, Target: n2)
    (No normalizers)
"""
    self.assertEqual(output, expected_output)

  def test_auto_normalize_timedelta(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "ts_delta": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.TIMEDELTA,
                        shape=(),
                    )
                }
            )
        },
        edge_sets={},
    )
    stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "nodes": statistics_lib.FeatureSetStatistics(
                features={
                    "ts_delta": statistics_lib.FeatureStatistics(
                        count=10, minimum=0.0, maximum=100.0
                    )
                },
            )
        }
    )
    normalizer = normalize_lib.auto_normalize(schema, stats)
    out_schema = normalizer.output_schema()
    self.assertIn("ts_delta_SINUSOID", out_schema.node_sets["nodes"].features)

  def test_auto_normalize_timestamp(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "created_at": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        shape=(),
                    )
                }
            )
        },
        edge_sets={},
    )
    stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "nodes": statistics_lib.FeatureSetStatistics(
                features={
                    "created_at": statistics_lib.FeatureStatistics(
                        count=2, minimum=100.0, maximum=200.0
                    )
                },
            )
        }
    )
    normalizer = normalize_lib.auto_normalize(
        schema,
        stats,
        config=normalize_lib.AutoNormalizeConfig(
            timestamp_normalize=True, has_seed_timestamps=True
        ),
    )
    out_schema = normalizer.output_schema()
    self.assertIn(
        "created_at_seed_delta_SINUSOID",
        out_schema.node_sets["nodes"].features,
    )
    self.assertEqual(
        out_schema.node_sets["nodes"]
        .features["created_at_seed_delta_SINUSOID"]
        .shape,
        (32,),
    )

    # Without seed timestamps, timestamp normalization is skipped.
    normalizer_no_seed = normalize_lib.auto_normalize(
        schema,
        stats,
        config=normalize_lib.AutoNormalizeConfig(
            timestamp_normalize=True, has_seed_timestamps=False
        ),
    )
    self.assertNotIn(
        "created_at_seed_delta_SINUSOID",
        normalizer_no_seed.output_schema().node_sets["nodes"].features,
    )

    # End-to-end normalization execution
    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "nodes": in_memory_graph_lib.InMemoryNodeSet(
                features={"created_at": np.array([100, 200], dtype=np.int64)},
                num_nodes=2,
            )
        },
        edge_sets={},
    )
    out_graph = normalizer.normalize_numpy(
        graph,
        seed_timestamps={"nodes": np.array([500, 500], dtype=np.int64)},
    )
    self.assertIn(
        "created_at_seed_delta_SINUSOID",
        out_graph.node_sets["nodes"].features,
    )
    out_feature = out_graph.node_sets["nodes"].features[
        "created_at_seed_delta_SINUSOID"
    ]
    self.assertEqual(out_feature.shape, (2, 32))
    # Sinusoidal embeddings satisfy sin^2(x) + cos^2(x) == 1, ensuring non-zero
    # values that respect sinusoidal bounds.
    sin_part = out_feature[:, :16]
    cos_part = out_feature[:, 16:]
    np.testing.assert_allclose(
        sin_part**2 + cos_part**2, np.ones((2, 16)), atol=1e-5
    )
    # Distinct input timestamps produce distinct embeddings
    self.assertFalse(np.allclose(out_feature[0], out_feature[1]))

    # Single broadcast timestamp array produces the same result
    out_graph_single = normalizer.normalize_numpy(
        graph, seed_timestamps={"nodes": np.array([500], dtype=np.int64)}
    )
    np.testing.assert_allclose(
        out_graph_single.node_sets["nodes"].features[
            "created_at_seed_delta_SINUSOID"
        ],
        out_feature,
    )

  def test_auto_normalize_timestamp_disabled_by_default(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "created_at": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        shape=(),
                    )
                }
            )
        },
        edge_sets={},
    )
    stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "nodes": statistics_lib.FeatureSetStatistics(
                features={
                    "created_at": statistics_lib.FeatureStatistics(
                        count=2, minimum=100.0, maximum=200.0
                    )
                },
            )
        }
    )
    normalizer = normalize_lib.auto_normalize(schema, stats)
    out_schema = normalizer.output_schema()
    self.assertNotIn(
        "created_at_seed_delta_SINUSOID",
        out_schema.node_sets["nodes"].features,
    )

  def test_auto_normalize_timestamp_dynamic_shape_skipped(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "created_at": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        shape=(None,),
                    )
                }
            )
        },
        edge_sets={},
    )
    stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "nodes": statistics_lib.FeatureSetStatistics(
                features={
                    "created_at": statistics_lib.FeatureStatistics(count=2)
                },
            )
        }
    )
    normalizer = normalize_lib.auto_normalize(
        schema,
        stats,
        config=normalize_lib.AutoNormalizeConfig(
            timestamp_normalize=True, has_seed_timestamps=True
        ),
    )
    out_schema = normalizer.output_schema()
    self.assertNotIn(
        "created_at_seed_delta_SINUSOID",
        out_schema.node_sets["nodes"].features,
    )

  def _timestamp_schema_and_stats(self, shape=()):
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "created_at": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        shape=shape,
                    )
                }
            )
        },
        edge_sets={},
    )
    stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "nodes": statistics_lib.FeatureSetStatistics(
                features={
                    "created_at": statistics_lib.FeatureStatistics(count=2)
                },
            )
        }
    )
    return schema, stats

  def test_auto_normalize_calendar_disabled_by_default(self):
    schema, stats = self._timestamp_schema_and_stats()
    normalizer = normalize_lib.auto_normalize(schema, stats)
    features = normalizer.output_schema().node_sets["nodes"].features
    self.assertNotIn("created_at_hour_CALENDAR", features)
    self.assertNotIn("created_at_day_of_week_CALENDAR", features)
    self.assertNotIn("created_at_day_of_month_CALENDAR", features)
    self.assertNotIn("created_at_month_CALENDAR", features)

  def test_auto_normalize_calendar(self):
    schema, stats = self._timestamp_schema_and_stats()
    normalizer = normalize_lib.auto_normalize(
        schema,
        stats,
        config=normalize_lib.AutoNormalizeConfig(calendar_normalize=True),
    )
    self.assertCountEqual(
        normalizer.get_normalized_feature_names("nodes", "created_at"),
        [
            "created_at_hour_CALENDAR",
            "created_at_day_of_week_CALENDAR",
            "created_at_day_of_month_CALENDAR",
            "created_at_month_CALENDAR",
        ],
    )

    # Calendar normalization needs no seed timestamps.
    self.assertEmpty(normalizer.accepted_kwargs)

    # End-to-end normalization execution. 0 is 1970-01-01 00:00 UTC (Thursday),
    # 45296 is 1970-01-01 12:34 UTC.
    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "nodes": in_memory_graph_lib.InMemoryNodeSet(
                features={"created_at": np.array([0, 45296], dtype=np.int64)},
                num_nodes=2,
            )
        },
        edge_sets={},
    )
    out_features = normalizer.normalize_numpy(graph).node_sets["nodes"].features
    np.testing.assert_allclose(
        out_features["created_at_hour_CALENDAR"],
        np.array([0.0 / 24.0 - 0.5, 12.0 / 24.0 - 0.5], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out_features["created_at_day_of_week_CALENDAR"],
        np.array([3.0 / 7.0 - 0.5] * 2, dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out_features["created_at_day_of_month_CALENDAR"],
        np.array([(1.0 - 1.0) / 31.0 - 0.5] * 2, dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out_features["created_at_month_CALENDAR"],
        np.array([(1.0 - 1.0) / 12.0 - 0.5] * 2, dtype=np.float32),
        atol=1e-6,
    )

  def test_auto_normalize_calendar_subset(self):
    schema, stats = self._timestamp_schema_and_stats()
    normalizer = normalize_lib.auto_normalize(
        schema,
        stats,
        config=normalize_lib.AutoNormalizeConfig(
            calendar_normalize=True,
            calendar_features=(normalize_lib.CalendarFeature.HOUR,),
        ),
    )
    self.assertEqual(
        normalizer.get_normalized_feature_names("nodes", "created_at"),
        ["created_at_hour_CALENDAR"],
    )

  def test_auto_normalize_calendar_and_timestamp_compose(self):
    schema, stats = self._timestamp_schema_and_stats()
    normalizer = normalize_lib.auto_normalize(
        schema,
        stats,
        config=normalize_lib.AutoNormalizeConfig(
            calendar_normalize=True,
            timestamp_normalize=True,
            has_seed_timestamps=True,
        ),
    )
    self.assertCountEqual(
        normalizer.get_normalized_feature_names("nodes", "created_at"),
        [
            "created_at_seed_delta_SINUSOID",
            "created_at_hour_CALENDAR",
            "created_at_day_of_week_CALENDAR",
            "created_at_day_of_month_CALENDAR",
            "created_at_month_CALENDAR",
        ],
    )

    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "nodes": in_memory_graph_lib.InMemoryNodeSet(
                features={"created_at": np.array([0, 45296], dtype=np.int64)},
                num_nodes=2,
            )
        },
        edge_sets={},
    )
    out_features = (
        normalizer.normalize_numpy(
            graph,
            seed_timestamps={"nodes": np.array([86400, 86400], dtype=np.int64)},
        )
        .node_sets["nodes"]
        .features
    )
    self.assertIn("created_at_seed_delta_SINUSOID", out_features)
    self.assertIn("created_at_hour_CALENDAR", out_features)

  def test_auto_normalize_calendar_dynamic_shape_raises(self):
    schema, stats = self._timestamp_schema_and_stats(shape=(None,))
    with self.assertRaisesRegex(
        ValueError, "requires fixed-length feature tensors"
    ):
      normalize_lib.auto_normalize(
          schema,
          stats,
          config=normalize_lib.AutoNormalizeConfig(calendar_normalize=True),
      )

  def test_auto_normalize_calendar_serialization_round_trip(self):
    schema, stats = self._timestamp_schema_and_stats()
    normalizer = normalize_lib.auto_normalize(
        schema,
        stats,
        config=normalize_lib.AutoNormalizeConfig(calendar_normalize=True),
    )
    restored = normalize_lib.GraphNormalizerConfig.from_json(  # pyrefly: ignore[missing-attribute]
        normalizer.config.to_json()  # pyrefly: ignore[missing-attribute]
    ).make()
    self.assertEqual(
        restored.output_schema().node_sets["nodes"].features.keys(),
        normalizer.output_schema().node_sets["nodes"].features.keys(),
    )

    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "nodes": in_memory_graph_lib.InMemoryNodeSet(
                features={"created_at": np.array([0, 45296], dtype=np.int64)},
                num_nodes=2,
            )
        },
        edge_sets={},
    )
    test_util.assert_are_equal(
        self,
        restored.normalize_numpy(graph),
        normalizer.normalize_numpy(graph),
    )

  def test_auto_normalize_mask(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "mask": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BOOL,
                        semantic=schema_lib.FeatureSemantic.MASK,
                        shape=(),
                    )
                }
            )
        },
        edge_sets={},
    )
    stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "nodes": statistics_lib.FeatureSetStatistics(
                features={},
            )
        }
    )
    normalizer = normalize_lib.auto_normalize(schema, stats)
    out_schema = normalizer.output_schema()
    self.assertIn("mask", out_schema.node_sets["nodes"].features)
    self.assertEqual(
        out_schema.node_sets["nodes"].features["mask"].semantic,
        schema_lib.FeatureSemantic.MASK,
    )


def _missing_value_schema_and_stats(
    **feature_stats_kwargs,
) -> tuple[schema_lib.GraphSchema, statistics_lib.GraphFeatureStatistics]:
  schema = schema_lib.GraphSchema(
      node_sets={
          "nodes": schema_lib.NodeSchema(
              features={
                  "f": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                      shape=(),
                  )
              }
          )
      },
      edge_sets={},
  )
  stats = statistics_lib.GraphFeatureStatistics(
      node_sets={
          "nodes": statistics_lib.FeatureSetStatistics(
              features={
                  "f": statistics_lib.FeatureStatistics(
                      count=10,
                      minimum=0.0,
                      maximum=1.0,
                      mean=0.5,
                      quantiles=[0.0, 0.25, 0.5, 0.75, 1.0],
                      **feature_stats_kwargs,
                  )
              }
          )
      }
  )
  return schema, stats


class NumericalMissingValuePolicyTest(parameterized.TestCase):

  def test_default_is_error(self):
    self.assertEqual(
        normalize_lib.AutoNormalizeConfig().numerical_missing_value_policy,
        normalize_lib.NumericalMissingValuePolicy.ERROR,
    )

  def test_clean_stats_build(self):
    schema, stats = _missing_value_schema_and_stats()
    normalizer = normalize_lib.auto_normalize(schema, stats)
    self.assertIn(
        "f_SOFT_QUANTILE",
        normalizer.output_schema().node_sets["nodes"].features,
    )

  @parameterized.parameters(
      dict(num_nan=1), dict(num_posinf=2), dict(num_neginf=3)
  )
  def test_non_finite_stats_raise(self, **kwargs):
    schema, stats = _missing_value_schema_and_stats(**kwargs)
    with self.assertRaisesRegex(
        ValueError,
        r"Feature 'f' in node set 'nodes' has .*"
        r"numerical_missing_value_policy='error'",
    ):
      normalize_lib.auto_normalize(schema, stats)

  def test_kept_raw_feature_is_checked(self):
    schema, stats = _missing_value_schema_and_stats(num_nan=1)
    config = normalize_lib.AutoNormalizeConfig(
        keep_raw_features={("nodes", "f")}
    )
    with self.assertRaisesRegex(ValueError, "numerical_missing_value_policy"):
      normalize_lib.auto_normalize(schema, stats, config)

  def test_config_json_round_trip(self):
    config = normalize_lib.AutoNormalizeConfig.from_json(  # pyrefly: ignore[missing-attribute]
        normalize_lib.AutoNormalizeConfig().to_json()  # pyrefly: ignore[missing-attribute]
    )
    self.assertIs(
        config.numerical_missing_value_policy,
        normalize_lib.NumericalMissingValuePolicy.ERROR,
    )


class TimedeltaNormalizerTest(parameterized.TestCase):

  def test_output_schema(self):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(3,),
        is_timeseries=True,
        group="sensor",
    )
    normalizer = normalize_lib.TimedeltaNormalizer.create("time", schema)
    expected_schema = {
        "time_seed_delta": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.TIMEDELTA,
            shape=(3,),
            is_timeseries=True,
            group="sensor",
        )
    }
    self.assertEqual(normalizer.output_schema(), expected_schema)

  def test_timedelta_normalizer_numpy_1d(self):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(),
    )
    normalizer = normalize_lib.TimedeltaNormalizer.create("created_at", schema)
    raw_val = np.array([100, 300], dtype=np.int64)
    seed_timestamps = np.array([500, 500], dtype=np.int64)
    out = normalizer.normalize_numpy(raw_val, seed_timestamps=seed_timestamps)
    self.assertIn("created_at_seed_delta", out)
    np.testing.assert_array_equal(
        out["created_at_seed_delta"], np.array([400, 200], dtype=np.int64)
    )

  def test_timedelta_normalizer_numpy_timeseries(self):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(3,),
        is_timeseries=True,
        group="time",
    )
    normalizer = normalize_lib.TimedeltaNormalizer.create("time", schema)
    raw_val = np.array([[100, 200, 300], [400, 450, 500]], dtype=np.int64)
    seed_timestamps = np.array([500, 600], dtype=np.int64)
    out = normalizer.normalize_numpy(raw_val, seed_timestamps=seed_timestamps)
    expected = np.array([[400, 300, 200], [200, 150, 100]], dtype=np.int64)
    np.testing.assert_array_equal(out["time_seed_delta"], expected)

  def test_timedelta_normalizer_tensorflow(self):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(2,),
        is_timeseries=True,
    )
    normalizer = normalize_lib.TimedeltaNormalizer.create("time", schema)
    raw_np = np.array([[100, 200], [300, 400]], dtype=np.int64)
    seeds_np = np.array([500, 1000], dtype=np.int64)
    raw_tf = tf.constant(raw_np)
    seeds_tf = tf.constant(seeds_np)

    out_tf = normalizer.normalize_tensorflow(raw_tf, seed_timestamps=seeds_tf)
    out_np = normalizer.normalize_numpy(raw_np, seed_timestamps=seeds_np)
    expected = np.array([[400, 300], [700, 600]], dtype=np.int64)

    np.testing.assert_array_equal(out_np["time_seed_delta"], expected)
    np.testing.assert_array_equal(
        out_tf["time_seed_delta"].numpy(), out_np["time_seed_delta"]
    )

  def test_timedelta_normalizer_missing_seed_timestamps_asserts(self):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(),
    )
    normalizer = normalize_lib.TimedeltaNormalizer.create("time", schema)
    raw_val = np.array([100], dtype=np.int64)
    with self.assertRaises(AssertionError):
      normalizer.normalize_numpy(raw_val, seed_timestamps=None)

    with self.assertRaises(AssertionError):
      normalizer.normalize_tensorflow(tf.constant([100]), seed_timestamps=None)

  def test_timedelta_normalizer_object_array_raises(self):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(2,),
    )
    normalizer = normalize_lib.TimedeltaNormalizer.create("time", schema)
    with self.assertRaisesRegex(AssertionError, "requires fixed-length"):
      normalizer.normalize_numpy(
          np.array([np.array([100])], dtype=object),
          seed_timestamps=np.array([500]),
      )

  def test_timedelta_normalizer_tensorflow_unknown_rank_asserts(self):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(2,),
    )
    normalizer = normalize_lib.TimedeltaNormalizer.create("time", schema)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int64)])
    def normalize_fn(val):
      return normalizer.normalize_tensorflow(
          val, seed_timestamps=tf.constant([500], dtype=tf.int64)
      )

    with self.assertRaisesRegex(AssertionError, "unknown rank"):
      normalize_fn(tf.constant([100, 200], dtype=tf.int64))

  def test_timedelta_normalizer_invalid_semantic_raises(self):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    with self.assertRaisesRegex(ValueError, "only supports TIMESTAMP"):
      normalize_lib.TimedeltaNormalizer.create("num", schema)

  def test_timedelta_normalizer_dynamic_shape_raises(self):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(None,),
    )
    with self.assertRaisesRegex(ValueError, "requires fixed-length"):
      normalize_lib.TimedeltaNormalizer.create("time", schema)


class SequentialNormalizerTest(parameterized.TestCase):

  def test_sequential_normalizer_numpy_and_tensorflow(self):
    ts_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(2,),
        is_timeseries=True,
    )
    stage1 = normalize_lib.TimedeltaNormalizer.create("time", ts_schema)
    delta_schema = stage1.output_schema()["time_seed_delta"]
    stage2 = normalize_lib.SinusoidTimedeltaNormalizer.create(
        "time_seed_delta", delta_schema, embedding_dim=4
    )
    sequential = normalize_lib.SequentialNormalizer.create([stage1, stage2])

    # Output schema should only contain final stage output.
    out_schema = sequential.output_schema()
    self.assertLen(out_schema, 1)
    self.assertIn("time_seed_delta_SINUSOID", out_schema)
    self.assertNotIn("time_seed_delta", out_schema)
    self.assertEqual(out_schema["time_seed_delta_SINUSOID"].shape, (2, 4))

    raw_np = np.array([[100, 200], [300, 400]], dtype=np.int64)
    seeds_np = np.array([500, 1000], dtype=np.int64)

    # NumPy normalization.
    out_np = sequential.normalize_numpy(raw_np, seed_timestamps=seeds_np)
    self.assertEqual(list(out_np.keys()), ["time_seed_delta_SINUSOID"])
    self.assertEqual(out_np["time_seed_delta_SINUSOID"].shape, (2, 2, 4))

    # TensorFlow normalization.
    raw_tf = tf.constant(raw_np)
    seeds_tf = tf.constant(seeds_np)
    out_tf = sequential.normalize_tensorflow(raw_tf, seed_timestamps=seeds_tf)
    self.assertEqual(list(out_tf.keys()), ["time_seed_delta_SINUSOID"])
    np.testing.assert_allclose(
        out_tf["time_seed_delta_SINUSOID"].numpy(),
        out_np["time_seed_delta_SINUSOID"],
        rtol=1e-5,
    )

  def test_sequential_normalizer_json_serialization(self):
    ts_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(),
    )
    stage1 = normalize_lib.TimedeltaNormalizer.create("time", ts_schema)
    delta_schema = stage1.output_schema()["time_seed_delta"]
    stage2 = normalize_lib.SinusoidTimedeltaNormalizer.create(
        "time_seed_delta", delta_schema, embedding_dim=4
    )
    sequential = normalize_lib.SequentialNormalizer.create([stage1, stage2])

    json_str = sequential.to_json()  # pyrefly: ignore[missing-attribute]
    # pyrefly: ignore[missing-attribute]
    reconstructed = normalize_lib.SequentialNormalizer.from_json(json_str)
    self.assertLen(reconstructed.stages, 2)
    self.assertEqual(reconstructed.input_feature, "time")
    self.assertIn("time_seed_delta_SINUSOID", reconstructed.output_schema())
    # Verify execution on reconstructed normalizer.
    out = reconstructed.normalize_numpy(
        np.array([100], dtype=np.int64),
        seed_timestamps=np.array([500], dtype=np.int64),
    )
    self.assertIn("time_seed_delta_SINUSOID", out)

  def test_validate_stages(self):
    ts_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(),
    )
    stage1 = normalize_lib.TimedeltaNormalizer.create("time", ts_schema)
    delta_schema = stage1.output_schema()["time_seed_delta"]
    stage2 = normalize_lib.SinusoidTimedeltaNormalizer.create(
        "time_seed_delta", delta_schema, embedding_dim=4
    )

    # Valid stages.
    normalize_lib._validate_stages("time", [stage1, stage2])

    # Empty stages raises.
    with self.assertRaisesRegex(ValueError, "at least one stage"):
      normalize_lib._validate_stages("time", [])

    # Mismatched input feature raises.
    with self.assertRaisesRegex(ValueError, "does not match first stage"):
      normalize_lib._validate_stages("other_feature", [stage1, stage2])

    # Missing intermediate feature raises.
    unconnected_stage = normalize_lib.SinusoidTimedeltaNormalizer.create(
        "non_existent_feature", delta_schema, embedding_dim=4
    )
    with self.assertRaisesRegex(ValueError, "expects input feature"):
      normalize_lib._validate_stages("time", [stage1, unconnected_stage])

  def test_sequential_normalizer_empty_stages_raises(self):
    with self.assertRaisesRegex(ValueError, "at least one stage"):
      normalize_lib.SequentialNormalizer.create([])

  def test_sequential_normalizer_mismatched_input_feature_raises(self):
    ts_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(),
    )
    stage = normalize_lib.TimedeltaNormalizer.create("time", ts_schema)
    with self.assertRaisesRegex(ValueError, "does not match first stage"):
      normalize_lib.SequentialNormalizer(
          input_feature="mismatched_feature",
          stages=[stage],
      )

  def test_sequential_normalizer_missing_intermediate_feature_raises(self):
    ts_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(),
    )
    stage1 = normalize_lib.TimedeltaNormalizer.create("time", ts_schema)
    delta_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMEDELTA,
        shape=(),
    )
    # Stage 2 expects "non_existent_feature" which stage 1 does not output.
    stage2 = normalize_lib.SinusoidTimedeltaNormalizer.create(
        "non_existent_feature", delta_schema, embedding_dim=4
    )
    with self.assertRaisesRegex(ValueError, "expects input feature"):
      normalize_lib.SequentialNormalizer.create([stage1, stage2])

  def test_sequential_normalizer_nested(self):
    ts_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(),
    )
    stage1 = normalize_lib.TimedeltaNormalizer.create("time", ts_schema)
    delta_schema = stage1.output_schema()["time_seed_delta"]
    stage2 = normalize_lib.SinusoidTimedeltaNormalizer.create(
        "time_seed_delta", delta_schema, embedding_dim=4
    )
    inner = normalize_lib.SequentialNormalizer.create([stage1])
    outer = normalize_lib.SequentialNormalizer.create([inner, stage2])
    out = outer.normalize_numpy(
        np.array([100], dtype=np.int64),
        seed_timestamps=np.array([500], dtype=np.int64),
    )
    self.assertIn("time_seed_delta_SINUSOID", out)

  def test_sequential_normalizer_tensorflow_resources(self):
    ts_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(),
    )
    stage1 = normalize_lib.TimedeltaNormalizer.create("time", ts_schema)
    delta_schema = stage1.output_schema()["time_seed_delta"]
    stage2 = normalize_lib.SinusoidTimedeltaNormalizer.create(
        "time_seed_delta", delta_schema, embedding_dim=4
    )
    sequential = normalize_lib.SequentialNormalizer.create([stage1, stage2])
    self.assertEmpty(sequential.tensorflow_resources())

  def test_sequential_normalizer_extra_kwargs_ignored(self):
    ts_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(),
    )
    stage1 = normalize_lib.TimedeltaNormalizer.create("time", ts_schema)
    delta_schema = stage1.output_schema()["time_seed_delta"]
    stage2 = normalize_lib.SinusoidTimedeltaNormalizer.create(
        "time_seed_delta", delta_schema, embedding_dim=4
    )
    sequential = normalize_lib.SequentialNormalizer.create([stage1, stage2])
    out = sequential.normalize_numpy(
        np.array([100], dtype=np.int64),
        seed_timestamps=np.array([500], dtype=np.int64),
        unrelated_kwarg="ignored",
    )
    self.assertIn("time_seed_delta_SINUSOID", out)


class CreateTimestampNormalizerTest(absltest.TestCase):

  def test_create_timestamp_normalizer(self):
    ts_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(2,),
    )
    normalizer = normalize_lib._create_timestamp_normalizer(
        "time", ts_schema, embedding_dim=4
    )
    self.assertLen(normalizer.stages, 2)
    self.assertEqual(normalizer.input_feature, "time")
    self.assertIn("time_seed_delta_SINUSOID", normalizer.output_schema())
    self.assertEqual(
        normalizer.output_schema()["time_seed_delta_SINUSOID"].shape, (2, 4)
    )


class GraphNormalizerKwargsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ts_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        shape=(),
    )
    self.id_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
        shape=(),
    )
    self.timedelta_normalizer = normalize_lib.TimedeltaNormalizer.create(
        "time", self.ts_schema
    )
    self.identity_normalizer = normalize_lib.IdentityNormalizer(
        input_feature="id", input_schema=self.id_schema
    )
    self.graph_normalizer = normalize_lib.GraphNormalizer(
        config=normalize_lib.GraphNormalizerConfig(
            nodesets={
                "n": normalize_lib.NodeSetNormalizerConfig([
                    self.timedelta_normalizer,
                    self.identity_normalizer,
                ])
            },
            edgesets={},
        )
    )
    self.graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n": in_memory_graph_lib.InMemoryNodeSet(
                features={
                    "time": np.array([100, 200], dtype=np.int64),
                    "id": np.array([1, 2], dtype=np.int64),
                },
                num_nodes=2,
            )
        },
        edge_sets={},
    )

  def test_normalize_numpy_with_kwargs(self):
    seed_ts = np.array([500, 500], dtype=np.int64)
    out_graph = self.graph_normalizer.normalize_numpy(
        self.graph, seed_timestamps={"n": seed_ts}
    )
    expected_deltas = np.array([400, 300], dtype=np.int64)
    np.testing.assert_array_equal(
        out_graph.node_sets["n"].features["time_seed_delta"], expected_deltas
    )
    np.testing.assert_array_equal(
        out_graph.node_sets["n"].features["id"],
        np.array([1, 2], dtype=np.int64),
    )

  def test_normalize_tensorflow_with_kwargs(self):
    tf_graph = tf_io.graph_to_tf_graph(self.graph)
    seed_ts = tf.constant([500, 500], dtype=tf.int64)
    out_graph = self.graph_normalizer.normalize_tensorflow(
        tf_graph, seed_timestamps={"n": seed_ts}
    )
    expected_deltas = np.array([400, 300], dtype=np.int64)
    np.testing.assert_array_equal(
        out_graph.node_sets["n"].features["time_seed_delta"].numpy(),
        expected_deltas,
    )

  def test_normalize_numpy_to_jax_with_kwargs(self):
    seed_ts = np.array([500, 500], dtype=np.int64)
    out_graph = self.graph_normalizer.normalize_numpy_to_jax(
        self.graph, seed_timestamps={"n": seed_ts}
    )
    expected_deltas = np.array([400, 300], dtype=np.int64)
    np.testing.assert_array_equal(
        np.asarray(out_graph.node_sets["n"].features["time_seed_delta"]),
        expected_deltas,
    )

  def test_unexpected_kwargs_raises(self):
    seed_ts = np.array([500, 500], dtype=np.int64)
    with self.assertRaisesRegex(
        ValueError,
        "Keyword argument 'seed_timestamps' to GraphNormalizer must be a dict",
    ):
      self.graph_normalizer.normalize_numpy(self.graph, seed_timestamps=seed_ts)

    with self.assertRaisesRegex(
        ValueError,
        "Unexpected keyword arguments for GraphNormalizer.*bad_kwarg",
    ):
      self.graph_normalizer.normalize_numpy(self.graph, bad_kwarg="val")

    tf_graph = tf_io.graph_to_tf_graph(self.graph)
    with self.assertRaisesRegex(
        ValueError,
        "Unexpected keyword arguments for GraphNormalizer.*bad_kwarg",
    ):
      self.graph_normalizer.normalize_tensorflow(tf_graph, bad_kwarg="val")

    with self.assertRaisesRegex(
        ValueError,
        "Unexpected keyword arguments for GraphNormalizer.*bad_kwarg",
    ):
      self.graph_normalizer.normalize_numpy_to_jax(self.graph, bad_kwarg="val")

  def test_chained_normalizer_in_graph_with_kwargs(self):
    delta_schema = self.timedelta_normalizer.output_schema()["time_seed_delta"]
    sinusoid_normalizer = normalize_lib.SinusoidTimedeltaNormalizer.create(
        "time_seed_delta", delta_schema, embedding_dim=4
    )
    seq = normalize_lib.SequentialNormalizer.create(
        [self.timedelta_normalizer, sinusoid_normalizer]
    )
    normalizer = normalize_lib.GraphNormalizer(
        config=normalize_lib.GraphNormalizerConfig(
            nodesets={"n": normalize_lib.NodeSetNormalizerConfig([seq])},
            edgesets={},
        )
    )
    seed_ts = np.array([500, 500], dtype=np.int64)
    out_graph = normalizer.normalize_numpy(
        self.graph, seed_timestamps={"n": seed_ts}
    )
    self.assertIn("time_seed_delta_SINUSOID", out_graph.node_sets["n"].features)
    self.assertEqual(
        out_graph.node_sets["n"].features["time_seed_delta_SINUSOID"].shape,
        (2, 4),
    )

    tf_graph = tf_io.graph_to_tf_graph(self.graph)
    tf_seed_ts = tf.constant([500, 500], dtype=tf.int64)
    tf_out_graph = normalizer.normalize_tensorflow(
        tf_graph, seed_timestamps={"n": tf_seed_ts}
    )
    self.assertIn(
        "time_seed_delta_SINUSOID", tf_out_graph.node_sets["n"].features
    )
    self.assertEqual(
        tf_out_graph.node_sets["n"].features["time_seed_delta_SINUSOID"].shape,
        (2, 4),
    )

    jax_out_graph = normalizer.normalize_numpy_to_jax(
        self.graph, seed_timestamps={"n": seed_ts}
    )
    self.assertIn(
        "time_seed_delta_SINUSOID", jax_out_graph.node_sets["n"].features
    )
    self.assertEqual(
        jax_out_graph.node_sets["n"].features["time_seed_delta_SINUSOID"].shape,
        (2, 4),
    )

  def test_normalize_with_edgeset_kwargs(self):
    edge_normalizer = normalize_lib.GraphNormalizer(
        config=normalize_lib.GraphNormalizerConfig(
            nodesets={},
            edgesets={
                "e": normalize_lib.EdgeSetNormalizerConfig(
                    source="n",
                    target="n",
                    normalizers=[self.timedelta_normalizer],
                )
            },
        )
    )
    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n": in_memory_graph_lib.InMemoryNodeSet(features={}, num_nodes=2)
        },
        edge_sets={
            "e": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0, 1], [1, 0]]),
                features={"time": np.array([100, 200], dtype=np.int64)},
            )
        },
    )
    seed_ts = np.array([500, 500], dtype=np.int64)
    expected_deltas = np.array([400, 300], dtype=np.int64)

    # NumPy
    out_graph = edge_normalizer.normalize_numpy(
        graph, seed_timestamps={"e": seed_ts}
    )
    np.testing.assert_array_equal(
        out_graph.edge_sets["e"].features["time_seed_delta"], expected_deltas
    )

    # TensorFlow
    tf_graph = tf_io.graph_to_tf_graph(graph)
    tf_seed_ts = tf.constant([500, 500], dtype=tf.int64)
    tf_out_graph = edge_normalizer.normalize_tensorflow(
        tf_graph, seed_timestamps={"e": tf_seed_ts}
    )
    np.testing.assert_array_equal(
        tf_out_graph.edge_sets["e"].features["time_seed_delta"].numpy(),
        expected_deltas,
    )

    # JAX
    jax_out_graph = edge_normalizer.normalize_numpy_to_jax(
        graph, seed_timestamps={"e": seed_ts}
    )
    np.testing.assert_array_equal(
        np.asarray(jax_out_graph.edge_sets["e"].features["time_seed_delta"]),
        expected_deltas,
    )


class OneHotNormalizerTest(parameterized.TestCase):

  def test_multiclass_and_oov_numpy_tf_and_json(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        shape=(),
        num_categorical_values=3,
    )
    normalizer = normalize_lib.OneHotNormalizer.create(
        "cat", input_schema, drop_if_binary=True
    )
    values = np.array([0, 2, 3, -1, 1], dtype=np.int64)
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np_out = normalizer.normalize_numpy(values)
    np.testing.assert_array_equal(np_out["cat_ONE_HOT"], expected)

    tf_out = normalizer.normalize_tensorflow(tf.constant(values))
    np.testing.assert_array_equal(tf_out["cat_ONE_HOT"].numpy(), expected)

    self.assertEqual(
        normalizer.output_schema(),
        {
            "cat_ONE_HOT": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.EMBEDDING,
                shape=(3,),
            )
        },
    )

    cfg = normalize_lib.GraphNormalizerConfig(
        nodesets={
            "n": normalize_lib.NodeSetNormalizerConfig(normalizers=[normalizer])
        },
        edgesets={},
    )
    restored = normalize_lib.GraphNormalizerConfig.from_json(  # pyrefly: ignore[missing-attribute]
        cfg.to_json()  # pyrefly: ignore[missing-attribute]
    )
    test_util.assert_are_equal(self, restored, cfg)

  @parameterized.named_parameters(
      (
          "int_drop",
          schema_lib.FeatureFormat.INTEGER_64,
          True,
          [0, 1, 0, 1],
          [[0.0], [1.0], [0.0], [1.0]],
      ),
      (
          "int_keep",
          schema_lib.FeatureFormat.INTEGER_64,
          False,
          [0, 1, 2, -1],
          [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
      ),
      (
          "bool_drop",
          schema_lib.FeatureFormat.BOOL,
          True,
          [False, True, False, True],
          [[0.0], [1.0], [0.0], [1.0]],
      ),
      (
          "bool_keep",
          schema_lib.FeatureFormat.BOOL,
          False,
          [False, True, False, True],
          [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
      ),
  )
  def test_binary_drop_and_bool(self, fmt, drop_if_binary, values, expected):
    input_schema = schema_lib.FeatureSchema(
        format=fmt,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        shape=(),
    )
    stats = statistics_lib.FeatureStatistics(
        count=4, minimum=0.0, maximum=1.0, dictionary={}, quantiles=[]
    )
    normalizer = normalize_lib.OneHotNormalizer.create(
        "flag",
        input_schema,
        stats if fmt != schema_lib.FeatureFormat.BOOL else None,
        drop_if_binary=drop_if_binary,
    )
    values = np.array(
        values,
        dtype=np.bool_ if fmt == schema_lib.FeatureFormat.BOOL else np.int64,
    )
    expected = np.array(expected, dtype=np.float32)

    np_out = normalizer.normalize_numpy(values)["flag_ONE_HOT"]
    tf_out = (
        normalizer.normalize_tensorflow(tf.constant(values))["flag_ONE_HOT"]
        .numpy()
    )
    np.testing.assert_array_equal(np_out, expected)
    np.testing.assert_array_equal(tf_out, expected)
    self.assertEqual(np_out.dtype, np.float32)
    self.assertEqual(tf_out.dtype, np.float32)

  def test_binary_drop_enforces_binary_numpy_tf_and_graph_mode(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        shape=(),
        num_categorical_values=2,
    )
    normalizer = normalize_lib.OneHotNormalizer.create(
        "flag", input_schema, drop_if_binary=True
    )

    @tf.function
    def _tf_fn(tensor):
      return normalizer.normalize_tensorflow(tensor)["flag_ONE_HOT"]

    for bad_values in (
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0, -1], dtype=np.int64),
    ):
      with self.assertRaisesRegex(ValueError, r"must be in \[0, 2\)"):
        normalizer.normalize_numpy(bad_values)
      with self.assertRaises(tf.errors.InvalidArgumentError):
        normalizer.normalize_tensorflow(tf.constant(bad_values))
      with self.assertRaises(tf.errors.InvalidArgumentError):
        _tf_fn(tf.constant(bad_values))

  def test_invalid_output_shape_rejected(self):
    with self.assertRaisesRegex(ValueError, "trailing output_shape"):
      normalize_lib.OneHotNormalizer(
          input_feature="flag",
          num_categories=2,
          output_shape=(2,),
          output_feature_name="flag_ONE_HOT",
          drop_if_binary=True,
      )
    with self.assertRaisesRegex(ValueError, "trailing output_shape"):
      normalize_lib.OneHotNormalizer(
          input_feature="flag",
          num_categories=2,
          output_shape=(),
          output_feature_name="flag_ONE_HOT",
      )
    with self.assertRaisesRegex(ValueError, "num_categories >= 1"):
      normalize_lib.OneHotNormalizer(
          input_feature="flag",
          num_categories=0,
          output_shape=(0,),
          output_feature_name="flag_ONE_HOT",
      )

  def test_multidim_feature_and_invalid_inputs(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_32,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        shape=(2,),
        num_categorical_values=3,
    )
    normalizer = normalize_lib.OneHotNormalizer.create("pair", input_schema)
    values = np.array([[0, 2], [1, 0]], dtype=np.int32)
    expected = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np_out = normalizer.normalize_numpy(values)["pair_ONE_HOT"]
    tf_out = (
        normalizer.normalize_tensorflow(tf.constant(values))["pair_ONE_HOT"]
        .numpy()
    )
    self.assertEqual(np_out.dtype, np.float32)
    self.assertEqual(tf_out.dtype, np.float32)
    np.testing.assert_array_equal(np_out, expected)
    np.testing.assert_array_equal(tf_out, expected)

    with self.assertRaisesRegex(ValueError, "INTEGER or BOOL"):
      normalize_lib.OneHotNormalizer.create(
          "bad",
          schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.FLOAT_32,
              semantic=schema_lib.FeatureSemantic.CATEGORICAL,
          ),
          num_categories=3,
      )
    with self.assertRaisesRegex(ValueError, "static shape"):
      normalize_lib.OneHotNormalizer.create(
          "dyn",
          schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.INTEGER_64,
              semantic=schema_lib.FeatureSemantic.CATEGORICAL,
              shape=(None,),
          ),
          num_categories=3,
      )
    with self.assertRaisesRegex(ValueError, "Cannot infer num_categories"):
      normalize_lib.OneHotNormalizer.create(
          "missing",
          schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.INTEGER_64,
              semantic=schema_lib.FeatureSemantic.CATEGORICAL,
              shape=(),
          ),
      )
    with self.assertRaisesRegex(ValueError, "invalid num_categories=0"):
      normalize_lib.OneHotNormalizer.create(
          "zero_cats", input_schema, num_categories=0
      )

  def test_empty_or_non_finite_stats_rejected(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        shape=(),
    )
    for bad_stats in (
        statistics_lib.FeatureStatistics(
            count=0,
            minimum=float("nan"),
            maximum=float("nan"),
            dictionary={},
            quantiles=[],
        ),
        statistics_lib.FeatureStatistics(
            count=3,
            minimum=0.0,
            maximum=float("nan"),
            dictionary={},
            quantiles=[],
        ),
        statistics_lib.FeatureStatistics(
            count=3,
            minimum=0.0,
            maximum=float("inf"),
            dictionary={},
            quantiles=[],
        ),
    ):
      with self.assertRaisesRegex(ValueError, "No finite maximum observed"):
        normalize_lib.OneHotNormalizer.create("cat", input_schema, bad_stats)

    # Only -1 sentinels: no category was observed, so there is nothing to
    # encode (instead of a single all-zero column).
    sentinels_only = statistics_lib.FeatureStatistics(
        count=3, minimum=-1.0, maximum=-1.0, dictionary={}, quantiles=[]
    )
    with self.assertRaisesRegex(ValueError, "no non-negative value"):
      normalize_lib.OneHotNormalizer.create("cat", input_schema, sentinels_only)

    # Providing an explicit category count or schema num_categorical_values
    # succeeds even when stats are empty.
    empty_stats = statistics_lib.FeatureStatistics(
        count=0,
        minimum=float("nan"),
        maximum=float("nan"),
        dictionary={},
        quantiles=[],
    )
    explicit = normalize_lib.OneHotNormalizer.create(
        "cat", input_schema, empty_stats, num_categories=3
    )
    self.assertEqual(explicit.num_categories, 3)

  def test_max_one_hot_categories_rejects_large_id(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        shape=(),
    )
    large_id_stats = statistics_lib.FeatureStatistics(
        count=10,
        minimum=0.0,
        maximum=1e9,
        dictionary={},
        quantiles=[],
    )
    with self.assertRaisesRegex(
        ValueError, "exceeds max_one_hot_categories=1024.*DictionaryIndex"
    ):
      normalize_lib.OneHotNormalizer.create(
          "hashed_id", input_schema, large_id_stats
      )

    # Explicit / schema category counts that exceed the cap report a message
    # tailored to explicit widths (not suggesting num_categorical_values or
    # DictionaryIndexNormalizer).
    for schema, explicit_num_cats in (
        (input_schema, 10),
        (
            schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.INTEGER_64,
                semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                shape=(),
                num_categorical_values=10,
            ),
            None,
        ),
    ):
      with self.assertRaises(ValueError) as cm:
        normalize_lib.OneHotNormalizer.create(
            "cat",
            schema,
            num_categories=explicit_num_cats,
            max_one_hot_categories=4,
        )
      msg = str(cm.exception)
      self.assertIn("exceeds max_one_hot_categories=4", msg)
      self.assertIn("disable `categorical_one_hot`", msg)
      self.assertNotIn("DictionaryIndexNormalizer", msg)
      self.assertNotIn("num_categorical_values", msg)

    # Explicit smaller cap is also enforced, while passing an explicit
    # num_categorical_values within the cap succeeds.
    small_stats = statistics_lib.FeatureStatistics(
        count=4, minimum=0.0, maximum=10.0, dictionary={}, quantiles=[]
    )
    with self.assertRaisesRegex(ValueError, "exceeds max_one_hot_categories=4"):
      normalize_lib.OneHotNormalizer.create(
          "cat", input_schema, small_stats, max_one_hot_categories=4
      )
    capped_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        shape=(),
        num_categorical_values=4,
    )
    ok_norm = normalize_lib.OneHotNormalizer.create(
        "cat", capped_schema, large_id_stats, max_one_hot_categories=4
    )
    self.assertEqual(ok_norm.num_categories, 4)

    # `None` disables the cap (2001 categories from maximum=2000), and a cap
    # below 1 is rejected.
    wide_stats = statistics_lib.FeatureStatistics(
        count=4, minimum=0.0, maximum=2000.0, dictionary={}, quantiles=[]
    )
    uncapped = normalize_lib.OneHotNormalizer.create(
        "cat", input_schema, wide_stats, max_one_hot_categories=None
    )
    self.assertEqual(uncapped.num_categories, 2001)
    with self.assertRaisesRegex(ValueError, ">= 1 or None, got 0"):
      normalize_lib.OneHotNormalizer.create(
          "cat", input_schema, wide_stats, max_one_hot_categories=0
      )

  def test_drop_if_binary_rejects_negative_sentinels_in_stats(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        shape=(),
    )
    sentinel_stats = statistics_lib.FeatureStatistics(
        count=5, minimum=-1.0, maximum=1.0, dictionary={}, quantiles=[]
    )
    with self.assertRaisesRegex(ValueError, "minimum=-1.0 < 0"):
      normalize_lib.OneHotNormalizer.create(
          "flag", input_schema, sentinel_stats, drop_if_binary=True
      )
    # A third category observed in the stats contradicts a feature declared
    # binary (num_categorical_values=2) and committed to [0, 2).
    binary_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        shape=(),
        num_categorical_values=2,
    )
    third_class_stats = statistics_lib.FeatureStatistics(
        count=5, minimum=0.0, maximum=2.0, dictionary={}, quantiles=[]
    )
    with self.assertRaisesRegex(ValueError, "maximum=2.0 >= 2"):
      normalize_lib.OneHotNormalizer.create(
          "flag", binary_schema, third_class_stats, drop_if_binary=True
      )

    # With drop_if_binary=False, -1 sentinels are allowed and encode as zeros.
    safe_norm = normalize_lib.OneHotNormalizer.create(
        "flag", input_schema, sentinel_stats, drop_if_binary=False
    )
    np.testing.assert_array_equal(
        safe_norm.normalize_numpy(np.array([-1, 0, 1], dtype=np.int64))[
            "flag_ONE_HOT"
        ],
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

  def test_timeseries_implicit_group_preserved(self):
    ts_implicit = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        shape=(4,),
        num_categorical_values=3,
        is_timeseries=True,
        group=None,
    )
    expected_implicit = {
        "events_ONE_HOT": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=(4, 3),
            is_timeseries=True,
            group="events",
        )
    }
    norm_implicit = normalize_lib.OneHotNormalizer.create("events", ts_implicit)
    self.assertEqual(norm_implicit.output_schema(), expected_implicit)

    # Direct construction and JSON with `group=None` also populate the implicit
    # group via `__post_init__`.
    direct = normalize_lib.OneHotNormalizer(
        input_feature="events",
        num_categories=3,
        output_shape=(4, 3),
        output_feature_name="events_ONE_HOT",
        is_timeseries=True,
        group=None,
    )
    self.assertEqual(direct.output_schema(), expected_implicit)
    cfg = normalize_lib.GraphNormalizerConfig(
        nodesets={"n": normalize_lib.NodeSetNormalizerConfig([direct])},
        edgesets={},
    )
    null_group_json = cfg.to_json().replace('"group": "events"', '"group": null')  # pyrefly: ignore[missing-attribute]
    self.assertIn('"group": null', null_group_json)
    restored = (
        normalize_lib.GraphNormalizerConfig.from_json(null_group_json)  # pyrefly: ignore[missing-attribute]
        .nodesets["n"]
        .normalizers[0]
    )
    self.assertEqual(restored.output_schema(), expected_implicit)

    ts_explicit = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        shape=(4,),
        num_categorical_values=3,
        is_timeseries=True,
        group="seq_a",
    )
    norm_explicit = normalize_lib.OneHotNormalizer.create("events", ts_explicit)
    self.assertEqual(
        norm_explicit.output_schema(),
        {
            "events_ONE_HOT": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.EMBEDDING,
                shape=(4, 3),
                is_timeseries=True,
                group="seq_a",
            )
        },
    )


class ImputeNanNormalizerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("median", "median", [2.0, 1.0]),
      ("mean", "mean", [2.25, 1.0]),
      ("zero", "zero", [0.0, 0.0]),
  )
  def test_from_values_strategies_and_tf_parity(self, strategy, expected_fills):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(2,),
    )
    train_values = np.array(
        [
            [2.0, -3.0],
            [2.0, 5.0],
            [1.0, -3.0],
            [4.0, 5.0],
            [np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    normalizer = normalize_lib.ImputeNanNormalizer.from_values(
        "num", input_schema, train_values, strategy=strategy
    )
    np.testing.assert_allclose(
        normalizer.fill_values, np.array(expected_fills, dtype=np.float32)
    )

    test_values = np.array([[np.nan, 10.0], [7.0, np.nan]], dtype=np.float32)
    np_out = normalizer.normalize_numpy(test_values)["num_IMPUTED"]
    tf_out = (
        normalizer.normalize_tensorflow(tf.constant(test_values))["num_IMPUTED"]
        .numpy()
    )
    expected = np.array(
        [[expected_fills[0], 10.0], [7.0, expected_fills[1]]], dtype=np.float32
    )
    np.testing.assert_allclose(np_out, expected)
    np.testing.assert_allclose(tf_out, expected)
    self.assertEqual(np_out.dtype, np.float32)
    self.assertEqual(tf_out.dtype, np.float32)

    cfg = normalize_lib.GraphNormalizerConfig(
        nodesets={
            "n": normalize_lib.NodeSetNormalizerConfig(normalizers=[normalizer])
        },
        edgesets={},
    )
    restored = normalize_lib.GraphNormalizerConfig.from_json(  # pyrefly: ignore[missing-attribute]
        cfg.to_json()  # pyrefly: ignore[missing-attribute]
    )
    test_util.assert_are_equal(self, restored, cfg)

  def test_create_from_stats_and_all_nan_rejected(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    stats = statistics_lib.FeatureStatistics(
        count=3,
        minimum=1.0,
        maximum=9.0,
        dictionary={},
        quantiles=[1.0, 4.0, 9.0],
        mean=4.5,
    )
    med_norm = normalize_lib.ImputeNanNormalizer.create(
        "x", input_schema, stats, strategy="median"
    )
    np.testing.assert_allclose(
        med_norm.normalize_numpy(np.array([np.nan, 2.0], dtype=np.float32))[
            "x_IMPUTED"
        ],
        np.array([4.0, 2.0], dtype=np.float32),
    )

    with self.assertRaisesRegex(ValueError, "entirely non-finite"):
      normalize_lib.ImputeNanNormalizer.from_values(
          "x", input_schema, np.array([np.nan, np.nan], dtype=np.float32)
      )

  def test_create_median_even_length_matches_numpy_median(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    even_stats = statistics_lib.FeatureStatistics(
        count=4,
        minimum=1.0,
        maximum=10.0,
        dictionary={},
        quantiles=[1.0, 2.0, 6.0, 10.0],
        mean=4.75,
    )
    norm = normalize_lib.ImputeNanNormalizer.create(
        "x", input_schema, even_stats, strategy="median"
    )
    np.testing.assert_allclose(
        norm.fill_values, np.array(4.0, dtype=np.float32)
    )

  def test_create_fails_loudly_and_median_with_fallbacks(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    no_quantiles_stats = statistics_lib.FeatureStatistics(
        count=3,
        minimum=1.0,
        maximum=5.0,
        dictionary={},
        quantiles=[],
        mean=3.0,
    )
    empty_stats = statistics_lib.FeatureStatistics(
        count=0,
        minimum=float("nan"),
        maximum=float("nan"),
        dictionary={},
        quantiles=[],
        mean=float("nan"),
    )

    with self.assertRaisesRegex(ValueError, "No finite quantiles"):
      normalize_lib.ImputeNanNormalizer.create(
          "x", input_schema, no_quantiles_stats, strategy="median"
      )
    with self.assertRaisesRegex(ValueError, "No finite mean"):
      normalize_lib.ImputeNanNormalizer.create(
          "x", input_schema, empty_stats, strategy="mean"
      )

    # Explicit fallback strategy: quantiles -> mean -> 0.0.
    mean_fb = normalize_lib.ImputeNanNormalizer.create(
        "x", input_schema, no_quantiles_stats, strategy="median_with_fallbacks"
    )
    np.testing.assert_allclose(
        mean_fb.fill_values, np.array(3.0, dtype=np.float32)
    )
    zero_fb = normalize_lib.ImputeNanNormalizer.create(
        "x", input_schema, empty_stats, strategy="median_with_fallbacks"
    )
    np.testing.assert_allclose(
        zero_fb.fill_values, np.array(0.0, dtype=np.float32)
    )
    # With finite quantiles the fallback strategy uses their median (4.0),
    # not the mean (9.0).
    skewed_stats = statistics_lib.FeatureStatistics(
        count=3,
        minimum=1.0,
        maximum=22.0,
        dictionary={},
        quantiles=[1.0, 4.0, 22.0],
        mean=9.0,
    )
    quantile_fb = normalize_lib.ImputeNanNormalizer.create(
        "x", input_schema, skewed_stats, strategy="median_with_fallbacks"
    )
    self.assertEqual(float(quantile_fb.fill_values), 4.0)
    # `mean` and `zero` use exactly the mean and 0.0.
    self.assertEqual(
        float(
            normalize_lib.ImputeNanNormalizer.create(
                "x", input_schema, skewed_stats, strategy="mean"
            ).fill_values
        ),
        9.0,
    )
    self.assertEqual(
        float(
            normalize_lib.ImputeNanNormalizer.create(
                "x", input_schema, skewed_stats, strategy="zero"
            ).fill_values
        ),
        0.0,
    )
    # A non-finite quantile makes the quantiles unusable even when the middle
    # one is finite: `median` raises and the fallback uses the mean.
    inf_quantile_stats = statistics_lib.FeatureStatistics(
        count=3,
        minimum=1.0,
        maximum=float("inf"),
        dictionary={},
        quantiles=[1.0, 2.0, float("inf")],
        mean=5.0,
    )
    with self.assertRaisesRegex(ValueError, "No finite quantiles"):
      normalize_lib.ImputeNanNormalizer.create(
          "x", input_schema, inf_quantile_stats, strategy="median"
      )
    self.assertEqual(
        float(
            normalize_lib.ImputeNanNormalizer.create(
                "x",
                input_schema,
                inf_quantile_stats,
                strategy="median_with_fallbacks",
            ).fill_values
        ),
        5.0,
    )

  def test_imputes_posinf_and_neginf_across_paths(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(2,),
    )
    train_values = np.array(
        [
            [2.0, 10.0],
            [4.0, 20.0],
            [np.inf, -np.inf],
            [-np.inf, np.nan],
        ],
        dtype=np.float32,
    )
    normalizer = normalize_lib.ImputeNanNormalizer.from_values(
        "x", input_schema, train_values, strategy="median"
    )
    np.testing.assert_allclose(
        normalizer.fill_values, np.array([3.0, 15.0], dtype=np.float32)
    )

    test_values = np.array(
        [[np.nan, np.inf], [-np.inf, 7.0]], dtype=np.float32
    )
    expected = np.array([[3.0, 15.0], [3.0, 7.0]], dtype=np.float32)
    np.testing.assert_allclose(
        normalizer.normalize_numpy(test_values)["x_IMPUTED"], expected
    )
    np.testing.assert_allclose(
        normalizer.normalize_tensorflow(tf.constant(test_values))["x_IMPUTED"]
        .numpy(),
        expected,
    )

    with self.assertRaisesRegex(ValueError, "entirely non-finite"):
      normalize_lib.ImputeNanNormalizer.from_values(
          "x",
          input_schema,
          np.array([[np.inf, 1.0], [-np.inf, 2.0]], dtype=np.float32),
      )
    for bad_fill in (np.nan, np.inf, -np.inf):
      with self.assertRaisesRegex(ValueError, "finite fill_values"):
        normalize_lib.ImputeNanNormalizer(
            input_feature="x",
            output_feature_name="x_IMPUTED",
            output_shape=(2,),
            fill_values=np.array([1.0, bad_fill], dtype=np.float32),
        )

  def test_float64_beyond_float32_range_is_filled_on_both_paths(self):
    normalizer = normalize_lib.ImputeNanNormalizer(
        input_feature="x",
        output_feature_name="x_IMPUTED",
        output_shape=(),
        fill_values=np.array(2.0, dtype=np.float32),
    )
    # 1e40 overflows float32, so it is filled like ±inf on both paths, and the
    # output stays finite.
    values = np.array([1e40, 3.0, np.nan], dtype=np.float64)
    expected = np.array([2.0, 3.0, 2.0], dtype=np.float32)
    for out in (
        normalizer.normalize_numpy(values)["x_IMPUTED"],
        normalizer.normalize_tensorflow(tf.constant(values))["x_IMPUTED"].numpy(),
    ):
      self.assertEqual(out.dtype, np.float32)
      np.testing.assert_array_equal(out, expected)

  def test_parameter_and_schema_validation_raises(self):
    with self.assertRaisesRegex(ValueError, r"fill_values of shape \(\)"):
      normalize_lib.ImputeNanNormalizer(
          input_feature="x",
          output_feature_name="x_IMPUTED",
          output_shape=(),
          fill_values=np.array([[1.0], [2.0]], dtype=np.float32),
      )
    stats = statistics_lib.FeatureStatistics(
        count=2,
        minimum=1.0,
        maximum=3.0,
        dictionary={},
        quantiles=[1.0, 3.0],
        mean=2.0,
    )
    with self.assertRaisesRegex(ValueError, "INTEGER or FLOAT"):
      normalize_lib.ImputeNanNormalizer.create(
          "x",
          schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.BYTES,
              semantic=schema_lib.FeatureSemantic.NUMERICAL,
              shape=(),
          ),
          stats,
      )
    with self.assertRaisesRegex(ValueError, "static shape"):
      normalize_lib.ImputeNanNormalizer.create(
          "x",
          schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.FLOAT_32,
              semantic=schema_lib.FeatureSemantic.NUMERICAL,
              shape=(None,),
          ),
          stats,
      )
    fixed_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(2,),
    )
    with self.assertRaisesRegex(ValueError, "at least 1 dimension"):
      normalize_lib.ImputeNanNormalizer.from_values(
          "x", fixed_schema, np.array(1.0, dtype=np.float32)
      )
    with self.assertRaisesRegex(
        ValueError, r"item shape \(3,\), expected \(2,\)"
    ):
      normalize_lib.ImputeNanNormalizer.from_values(
          "x", fixed_schema, np.zeros((4, 3), dtype=np.float32)
      )

  def test_timeseries_implicit_group_preserved(self):
    ts_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(3,),
        is_timeseries=True,
        group=None,
    )
    expected_implicit = {
        "seq_IMPUTED": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.NUMERICAL,
            shape=(3,),
            is_timeseries=True,
            group="seq",
        )
    }
    stats = statistics_lib.FeatureStatistics(
        count=3,
        minimum=1.0,
        maximum=3.0,
        dictionary={},
        quantiles=[1.0, 2.0, 3.0],
        mean=2.0,
    )
    from_stats = normalize_lib.ImputeNanNormalizer.create(
        "seq", ts_schema, stats
    )
    self.assertEqual(from_stats.output_schema(), expected_implicit)
    from_vals = normalize_lib.ImputeNanNormalizer.from_values(
        "seq", ts_schema, np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    )
    self.assertEqual(from_vals.output_schema(), expected_implicit)
    direct = normalize_lib.ImputeNanNormalizer(
        input_feature="seq",
        output_feature_name="seq_IMPUTED",
        output_shape=(3,),
        fill_values=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        is_timeseries=True,
        group=None,
    )
    self.assertEqual(direct.output_schema(), expected_implicit)
    cfg = normalize_lib.GraphNormalizerConfig(
        nodesets={"n": normalize_lib.NodeSetNormalizerConfig([direct])},
        edgesets={},
    )
    null_group_json = cfg.to_json().replace('"group": "seq"', '"group": null')  # pyrefly: ignore[missing-attribute]
    self.assertIn('"group": null', null_group_json)
    restored = (
        normalize_lib.GraphNormalizerConfig.from_json(null_group_json)  # pyrefly: ignore[missing-attribute]
        .nodesets["n"]
        .normalizers[0]
    )
    self.assertEqual(restored.output_schema(), expected_implicit)

    explicit_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(3,),
        is_timeseries=True,
        group="seq_a",
    )
    explicit_norm = normalize_lib.ImputeNanNormalizer.create(
        "seq", explicit_schema, stats
    )
    self.assertEqual(
        explicit_norm.output_schema(),
        {
            "seq_IMPUTED": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.NUMERICAL,
                shape=(3,),
                is_timeseries=True,
                group="seq_a",
            )
        },
    )

  def test_output_schema_preserves_input_semantic(self):
    stats = statistics_lib.FeatureStatistics(
        count=2,
        minimum=1.0,
        maximum=3.0,
        dictionary={},
        quantiles=[1.0, 3.0],
        mean=2.0,
    )
    vals = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    normalizers = []
    for semantic in (
        schema_lib.FeatureSemantic.NUMERICAL,
        schema_lib.FeatureSemantic.EMBEDDING,
    ):
      schema = schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=semantic,
          shape=(2,),
      )
      for name, norm in (
          (
              f"stats_{semantic.name}",
              normalize_lib.ImputeNanNormalizer.create(
                  f"stats_{semantic.name}", schema, stats
              ),
          ),
          (
              f"vals_{semantic.name}",
              normalize_lib.ImputeNanNormalizer.from_values(
                  f"vals_{semantic.name}", schema, vals
              ),
          ),
      ):
        self.assertEqual(
            norm.output_schema(),
            {
                f"{name}_IMPUTED": schema_lib.FeatureSchema(
                    format=schema_lib.FeatureFormat.FLOAT_32,
                    semantic=semantic,
                    shape=(2,),
                )
            },
        )
        normalizers.append(norm)

    cfg = normalize_lib.GraphNormalizerConfig(
        nodesets={
            "n": normalize_lib.NodeSetNormalizerConfig(normalizers=normalizers)
        },
        edgesets={},
    )
    restored = normalize_lib.GraphNormalizerConfig.from_json(  # pyrefly: ignore[missing-attribute]
        cfg.to_json()  # pyrefly: ignore[missing-attribute]
    )
    test_util.assert_are_equal(self, restored, cfg)

  def test_from_values_defaults_to_median(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    train_values = np.array([1.0, 2.0, 9.0, np.nan], dtype=np.float32)
    normalizer = normalize_lib.ImputeNanNormalizer.from_values(
        "x", input_schema, train_values
    )
    np.testing.assert_allclose(
        normalizer.fill_values, np.array(2.0, dtype=np.float32)
    )

  def test_unsupported_strategy_rejected(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    with self.assertRaisesRegex(ValueError, "Unsupported ImputeNanNormalizer"):
      normalize_lib.ImputeNanNormalizer.from_values(
          "x",
          input_schema,
          np.array([1.0, 2.0, 2.0], dtype=np.float32),
          strategy="most_frequent",  # pyrefly: ignore[bad-argument-type]
      )
    stats = statistics_lib.FeatureStatistics(
        count=3,
        minimum=1.0,
        maximum=2.0,
        dictionary={},
        quantiles=[1.0, 2.0, 2.0],
        mean=5.0 / 3.0,
    )
    for bad_strategy in ("most_frequent", "bogus"):
      with self.assertRaisesRegex(
          ValueError, "Unsupported ImputeNanNormalizer.create strategy"
      ):
        normalize_lib.ImputeNanNormalizer.create(
            "x",
            input_schema,
            stats,
            strategy=bad_strategy,  # pyrefly: ignore[bad-argument-type]
        )
    # The strategy is validated before looking at the data, so even an empty
    # feature or an all-NaN column reports the unsupported strategy.
    for schema, values in (
        (
            schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.NUMERICAL,
                shape=(0,),
            ),
            np.zeros((3, 0), dtype=np.float32),
        ),
        (input_schema, np.array([np.nan], dtype=np.float32)),
    ):
      with self.assertRaisesRegex(
          ValueError, "Unsupported ImputeNanNormalizer"
      ):
        normalize_lib.ImputeNanNormalizer.from_values(
            "x",
            schema,
            values,
            strategy="bogus",  # pyrefly: ignore[bad-argument-type]
        )


class ZScoreNormalizerTest(absltest.TestCase):

  def test_z_score_with_zero_variance_clamping_and_json(self):
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.EMBEDDING,
        shape=(3,),
    )
    # Col 0: [1, 5, nan] -> mean 3, std 2 (distinct from 1.0).
    # Col 1: [5, 5, 5]   -> mean 5, constant -> std 1.0.
    # Col 2: [2, 6, 4]   -> mean 4, std sqrt(8/3).
    train_values = np.array(
        [
            [1.0, 5.0, 2.0],
            [5.0, 5.0, 6.0],
            [np.nan, 5.0, 4.0],
        ],
        dtype=np.float32,
    )
    normalizer = normalize_lib.ZScoreNormalizer.from_values(
        "emb", input_schema, train_values
    )
    np.testing.assert_allclose(
        normalizer.mean, np.array([3.0, 5.0, 4.0], dtype=np.float64), rtol=1e-12
    )
    np.testing.assert_allclose(
        normalizer.std,
        np.array([2.0, 1.0, math.sqrt(8.0 / 3.0)], dtype=np.float64),
        rtol=1e-12,
    )

    test_values = np.array(
        [[7.0, 5.0, 6.0], [-1.0, 7.0, 2.0]], dtype=np.float32
    )
    np_out = normalizer.normalize_numpy(test_values)["emb_Z_SCORE"]
    tf_out = (
        normalizer.normalize_tensorflow(tf.constant(test_values))["emb_Z_SCORE"]
        .numpy()
    )
    s = math.sqrt(1.5)  # 2 / sqrt(8/3)
    expected = np.array([[2.0, 0.0, s], [-2.0, 2.0, -s]], dtype=np.float32)
    for out in (np_out, tf_out):
      self.assertEqual(out.dtype, np.float32)
      np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)

    cfg = normalize_lib.GraphNormalizerConfig(
        nodesets={
            "n": normalize_lib.NodeSetNormalizerConfig(normalizers=[normalizer])
        },
        edgesets={},
    )
    restored = normalize_lib.GraphNormalizerConfig.from_json(  # pyrefly: ignore[missing-attribute]
        cfg.to_json()  # pyrefly: ignore[missing-attribute]
    )
    test_util.assert_are_equal(self, restored, cfg)

  def test_constant_std_uses_unit_scale_on_all_paths(self):
    # 1. Construction: exact zero std becomes 1.0, while any positive finite
    # std is kept, including at a large mean.
    constructed = normalize_lib.ZScoreNormalizer(
        input_feature="x",
        output_feature_name="x_Z_SCORE",
        output_shape=(3,),
        mean=np.array([1.0, 1e6, 0.0], dtype=np.float64),
        std=np.array([0.0, 1e-3, 1e-3], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        constructed.std,
        np.array([1.0, 1e-3, 1e-3], dtype=np.float64),
    )

    # 2. Fitting: column 0 is constant (`max == min`), so it gets scale 1.0 and
    # normalizes to 0.0. Columns 1-3 have spread `sqrt(2/3) * step` at offsets
    # 0, 1e6, and 1e6 with a float32-ulp step (0.0625); all keep their std.
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_64,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(4,),
    )
    train_values = np.array(
        [
            [0.1, 0.0, 1e6, 1e6],
            [0.1, 1.0, 1e6 + 1.0, 1e6 + 0.0625],
            [0.1, 2.0, 1e6 + 2.0, 1e6 - 0.0625],
        ],
        dtype=np.float64,
    )
    fitted = normalize_lib.ZScoreNormalizer.from_values(
        "x", input_schema, train_values
    )
    np.testing.assert_allclose(
        fitted.std,
        [1.0, math.sqrt(2 / 3), math.sqrt(2 / 3), 0.0625 * math.sqrt(2 / 3)],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        fitted.mean, [0.1, 1.0, 1e6 + 1.0, 1e6], rtol=1e-15
    )

    s = math.sqrt(1.5)  # (+-1) / sqrt(2/3)
    expected_train = np.array(
        [[0.0, -s, -s, 0.0], [0.0, 0.0, 0.0, s], [0.0, s, s, -s]],
        dtype=np.float32,
    )
    # Unseen values: the constant column is shifted by 0.5 (scale 1.0), and the
    # others move two steps above their mean, i.e. 2 / sqrt(2/3) = sqrt(6).
    test_values = np.array([[0.6, 3.0, 1e6 + 3.0, 1e6 + 0.125]])
    expected_test = np.array(
        [[0.5, math.sqrt(6), math.sqrt(6), math.sqrt(6)]], dtype=np.float32
    )
    for values, expected in (
        (train_values, expected_train),
        (test_values, expected_test),
    ):
      np_out = fitted.normalize_numpy(values)["x_Z_SCORE"]
      tf_out = fitted.normalize_tensorflow(tf.constant(values))[
          "x_Z_SCORE"
      ].numpy()
      for out in (np_out, tf_out):
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)

    # 3. JSON deserialization maps 0.0 std to 1.0 and rejects negative std.
    cfg = normalize_lib.GraphNormalizerConfig(
        nodesets={
            "n": normalize_lib.NodeSetNormalizerConfig(
                normalizers=[
                    normalize_lib.ZScoreNormalizer(
                        input_feature="x",
                        output_feature_name="x_Z_SCORE",
                        output_shape=(3,),
                        mean=np.array([1.0, 2.0, 3.0], dtype=np.float64),
                        std=np.array([2.0, 4.0, 5.0], dtype=np.float64),
                    )
                ]
            )
        },
        edgesets={},
    )
    zero_json = cfg.to_json().replace("[2.0, 4.0, 5.0]", "[0.0, 4.0, 5.0]")  # pyrefly: ignore[missing-attribute]
    restored = normalize_lib.GraphNormalizerConfig.from_json(zero_json)  # pyrefly: ignore[missing-attribute]
    restored_norm = restored.nodesets["n"].normalizers[0]
    self.assertIsInstance(restored_norm, normalize_lib.ZScoreNormalizer)
    np.testing.assert_array_equal(
        restored_norm.std, np.array([1.0, 4.0, 5.0], dtype=np.float64)
    )
    self.assertEqual(restored_norm.std.dtype, np.float64)

    neg_json = cfg.to_json().replace("[2.0, 4.0, 5.0]", "[-1.0, 4.0, 5.0]")  # pyrefly: ignore[missing-attribute]
    with self.assertRaisesRegex(ValueError, "non-negative std"):
      normalize_lib.GraphNormalizerConfig.from_json(neg_json)  # pyrefly: ignore[missing-attribute]

    # 4. Normalization (NumPy, TF eager, TF graph, JAX) relies on the
    # `__post_init__` zero-variance invariant across every backend.
    constructed = normalize_lib.ZScoreNormalizer(
        input_feature="x",
        output_feature_name="x_Z_SCORE",
        output_shape=(2,),
        mean=np.array([1.0, 3.0], dtype=np.float64),
        std=np.array([0.0, 2.0], dtype=np.float64),
    )
    values = np.array([[4.0, 7.0]], dtype=np.float32)
    expected = np.array([[3.0, 2.0]], dtype=np.float32)
    np_out = constructed.normalize_numpy(values)["x_Z_SCORE"]
    tf_out = constructed.normalize_tensorflow(tf.constant(values))[
        "x_Z_SCORE"
    ].numpy()

    @tf.function
    def _tf_fn(tensor):
      return constructed.normalize_tensorflow(tensor)["x_Z_SCORE"]

    tf_graph_out = _tf_fn(tf.constant(values)).numpy()
    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=1, features={"x": values}
            )
        },
        edge_sets={},
    )
    jax_out = (
        normalize_lib.GraphNormalizer(
            config=normalize_lib.GraphNormalizerConfig(
                nodesets={
                    "n": normalize_lib.NodeSetNormalizerConfig(
                        normalizers=[constructed]
                    )
                },
                edgesets={},
            )
        )
        .normalize_numpy_to_jax(graph)
        .node_sets["n"]
        .features["x_Z_SCORE"]
    )
    for out in (np_out, tf_out, tf_graph_out, np.asarray(jax_out)):
      self.assertEqual(out.dtype, np.float32)
      np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)

  def test_invalid_mean_or_std_rejected(self):
    for bad_mean in (np.nan, np.inf, -np.inf):
      with self.assertRaisesRegex(ValueError, "finite mean"):
        normalize_lib.ZScoreNormalizer(
            input_feature="x",
            output_feature_name="x_Z_SCORE",
            output_shape=(2,),
            mean=np.array([1.0, bad_mean], dtype=np.float64),
            std=np.array([1.0, 1.0], dtype=np.float64),
        )
    for bad_std in (np.nan, np.inf, -np.inf, -1e-6, -2.0):
      with self.assertRaisesRegex(ValueError, "non-negative std"):
        normalize_lib.ZScoreNormalizer(
            input_feature="x",
            output_feature_name="x_Z_SCORE",
            output_shape=(2,),
            mean=np.array([1.0, 2.0], dtype=np.float64),
            std=np.array([1.0, bad_std], dtype=np.float64),
        )

  def test_parameter_shapes_must_match_output_shape(self):
    with self.assertRaisesRegex(ValueError, r"mean of shape \(3,\)"):
      normalize_lib.ZScoreNormalizer(
          input_feature="x",
          output_feature_name="x_Z_SCORE",
          output_shape=(3,),
          mean=np.array([[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]]),
          std=np.array([1.0, 1.0, 1.0]),
      )
    with self.assertRaisesRegex(ValueError, r"std of shape \(3,\)"):
      normalize_lib.ZScoreNormalizer(
          input_feature="x",
          output_feature_name="x_Z_SCORE",
          output_shape=(3,),
          mean=np.array([10.0, 20.0, 30.0]),
          std=np.array(1.0),
      )

  def test_from_values_fails_loudly_on_unusable_data(self):
    schema_2d = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_64,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(2,),
    )
    schema_1d = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_64,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    # No rows, or one channel without any finite value.
    for schema, values in (
        (schema_2d, np.zeros((0, 2), dtype=np.float64)),
        (schema_2d, np.array([[np.nan, 1.0], [np.inf, 3.0]])),
    ):
      with self.assertRaisesRegex(ValueError, "entirely non-finite"):
        normalize_lib.ZScoreNormalizer.from_values("x", schema, values)
    # Non-numeric values.
    for values in (np.array([b"1.0", b"3.0"]), np.array([True, False])):
      with self.assertRaisesRegex(ValueError, "dense integer or float array"):
        normalize_lib.ZScoreNormalizer.from_values("x", schema_1d, values)

  def test_categorical_input_semantic_rejected(self):
    values = np.array([0, 1, 2], dtype=np.int64)
    stats = statistics_lib.FeatureStatistics(
        count=3,
        minimum=0.0,
        maximum=2.0,
        dictionary={},
        quantiles=[0.0, 1.0, 2.0],
        mean=1.0,
    )
    numerical = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    for fit in (
        lambda s: normalize_lib.ZScoreNormalizer.from_values("x", s, values),
        lambda s: normalize_lib.ImputeNanNormalizer.from_values("x", s, values),
        lambda s: normalize_lib.ImputeNanNormalizer.create("x", s, stats),
    ):
      for semantic in (
          schema_lib.FeatureSemantic.CATEGORICAL,
          schema_lib.FeatureSemantic.TIMESTAMP,
          schema_lib.FeatureSemantic.PRIMARY_ID,
          schema_lib.FeatureSemantic.MASK,
      ):
        with self.assertRaisesRegex(ValueError, f"semantic {semantic.value}"):
          fit(
              schema_lib.FeatureSchema(
                  format=schema_lib.FeatureFormat.INTEGER_64,
                  semantic=semantic,
                  shape=(),
              )
          )
      # The same feature declared as numerical is accepted.
      fit(numerical)

  def test_large_integers_keep_exact_statistics_or_raise(self):
    int_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    # 2**52 + 1 + (0, 3, 6), 5 times: mean 2**52 + 4, std sqrt(6). The plain
    # float64 sum (~6.8e16) exceeds 2**53 and would round the mean by 1.
    values = np.tile(np.array([0, 3, 6], dtype=np.int64), 5) + 2**52 + 1
    fitted = normalize_lib.ZScoreNormalizer.from_values("x", int_schema, values)
    self.assertEqual(float(fitted.mean), float(2**52 + 4))
    np.testing.assert_allclose(float(fitted.std), math.sqrt(6), rtol=1e-12)
    np.testing.assert_allclose(
        fitted.normalize_numpy(values[:3])["x_Z_SCORE"],
        np.array([-1.0, 0.0, 1.0], dtype=np.float32) * np.sqrt(np.float32(1.5)),
        rtol=0,
        atol=1e-6,
    )

    # Beyond 2**53 float64 merges integers, e.g. nanosecond timestamps.
    with self.assertRaisesRegex(ValueError, r"beyond 2\*\*53"):
      normalize_lib.ZScoreNormalizer.from_values(
          "x", int_schema, np.array([0, 50, 100], dtype=np.int64) + 1_700 * 10**15
      )

  def test_std_underflow_or_overflow_raises(self):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_64,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    # A non-constant channel whose std underflows to 0 or overflows to inf
    # would otherwise be scaled by 1 or fail with a misleading message.
    for values in (np.array([0.0, 1e-200]), np.array([-1e200, 1e200])):
      with self.assertRaisesRegex(ValueError, "rescale the feature"):
        normalize_lib.ZScoreNormalizer.from_values("x", schema, values)

  def test_z_score_then_impute_zero_preserves_large_offset_precision(self):
    schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_64,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    values = np.array(
        [1e8, 1e8 + 20.5, 1e8 + 41.0, 1e8 + 61.5, np.nan, np.inf],
        dtype=np.float64,
    )
    z_norm = normalize_lib.ZScoreNormalizer.from_values("x", schema, values)
    z_out_schema = z_norm.output_schema()["x_Z_SCORE"]
    imp_norm = normalize_lib.ImputeNanNormalizer.from_values(
        "x_Z_SCORE",
        z_out_schema,
        z_norm.normalize_numpy(values)["x_Z_SCORE"],
        strategy="zero",
    )
    seq = normalize_lib.SequentialNormalizer.create([z_norm, imp_norm])
    expected = np.array(
        [
            -3.0 / np.sqrt(5.0),
            -1.0 / np.sqrt(5.0),
            1.0 / np.sqrt(5.0),
            3.0 / np.sqrt(5.0),
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    for out in (
        seq.normalize_numpy(values)["x_Z_SCORE_IMPUTED"],
        seq.normalize_tensorflow(tf.constant(values))[
            "x_Z_SCORE_IMPUTED"
        ].numpy(),
    ):
      self.assertEqual(out.dtype, np.float32)
      np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

  def test_large_offset_keeps_precision_on_all_paths(self):
    # The signal [0, 20.5, 41, 61.5] has mean 30.75, deviations (+-30.75,
    # +-10.25) = 10.25 * (+-3, +-1), and std 10.25 * sqrt(5); so the expected
    # outputs are (-3, -1, 1, 3) / sqrt(5) at any offset. In float32 the ulp
    # is 1 at 1e7 and 8 at 1e8, so a float32 fit or a float32 `x - mean`
    # visibly distorts them (e.g. +-32 instead of +-1.34 at 1e8).
    input_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_64,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(),
    )
    signal = np.array([0.0, 20.5, 41.0, 61.5], dtype=np.float64)
    expected = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32) / np.sqrt(
        np.float32(5.0)
    )
    for base in (1e7, 1e8):
      train_values = np.concatenate([
          base + signal,
          np.array([np.inf, -np.inf, np.nan], dtype=np.float64),
      ])
      fitted = normalize_lib.ZScoreNormalizer.from_values(
          "big", input_schema, train_values
      )
      self.assertEqual(float(fitted.mean), base + 30.75)
      np.testing.assert_allclose(
          float(fitted.std), 10.25 * math.sqrt(5), rtol=1e-12
      )

      # JSON keeps the float64 statistics bit-exact.
      cfg = normalize_lib.GraphNormalizerConfig(
          nodesets={
              "n": normalize_lib.NodeSetNormalizerConfig(normalizers=[fitted])
          },
          edgesets={},
      )
      restored = (
          normalize_lib.GraphNormalizerConfig.from_json(  # pyrefly: ignore[missing-attribute]
              cfg.to_json()  # pyrefly: ignore[missing-attribute]
          )
          .nodesets["n"]
          .normalizers[0]
      )
      self.assertEqual(float(restored.mean), float(fitted.mean))
      self.assertEqual(float(restored.std), float(fitted.std))
      self.assertEqual(restored.mean.dtype, np.float64)
      self.assertEqual(restored.std.dtype, np.float64)

      # Both the fitted and the deserialized normalizer keep the signal.
      values = base + signal
      for norm in (fitted, restored):
        np_out = norm.normalize_numpy(values)["big_Z_SCORE"]
        tf_out = norm.normalize_tensorflow(tf.constant(values))[
            "big_Z_SCORE"
        ].numpy()
        for out in (np_out, tf_out):
          self.assertEqual(out.dtype, np.float32)
          np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)

    # A constant channel at a huge offset takes its value exactly as the mean,
    # so it normalizes to exactly 0 (a mean 1 ulp off at 1e60 would be ~1e44).
    const_fitted = normalize_lib.ZScoreNormalizer.from_values(
        "c", input_schema, np.full((10,), 1e60)
    )
    self.assertEqual(float(const_fitted.mean), 1e60)
    np.testing.assert_array_equal(
        const_fitted.normalize_numpy(np.full((10,), 1e60))["c_Z_SCORE"],
        np.zeros((10,), dtype=np.float32),
    )

  def test_output_schema_matches_input_semantic_and_timeseries_group(self):
    num_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(2,),
        is_timeseries=True,
        group=None,
    )
    explicit_ts_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        shape=(2,),
        is_timeseries=True,
        group="seq_a",
    )
    emb_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.EMBEDDING,
        shape=(2,),
    )
    vals = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    num_norm = normalize_lib.ZScoreNormalizer.from_values(
        "num_ts", num_schema, vals
    )
    explicit_ts_norm = normalize_lib.ZScoreNormalizer.from_values(
        "num_ts", explicit_ts_schema, vals
    )
    emb_norm = normalize_lib.ZScoreNormalizer.from_values(
        "emb", emb_schema, vals
    )
    expected_implicit = {
        "num_ts_Z_SCORE": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.NUMERICAL,
            shape=(2,),
            is_timeseries=True,
            group="num_ts",
        )
    }
    self.assertEqual(num_norm.output_schema(), expected_implicit)
    self.assertEqual(
        explicit_ts_norm.output_schema(),
        {
            "num_ts_Z_SCORE": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.NUMERICAL,
                shape=(2,),
                is_timeseries=True,
                group="seq_a",
            )
        },
    )
    self.assertEqual(
        emb_norm.output_schema(),
        {
            "emb_Z_SCORE": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32,
                semantic=schema_lib.FeatureSemantic.EMBEDDING,
                shape=(2,),
                is_timeseries=False,
                group=None,
            )
        },
    )

    # Direct construction and JSON with `group=None` also populate the implicit
    # timeseries group.
    direct = normalize_lib.ZScoreNormalizer(
        input_feature="num_ts",
        output_feature_name="num_ts_Z_SCORE",
        output_shape=(2,),
        mean=np.zeros((2,), dtype=np.float64),
        std=np.ones((2,), dtype=np.float64),
        is_timeseries=True,
        group=None,
    )
    self.assertEqual(direct.output_schema(), expected_implicit)

    cfg = normalize_lib.GraphNormalizerConfig(
        nodesets={
            "n": normalize_lib.NodeSetNormalizerConfig(
                normalizers=[num_norm, emb_norm]
            )
        },
        edgesets={},
    )
    restored = normalize_lib.GraphNormalizerConfig.from_json(  # pyrefly: ignore[missing-attribute]
        cfg.to_json()  # pyrefly: ignore[missing-attribute]
    )
    test_util.assert_are_equal(self, restored, cfg)
    null_group_json = cfg.to_json().replace('"group": "num_ts"', '"group": null')  # pyrefly: ignore[missing-attribute]
    self.assertIn('"group": null', null_group_json)
    restored_null = (
        normalize_lib.GraphNormalizerConfig.from_json(null_group_json)  # pyrefly: ignore[missing-attribute]
        .nodesets["n"]
        .normalizers[0]
    )
    self.assertEqual(restored_null.output_schema(), expected_implicit)


class AutoNormalizeOneHotTest(absltest.TestCase):

  def test_auto_normalize_categorical_one_hot_bytes_int_bool(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "color": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                        shape=(),
                    ),
                    "shape": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                        shape=(),
                    ),
                    "flag": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                        shape=(),
                    ),
                    "active": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BOOL,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                        shape=(),
                    ),
                }
            )
        },
        edge_sets={},
    )
    stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "nodes": statistics_lib.FeatureSetStatistics(
                features={
                    "color": statistics_lib.FeatureStatistics(
                        count=4,
                        minimum=0.0,
                        maximum=0.0,
                        dictionary={
                            "red": statistics_lib.DictionaryItem(0, 2),
                            "green": statistics_lib.DictionaryItem(1, 1),
                            "blue": statistics_lib.DictionaryItem(2, 1),
                        },
                        quantiles=[],
                    ),
                    "shape": statistics_lib.FeatureStatistics(
                        count=4,
                        minimum=0.0,
                        maximum=0.0,
                        dictionary={
                            "circle": statistics_lib.DictionaryItem(0, 2),
                            "square": statistics_lib.DictionaryItem(1, 2),
                        },
                        quantiles=[],
                    ),
                    "flag": statistics_lib.FeatureStatistics(
                        count=4,
                        minimum=0.0,
                        maximum=1.0,
                        dictionary={},
                        quantiles=[],
                    ),
                    "active": statistics_lib.FeatureStatistics(
                        count=4,
                        minimum=float("nan"),
                        maximum=float("nan"),
                        dictionary={},
                        quantiles=[],
                    ),
                }
            )
        }
    )
    normalizer = normalize_lib.auto_normalize(
        schema,
        stats,
        config=normalize_lib.AutoNormalizeConfig(
            categorical_bytes_to_index=True,
            categorical_one_hot=True,
            categorical_one_hot_drop_if_binary=True,
        ),
    )
    self.assertEqual(
        normalizer.output_schema(),
        schema_lib.GraphSchema(
            node_sets={
                "nodes": schema_lib.NodeSchema(
                    features={
                        "color_INDEX_ONE_HOT": schema_lib.FeatureSchema(
                            format=schema_lib.FeatureFormat.FLOAT_32,
                            semantic=schema_lib.FeatureSemantic.EMBEDDING,
                            shape=(3,),
                        ),
                        "shape_INDEX_ONE_HOT": schema_lib.FeatureSchema(
                            format=schema_lib.FeatureFormat.FLOAT_32,
                            semantic=schema_lib.FeatureSemantic.EMBEDDING,
                            shape=(1,),
                        ),
                        "flag_ONE_HOT": schema_lib.FeatureSchema(
                            format=schema_lib.FeatureFormat.FLOAT_32,
                            semantic=schema_lib.FeatureSemantic.EMBEDDING,
                            shape=(1,),
                        ),
                        "active_ONE_HOT": schema_lib.FeatureSchema(
                            format=schema_lib.FeatureFormat.FLOAT_32,
                            semantic=schema_lib.FeatureSemantic.EMBEDDING,
                            shape=(1,),
                        ),
                    }
                )
            },
            edge_sets={},
        ),
    )
    restored_cfg = normalize_lib.GraphNormalizerConfig.from_json(  # pyrefly: ignore[missing-attribute]
        normalizer.config.to_json()  # pyrefly: ignore[missing-attribute]
    )
    test_util.assert_are_equal(self, restored_cfg, normalizer.config)

    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "nodes": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=3,
                features={
                    "color": np.array([b"red", b"purple", b"blue"]),
                    "shape": np.array([b"circle", b"square", b"circle"]),
                    "flag": np.array([0, 1, 0], dtype=np.int64),
                    "active": np.array([True, False, True], dtype=np.bool_),
                },
            )
        },
        edge_sets={},
    )
    out_np = normalizer.normalize_numpy(graph).node_sets["nodes"].features
    out_tf = (
        normalizer.normalize_tensorflow(tf_io.graph_to_tf_graph(graph))
        .node_sets["nodes"]
        .features
    )
    out_jax = (
        normalizer.normalize_numpy_to_jax(graph).node_sets["nodes"].features
    )
    self.assertEqual(
        set(out_np.keys()),
        {
            "color_INDEX_ONE_HOT",
            "shape_INDEX_ONE_HOT",
            "flag_ONE_HOT",
            "active_ONE_HOT",
        },
    )

    # 3-way color: OOV b"purple" -> [0, 0, 0].
    np.testing.assert_array_equal(
        out_np["color_INDEX_ONE_HOT"],
        np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        out_np["shape_INDEX_ONE_HOT"],
        np.array([[0.0], [1.0], [0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        out_np["flag_ONE_HOT"],
        np.array([[0.0], [1.0], [0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        out_np["active_ONE_HOT"],
        np.array([[1.0], [0.0], [1.0]], dtype=np.float32),
    )
    for key, expected in out_np.items():
      self.assertEqual(expected.dtype, np.float32)
      np.testing.assert_array_equal(out_tf[key].numpy(), expected)
      np.testing.assert_array_equal(np.asarray(out_jax[key]), expected)

    # Committed binary features fail fast if a third class / OOV appears.
    oov_graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "nodes": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=1,
                features={
                    "color": np.array([b"red"]),
                    "shape": np.array([b"triangle"]),
                    "flag": np.array([0], dtype=np.int64),
                    "active": np.array([True], dtype=np.bool_),
                },
            )
        },
        edge_sets={},
    )
    with self.assertRaisesRegex(ValueError, r"must be in \[0, 2\)"):
      normalizer.normalize_numpy(oov_graph)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      normalizer.normalize_tensorflow(tf_io.graph_to_tf_graph(oov_graph))

    # Default drop_if_binary=False encodes 2-class features as vectors so
    # OOV/third-class inputs encode cleanly as all zeros.
    vec_normalizer = normalize_lib.auto_normalize(
        schema,
        stats,
        config=normalize_lib.AutoNormalizeConfig(
            categorical_bytes_to_index=True,
            categorical_one_hot=True,
        ),
    )
    vec_out_np = (
        vec_normalizer.normalize_numpy(oov_graph).node_sets["nodes"].features
    )
    vec_out_tf = (
        vec_normalizer.normalize_tensorflow(tf_io.graph_to_tf_graph(oov_graph))
        .node_sets["nodes"]
        .features
    )
    expected_vec = {
        "color_INDEX_ONE_HOT": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        "shape_INDEX_ONE_HOT": np.array([[0.0, 0.0]], dtype=np.float32),
        "flag_ONE_HOT": np.array([[1.0, 0.0]], dtype=np.float32),
        "active_ONE_HOT": np.array([[0.0, 1.0]], dtype=np.float32),
    }
    self.assertEqual(set(vec_out_np.keys()), set(expected_vec.keys()))
    for key, expected in expected_vec.items():
      np.testing.assert_array_equal(vec_out_np[key], expected)
      np.testing.assert_array_equal(vec_out_tf[key].numpy(), expected)

    # `max_one_hot_categories` is forwarded on both the BYTES path (3-way
    # "color" > 2) and the INTEGER/BOOL path.
    with self.assertRaises(ValueError) as cm_bytes:
      normalize_lib.auto_normalize(
          schema,
          stats,
          config=normalize_lib.AutoNormalizeConfig(
              categorical_bytes_to_index=True,
              categorical_one_hot=True,
              max_one_hot_categories=2,
          ),
      )
    self.assertIn("exceeds max_one_hot_categories=2", str(cm_bytes.exception))
    self.assertIn("disable `categorical_one_hot`", str(cm_bytes.exception))
    self.assertNotIn("DictionaryIndexNormalizer", str(cm_bytes.exception))
    int_only_schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "cat_int": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                        shape=(),
                    )
                }
            )
        },
        edge_sets={},
    )
    int_only_stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "nodes": statistics_lib.FeatureSetStatistics(
                features={
                    "cat_int": statistics_lib.FeatureStatistics(
                        count=5,
                        minimum=0.0,
                        maximum=5.0,
                        dictionary={},
                        quantiles=[],
                    )
                }
            )
        }
    )
    with self.assertRaisesRegex(ValueError, "exceeds max_one_hot_categories=4"):
      normalize_lib.auto_normalize(
          int_only_schema,
          int_only_stats,
          config=normalize_lib.AutoNormalizeConfig(
              categorical_one_hot=True,
              max_one_hot_categories=4,
          ),
      )

    # Empty INTEGER stats without explicit `num_categorical_values` fail loudly.
    empty_int_stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "nodes": statistics_lib.FeatureSetStatistics(
                features={
                    "cat_int": statistics_lib.FeatureStatistics(
                        count=0,
                        minimum=float("nan"),
                        maximum=float("nan"),
                        dictionary={},
                        quantiles=[],
                    )
                }
            )
        }
    )
    with self.assertRaisesRegex(ValueError, "No finite maximum observed"):
      normalize_lib.auto_normalize(
          int_only_schema,
          empty_int_stats,
          config=normalize_lib.AutoNormalizeConfig(categorical_one_hot=True),
      )

  def test_auto_normalize_one_hot_bytes_without_observed_values(self):
    # With no observed value the dictionary is a 1-entry placeholder. Unseen
    # strings must still one-hot encode as zeros, not as that entry.
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "c": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES,
                        semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                        shape=(),
                    ),
                }
            )
        },
        edge_sets={},
    )
    stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            "nodes": statistics_lib.FeatureSetStatistics(
                features={
                    "c": statistics_lib.FeatureStatistics(
                        count=0, dictionary={}
                    ),
                }
            )
        }
    )
    normalizer = normalize_lib.auto_normalize(
        schema,
        stats,
        config=normalize_lib.AutoNormalizeConfig(
            categorical_bytes_to_index=True, categorical_one_hot=True
        ),
    )
    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "nodes": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=3,
                features={"c": np.array([b"red", b"blue", b"SENTINEL"])},
            )
        },
        edge_sets={},
    )
    # Not even the placeholder key of the empty dictionary gets a column.
    expected = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        normalizer.normalize_numpy(graph)
        .node_sets["nodes"]
        .features["c_INDEX_ONE_HOT"],
        expected,
    )
    np.testing.assert_array_equal(
        normalizer.normalize_tensorflow(tf_io.graph_to_tf_graph(graph))
        .node_sets["nodes"]
        .features["c_INDEX_ONE_HOT"]
        .numpy(),
        expected,
    )

  def test_auto_normalize_one_hot_bytes_timeseries_keeps_group(self):
    # The BYTES path goes through `DictionaryIndexNormalizer`, whose index
    # schema drops the timeseries flag; the one-hot output must still be a
    # timeseries in the input's (implicit or explicit) group.
    for group, expected_group in ((None, "events"), ("seq_a", "seq_a")):
      schema = schema_lib.GraphSchema(
          node_sets={
              "nodes": schema_lib.NodeSchema(
                  features={
                      "events": schema_lib.FeatureSchema(
                          format=schema_lib.FeatureFormat.BYTES,
                          semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                          shape=(4,),
                          is_timeseries=True,
                          group=group,
                      ),
                  }
              )
          },
          edge_sets={},
      )
      stats = statistics_lib.GraphFeatureStatistics(
          node_sets={
              "nodes": statistics_lib.FeatureSetStatistics(
                  features={
                      "events": statistics_lib.FeatureStatistics(
                          count=2,
                          dictionary={
                              "click": statistics_lib.DictionaryItem(0, 5),
                              "view": statistics_lib.DictionaryItem(1, 3),
                          },
                      ),
                  }
              )
          }
      )
      normalizer = normalize_lib.auto_normalize(
          schema,
          stats,
          config=normalize_lib.AutoNormalizeConfig(
              categorical_bytes_to_index=True, categorical_one_hot=True
          ),
      )
      out = (
          normalizer.output_schema()
          .node_sets["nodes"]
          .features["events_INDEX_ONE_HOT"]
      )
      self.assertTrue(out.is_timeseries)
      self.assertEqual(out.group, expected_group)
      self.assertEqual(out.shape, (4, 2))


if __name__ == "__main__":
  absltest.main()

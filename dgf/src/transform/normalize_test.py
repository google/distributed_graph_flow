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


class SoftQuantileNormalizerTest(absltest.TestCase):

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

  def test_nan_input(self):
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
    input_values = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    with self.assertRaisesRegex(ValueError, "contains NaN values"):
      normalizer.normalize_numpy(input_values)

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


if __name__ == "__main__":
  absltest.main()

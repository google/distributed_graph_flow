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
    self.assertEqual(
        out_schema.semantic, schema_lib.FeatureSemantic.EMBEDDING
    )
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
      out = normalizer.normalize_numpy(
          np.array(input_values, dtype=np.float32)
      )

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
    input_np = np.array(
        [np.array([0.0, 1.0]), np.array([2.0])], dtype=object
    )
    with self.assertRaisesRegex(
        ValueError, "requires fixed-length feature tensors"
    ):
      normalizer.normalize_numpy(input_np)


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
        normalize_lib.AutoNormalizeConfig(keep_raw_features=set(["f1"])),
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
        config=normalize_lib.AutoNormalizeConfig(timestamp_normalize=True),
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
        graph, seed_timestamps=np.array([500, 500], dtype=np.int64)
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
        graph, seed_timestamps=np.array([500], dtype=np.int64)
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
        config=normalize_lib.AutoNormalizeConfig(timestamp_normalize=True),
    )
    out_schema = normalizer.output_schema()
    self.assertNotIn(
        "created_at_seed_delta_SINUSOID",
        out_schema.node_sets["nodes"].features,
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
    expected = np.array(
        [[400, 300, 200], [200, 150, 100]], dtype=np.int64
    )
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
        self.graph, seed_timestamps=seed_ts
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
        tf_graph, seed_timestamps=seed_ts
    )
    expected_deltas = np.array([400, 300], dtype=np.int64)
    np.testing.assert_array_equal(
        out_graph.node_sets["n"].features["time_seed_delta"].numpy(),
        expected_deltas,
    )

  def test_normalize_numpy_to_jax_with_kwargs(self):
    seed_ts = np.array([500, 500], dtype=np.int64)
    out_graph = self.graph_normalizer.normalize_numpy_to_jax(
        self.graph, seed_timestamps=seed_ts
    )
    expected_deltas = np.array([400, 300], dtype=np.int64)
    np.testing.assert_array_equal(
        np.asarray(out_graph.node_sets["n"].features["time_seed_delta"]),
        expected_deltas,
    )

  def test_unexpected_kwargs_raises(self):
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
    out_graph = normalizer.normalize_numpy(self.graph, seed_timestamps=seed_ts)
    self.assertIn("time_seed_delta_SINUSOID", out_graph.node_sets["n"].features)
    self.assertEqual(
        out_graph.node_sets["n"].features["time_seed_delta_SINUSOID"].shape,
        (2, 4),
    )

    tf_graph = tf_io.graph_to_tf_graph(self.graph)
    tf_seed_ts = tf.constant([500, 500], dtype=tf.int64)
    tf_out_graph = normalizer.normalize_tensorflow(
        tf_graph, seed_timestamps=tf_seed_ts
    )
    self.assertIn(
        "time_seed_delta_SINUSOID", tf_out_graph.node_sets["n"].features
    )
    self.assertEqual(
        tf_out_graph.node_sets["n"].features["time_seed_delta_SINUSOID"].shape,
        (2, 4),
    )

    jax_out_graph = normalizer.normalize_numpy_to_jax(
        self.graph, seed_timestamps=seed_ts
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
            "n": in_memory_graph_lib.InMemoryNodeSet(
                features={}, num_nodes=2
            )
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
    out_graph = edge_normalizer.normalize_numpy(graph, seed_timestamps=seed_ts)
    np.testing.assert_array_equal(
        out_graph.edge_sets["e"].features["time_seed_delta"], expected_deltas
    )

    # TensorFlow
    tf_graph = tf_io.graph_to_tf_graph(graph)
    tf_seed_ts = tf.constant([500, 500], dtype=tf.int64)
    tf_out_graph = edge_normalizer.normalize_tensorflow(
        tf_graph, seed_timestamps=tf_seed_ts
    )
    np.testing.assert_array_equal(
        tf_out_graph.edge_sets["e"].features["time_seed_delta"].numpy(),
        expected_deltas,
    )

    # JAX
    jax_out_graph = edge_normalizer.normalize_numpy_to_jax(
        graph, seed_timestamps=seed_ts
    )
    np.testing.assert_array_equal(
        np.asarray(jax_out_graph.edge_sets["e"].features["time_seed_delta"]),
        expected_deltas,
    )


if __name__ == "__main__":
  absltest.main()


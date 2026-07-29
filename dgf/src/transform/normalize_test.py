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


if __name__ == "__main__":
  absltest.main()

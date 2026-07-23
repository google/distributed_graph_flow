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

"""Tests for temporal schema cache extraction utilities."""

from absl.testing import absltest
from dgf.src.data import schema as schema_lib
from dgf.src.util import temporal
import numpy as np


class TemporalTest(absltest.TestCase):

  def test_extract_timeseries_schema_cache_empty(self):
    schema = schema_lib.GraphSchema(
        node_sets={"nodes": schema_lib.NodeSchema(features={})},
        edge_sets={},
    )
    cache = temporal.extract_timeseries_schema_cache(schema)
    self.assertEqual(cache.node_sets, {"nodes": []})
    self.assertEqual(cache.edge_sets, {})
    self.assertFalse(cache.has_timeseries)

  def test_extract_timeseries_schema_cache_no_timeseries_features(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "nodes": schema_lib.NodeSchema(
                features={
                    "feat1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=False,
                    )
                }
            )
        },
        edge_sets={
            "edges": schema_lib.EdgeSchema(
                source="nodes",
                target="nodes",
                features={
                    "feat2": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=False,
                    )
                },
            )
        },
    )
    cache = temporal.extract_timeseries_schema_cache(schema)
    self.assertEqual(cache.node_sets, {"nodes": []})
    self.assertEqual(cache.edge_sets, {"edges": []})
    self.assertFalse(cache.has_timeseries)

  def test_extract_timeseries_schema_cache_grouped(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "hardware": schema_lib.NodeSchema(
                features={
                    "time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_timeseries=True,
                        is_creation_time=True,
                        group="g1",
                    ),
                    "signal": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        group="g1",
                    ),
                    "non_ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=False,
                    ),
                }
            )
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="hardware",
                target="hardware",
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_timeseries=True,
                        is_creation_time=True,
                        group="e1_g",
                    ),
                    "weight_series": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        group="e1_g",
                    ),
                },
            )
        },
    )

    cache = temporal.extract_timeseries_schema_cache(schema)
    self.assertTrue(cache.has_timeseries)

    self.assertIn("hardware", cache.node_sets)
    self.assertLen(cache.node_sets["hardware"], 1)
    hw_group = cache.node_sets["hardware"][0]
    self.assertEqual(hw_group.timestamp_feature_name, "time")
    self.assertCountEqual(hw_group.feature_names, ["time", "signal"])

    self.assertIn("e1", cache.edge_sets)
    self.assertLen(cache.edge_sets["e1"], 1)
    e1_group = cache.edge_sets["e1"][0]
    self.assertEqual(e1_group.timestamp_feature_name, "ts")
    self.assertCountEqual(e1_group.feature_names, ["ts", "weight_series"])

  def test_feature_schema_helpers(self):
    scalar_ts = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        shape=(None,),
        is_timeseries=True,
    )
    self.assertEqual(temporal.get_timeseries_step_shape(scalar_ts), ())
    self.assertEqual(temporal.with_sequence_length(scalar_ts, 30).shape, (30,))
    self.assertTrue(temporal.with_sequence_length(scalar_ts, 30).is_timeseries)

    vector_ts = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        shape=(None, 8),
        is_timeseries=True,
    )
    self.assertEqual(temporal.get_timeseries_step_shape(vector_ts), (8,))
    self.assertEqual(
        temporal.with_sequence_length(vector_ts, 30).shape, (30, 8)
    )

    unknown_ts = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        shape=None,
        is_timeseries=True,
    )
    with self.assertRaisesRegex(
        ValueError,
        r"Timeseries feature schema must have at least 1 dimension \(sequence"
        r" length at shape\[0\]\), but got shape=None\.",
    ):
      temporal.get_timeseries_step_shape(unknown_ts)

    empty_tuple_ts = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        shape=(),
        is_timeseries=True,
    )
    with self.assertRaisesRegex(
        ValueError,
        r"Timeseries feature schema must have at least 1 dimension \(sequence"
        r" length at shape\[0\]\), but got shape=\(\)\.",
    ):
      temporal.get_timeseries_step_shape(empty_tuple_ts)

    non_ts = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        shape=(10,),
        is_timeseries=False,
    )
    with self.assertRaisesRegex(
        ValueError, r"Feature schema must be a timeseries feature\."
    ):
      temporal.get_timeseries_step_shape(non_ts)

  def test_expand_mask_dims(self):
    # Create a mask with alternating True/False entries.
    mask = np.arange(50).reshape(5, 10) % 2 == 0
    target_2d = np.zeros((5, 10), dtype=np.float32)
    target_4d = np.ones((5, 10, 3, 4), dtype=np.float32) * 42.0

    # Check 2D target (no extra dimensions needed).
    expanded_2d = temporal.expand_mask_dims(mask, target_2d)
    self.assertIs(expanded_2d, mask)
    self.assertEqual(expanded_2d.shape, (5, 10))

    # Check 4D target (two extra dimensions added).
    expanded_4d = temporal.expand_mask_dims(mask, target_4d)
    self.assertEqual(expanded_4d.shape, (5, 10, 1, 1))
    np.testing.assert_array_equal(expanded_4d[:, :, 0, 0], mask)

  def test_timeseries_group(self):
    schemas = {
        "time": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.TIMESTAMP,
            is_timeseries=True,
            is_creation_time=True,
            group="group",
        ),
        "feature": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            is_timeseries=True,
            group="group",
        ),
    }

    self.assertEqual(schemas["feature"].group, "group")

  def test_get_group_creation_time_feature_name(self):
    schemas = {
        "explicit_time": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64,
            semantic=schema_lib.FeatureSemantic.TIMESTAMP,
            is_timeseries=True,
            is_creation_time=True,
            group="explicit_group",
        ),
    }

    self.assertEqual(
        temporal.get_group_creation_time_feature_name(
            "explicit_group", schemas
        ),
        "explicit_time",
    )

  def test_get_edgeset_creation_time_feature_name_heterogeneous(self):
    node_ts = lambda: schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_creation_time=True,
    )
    edge_ts = lambda: schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_creation_time=False,
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(features={"t1": node_ts()}),
            "n2": schema_lib.NodeSchema(features={"t2": node_ts()}),
        },
        edge_sets={},
    )
    es = lambda f: schema_lib.EdgeSchema(
        source="n1", target="n2", features={f: edge_ts()}
    )
    self.assertEqual(
        temporal.get_edgeset_creation_time_feature_name(es("t1"), schema), "t1"
    )
    self.assertEqual(
        temporal.get_edgeset_creation_time_feature_name(es("t2"), schema), "t2"
    )


if __name__ == "__main__":
  absltest.main()

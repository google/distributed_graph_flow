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


class TemporalTest(absltest.TestCase):

  def test_extract_timeseries_schema_cache_empty(self):
    schema = schema_lib.GraphSchema(
        node_sets={"nodes": schema_lib.NodeSchema(features={})},
        edge_sets={},
    )
    cache = temporal.extract_timeseries_schema_cache(schema)
    self.assertEqual(cache.node_sets, {})
    self.assertEqual(cache.edge_sets, {})

  def test_extract_timeseries_schema_cache_grouped(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "hardware": schema_lib.NodeSchema(
                features={
                    "time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_timeseries=True,
                    ),
                    "signal": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        timestamps="time",
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
                    ),
                    "weight_series": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        timestamps="ts",
                    ),
                },
            )
        },
    )

    cache = temporal.extract_timeseries_schema_cache(schema)

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


if __name__ == "__main__":
  absltest.main()

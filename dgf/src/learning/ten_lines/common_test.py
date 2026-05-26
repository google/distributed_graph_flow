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

"""Test for the basic buildable config class."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import schema as schema_lib
from dgf.src.learning.ten_lines import common
from dgf.src.util import gen_test_graph
import numpy as np


class TenLines(parameterized.TestCase):

  def test_parse_temporal_config_auto_detect_success(self):
    _, schema = gen_test_graph.generate_temporal_in_memory_graph(True)
    nodeset_ts, edgeset_ts = common.parse_temporal_config(
        schema=schema, timestamp_features=None, target_nodeset="n1"
    )
    self.assertEqual(nodeset_ts, {"n1": "timestamp"})
    self.assertEqual(edgeset_ts, {"e1": "timestamp"})

  def test_parse_temporal_config_auto_detect_multiple_timestamps_fails(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "ts1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    ),
                    "ts2": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    ),
                }
            )
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1",
                target="n1",
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    ),
                },
            )
        },
    )

    with self.assertRaisesRegex(
        ValueError, "Multiple timestamp features found in nodeset 'n1'"
    ):
      common.parse_temporal_config(
          schema=schema, timestamp_features=None, target_nodeset="n1"
      )

  def test_parse_temporal_config_auto_detect_invalid_format_fails(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    ),
                }
            )
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1",
                target="n1",
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    ),
                },
            )
        },
    )

    with self.assertRaisesRegex(
        ValueError,
        "Timestamp feature 'ts' in nodeset 'n1' must have format INTEGER_64",
    ):
      common.parse_temporal_config(
          schema=schema, timestamp_features=None, target_nodeset="n1"
      )

  def test_parse_temporal_config_manual_dict_success(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    ),
                }
            )
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1",
                target="n1",
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    ),
                },
            )
        },
    )

    nodeset_ts, edgeset_ts = common.parse_temporal_config(
        schema=schema,
        timestamp_features={"n1": "ts", "e1": "ts"},
        target_nodeset="n1",
    )
    self.assertEqual(nodeset_ts, {"n1": "ts"})
    self.assertEqual(edgeset_ts, {"e1": "ts"})

  def test_parse_temporal_config_manual_dict_invalid_name_fails(self):
    schema = schema_lib.GraphSchema(node_sets={}, edge_sets={})
    with self.assertRaisesRegex(
        ValueError,
        "Name 'invalid_name' in temporal configuration is neither a nodeset nor"
        " an edgeset",
    ):
      common.parse_temporal_config(
          schema=schema,
          timestamp_features={"invalid_name": "ts"},
          target_nodeset="n1",
      )

  def test_parse_temporal_config_manual_dict_invalid_feature_fails(self):
    schema = schema_lib.GraphSchema(
        node_sets={"n1": schema_lib.NodeSchema(features={})},
        edge_sets={},
    )
    with self.assertRaisesRegex(
        ValueError, "Feature 'invalid_feat' not found in nodeset 'n1'"
    ):
      common.parse_temporal_config(
          schema=schema,
          timestamp_features={"n1": "invalid_feat"},
          target_nodeset="n1",
      )

  def test_parse_temporal_config_validation_no_target_timestamp_fails(self):
    schema = schema_lib.GraphSchema(
        node_sets={"n1": schema_lib.NodeSchema(features={})},
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1",
                target="n1",
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    ),
                },
            )
        },
    )
    with self.assertRaisesRegex(
        ValueError,
        "The target nodeset 'n1' must have a timestamp feature",
    ):
      common.parse_temporal_config(
          schema=schema, timestamp_features=None, target_nodeset="n1"
      )

  def test_parse_temporal_config_validation_no_edgeset_timestamp_fails(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                    ),
                }
            )
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(
                source="n1",
                target="n1",
                features={},
            )
        },
    )
    with self.assertRaisesRegex(
        ValueError,
        "At least one edgeset must have a timestamp feature",
    ):
      common.parse_temporal_config(
          schema=schema, timestamp_features=None, target_nodeset="n1"
      )

  def test_num_model_weights(self):
    params = {
        "layer1": {
            "w": np.ones((10, 5), dtype=np.float32),
            "b": np.zeros((5,), dtype=np.float32),
        },
        "layer2": {
            "w": np.ones((5, 2), dtype=np.int32),
            "b": np.zeros((2,), dtype=np.int32),
        },
    }
    weights = common.num_model_weights(params)
    self.assertEqual(weights, {"float32": 55, "int32": 12})

  def test_num_model_weights_none(self):
    self.assertEqual(common.num_model_weights(None), {})

  @parameterized.named_parameters(
      (
          "hmp",
          "heterogeneous_message_passing",
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
      ),
      (
          "hmp_alias_hmpnn",
          "hmpnn",
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
      ),
      (
          "hmp_upper",
          "HETEROGENEOUS_MESSAGE_PASSING",
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
      ),
      (
          "hmp_enum",
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
      ),
      (
          "hgat",
          "heterogeneous_graph_attention_network",
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
      ),
      (
          "hgat_alias_hgat",
          "hgat",
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
      ),
      (
          "hgat_alias_han",
          "han",
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
      ),
      (
          "hgat_enum",
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
      ),
  )
  def test_parse_architecture_success(self, input_val, expected):
    self.assertEqual(common.parse_architecture(input_val), expected)

  def test_parse_architecture_invalid_fails(self):
    with self.assertRaisesRegex(ValueError, "Unknown architecture: invalid"):
      common.parse_architecture("invalid")

    with self.assertRaisesRegex(TypeError, "Expected Architecture or str"):
      common.parse_architecture(123)  # pytype: disable=wrong-arg-types


if __name__ == "__main__":
  absltest.main()

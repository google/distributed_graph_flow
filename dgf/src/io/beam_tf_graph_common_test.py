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

import absl.testing.absltest as absltest
from dgf.src.data import distributed_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import beam_tf_graph_common
from dgf.src.io import tf_graph_common
import numpy as np


class BeamTfGraphCommonTest(absltest.TestCase):

  def test_node_to_tf_example(self):
    node = distributed_graph.Node(id=b"n1", features={"f1": np.array([1, 2])})
    schema = {
        "f1": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64, shape=(2,)
        ),
        tf_graph_common.DEFAULT_KEY_ID: schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BYTES, shape=()
        ),
    }
    nodeset_schema = schema_lib.NodeSchema(features=schema)
    example = beam_tf_graph_common.node_to_tf_example(
        node, tf_graph_common.DEFAULT_KEY_ID, nodeset_schema
    )
    self.assertIn("f1", example.features.feature)
    self.assertIn(tf_graph_common.DEFAULT_KEY_ID, example.features.feature)
    self.assertEqual(example.features.feature["f1"].int64_list.value[:], [1, 2])
    self.assertEqual(
        example.features.feature[
            tf_graph_common.DEFAULT_KEY_ID
        ].bytes_list.value[0],
        b"n1",
    )


if __name__ == "__main__":
  absltest.main()

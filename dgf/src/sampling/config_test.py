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
from dgf.src.data import schema as schema_lib
from dgf.src.sampling import config as config_lib


class ConfigTest(parameterized.TestCase):

  def test_simple_sampling_config_to_sampling_config(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(features={}),
            "n2": schema_lib.NodeSchema(features={}),
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(source="n1", target="n2", features={}),
            "e2": schema_lib.EdgeSchema(source="n1", target="n1", features={}),
        },
    )
    simple_sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1", num_hops=2
    )
    sampling_config = config_lib.simple_sampling_config_to_sampling_plan(
        simple_sampling_config, schema
    )
    PlanNode = config_lib.PlanNode
    PlanEdge = config_lib.PlanEdge
    expected_sampling_config = config_lib.SamplingPlan(
        root=PlanNode(
            nodeset="n1",
            children=[
                PlanEdge(
                    edgeset="e1",
                    reversed=False,
                    node=PlanNode(
                        nodeset="n2",
                        children=[
                            PlanEdge(
                                edgeset="e1",
                                reversed=True,
                                node=PlanNode(nodeset="n1", children=[]),
                                hop_width=5,
                            )
                        ],
                    ),
                    hop_width=5,
                ),
                PlanEdge(
                    edgeset="e2",
                    reversed=False,
                    node=PlanNode(
                        nodeset="n1",
                        children=[
                            PlanEdge(
                                edgeset="e1",
                                reversed=False,
                                node=PlanNode(nodeset="n2", children=[]),
                                hop_width=5,
                            ),
                            PlanEdge(
                                edgeset="e2",
                                reversed=False,
                                node=PlanNode(nodeset="n1", children=[]),
                                hop_width=5,
                            ),
                            PlanEdge(
                                edgeset="e2",
                                reversed=True,
                                node=PlanNode(nodeset="n1", children=[]),
                                hop_width=5,
                            ),
                        ],
                    ),
                    hop_width=5,
                ),
                PlanEdge(
                    edgeset="e2",
                    reversed=True,
                    node=PlanNode(
                        nodeset="n1",
                        children=[
                            PlanEdge(
                                edgeset="e1",
                                reversed=False,
                                node=PlanNode(nodeset="n2", children=[]),
                                hop_width=5,
                            ),
                            PlanEdge(
                                edgeset="e2",
                                reversed=False,
                                node=PlanNode(nodeset="n1", children=[]),
                                hop_width=5,
                            ),
                            PlanEdge(
                                edgeset="e2",
                                reversed=True,
                                node=PlanNode(nodeset="n1", children=[]),
                                hop_width=5,
                            ),
                        ],
                    ),
                    hop_width=5,
                ),
            ],
        )
    )
    self.assertEqual(sampling_config, expected_sampling_config)


if __name__ == "__main__":
  absltest.main()

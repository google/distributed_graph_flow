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

  def test_simple_sampling_config_to_sampling_plan_with_max_timeseries_len(
      self,
  ):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(features={}),
        },
        edge_sets={},
    )
    simple_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1", num_hops=0, max_timeseries_len=10
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        simple_config, schema
    )
    self.assertEqual(plan.max_timeseries_len, 10)

  def test_default_max_timeseries_len(self):
    simple_config = config_lib.SimpleSamplingConfig(seed_nodeset="n1")
    self.assertEqual(simple_config.max_timeseries_len, 32)
    plan = config_lib.SamplingPlan(root=config_lib.PlanNode(nodeset="n1"))
    self.assertEqual(plan.max_timeseries_len, 32)

  @parameterized.parameters(True, False)
  def test_simple_sampling_config_to_sampling_plan_propagate_timestamp(
      self, propagate
  ):
    schema = schema_lib.GraphSchema(
        node_sets={"n1": schema_lib.NodeSchema(features={})},
        edge_sets={},
    )
    simple_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1",
        num_hops=0,
        propagate_timestamp_to_edges=propagate,
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        simple_config, schema
    )
    self.assertEqual(plan.propagate_timestamp_to_edges, propagate)

  def test_edgesets_in_plan(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(features={}),
            "n2": schema_lib.NodeSchema(features={}),
        },
        edge_sets={
            "e1": schema_lib.EdgeSchema(source="n1", target="n2", features={}),
            # "e2" is not reachable from the seed nodeset "n1".
            "e2": schema_lib.EdgeSchema(source="n3", target="n3", features={}),
        },
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        config_lib.SimpleSamplingConfig(
            seed_nodeset="n1", num_hops=2, reverse=True
        ),
        schema,
    )
    self.assertEqual(config_lib.edgesets_in_plan(plan), {"e1"})

  def test_edgesets_in_plan_without_edges(self):
    plan = config_lib.SamplingPlan(root=config_lib.PlanNode(nodeset="n1"))
    self.assertEmpty(config_lib.edgesets_in_plan(plan))

  def _two_hop_plan(self, **kwargs) -> config_lib.SamplingPlan:
    schema = schema_lib.GraphSchema(
        node_sets={"n1": schema_lib.NodeSchema(features={})},
        edge_sets={
            "e1": schema_lib.EdgeSchema(source="n1", target="n1", features={})
        },
    )
    return config_lib.simple_sampling_config_to_sampling_plan(
        config_lib.SimpleSamplingConfig(
            seed_nodeset="n1", num_hops=2, reverse=False, **kwargs
        ),
        schema,
    )

  def test_temporal_strategy_propagated_and_serialized(self):
    plan = self._two_hop_plan(
        temporal_sampling=True,
        temporal_strategy=config_lib.TemporalStrategy.LAST,
    )
    edge = plan.root.children[0]
    self.assertEqual(edge.temporal_strategy, config_lib.TemporalStrategy.LAST)
    self.assertEqual(
        edge.node.children[0].temporal_strategy,
        config_lib.TemporalStrategy.LAST,
    )
    self.assertIn('"temporal_strategy": "LAST"', plan.to_json())  # pyrefly: ignore[missing-attribute]
    self.assertEqual(
        config_lib.SamplingPlan.from_json(plan.to_json()),  # pyrefly: ignore[missing-attribute]
        plan,
    )

  def test_check_temporal_strategies(self):
    config_lib.check_temporal_strategies(self._two_hop_plan(), set())
    config_lib.check_temporal_strategies(
        self._two_hop_plan(
            temporal_sampling=True,
            temporal_strategy=config_lib.TemporalStrategy.LAST,
        ),
        {"e1"},
    )
    with self.assertRaisesRegex(ValueError, "requires temporal_sampling=True"):
      config_lib.check_temporal_strategies(
          self._two_hop_plan(
              temporal_strategy=config_lib.TemporalStrategy.LAST
          ),
          {"e1"},
      )
    with self.assertRaisesRegex(ValueError, "does not have a creation time"):
      config_lib.check_temporal_strategies(
          self._two_hop_plan(
              temporal_sampling=True,
              temporal_strategy=config_lib.TemporalStrategy.LAST,
          ),
          set(),
      )

  def test_check_supported_options(self):
    config_lib.check_supported_options(
        self._two_hop_plan(multi_visit=False),
        None,
        "my sampler",
        support_non_with_replacement=True,
        support_non_multi_visit=True,
    )
    with self.assertRaisesRegex(
        NotImplementedError,
        "The my sampler does not support the following options:"
        " with_replacement=True, temporal_sampling=True,"
        " temporal_strategy=LAST",
    ):
      config_lib.check_supported_options(
          self._two_hop_plan(
              with_replacement=True,
              temporal_sampling=True,
              temporal_strategy=config_lib.TemporalStrategy.LAST,
          ),
          None,
          "my sampler",
          support_non_with_replacement=True,
          support_multi_visit=True,
          support_non_multi_visit=True,
      )
    with self.assertRaisesRegex(NotImplementedError, "multi_visit=True"):
      config_lib.check_supported_options(
          self._two_hop_plan(),
          None,
          "my sampler",
          support_non_with_replacement=True,
          support_non_multi_visit=True,
      )
    with self.assertRaisesRegex(NotImplementedError, "multi_visit=False"):
      config_lib.check_supported_options(
          self._two_hop_plan(multi_visit=False),
          None,
          "my sampler",
          support_non_with_replacement=True,
          support_multi_visit=True,
      )
    with self.assertRaisesRegex(NotImplementedError, "with_replacement=False"):
      config_lib.check_supported_options(
          self._two_hop_plan(),
          None,
          "my sampler",
          support_multi_visit=True,
      )

  def test_check_supported_options_timeseries(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "ts": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        shape=(None,),
                        is_timeseries=True,
                        group="g",
                    )
                }
            )
        },
        edge_sets={},
    )
    plan = self._two_hop_plan()
    config_lib.check_supported_options(
        plan,
        schema,
        "my sampler",
        support_non_with_replacement=True,
        support_multi_visit=True,
        support_timeseries=True,
    )
    with self.assertRaisesRegex(NotImplementedError, "timeseries features"):
      config_lib.check_supported_options(
          plan,
          schema,
          "my sampler",
          support_non_with_replacement=True,
          support_multi_visit=True,
      )


if __name__ == "__main__":
  absltest.main()

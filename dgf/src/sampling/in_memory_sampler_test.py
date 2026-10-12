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

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import padding as padding_lib
from dgf.src.data import schema as schema_lib
from dgf.src.sampling import _in_memory_sampler_ext
from dgf.src.sampling import config as config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.sampling import temporal as sampling_temporal_lib
from dgf.src.transform import merge as merge_lib
from dgf.src.transform import temporal as temporal_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np

InMemoryNodeSet = in_memory_graph_lib.InMemoryNodeSet
InMemoryEdgeSet = in_memory_graph_lib.InMemoryEdgeSet
InMemoryGraph = in_memory_graph_lib.InMemoryGraph


class InMemorySamplerTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64
                    )
                }
            ),
            "n2": schema_lib.NodeSchema(
                features={
                    "f2": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.BYTES
                    )
                }
            ),
        },
        edge_sets={
            "e12": schema_lib.EdgeSchema(source="n1", target="n2"),
            "e11": schema_lib.EdgeSchema(source="n1", target="n1"),
            "e22": schema_lib.EdgeSchema(source="n2", target="n2"),
        },
    )

    cls.graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                features={"f1": np.array([10, 11])}, num_nodes=2
            ),
            "n2": in_memory_graph_lib.InMemoryNodeSet(
                features={"f2": np.array([20.0, 21.0, 22.0])}, num_nodes=3
            ),
        },
        edge_sets={
            "e11": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[1], [0]])
            ),
            "e12": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0, 0, 0], [0, 1, 2]])
            ),
            "e22": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0], [2]])
            ),
        },
    )

  def test_create_sampler(self):
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1", num_hops=2
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, self.schema
    )
    sampler = in_memory_sampler_lib.create_sampler(
        self.graph, plan, self.schema, batch_size=5
    )
    self.assertEqual(
        str(sampler),
        """\
Sampler(
  plan=SamplingPlan(root=
Node(nodeset_idx=0, children=[
  Edge(edgeset_idx=0, reversed=0, hop_width=5, node=
    Node(nodeset_idx=0, children=[
      Edge(edgeset_idx=0, reversed=0, hop_width=5, node=
        Node(nodeset_idx=0, children=[])),
      Edge(edgeset_idx=0, reversed=1, hop_width=5, node=
        Node(nodeset_idx=0, children=[])),
      Edge(edgeset_idx=1, reversed=0, hop_width=5, node=
        Node(nodeset_idx=1, children=[]))
    ])),
  Edge(edgeset_idx=0, reversed=1, hop_width=5, node=
    Node(nodeset_idx=0, children=[
      Edge(edgeset_idx=0, reversed=0, hop_width=5, node=
        Node(nodeset_idx=0, children=[])),
      Edge(edgeset_idx=0, reversed=1, hop_width=5, node=
        Node(nodeset_idx=0, children=[])),
      Edge(edgeset_idx=1, reversed=0, hop_width=5, node=
        Node(nodeset_idx=1, children=[]))
    ])),
  Edge(edgeset_idx=1, reversed=0, hop_width=5, node=
    Node(nodeset_idx=1, children=[
      Edge(edgeset_idx=1, reversed=1, hop_width=5, node=
        Node(nodeset_idx=0, children=[])),
      Edge(edgeset_idx=2, reversed=0, hop_width=5, node=
        Node(nodeset_idx=1, children=[])),
      Edge(edgeset_idx=2, reversed=1, hop_width=5, node=
        Node(nodeset_idx=1, children=[]))
    ]))
]),
  with_replacement=0
),
  nodeset_index={n1: 0, n2: 1},
  edgeset_index={e11: 0, e12: 1, e22: 2},
  nodesets=[{idx=0, num_nodes=2}, {idx=1, num_nodes=3}]
)""",
    )

  @parameterized.parameters(
      (
          np.array([[], []]),
          False,
          2,
          2,
          "AdjacencyIndex(source_blocks=(3)[0, 0, 0], target_node_idxs=(0)[])",
      ),
      (
          np.array([[0, 0, 0], [0, 1, 2]]),
          False,
          2,
          3,
          (
              "AdjacencyIndex(source_blocks=(3)[0, 3, 3],"
              " target_node_idxs=(3)[0, 1, 2])"
          ),
      ),
      (
          np.array([[0, 0, 0], [0, 1, 2]]),
          True,
          2,
          3,
          (
              "AdjacencyIndex(source_blocks=(4)[0, 1, 2, 3],"
              " target_node_idxs=(3)[0, 0, 0])"
          ),
      ),
      (
          np.array([[0, 0, 1, 1], [0, 1, 0, 1]]),
          False,
          2,
          2,
          (
              "AdjacencyIndex(source_blocks=(3)[0, 2, 4],"
              " target_node_idxs=(4)[0, 1, 0, 1])"
          ),
      ),
  )
  def test_adjacency(
      self,
      adjacency,
      reversed_edge,
      num_source_nodes,
      num_target_nodes,
      expected_str,
  ):

    adjacency_index = (
        in_memory_sampler_lib._in_memory_sampler_ext.BuildAdjacencyIndex(
            py_adjacency=adjacency,
            reversed=reversed_edge,
            num_source_nodes=num_source_nodes,
            num_target_nodes=num_target_nodes,
        )
    )
    self.assertEqual(
        str(adjacency_index),
        expected_str,
    )

  def test_sample_0_hop(self):
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1", num_hops=0, hop_width=2
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, self.schema
    )
    sampler = in_memory_sampler_lib.create_sampler(
        self.graph,
        plan,
        self.schema,
        return_features=False,
        return_node_idxs=True,
        batch_size=5,
    )
    sample = sampler.sample(0)
    self.assertEqual(sample.node_sets.keys(), {"n1", "n2"})
    self.assertEqual(sample.edge_sets.keys(), {"e11", "e12", "e22"})
    self.assertTrue(
        np.array_equal(sample.node_sets["n1"].features["#idx"], np.array([0]))
    )

  def test_sample_1_hop(self):
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1", num_hops=1, hop_width=2
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, self.schema
    )
    sampler = in_memory_sampler_lib.create_sampler(
        self.graph,
        plan,
        self.schema,
        return_features=False,
        return_node_idxs=True,
        batch_size=5,
    )
    sample = sampler.sample(0)
    self.assertEqual(sample.node_sets.keys(), {"n1", "n2"})
    self.assertEqual(sample.edge_sets.keys(), {"e11", "e12", "e22"})
    self.assertTrue(
        np.array_equal(
            sample.node_sets["n1"].features["#idx"], np.array([0, 1])
        )
    )
    test_util.assert_unique_subset_of_length(
        self, sample.node_sets["n2"].features["#idx"].tolist(), [0, 1, 2], 2
    )

    self.assertTrue(
        np.array_equal(sample.edge_sets["e11"].adjacency, np.array([[1], [0]]))
    )
    self.assertTrue(
        np.array_equal(
            sample.edge_sets["e12"].adjacency, np.array([[0, 0], [0, 1]])
        )
    )

  def test_sample_1_hop_deterministic(self):
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1", num_hops=1, hop_width=2
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, self.schema
    )
    sampler = in_memory_sampler_lib.create_sampler(
        self.graph,
        plan,
        self.schema,
        debug_sampling=True,
        return_features=False,
        return_node_idxs=True,
        batch_size=5,
    )
    sample = sampler.sample(0)
    array = np.array
    test_util.assert_are_equal(
        self,
        sample,
        in_memory_graph_lib.InMemoryGraph(
            node_sets={
                "n1": InMemoryNodeSet(
                    features={"#idx": array([0, 1])},
                    num_nodes=2,
                ),
                "n2": InMemoryNodeSet(
                    features={"#idx": array([0, 1])},
                    num_nodes=2,
                ),
            },
            edge_sets={
                "e11": InMemoryEdgeSet(adjacency=array([[1], [0]])),
                "e12": InMemoryEdgeSet(
                    adjacency=array([[0, 0], [0, 1]]),
                ),
                "e22": InMemoryEdgeSet(
                    adjacency=array([[], []], dtype=np.int64), features={}
                ),
            },
        ),
    )

  def test_sample_2_hop_deterministic(self):
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1", num_hops=2, hop_width=2
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, self.schema
    )
    sampler = in_memory_sampler_lib.create_sampler(
        self.graph,
        plan,
        self.schema,
        debug_sampling=True,
        return_features=False,
        return_node_idxs=True,
        batch_size=5,
    )
    sample = sampler.sample(0)
    array = np.array
    test_util.assert_are_equal(
        self,
        sample,
        in_memory_graph_lib.InMemoryGraph(
            node_sets={
                "n1": InMemoryNodeSet(
                    features={"#idx": array([0, 1])},
                    num_nodes=2,
                ),
                "n2": InMemoryNodeSet(
                    features={"#idx": array([0, 2, 1])},
                    num_nodes=3,
                ),
            },
            edge_sets={
                "e11": InMemoryEdgeSet(
                    adjacency=array([[1], [0]]),
                ),
                "e12": InMemoryEdgeSet(
                    adjacency=array([[0, 0], [0, 2]]),
                ),
                "e22": InMemoryEdgeSet(
                    adjacency=array([[0], [1]]),
                ),
            },
        ),
    )

  def test_sample_2_hop_deterministic_with_replacement(self):
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1", num_hops=2, hop_width=2, with_replacement=True
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, self.schema
    )
    sampler = in_memory_sampler_lib.create_sampler(
        self.graph,
        plan,
        self.schema,
        debug_sampling=True,
        return_features=False,
        return_node_idxs=True,
        batch_size=5,
    )
    sample = sampler.sample(0)
    array = np.array
    test_util.assert_are_equal(
        self,
        sample,
        in_memory_graph_lib.InMemoryGraph(
            node_sets={
                "n1": InMemoryNodeSet(
                    features={"#idx": array([0, 1, 0, 0, 0])},
                    num_nodes=5,
                ),
                "n2": InMemoryNodeSet(
                    features={"#idx": array([0, 2, 1])},
                    num_nodes=3,
                ),
            },
            edge_sets={
                "e11": InMemoryEdgeSet(
                    adjacency=array([[1, 1], [0, 2]]),
                ),
                "e12": InMemoryEdgeSet(
                    adjacency=array([[0, 0, 3, 4], [0, 2, 0, 2]]),
                ),
                "e22": InMemoryEdgeSet(
                    adjacency=array([[0], [1]]),
                ),
            },
        ),
    )

  def test_sample_with_edge_masking(self):
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1", num_hops=1, hop_width=2
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, self.schema
    )
    sampler = in_memory_sampler_lib.create_sampler(
        self.graph,
        plan,
        self.schema,
        debug_sampling=True,
        return_features=False,
        return_node_idxs=True,
        batch_size=5,
        edgeset_to_mask="e12",
    )
    # Without masking, sampling from node 0 would give neighbors 0 and 1 in n2.
    # We mask edge 1 (which connects n1:0 to n2:1).
    # So it should give neighbors 0 and 2 in n2.
    sample = sampler.sample(0, masked_edge_idxs=1)

    self.assertEqual(sample.node_sets.keys(), {"n1", "n2"})
    self.assertEqual(sample.edge_sets.keys(), {"e11", "e12", "e22"})

    # Check n2 nodes. Should be 0 and 2 (original indices).
    n2_idxs = sample.node_sets["n2"].features["#idx"]
    np.testing.assert_array_equal(n2_idxs, np.array([0, 2]))

  def test_create_sampler_temporal_and_masking_error(self):
    graph, schema = gen_test_graph.generate_temporal_in_memory_graph(
        include_e2=True
    )
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1",
        num_hops=1,
        hop_width=2,
        temporal_sampling=True,
    )
    with self.assertRaisesRegex(
        ValueError,
        "Temporal filtering and edge masking cannot be used at the same time.",
    ):
      in_memory_sampler_lib.create_sampler(
          graph,
          sampling_config,
          schema,
          edgeset_to_mask="e1",
          debug_sampling=True,
          batch_size=5,
      )

  def test_sample_with_edge_masking_reverse(self):
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n2", num_hops=1, hop_width=2, reverse=True
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, self.schema
    )
    sampler = in_memory_sampler_lib.create_sampler(
        self.graph,
        plan,
        self.schema,
        debug_sampling=True,
        return_features=False,
        return_node_idxs=True,
        batch_size=5,
        edgeset_to_mask="e12",
    )
    # In e12, edge 0 is (n1:0, n2:0).
    # Sampling from n2:0 in reverse would give neighbor 0 in n1.
    # We mask edge 0.
    # So it should NOT give neighbor 0 in n1.
    sample = sampler.sample(0, masked_edge_idxs=0)

    self.assertEqual(sample.node_sets.keys(), {"n1", "n2"})

    # Check n1 nodes. Should be empty (no neighbors found after masking).
    n1_idxs = sample.node_sets["n1"].features["#idx"]
    np.testing.assert_array_equal(n1_idxs, np.array([], dtype=np.int64))

  def test_sample_multiple(self):
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1", num_hops=1, hop_width=2
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, self.schema
    )
    sampler = in_memory_sampler_lib.create_sampler(
        self.graph,
        plan,
        self.schema,
        return_features=False,
        return_node_idxs=True,
        debug_sampling=True,
        batch_size=5,
    )
    samples = sampler.sample([0, 1])
    self.assertLen(samples, 2)
    array = np.array
    test_util.assert_are_equal(
        self,
        samples[0],
        in_memory_graph_lib.InMemoryGraph(
            node_sets={
                "n1": InMemoryNodeSet(
                    features={"#idx": array([0, 1])},
                    num_nodes=2,
                ),
                "n2": InMemoryNodeSet(
                    features={"#idx": array([0, 1])},
                    num_nodes=2,
                ),
            },
            edge_sets={
                "e11": InMemoryEdgeSet(adjacency=array([[1], [0]])),
                "e12": InMemoryEdgeSet(
                    adjacency=array([[0, 0], [0, 1]]),
                ),
                "e22": InMemoryEdgeSet(
                    adjacency=array([[], []], dtype=np.int64), features={}
                ),
            },
        ),
    )
    test_util.assert_are_equal(
        self,
        samples[1],
        in_memory_graph_lib.InMemoryGraph(
            node_sets={
                "n1": InMemoryNodeSet(
                    features={"#idx": array([1, 0])},
                    num_nodes=2,
                ),
                "n2": InMemoryNodeSet(
                    features={"#idx": array([], dtype=np.int64)},
                    num_nodes=0,
                ),
            },
            edge_sets={
                "e11": InMemoryEdgeSet(adjacency=array([[0], [1]])),
                "e12": InMemoryEdgeSet(
                    adjacency=array([[], []], dtype=np.int64),
                ),
                "e22": InMemoryEdgeSet(
                    adjacency=array([[], []], dtype=np.int64), features={}
                ),
            },
        ),
    )

  @parameterized.parameters(
      (True, True),
      (True, False),
      (False, True),
      (False, False),
  )
  def test_sample_select_output(self, return_features, return_node_idxs):
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1", num_hops=0, hop_width=2
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, self.schema
    )
    sampler = in_memory_sampler_lib.create_sampler(
        self.graph,
        plan,
        self.schema,
        return_features=return_features,
        return_node_idxs=return_node_idxs,
        batch_size=5,
    )
    sample = sampler.sample(0)
    n1_features = sample.node_sets["n1"].features

    if return_features:
      self.assertIn("f1", n1_features)
      self.assertTrue(np.array_equal(n1_features["f1"], np.array([10])))
    else:
      self.assertNotIn("f1", n1_features)

    if return_node_idxs:
      self.assertIn("#idx", n1_features)
      self.assertTrue(np.array_equal(n1_features["#idx"], np.array([0])))
    else:
      self.assertNotIn("#idx", n1_features)

  def test_is_deterministic(self):
    """Tests that the sampler is deterministic when "seed" if provided."""
    for seed in [1234]:
      for batch_size in [1, 2]:
        for seed_idxs in [[0, 1], [1]]:
          for num_hops in [0, 1, 2]:
            for hop_width in [1, 2]:

              ground_truth_samples = None

              # Check that all the 100 samples are equalty the same.
              for i in range(100):
                plan = config_lib.simple_sampling_config_to_sampling_plan(
                    config_lib.SimpleSamplingConfig(
                        seed_nodeset="n1",
                        num_hops=num_hops,
                        hop_width=hop_width,
                    ),
                    self.schema,
                )
                sampler = in_memory_sampler_lib.create_sampler(
                    self.graph,
                    plan,
                    self.schema,
                    return_node_idxs=True,
                    batch_size=batch_size,
                    seed=seed,
                )
                samples = sampler.sample(seed_idxs)
                if i == 0:
                  ground_truth_samples = samples
                else:
                  test_util.assert_are_equal(
                      self, ground_truth_samples, samples
                  )

  @parameterized.parameters(
      dict(
          depth=0,
          seed_nodeset="n1",
          seed_idxs=[0, 1],
          expected_subgraph=in_memory_graph_lib.InMemoryGraph(
              node_sets={
                  "n1": InMemoryNodeSet(
                      num_nodes=2,
                      features={"#idx": np.array([0, 1], dtype=np.int64)},
                  ),
                  "n2": InMemoryNodeSet(
                      num_nodes=0,
                      features={"#idx": np.array([], dtype=np.int64)},
                  ),
              },
              edge_sets={
                  "e11": InMemoryEdgeSet(
                      adjacency=np.array([[], []], dtype=np.int64),
                  ),
                  "e12": InMemoryEdgeSet(
                      adjacency=np.array([[], []], dtype=np.int64),
                  ),
                  "e22": InMemoryEdgeSet(
                      adjacency=np.array([[], []], dtype=np.int64),
                  ),
              },
          ),
      ),
      dict(
          depth=1,
          seed_nodeset="n1",
          seed_idxs=[0, 1],
          expected_subgraph=in_memory_graph_lib.InMemoryGraph(
              node_sets={
                  "n1": InMemoryNodeSet(
                      num_nodes=2,
                      features={"#idx": np.array([0, 1], dtype=np.int64)},
                  ),
                  "n2": InMemoryNodeSet(
                      num_nodes=3,
                      features={"#idx": np.array([0, 1, 2], dtype=np.int64)},
                  ),
              },
              edge_sets={
                  "e11": InMemoryEdgeSet(
                      adjacency=np.array([[1], [0]], dtype=np.int64),
                  ),
                  "e12": InMemoryEdgeSet(
                      adjacency=np.array(
                          [[0, 0, 0], [0, 1, 2]], dtype=np.int64
                      ),
                  ),
                  "e22": InMemoryEdgeSet(
                      adjacency=np.array([[], []], dtype=np.int64),
                  ),
              },
          ),
      ),
      dict(
          depth=2,
          seed_nodeset="n1",
          seed_idxs=[0, 1],
          expected_subgraph=in_memory_graph_lib.InMemoryGraph(
              node_sets={
                  "n1": InMemoryNodeSet(
                      num_nodes=2,
                      features={"#idx": np.array([0, 1], dtype=np.int64)},
                  ),
                  "n2": InMemoryNodeSet(
                      num_nodes=3,
                      features={"#idx": np.array([0, 2, 1], dtype=np.int64)},
                  ),
              },
              edge_sets={
                  "e11": InMemoryEdgeSet(
                      adjacency=np.array([[1], [0]], dtype=np.int64),
                  ),
                  "e12": InMemoryEdgeSet(
                      adjacency=np.array(
                          [[0, 0, 0], [0, 1, 2]], dtype=np.int64
                      ),
                  ),
                  "e22": InMemoryEdgeSet(
                      adjacency=np.array([[0], [1]], dtype=np.int64),
                  ),
              },
          ),
      ),
      dict(
          depth=2,
          seed_nodeset="n2",
          seed_idxs=[2, 0],
          expected_subgraph=in_memory_graph_lib.InMemoryGraph(
              node_sets={
                  "n1": InMemoryNodeSet(
                      num_nodes=2,
                      features={"#idx": np.array([0, 1], dtype=np.int64)},
                  ),
                  "n2": InMemoryNodeSet(
                      num_nodes=3,
                      features={"#idx": np.array([2, 0, 1], dtype=np.int64)},
                  ),
              },
              edge_sets={
                  "e11": InMemoryEdgeSet(
                      adjacency=np.array([[1], [0]], dtype=np.int64),
                      features={},
                  ),
                  "e12": InMemoryEdgeSet(
                      adjacency=np.array(
                          [[0, 0, 0], [0, 1, 2]], dtype=np.int64
                      ),
                      features={},
                  ),
                  "e22": InMemoryEdgeSet(
                      adjacency=np.array([[1], [0]], dtype=np.int64),
                      features={},
                  ),
              },
          ),
      ),
  )
  def test_subgraph(
      self,
      depth: int,
      seed_nodeset: str,
      seed_idxs: list[int],
      expected_subgraph: in_memory_graph_lib.InMemoryGraph,
  ):
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset=seed_nodeset,
        num_hops=depth,
        hop_width=100,  # Large enough to cover all neighbors.
    )
    sampler = in_memory_sampler_lib.create_sampler(
        self.graph,
        sampling_config,
        schema=self.schema,
        return_features=False,
        return_node_idxs=True,
        batch_size=2,
    )
    subgraph = sampler.subgraph(seed_idxs)
    test_util.assert_are_equal(
        self,
        subgraph,
        expected_subgraph,
    )

  def test_multi_subgraphs(self):
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1",
        num_hops=1,
        hop_width=100,
    )
    sampler = in_memory_sampler_lib.create_sampler(
        self.graph,
        sampling_config,
        schema=self.schema,
        return_features=False,
        return_node_idxs=True,
        batch_size=2,
    )
    subgraphs = sampler.multisubgraph([0, 1])
    self.assertLen(subgraphs, 2)

    expected_subgraph_0 = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": InMemoryNodeSet(
                num_nodes=2,
                features={"#idx": np.array([0, 1], dtype=np.int64)},
            ),
            "n2": InMemoryNodeSet(
                num_nodes=3,
                features={"#idx": np.array([0, 1, 2], dtype=np.int64)},
            ),
        },
        edge_sets={
            "e11": InMemoryEdgeSet(
                adjacency=np.array([[1], [0]], dtype=np.int64),
            ),
            "e12": InMemoryEdgeSet(
                adjacency=np.array([[0, 0, 0], [0, 1, 2]], dtype=np.int64),
            ),
            "e22": InMemoryEdgeSet(
                adjacency=np.array([[], []], dtype=np.int64),
            ),
        },
    )

    expected_subgraph_1 = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": InMemoryNodeSet(
                num_nodes=2,
                features={"#idx": np.array([1, 0], dtype=np.int64)},
            ),
            "n2": InMemoryNodeSet(
                num_nodes=0,
                features={"#idx": np.array([], dtype=np.int64)},
            ),
        },
        edge_sets={
            "e11": InMemoryEdgeSet(
                adjacency=np.array([[0], [1]], dtype=np.int64),
            ),
            "e12": InMemoryEdgeSet(
                adjacency=np.array([[], []], dtype=np.int64),
            ),
            "e22": InMemoryEdgeSet(
                adjacency=np.array([[], []], dtype=np.int64),
            ),
        },
    )

    test_util.assert_are_equal(self, subgraphs[0], expected_subgraph_0)
    test_util.assert_are_equal(self, subgraphs[1], expected_subgraph_1)

  def test_ego_sampling_matches_subgraph(self):
    """Tests that k-hop ego graph extracted with the sample() method is the same as k-hop subgraph for the same seed node."""
    # [[0 0 0 1 1 1 2 2 3 4 4 4 4 5 6 6 6 7],
    # [1 4 6 0 2 3 1 4 1 0 2 5 6 4 0 4 7 6]]
    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": InMemoryNodeSet(
                features={"#idx": np.array([0, 1, 2, 3, 4, 5, 6, 7])},
                num_nodes=8,
            ),
        },
        edge_sets={
            "e11": InMemoryEdgeSet(
                adjacency=np.array([
                    [0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5, 6, 6, 6, 7],
                    [1, 4, 6, 0, 2, 3, 1, 4, 1, 0, 2, 5, 6, 4, 0, 4, 7, 6],
                ])
            ),
        },
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64
                    )
                }
            ),
        },
        edge_sets={
            "e11": schema_lib.EdgeSchema(source="n1", target="n1"),
        },
    )
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1",
        num_hops=2,
        hop_width=20,  # Does not matter for subgraph(), matters for sample().
    )
    sampler = in_memory_sampler_lib.create_sampler(
        graph,
        sampling_config,
        schema,
        return_features=False,
        return_node_idxs=True,
        batch_size=1,
    )
    sample = sampler.sample(0)
    subgraph = sampler.subgraph([0])
    test_util.assert_are_equal(self, sample, subgraph)

  def test_subgraph_line(self):
    """Tests subgraph extraction with a circular graph and a linear plan.

    The graph contains 2 nodesets (each containing one node) and two edges sets
    (each containing one edge):
        Edset e1: n1 -> n1
        Edset e2: n1 -> n2

    Plan:
      n1 -(e1)-> n1 -(e1)-> n1 -(e2)-> n2.

    The sampling starts on n1, and we want to make sure n2 is extracted.
    """

    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": InMemoryNodeSet(num_nodes=1),
            "n2": InMemoryNodeSet(num_nodes=1),
        },
        edge_sets={
            "e11": InMemoryEdgeSet(adjacency=np.array([[0], [0]])),
            "e12": InMemoryEdgeSet(adjacency=np.array([[0], [0]])),
        },
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(),
            "n2": schema_lib.NodeSchema(),
        },
        edge_sets={
            "e11": schema_lib.EdgeSchema(source="n1", target="n1"),
            "e12": schema_lib.EdgeSchema(source="n1", target="n2"),
        },
    )
    sampling_config = config_lib.SamplingPlan(
        root=config_lib.PlanNode(
            nodeset="n1",
            children=[
                config_lib.PlanEdge(
                    edgeset="e11",
                    reversed=False,
                    hop_width=5,
                    node=config_lib.PlanNode(
                        nodeset="n1",
                        children=[
                            config_lib.PlanEdge(
                                edgeset="e11",
                                reversed=False,
                                hop_width=5,
                                node=config_lib.PlanNode(
                                    nodeset="n1",
                                    children=[
                                        config_lib.PlanEdge(
                                            edgeset="e12",
                                            reversed=False,
                                            hop_width=5,
                                            node=config_lib.PlanNode(
                                                nodeset="n2"
                                            ),
                                        )
                                    ],
                                ),
                            )
                        ],
                    ),
                )
            ],
        )
    )
    sampler = in_memory_sampler_lib.create_sampler(
        graph,
        sampling_config,
        schema,
        return_features=False,
        return_node_idxs=True,
        batch_size=1,
    )
    subgraph = sampler.subgraph([0])
    test_util.assert_are_equal(
        self,
        subgraph,
        in_memory_graph_lib.InMemoryGraph(
            node_sets={
                "n1": InMemoryNodeSet(
                    num_nodes=1, features={"#idx": np.array([0])}
                ),
                "n2": InMemoryNodeSet(
                    num_nodes=1, features={"#idx": np.array([0])}
                ),
            },
            edge_sets={
                "e11": InMemoryEdgeSet(adjacency=np.array([[0], [0]])),
                "e12": InMemoryEdgeSet(adjacency=np.array([[0], [0]])),
            },
        ),
    )

  # The graph has 4 "n1" nodes with creation times 10, 20, 30 and 40. The "e1"
  # edges have a creation time: 0 -> 1 (15), 0 -> 2 (25), 1 -> 3 (35). The "e2"
  # edges don't: 2 -> 0 and 3 -> 1. Unless disabled, the sampler gives to the
  # "e2" edges the creation time of their connected nodes: 30 and 40.
  @parameterized.named_parameters(
      dict(
          testcase_name="only_the_seed_node_at_10",
          seed_timestamp=10,
          num_hops=2,
          propagate_timestamp_to_edges=True,
          remove_node_creation_time=False,
          expected_node_idxs=[0],
          expected_e1=[[], []],
          expected_e2=[[], []],
      ),
      dict(
          testcase_name="one_e1_edge_at_20",
          seed_timestamp=20,
          num_hops=2,
          propagate_timestamp_to_edges=True,
          remove_node_creation_time=False,
          expected_node_idxs=[0, 1],
          expected_e1=[[0], [1]],
          expected_e2=[[], []],
      ),
      dict(
          # The propagated "e2" edge 2 -> 0 (30) is filtered out.
          testcase_name="propagated_e2_edge_filtered_at_25",
          seed_timestamp=25,
          num_hops=2,
          propagate_timestamp_to_edges=True,
          remove_node_creation_time=False,
          expected_node_idxs=[0, 1, 2],
          expected_e1=[[0, 0], [1, 2]],
          expected_e2=[[], []],
      ),
      dict(
          testcase_name="all_the_nodes_at_40",
          seed_timestamp=40,
          num_hops=2,
          propagate_timestamp_to_edges=True,
          remove_node_creation_time=False,
          expected_node_idxs=[0, 1, 3, 2],
          expected_e1=[[0, 0, 1], [1, 3, 2]],
          expected_e2=[[3], [0]],
      ),
      dict(
          # Without propagation, the "e2" edges are never filtered.
          testcase_name="without_propagation_e2_edge_kept_at_25",
          seed_timestamp=25,
          num_hops=2,
          propagate_timestamp_to_edges=False,
          remove_node_creation_time=False,
          expected_node_idxs=[0, 1, 2],
          expected_e1=[[0, 0], [1, 2]],
          expected_e2=[[2], [0]],
      ),
      dict(
          # Without node creation time, there is nothing to propagate and the
          # "e2" edges are not filtered.
          testcase_name="without_node_creation_time_e2_edge_kept_at_25",
          seed_timestamp=25,
          num_hops=2,
          propagate_timestamp_to_edges=True,
          remove_node_creation_time=True,
          expected_node_idxs=[0, 1, 2],
          expected_e1=[[0, 0], [1, 2]],
          expected_e2=[[2], [0]],
      ),
      dict(
          # The creation time of the "e1" edge 0 -> 1 (15) is used instead of
          # the creation time of its connected nodes (i.e. max(10, 20) = 20),
          # which would have filtered out the edge.
          testcase_name="edge_creation_time_not_overridden_at_15",
          seed_timestamp=15,
          num_hops=1,
          propagate_timestamp_to_edges=True,
          remove_node_creation_time=False,
          expected_node_idxs=[0, 1],
          expected_e1=[[0], [1]],
          expected_e2=[[], []],
      ),
  )
  def test_temporal_sampling(
      self,
      seed_timestamp: int,
      num_hops: int,
      propagate_timestamp_to_edges: bool,
      remove_node_creation_time: bool,
      expected_node_idxs: list[int],
      expected_e1: list[list[int]],
      expected_e2: list[list[int]],
  ):
    graph, schema = gen_test_graph.generate_temporal_in_memory_graph(
        include_e2=True
    )
    if remove_node_creation_time:
      del graph.node_sets["n1"].features["timestamp"]
      del schema.node_sets["n1"].features["timestamp"]

    config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1",
        num_hops=num_hops,
        hop_width=2,
        reverse=False,
        temporal_sampling=True,
        propagate_timestamp_to_edges=propagate_timestamp_to_edges,
    )
    sampler = self._create_temporal_sampler(graph, schema, config)
    sample = sampler.sample([0], seed_timestamps=[seed_timestamp])[0]

    node_idxs = np.array(expected_node_idxs, dtype=np.int64)
    node_features = graph.node_sets["n1"].features
    expected_graph = InMemoryGraph(
        node_sets={
            "n1": InMemoryNodeSet(
                num_nodes=len(expected_node_idxs),
                features={
                    "#idx": node_idxs,
                    # The features of the sampled nodes.
                    **{
                        name: values[node_idxs]
                        for name, values in node_features.items()
                    },
                },
            )
        },
        edge_sets={
            "e1": InMemoryEdgeSet(
                adjacency=np.array(expected_e1, dtype=np.int64), features={}
            ),
            "e2": InMemoryEdgeSet(
                adjacency=np.array(expected_e2, dtype=np.int64), features={}
            ),
        },
    )
    test_util.assert_are_equal(self, expected_graph, sample)

    # The propagated creation times are internal to the sampler.
    for features in [
        graph.edge_sets["e2"].features,
        schema.edge_sets["e2"].features,
        sample.edge_sets["e2"].features,
    ]:
      self.assertNotIn(
          sampling_temporal_lib.PROPAGATED_CREATION_TIME_FEATURE, features
      )

    if not propagate_timestamp_to_edges or remove_node_creation_time:
      return

    # Propagating with "dgf.transform.propagate_timestamp_to_edges" instead of
    # with the sampler gives the same samples.
    manual_graph, manual_schema = temporal_lib.propagate_timestamp_to_edges(
        graph, schema, target_edgesets=["e2"], target_feature="ts"
    )
    manual_sampler = self._create_temporal_sampler(
        manual_graph,
        manual_schema,
        dataclasses.replace(config, propagate_timestamp_to_edges=False),
    )
    seeds = [0, 1, 2, 3]
    timestamps = [seed_timestamp] * len(seeds)
    test_util.assert_are_equal(
        self,
        sampler.sample(seeds, seed_timestamps=timestamps),
        manual_sampler.sample(seeds, seed_timestamps=timestamps),
    )

  def _create_temporal_sampler(
      self,
      graph: in_memory_graph_lib.InMemoryGraph,
      schema: schema_lib.GraphSchema,
      config: config_lib.SimpleSamplingConfig,
  ) -> in_memory_sampler_lib.Sampler:
    return in_memory_sampler_lib.create_sampler(
        graph,
        config,
        schema,
        return_features=True,
        return_node_idxs=True,
        batch_size=5,
        debug_sampling=True,
    )

  def test_temporal_sampling_missing_seed_timestamp(self):
    graph, schema = gen_test_graph.generate_temporal_in_memory_graph(
        include_e2=True
    )

    sampler = in_memory_sampler_lib.create_sampler(
        graph,
        config_lib.SimpleSamplingConfig(
            seed_nodeset="n1",
            num_hops=2,
            hop_width=2,
            reverse=False,
            temporal_sampling=True,
        ),
        schema,
        batch_size=5,
    )

    with self.assertRaisesRegex(
        ValueError,
        "Cannot use SampleRandomUniform when timestamps are available",
    ):
      _ = sampler.sample([0])

  def test_temporal_sampling_no_temporal_edgesets_succeeds(self):
    graph = gen_test_graph.generate_in_memory_graph()
    schema = gen_test_graph.generate_schema()
    sampler = in_memory_sampler_lib.create_sampler(
        graph,
        config_lib.SimpleSamplingConfig(
            seed_nodeset="n1",
            num_hops=2,
            hop_width=2,
            reverse=False,
        ),
        schema,
        batch_size=5,
    )

    res = sampler.sample([0], seed_timestamps=[5])
    self.assertLen(res, 1)

  def test_sample_numpy_array_input(self):
    graph, schema = gen_test_graph.generate_temporal_in_memory_graph(
        include_e2=True
    )
    sampler = in_memory_sampler_lib.create_sampler(
        graph,
        config_lib.SimpleSamplingConfig(
            seed_nodeset="n1",
            num_hops=2,
            hop_width=2,
            reverse=False,
            temporal_sampling=True,
        ),
        schema,
        return_features=True,
        return_node_idxs=True,
        batch_size=5,
        debug_sampling=True,
    )
    samples = sampler.sample(
        np.array([0, 1]),
        seed_timestamps=np.array([15, 25]),
    )
    self.assertLen(samples, 2)

  def test_edgeset_name_to_idx(self):
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1", num_hops=1
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, self.schema
    )
    sampler = in_memory_sampler_lib.create_sampler(
        self.graph, plan, self.schema, batch_size=5
    )
    idx = sampler._cc_sampler.EdgesetNameToEdgesetIdx("e12")
    self.assertEqual(idx, 1)

    with self.assertRaisesRegex(ValueError, "Edgeset 'invalid' not found"):
      sampler._cc_sampler.EdgesetNameToEdgesetIdx("invalid")

  def test_random_walk_negative_sampling(self):
    graph, schema = gen_test_graph.generate_recommender_like_in_memory_graph()
    target_edgeset = "e2"

    # Plan must index both forward and backward to allow walking.
    plan = config_lib.SamplingPlan(
        root=config_lib.PlanNode(
            nodeset="n1",
            children=[
                config_lib.PlanEdge(
                    edgeset=target_edgeset,
                    reversed=False,
                    hop_width=1,
                    node=config_lib.PlanNode(
                        nodeset="n2",
                        children=[
                            config_lib.PlanEdge(
                                edgeset=target_edgeset,
                                reversed=True,
                                hop_width=1,
                                node=config_lib.PlanNode(
                                    nodeset="n1",
                                    children=[
                                        config_lib.PlanEdge(
                                            edgeset=target_edgeset,
                                            reversed=False,
                                            hop_width=1,
                                            node=config_lib.PlanNode(
                                                nodeset="n2"
                                            ),
                                        )
                                    ],
                                ),
                            )
                        ],
                    ),
                )
            ],
        )
    )
    sampler = in_memory_sampler_lib.create_sampler(
        graph, plan, schema, batch_size=1, seed=1234
    )

    edgeset_idx = sampler._cc_sampler.EdgesetNameToEdgesetIdx(target_edgeset)

    # Seed n1:0 has direct neighbors {0, 1}. Reachable via walk is {2}.
    # Request 3 negatives with high walk budget to avoid fallback.
    negatives = sampler._cc_sampler.RandomWalkNegativeSampling(
        seed_node_idxs=np.array([0], dtype=np.int64),
        target_edgeset_idx=edgeset_idx,
        num_walks=500,
        num_negatives_per_seed=3,
    )

    self.assertEqual(negatives.shape, (1, 3))
    sampled = negatives[0]
    self.assertEqual(sampled[0], 2)
    for n in sampled[1:]:
      self.assertIn(n, [0, 1, 2, 3, 4, 5])

  def test_random_walk_negative_sampling_fallback(self):
    graph, schema = gen_test_graph.generate_recommender_like_in_memory_graph()
    target_edgeset = "e2"

    plan = config_lib.SamplingPlan(
        root=config_lib.PlanNode(
            nodeset="n1",
            children=[
                config_lib.PlanEdge(
                    edgeset=target_edgeset,
                    reversed=False,
                    hop_width=1,
                    node=config_lib.PlanNode(
                        nodeset="n2",
                        children=[
                            config_lib.PlanEdge(
                                edgeset=target_edgeset,
                                reversed=True,
                                hop_width=1,
                                node=config_lib.PlanNode(nodeset="n1"),
                            )
                        ],
                    ),
                )
            ],
        )
    )
    sampler = in_memory_sampler_lib.create_sampler(
        graph, plan, schema, batch_size=1, seed=1234
    )
    edgeset_idx = sampler._cc_sampler.EdgesetNameToEdgesetIdx(target_edgeset)

    # Request 3 negatives with 0 walks to force immediate fallback.
    # Simplified fallback samples with replacement from all target nodes,
    # ignoring exclusions.
    negatives = sampler._cc_sampler.RandomWalkNegativeSampling(
        seed_node_idxs=np.array([0], dtype=np.int64),
        target_edgeset_idx=edgeset_idx,
        num_walks=0,  # Force fallback
        num_negatives_per_seed=3,
    )

    self.assertEqual(negatives.shape, (1, 3))
    sampled = negatives[0]
    for n in sampled:
      self.assertIn(n, [0, 1, 2, 3, 4, 5])

  def _create_causal_timeseries_test_graph(self):
    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "alerts": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "#id": np.array([0, 1], dtype=np.int64),
                    "#creation_time": np.array([30, 60], dtype=np.int64),
                },
            ),
            "hardware": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=1,
                features={
                    "#id": np.array([0], dtype=np.int64),
                    "time": np.array(
                        [
                            np.array([10, 20, 30, 40, 50, 60, 70]),
                        ],
                        dtype=np.object_,
                    ),
                    "signal": np.array(
                        [
                            np.array(
                                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                                dtype=np.float32,
                            ),
                        ],
                        dtype=np.object_,
                    ),
                },
            ),
        },
        edge_sets={
            "hardware_to_alert": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0, 0], [0, 1]], dtype=np.int64),
                features={},
            )
        },
    )
    schema = schema_lib.GraphSchema(
        node_sets={
            "alerts": schema_lib.NodeSchema(
                features={
                    "#id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                    ),
                    "#creation_time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_creation_time=True,
                    ),
                }
            ),
            "hardware": schema_lib.NodeSchema(
                features={
                    "#id": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                    ),
                    "time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_timeseries=True,
                        is_creation_time=True,
                        group="time",
                    ),
                    "signal": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        group="time",
                    ),
                }
            ),
        },
        edge_sets={
            "hardware_to_alert": schema_lib.EdgeSchema(
                source="hardware", target="alerts"
            )
        },
    )
    # Propagate timestamps to edges so temporal sampling works.
    return temporal_lib.propagate_timestamp_to_edges(graph, schema)

  def test_sample_merged_with_timeseries_matches_graph_merger(self):
    graph, schema = self._create_causal_timeseries_test_graph()
    plan = config_lib.SimpleSamplingConfig(
        seed_nodeset="alerts",
        num_hops=1,
        hop_width=10,
        reverse=True,
        temporal_sampling=True,
        max_timeseries_len=3,
    )

    def create_sampler():
      return in_memory_sampler_lib.create_sampler(
          graph,
          plan,
          schema,
          batch_size=2,
          return_features=True,
          slice_timeseries_by_seed=True,
          seed=1234,
      )

    seed_node_idxs = np.array([0, 1], dtype=np.int64)
    seed_timestamps = np.array([30, 60], dtype=np.int64)
    expected = merge_lib.GraphMerger(schema)(
        create_sampler().sample(seed_node_idxs, seed_timestamps=seed_timestamps)
    )
    actual = create_sampler().sample_merged(
        seed_node_idxs, seed_timestamps=seed_timestamps
    )
    _assert_merged_equal(self, actual, expected)

  def test_causal_timeseries_filtering_in_sample(self):
    graph, schema = self._create_causal_timeseries_test_graph()
    plan = config_lib.SimpleSamplingConfig(
        seed_nodeset="alerts",
        num_hops=1,
        hop_width=10,
        reverse=True,
        temporal_sampling=True,
        max_timeseries_len=3,
    )
    sampler = in_memory_sampler_lib.create_sampler(
        graph,
        plan,
        schema,
        batch_size=2,
        return_features=True,
        slice_timeseries_by_seed=True,
    )
    samples = sampler.sample(
        [0, 1], seed_timestamps=np.array([30, 60], dtype=np.int64)
    )
    # Verify exact seed creation timestamps remain intact on seed nodes.
    self.assertEqual(
        samples[0].node_sets["alerts"].features["#creation_time"][0], 30
    )
    self.assertEqual(
        samples[1].node_sets["alerts"].features["#creation_time"][0], 60
    )
    # Sample 0 (alert 0 at target_timestamp=30): <= 30 is [10, 20, 30].
    np.testing.assert_array_equal(
        samples[0].node_sets["hardware"].features["time"][0], [10, 20, 30]
    )
    np.testing.assert_array_equal(
        samples[0].node_sets["hardware"].features["signal"][0], [1.0, 2.0, 3.0]
    )
    # Sample 1 (alert 1 at target_timestamp=60): <= 60 is [10, ..., 60].
    # max_timeseries_len=3 -> [40, 50, 60].
    np.testing.assert_array_equal(
        samples[1].node_sets["hardware"].features["time"][0], [40, 50, 60]
    )
    np.testing.assert_array_equal(
        samples[1].node_sets["hardware"].features["signal"][0], [4.0, 5.0, 6.0]
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="default_from_plan",
          temporal_sampling=True,
          slice_timeseries_by_seed=None,
          expected_slice=True,
          expected_time=[10, 20, 30],
          expected_signal=[1.0, 2.0, 3.0],
      ),
      dict(
          testcase_name="override_plan_to_false",
          temporal_sampling=True,
          slice_timeseries_by_seed=False,
          expected_slice=False,
          expected_time=[50, 60, 70],
          expected_signal=[5.0, 6.0, 7.0],
      ),
      dict(
          testcase_name="override_plan_to_true",
          temporal_sampling=False,
          slice_timeseries_by_seed=True,
          expected_slice=True,
          expected_time=None,
          expected_signal=None,
      ),
  )
  def test_slice_timeseries_by_seed(
      self,
      temporal_sampling,
      slice_timeseries_by_seed,
      expected_slice,
      expected_time,
      expected_signal,
  ):
    graph, schema = self._create_causal_timeseries_test_graph()
    plan = config_lib.SimpleSamplingConfig(
        seed_nodeset="alerts",
        num_hops=1,
        hop_width=10,
        reverse=True,
        temporal_sampling=temporal_sampling,
        max_timeseries_len=3,
    )
    kwargs = {}
    if slice_timeseries_by_seed is not None:
      kwargs["slice_timeseries_by_seed"] = slice_timeseries_by_seed

    sampler = in_memory_sampler_lib.create_sampler(
        graph,
        plan,
        schema,
        batch_size=2,
        return_features=True,
        **kwargs,
    )

    self.assertEqual(sampler._slice_timeseries_by_seed, expected_slice)
    if expected_time is not None:
      samples = sampler.sample(
          [0], seed_timestamps=np.array([30], dtype=np.int64)
      )
      np.testing.assert_array_equal(
          samples[0].node_sets["hardware"].features["time"][0], expected_time
      )
      np.testing.assert_array_equal(
          samples[0].node_sets["hardware"].features["signal"][0],
          expected_signal,
      )

  def test_slice_timeseries_by_seed_validation_errors(self):
    empty_graph = in_memory_graph_lib.InMemoryGraph(node_sets={}, edge_sets={})

    # 2. Invalid max_timeseries_len (<= 0)
    ts_schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_timeseries=True,
                    ),
                }
            ),
        },
        edge_sets={},
    )
    with self.assertRaisesRegex(
        ValueError,
        "max_timeseries_len must be positive",
    ):
      in_memory_sampler_lib.Sampler(
          cc_sampler=None,
          full_graph=empty_graph,
          return_features=True,
          return_node_idxs=False,
          schema=ts_schema,
          slice_timeseries_by_seed=True,
          max_timeseries_len=0,
          has_temporal_edgesets=False,
      )

    # 3. Missing seed_timestamps when slice_timeseries_by_seed=True and schema
    # contains is_timeseries=True features
    sampler_missing_timestamps = in_memory_sampler_lib.Sampler(
        cc_sampler=None,
        full_graph=empty_graph,
        return_features=True,
        return_node_idxs=False,
        schema=ts_schema,
        slice_timeseries_by_seed=True,
        max_timeseries_len=3,
        has_temporal_edgesets=False,
    )
    with self.assertRaisesRegex(
        AssertionError,
        "`seed_timestamps` must be provided when"
        " `slice_timeseries_by_seed=True` and the schema contains"
        " `is_timeseries=True` features.",
    ):
      sampler_missing_timestamps.sample(seed_node_idxs=0, seed_timestamps=None)

  def test_timeseries_subgraph_and_multisubgraph_clipping(self):
    graph, schema = self._create_causal_timeseries_test_graph()
    plan = config_lib.SimpleSamplingConfig(
        seed_nodeset="alerts",
        num_hops=1,
        hop_width=10,
        max_timeseries_len=3,
    )
    sampler = in_memory_sampler_lib.create_sampler(
        graph,
        plan,
        schema,
        batch_size=2,
        return_features=True,
    )

    # Test subgraph
    sub = sampler.subgraph([0])
    # Original hardware: [10, 20, 30, 40, 50, 60, 70] -> last 3: [50, 60, 70]
    np.testing.assert_array_equal(
        sub.node_sets["hardware"].features["time"][0], [50, 60, 70]
    )
    np.testing.assert_array_equal(
        sub.node_sets["hardware"].features["signal"][0], [5.0, 6.0, 7.0]
    )

    # Test multisubgraph
    multisubs = sampler.multisubgraph([0, 1])
    self.assertLen(multisubs, 2)
    np.testing.assert_array_equal(
        multisubs[0].node_sets["hardware"].features["time"][0], [50, 60, 70]
    )
    np.testing.assert_array_equal(
        multisubs[1].node_sets["hardware"].features["time"][0], [50, 60, 70]
    )

  def test_timeseries_unseeded_sample_clipping(self):
    graph, schema = self._create_causal_timeseries_test_graph()
    plan = config_lib.SimpleSamplingConfig(
        seed_nodeset="alerts",
        num_hops=1,
        hop_width=10,
        temporal_sampling=False,
        max_timeseries_len=4,
    )
    sampler = in_memory_sampler_lib.create_sampler(
        graph,
        plan,
        schema,
        batch_size=1,
        return_features=True,
        debug_sampling=True,
    )
    sample = sampler.sample(0)
    # Original hardware: 7 items -> last 4: [40, 50, 60, 70]
    np.testing.assert_array_equal(
        sample.node_sets["hardware"].features["time"][0], [40, 50, 60, 70]
    )
    np.testing.assert_array_equal(
        sample.node_sets["hardware"].features["signal"][0],
        [4.0, 5.0, 6.0, 7.0],
    )

  def test_add_features_to_samples_fused_direct(self):
    graph, schema = self._create_causal_timeseries_test_graph()
    # Create sample graph containing only #idx
    sample = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "hardware": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=1,
                features={"#idx": np.array([0], dtype=np.int64)},
            ),
            "alerts": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=1,
                features={"#idx": np.array([0], dtype=np.int64)},
            ),
        },
        edge_sets={
            "hardware_to_alert": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.empty((2, 0), dtype=np.int64),
                features={},
            ),
        },
    )
    sampler = in_memory_sampler_lib.Sampler(
        cc_sampler=None,
        full_graph=graph,
        schema=schema,
        return_features=True,
        return_node_idxs=False,
        slice_timeseries_by_seed=True,
        max_timeseries_len=2,
        has_temporal_edgesets=False,
    )
    sampler._add_finalize_graphs(
        [sample], seed_timestamps=np.array([30], dtype=np.int64)
    )
    self.assertNotIn("#idx", sample.node_sets["hardware"].features)
    # Original times <= 30: [10, 20, 30], max_len 2 -> [20, 30]
    np.testing.assert_array_equal(
        sample.node_sets["hardware"].features["time"][0], [20, 30]
    )
    np.testing.assert_array_equal(
        sample.node_sets["hardware"].features["signal"][0], [2.0, 3.0]
    )


def _padding(
    num_nodes: dict[str, int], num_edges: dict[str, int]
) -> padding_lib.Padding:
  return padding_lib.Padding(
      node_sets={
          k: padding_lib.NodeSetPadding(num_nodes=v)
          for k, v in num_nodes.items()
      },
      edge_sets={
          k: padding_lib.EdgeSetPadding(num_edges=v)
          for k, v in num_edges.items()
      },
  )


def _sampler_output_schema(
    schema: schema_lib.GraphSchema,
    return_features: bool,
    return_node_idxs: bool,
) -> schema_lib.GraphSchema:
  """Schema of the graphs returned by `sample_merged`."""
  node_sets = {}
  for name, node_set in schema.node_sets.items():
    features = dict(node_set.features) if return_features else {}
    if return_node_idxs:
      features["#idx"] = schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64
      )
    node_sets[name] = dataclasses.replace(node_set, features=features)
  return dataclasses.replace(schema, node_sets=node_sets)


def _generate_large_graph(
    num_nodes: int, num_edges: int
) -> tuple[InMemoryGraph, schema_lib.GraphSchema]:
  """Random graph with features of various dtypes."""
  rng = np.random.default_rng(0)
  graph = InMemoryGraph(
      node_sets={
          "a": InMemoryNodeSet(
              num_nodes=num_nodes,
              features={
                  "f32": rng.random((num_nodes, 8), dtype=np.float32),
                  "i32": rng.integers(0, 100, num_nodes, dtype=np.int32),
                  "i64": rng.integers(0, 100, (num_nodes, 1), dtype=np.int64),
                  "f16": rng.random(num_nodes).astype(np.float16),
                  "bool": rng.random(num_nodes) > 0.5,
                  "bytes": np.array(
                      [f"b{i % 997}".encode() for i in range(num_nodes)]
                  ),
                  "str": np.array([f"s{i % 13}" for i in range(num_nodes)]),
                  "ragged": np.array(
                      [np.arange(i % 3) for i in range(num_nodes)],
                      dtype=np.object_,
                  ),
              },
          ),
      },
      edge_sets={
          "aa": InMemoryEdgeSet(
              adjacency=rng.integers(0, num_nodes, (2, num_edges)),
          ),
      },
  )
  schema = schema_lib.GraphSchema(
      node_sets={
          "a": schema_lib.NodeSchema(
              features={
                  "f32": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32, shape=(8,)
                  ),
                  "i32": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_32
                  ),
                  "i64": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64, shape=(1,)
                  ),
                  "f16": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32
                  ),
                  "bool": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_32
                  ),
                  "bytes": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES
                  ),
                  "str": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES
                  ),
                  "ragged": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      shape=(None,),
                  ),
              }
          )
      },
      edge_sets={"aa": schema_lib.EdgeSchema(source="a", target="a")},
  )
  return graph, schema


def _assert_same_dtypes_and_keys(test: absltest.TestCase, a, b):
  """Checks that the arrays have the same dtypes and dicts the same keys."""
  if isinstance(a, np.ndarray):
    test.assertEqual(a.dtype, b.dtype)
  elif dataclasses.is_dataclass(a):
    for field in dataclasses.fields(a):
      _assert_same_dtypes_and_keys(
          test, getattr(a, field.name), getattr(b, field.name)
      )
  elif isinstance(a, dict):
    test.assertEqual(list(a), list(b))
    for key in a:
      _assert_same_dtypes_and_keys(test, a[key], b[key])
  elif isinstance(a, (list, tuple)):
    test.assertEqual(len(a), len(b))
    for x, y in zip(a, b):
      _assert_same_dtypes_and_keys(test, x, y)


def _assert_merged_equal(test: absltest.TestCase, actual, expected):
  """Checks that two merged graphs (+ offsets) are equal."""
  test_util.assert_are_equal(test, actual, expected)
  _assert_same_dtypes_and_keys(test, actual, expected)


class SampleMergedTest(parameterized.TestCase):
  """Tests that `sample_merged` is equivalent to `sample` + `GraphMerger`."""

  def _create_sampler(
      self,
      graph: InMemoryGraph,
      schema: schema_lib.GraphSchema,
      seed_nodeset: str,
      num_hops: int = 2,
      hop_width: int = 2,
      with_replacement: bool = False,
      temporal_sampling: bool = False,
      **kwargs,
  ) -> in_memory_sampler_lib.Sampler:
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        config_lib.SimpleSamplingConfig(
            seed_nodeset=seed_nodeset,
            num_hops=num_hops,
            hop_width=hop_width,
            with_replacement=with_replacement,
            temporal_sampling=temporal_sampling,
        ),
        schema,
    )
    return in_memory_sampler_lib.create_sampler(
        graph, plan, schema, seed=1234, batch_size=4, **kwargs
    )

  def _assert_matches_graph_merger(
      self,
      create_sampler,
      seed_node_idxs: np.ndarray,
      schema: schema_lib.GraphSchema,
      padding: padding_lib.Padding | None,
      return_features: bool,
      return_node_idxs: bool,
      **sample_kwargs,
  ):
    """Compares `sample_merged` with `sample` + `GraphMerger`.

    Both are computed with identically seeded samplers.

    Args:
      create_sampler: Creates a sampler. Should be deterministic.
      seed_node_idxs: The seed nodes.
      schema: Schema of the sampler.
      padding: Optional padding.
      return_features: Sampler option.
      return_node_idxs: Sampler option.
      **sample_kwargs: Extra arguments to the sampling methods.
    """
    reference_sampler = create_sampler(
        return_features=return_features, return_node_idxs=return_node_idxs
    )
    expected = merge_lib.GraphMerger(
        _sampler_output_schema(schema, return_features, return_node_idxs),
        padding=padding,
    )(reference_sampler.sample(seed_node_idxs, **sample_kwargs))

    sampler = create_sampler(
        return_features=return_features,
        return_node_idxs=return_node_idxs,
        padding=padding,
    )
    actual = sampler.sample_merged(seed_node_idxs, **sample_kwargs)
    _assert_merged_equal(self, actual, expected)

  @parameterized.product(
      use_padding=[False, True],
      output=[(True, False), (False, True), (True, True)],
      with_replacement=[False, True],
  )
  def test_matches_graph_merger(
      self,
      use_padding: bool,
      output: tuple[bool, bool],
      with_replacement: bool,
  ):
    graph = gen_test_graph.generate_in_memory_graph()
    schema = gen_test_graph.generate_schema()
    padding = (
        _padding({"n1": 100, "n2": 100}, {"e1": 100, "e2": 100})
        if use_padding
        else None
    )
    return_features, return_node_idxs = output
    self._assert_matches_graph_merger(
        lambda **kwargs: self._create_sampler(
            graph,
            schema,
            seed_nodeset="n1",
            with_replacement=with_replacement,
            **kwargs,
        ),
        np.array([0, 1, 1, 0], dtype=np.int64),
        schema,
        padding,
        return_features=return_features,
        return_node_idxs=return_node_idxs,
    )

  def test_subset_schema_and_partial_padding(self):
    graph = gen_test_graph.generate_in_memory_graph()
    schema = gen_test_graph.generate_schema()
    # Only keep some of the features, and only pad "n1" / "e1".
    subset_schema = dataclasses.replace(
        schema,
        node_sets={
            "n1": dataclasses.replace(
                schema.node_sets["n1"],
                features={"f2": schema.node_sets["n1"].features["f2"]},
            ),
            "n2": dataclasses.replace(schema.node_sets["n2"], features={}),
        },
    )
    self._assert_matches_graph_merger(
        lambda **kwargs: self._create_sampler(
            graph, subset_schema, seed_nodeset="n1", **kwargs
        ),
        np.array([1, 0], dtype=np.int64),
        subset_schema,
        _padding({"n1": 50}, {"e1": 60}),
        return_features=True,
        return_node_idxs=False,
    )

  @parameterized.product(num_threads=[0, 1, 4], return_node_idxs=[False, True])
  def test_large_graph_all_dtypes(
      self, num_threads: int, return_node_idxs: bool
  ):
    graph, schema = _generate_large_graph(num_nodes=3000, num_edges=20000)
    num_seeds = 32
    num_hops = 3
    hop_width = 5
    # Upper bound of the number of nodes per sample (the edges are sampled in
    # both directions).
    max_nodes = sum((2 * hop_width) ** i for i in range(num_hops + 1))
    padding = _padding(
        {"a": num_seeds * max_nodes + 1}, {"aa": num_seeds * max_nodes}
    )
    seed_node_idxs = np.random.default_rng(1).integers(
        0, 3000, num_seeds, dtype=np.int64
    )
    for use_padding in [False, True]:
      self._assert_matches_graph_merger(
          lambda **kwargs: self._create_sampler(
              graph,
              schema,
              seed_nodeset="a",
              num_hops=num_hops,
              hop_width=hop_width,
              num_threads=num_threads,
              **kwargs,
          ),
          seed_node_idxs,
          schema,
          padding if use_padding else None,
          return_features=True,
          return_node_idxs=return_node_idxs,
      )

  def test_temporal_sampling(self):
    graph, schema = gen_test_graph.generate_temporal_in_memory_graph(
        include_e2=True
    )
    self._assert_matches_graph_merger(
        lambda **kwargs: self._create_sampler(
            graph,
            schema,
            seed_nodeset="n1",
            temporal_sampling=True,
            **kwargs,
        ),
        np.array([0, 1, 2, 3], dtype=np.int64),
        schema,
        _padding({"n1": 100}, {"e1": 100, "e2": 100}),
        return_features=True,
        return_node_idxs=False,
        seed_timestamps=np.array([20, 30, 40, 50], dtype=np.int64),
    )

  def test_edge_masking(self):
    graph = gen_test_graph.generate_in_memory_graph()
    schema = gen_test_graph.generate_schema()
    self._assert_matches_graph_merger(
        lambda **kwargs: self._create_sampler(
            graph, schema, seed_nodeset="n1", edgeset_to_mask="e2", **kwargs
        ),
        np.array([0, 1], dtype=np.int64),
        schema,
        None,
        return_features=True,
        return_node_idxs=False,
        masked_edge_idxs=np.array([1, -1], dtype=np.int64),
    )

  def test_insufficient_padding_raises(self):
    graph = gen_test_graph.generate_in_memory_graph()
    schema = gen_test_graph.generate_schema()
    sampler = self._create_sampler(
        graph,
        schema,
        seed_nodeset="n1",
        padding=_padding({"n1": 2, "n2": 100}, {"e1": 100, "e2": 100}),
    )
    with self.assertRaisesRegex(
        merge_lib.InsufficientPaddingError,
        "Padding for node set 'n1' is insufficient",
    ):
      sampler.sample_merged([0, 1])

  def test_padding_missing_sentinel_raises(self):
    graph = gen_test_graph.generate_in_memory_graph()
    schema = gen_test_graph.generate_schema()
    sampler = self._create_sampler(
        graph,
        schema,
        seed_nodeset="n1",
        padding=_padding({"n1": 100}, {"e2": 100}),
    )
    with self.assertRaisesRegex(ValueError, "requires sentinel nodes"):
      sampler.sample_merged([0, 1])

  def test_unknown_padding_set_raises(self):
    graph = gen_test_graph.generate_in_memory_graph()
    schema = gen_test_graph.generate_schema()
    sampler = self._create_sampler(graph, schema, seed_nodeset="n1")
    with self.assertRaisesRegex(ValueError, "unknown node sets"):
      sampler.with_padding(_padding({"unknown": 10}, {}))

  @parameterized.parameters(
      (False, True),
      (True, False),
      (True, True),
  )
  def test_overflow_matches_merge_sub_batches(
      self, skip_overflow_padding_error: bool, split_overflow_padding_error: bool
  ):
    graph, schema = _generate_large_graph(num_nodes=200, num_edges=2000)
    seed_node_idxs = np.arange(16, dtype=np.int64)
    # A padding sufficient for each sample (at most 1 + 6 + 36 nodes and 42
    # edges), but not for the entire batch.
    padding = _padding({"a": 60}, {"aa": 100})

    def create_sampler():
      return self._create_sampler(
          graph, schema, seed_nodeset="a", num_hops=2, hop_width=3
      )

    expected_num_skipped = []
    expected = list(
        merge_lib.GraphMerger(schema, padding=padding).merge_sub_batches(
            create_sampler().sample(seed_node_idxs),
            skip_overflow_padding_error=skip_overflow_padding_error,
            split_overflow_padding_error=split_overflow_padding_error,
            on_skip_samples=expected_num_skipped.append,
        )
    )
    actual_num_skipped = []
    actual = (
        create_sampler()
        .with_padding(padding)
        .sample_merged_sub_batches(
            seed_node_idxs,
            skip_overflow_padding_error=skip_overflow_padding_error,
            split_overflow_padding_error=split_overflow_padding_error,
            on_skip_samples=actual_num_skipped.append,
        )
    )
    if split_overflow_padding_error:
      self.assertGreater(len(actual), 1)
    else:
      self.assertEqual(actual_num_skipped, [len(seed_node_idxs)])
    self.assertEqual(actual_num_skipped, expected_num_skipped)
    _assert_merged_equal(self, actual, expected)

  def test_with_padding(self):
    graph = gen_test_graph.generate_in_memory_graph()
    schema = gen_test_graph.generate_schema()
    padding = _padding({"n1": 100, "n2": 100}, {"e1": 100, "e2": 100})
    seed_node_idxs = np.array([0, 1], dtype=np.int64)
    sampler = self._create_sampler(graph, schema, seed_nodeset="n1")

    padded_sampler = sampler.with_padding(padding)
    self.assertIs(padded_sampler.padding, padding)
    _assert_merged_equal(
        self,
        padded_sampler.sample_merged(seed_node_idxs),
        self._create_sampler(
            graph, schema, seed_nodeset="n1", padding=padding
        ).sample_merged(seed_node_idxs),
    )

    # The original sampler is not padded.
    self.assertIsNone(sampler.padding)
    merged, offsets = sampler.sample_merged(seed_node_idxs)
    self.assertEqual(merged.node_sets["n1"].num_nodes, offsets["n1"][-1])

  def test_set_return_options_after_sample_merged(self):
    graph = gen_test_graph.generate_in_memory_graph()
    schema = gen_test_graph.generate_schema()
    sampler = self._create_sampler(graph, schema, seed_nodeset="n1")
    merged, _ = sampler.sample_merged([0])
    self.assertEqual(set(merged.node_sets["n1"].features), {"f1", "f2"})

    sampler.set_return_options(return_features=False, return_node_idxs=True)
    merged, _ = sampler.sample_merged([0])
    self.assertEqual(set(merged.node_sets["n1"].features), {"#idx"})

  def test_feature_with_missing_values_raises(self):
    graph = gen_test_graph.generate_in_memory_graph()
    schema = gen_test_graph.generate_schema()
    # "n1" has 2 nodes, but "f2" only has one value.
    graph.node_sets["n1"].features["f2"] = np.array([[0.0, 1.0]], np.float32)
    sampler = self._create_sampler(graph, schema, seed_nodeset="n1")
    with self.assertRaisesRegex(ValueError, "out of the bounds"):
      sampler.sample_merged([0, 1])


class CcGatherSourceTest(parameterized.TestCase):

  @parameterized.parameters(
      (np.array([1, 2], np.int32),),
      (np.array([[1.0, 2.0]], np.float16),),
      (np.array([True, False]),),
      (np.array([1, 2], np.uint64),),
  )
  def test_numerical(self, src: np.ndarray):
    self.assertIs(in_memory_sampler_lib._cc_gather_source(src), src)  # pylint: disable=protected-access

  @parameterized.parameters(
      (np.array([b"a", b"bcd"]), (2, 3)),
      (np.array([[b"a"], [b"bc"]]), (2, 2)),
      (np.array(["a", "bc"]), (2, 8)),
  )
  def test_bytes_and_strings(self, src: np.ndarray, expected_shape):
    result = in_memory_sampler_lib._cc_gather_source(src)  # pylint: disable=protected-access
    assert result is not None
    self.assertEqual(result.dtype, np.uint8)
    self.assertEqual(result.shape, expected_shape)
    self.assertTrue(np.shares_memory(result, src))
    np.testing.assert_array_equal(result.view(src.dtype).reshape(src.shape), src)

  @parameterized.named_parameters(
      ("object", np.array([np.array([1]), np.array([1, 2])], np.object_)),
      ("non_contiguous", np.arange(12).reshape(3, 4)[:, ::2]),
      ("scalar", np.array(1)),
      ("empty_rows", np.zeros((3, 0), np.float32)),
      ("complex", np.array([1j])),
      ("list", [1, 2]),
  )
  def test_unsupported(self, src):
    self.assertIsNone(in_memory_sampler_lib._cc_gather_source(src))  # pylint: disable=protected-access


class InMemorySamplerDiamon(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.schema = schema_lib.GraphSchema(
        node_sets={
            "n1": schema_lib.NodeSchema(
                features={
                    "f1": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64
                    )
                }
            ),
        },
        edge_sets={
            "e11": schema_lib.EdgeSchema(source="n1", target="n1"),
        },
    )
    # Create a diamon followed by 3 nodes.
    cls.graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                features={"f1": np.array([10, 11, 12, 13, 14, 15, 16])},
                num_nodes=7,
            ),
        },
        edge_sets={
            "e11": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array(
                    [[0, 0, 1, 2, 3, 3, 3], [1, 2, 3, 3, 4, 5, 6]]
                )
            ),
        },
    )

  @parameterized.parameters(True, False)
  def test_multi_visit(self, multi_visit: bool):
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        config_lib.SimpleSamplingConfig(
            seed_nodeset="n1",
            num_hops=5,
            hop_width=2,
            reverse=False,
            multi_visit=multi_visit,
        ),
        self.schema,
    )
    sampler = in_memory_sampler_lib.create_sampler(
        self.graph,
        plan,
        self.schema,
        return_features=True,
        return_node_idxs=False,
        batch_size=5,
        seed=0,
    )
    sample = sampler.sample(0)
    self.assertIsNotNone(sample)
    if multi_visit:
      self.assertEqual(sample.node_sets["n1"].num_nodes, 7)
    else:
      self.assertEqual(sample.node_sets["n1"].num_nodes, 6)


if __name__ == "__main__":
  absltest.main()

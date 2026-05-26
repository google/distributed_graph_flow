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

import logging
from typing import List
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.sampling import _in_memory_sampler_ext
from dgf.src.sampling import config as config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
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
    sampling_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1",
        num_hops=1,
        hop_width=2,
        edgeset_timestamp_features={"e12": "timestamp"},
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        sampling_config, self.schema
    )
    with self.assertRaisesRegex(
        ValueError,
        "Temporal filtering and edge masking cannot be used at the same time.",
    ):
      in_memory_sampler_lib.create_sampler(
          self.graph,
          plan,
          self.schema,
          debug_sampling=True,
          batch_size=5,
          edgeset_to_mask="e12",
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
      seed_idxs: List[int],
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

  @parameterized.parameters(10, 20, 25, 40)
  def test_temporal_sampling(self, seed_timestamp: int):

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
            edgeset_timestamp_features={"e1": "timestamp"},
        ),
        schema,
        return_features=True,
        return_node_idxs=True,
        batch_size=5,
        debug_sampling=True,
    )

    sample = sampler.sample([0], seed_timestamps=[seed_timestamp])

    # The edges are (source node idx, destination node idx, timestamp):
    # 0 -> 1 (15)
    # 0 -> 2 (25)
    # 1 -> 3 (35)
    if seed_timestamp == 10:
      # Should only have node idx 0
      expected_n1 = InMemoryNodeSet(
          num_nodes=1,
          features={
              "#idx": np.array([0], dtype=np.int64),
              "timestamp": np.array([10], dtype=np.int64),
              "feat": np.array([[1.0, 1.0]], dtype=np.float32),
          },
      )
      expected_e1 = InMemoryEdgeSet(
          adjacency=np.zeros((2, 0), dtype=np.int64), features={}
      )
      expected_e2 = InMemoryEdgeSet(
          adjacency=np.zeros((2, 0), dtype=np.int64), features={}
      )
    elif seed_timestamp == 20:
      # Should only have node idx 0 and 1
      expected_n1 = InMemoryNodeSet(
          num_nodes=2,
          features={
              "#idx": np.array([0, 1], dtype=np.int64),
              "timestamp": np.array([10, 20], dtype=np.int64),
              "feat": np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
          },
      )
      expected_e1 = InMemoryEdgeSet(
          adjacency=np.array([[0], [1]], dtype=np.int64), features={}
      )
      expected_e2 = InMemoryEdgeSet(
          adjacency=np.zeros((2, 0), dtype=np.int64), features={}
      )
    elif seed_timestamp == 25:
      # Should only have node idx 0, 1, and 2
      expected_n1 = InMemoryNodeSet(
          num_nodes=3,
          features={
              "#idx": np.array([0, 1, 2], dtype=np.int64),
              "timestamp": np.array([10, 20, 30], dtype=np.int64),
              "feat": np.array(
                  [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32
              ),
          },
      )
      expected_e1 = InMemoryEdgeSet(
          adjacency=np.array([[0, 0], [1, 2]], dtype=np.int64), features={}
      )
      expected_e2 = InMemoryEdgeSet(
          adjacency=np.array([[2], [0]], dtype=np.int64), features={}
      )
    elif seed_timestamp == 40:
      # Should have all the node idsx (0,1,2,3)
      expected_n1 = InMemoryNodeSet(
          num_nodes=4,
          features={
              "#idx": np.array([0, 1, 3, 2], dtype=np.int64),
              "timestamp": np.array([10, 20, 40, 30], dtype=np.int64),
              "feat": np.array(
                  [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [3.0, 3.0]],
                  dtype=np.float32,
              ),
          },
      )
      expected_e1 = InMemoryEdgeSet(
          adjacency=np.array([[0, 0, 1], [1, 3, 2]], dtype=np.int64),
          features={},
      )
      expected_e2 = InMemoryEdgeSet(
          adjacency=np.array([[3], [0]], dtype=np.int64), features={}
      )
    else:
      raise ValueError(f"Unexpected seed_timestamp: {seed_timestamp}")

    expected_graph = InMemoryGraph(
        node_sets={"n1": expected_n1},
        edge_sets={"e1": expected_e1, "e2": expected_e2},
    )
    test_util.assert_are_equal(self, expected_graph, sample[0])

  def test_temporal_sampling_wrong_edgeset(self):
    graph, schema = gen_test_graph.generate_temporal_in_memory_graph(
        include_e2=True
    )
    with self.assertRaisesRegex(
        ValueError, "Edgeset 'do_not_exist1' does not exist in schema"
    ):
      in_memory_sampler_lib.create_sampler(
          graph,
          config_lib.SimpleSamplingConfig(
              seed_nodeset="n1",
              num_hops=2,
              hop_width=2,
              reverse=False,
              edgeset_timestamp_features={"do_not_exist1": "feature"},
          ),
          schema,
          batch_size=5,
      )

  def test_temporal_sampling_wrong_feature(self):
    graph, schema = gen_test_graph.generate_temporal_in_memory_graph(
        include_e2=True
    )
    with self.assertRaisesRegex(ValueError, 'Key ".*" not found in map'):
      _ = in_memory_sampler_lib.create_sampler(
          graph,
          config_lib.SimpleSamplingConfig(
              seed_nodeset="n1",
              num_hops=2,
              hop_width=2,
              reverse=False,
              edgeset_timestamp_features={"e1": "do_not_exist"},
          ),
          schema,
          batch_size=5,
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
            edgeset_timestamp_features={"e1": "timestamp"},
        ),
        schema,
        batch_size=5,
    )

    with self.assertRaisesRegex(
        ValueError,
        "Cannot use SampleRandomUniform when timestamps are available",
    ):
      _ = sampler.sample([0])

  def test_temporal_sampling_unexpected_seed_timestamp(self):
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
        ),
        schema,
        batch_size=5,
    )

    with self.assertRaisesRegex(
        ValueError,
        "seed_timestamps provided but no temporal edgesets configured",
    ):
      _ = sampler.sample([0], seed_timestamps=[5])

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
            edgeset_timestamp_features={"e1": "timestamp"},
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


if __name__ == "__main__":
  absltest.main()

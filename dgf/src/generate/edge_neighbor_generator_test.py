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
from dgf.src.generate import edge_neighbor_generator
from dgf.src.sampling import config as config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.util import gen_test_graph
import numpy as np


class EdgeNeighborGeneratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.graph, self.schema = (
        gen_test_graph.generate_recommender_like_in_memory_graph()
    )
    self.target_edgeset = "e2"
    self.num_negative_neighbors = 3

  def test_negative_neighbor_sampler(self):
    sampler = edge_neighbor_generator.RandomNegativeSampler(
        self.graph,
        self.schema,
        self.target_edgeset,
        self.num_negative_neighbors,
    )
    seed_edge_idxs = np.array([0, 1], dtype=np.int64)
    neg_trg_nodes = sampler.sample(seed_edge_idxs)

    self.assertEqual(neg_trg_nodes.shape, (2, self.num_negative_neighbors))
    # Check that all node idxs are within valid range
    num_nodes = self.graph.node_sets[
        self.schema.edge_sets[self.target_edgeset].target
    ].num_nodes
    self.assertTrue(np.all(neg_trg_nodes >= 0))
    self.assertTrue(np.all(neg_trg_nodes < num_nodes))

  def test_random_walk_negative_sampler(self):
    plan = config_lib.SamplingPlan(
        root=config_lib.PlanNode(
            nodeset="n1",
            children=[
                config_lib.PlanEdge(
                    edgeset=self.target_edgeset,
                    reversed=False,
                    hop_width=1,
                    node=config_lib.PlanNode(
                        nodeset="n2",
                        children=[
                            config_lib.PlanEdge(
                                edgeset=self.target_edgeset,
                                reversed=True,
                                hop_width=1,
                                node=config_lib.PlanNode(
                                    nodeset="n1",
                                    children=[
                                        config_lib.PlanEdge(
                                            edgeset=self.target_edgeset,
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
        self.graph, plan, self.schema, batch_size=1, seed=1234
    )

    rw_sampler = edge_neighbor_generator.RandomWalkNegativeSampler(
        graph=self.graph,
        schema=self.schema,
        target_edgeset=self.target_edgeset,
        num_negative_neighbors=self.num_negative_neighbors,
        sampler=sampler,
        num_walks=500,
    )

    # Edge 0 maps to source node 0. Edge 2 maps to source node 1.
    seed_edge_idxs = np.array([0, 2], dtype=np.int64)
    neg_trg_nodes = rw_sampler.sample(seed_edge_idxs)

    self.assertEqual(neg_trg_nodes.shape, (2, self.num_negative_neighbors))

    # Node 0 samples: top ranked is 2, fallbacks can be any target node.
    sampled0 = neg_trg_nodes[0]
    self.assertEqual(sampled0[0], 2)
    for n in sampled0[1:]:
      self.assertIn(n, [0, 1, 2, 3, 4, 5])

    # Node 1 samples: top ranked is 0, fallbacks can be any target node.
    sampled1 = neg_trg_nodes[1]
    self.assertEqual(sampled1[0], 0)
    for n in sampled1[1:]:
      self.assertIn(n, [0, 1, 2, 3, 4, 5])

    # Test seed node 2 directly via the private node sampling method, as it has
    # no edges. Fallback samples with replacement from 0 to 5.
    neg_node2 = rw_sampler._sample_from_seed_node_idxs(
        np.array([2], dtype=np.int64)
    )
    self.assertEqual(neg_node2.shape, (1, self.num_negative_neighbors))
    sampled2 = neg_node2[0]
    for n in sampled2:
      self.assertIn(n, [0, 1, 2, 3, 4, 5])

  def test_edge_neighbor_generator(self):
    generator = edge_neighbor_generator.EdgeNeighborGenerator(
        self.graph,
        self.schema,
        self.target_edgeset,
        self.num_negative_neighbors,
    )
    seed_edge_idxs = np.array([0, 1], dtype=np.int64)
    edge_neighbor_idxs = generator.generate(seed_edge_idxs)

    self.assertEqual(edge_neighbor_idxs.pos_src_node_idxs.shape, (2,))
    self.assertEqual(edge_neighbor_idxs.pos_trg_node_idxs.shape, (2,))
    self.assertEqual(
        edge_neighbor_idxs.neg_trg_node_idxs.shape,
        (2, self.num_negative_neighbors),
    )

    # Verify values against ground truth from graph
    target_edgeset_data = self.graph.edge_sets[self.target_edgeset]
    expected_pos_src = target_edgeset_data.adjacency[0][seed_edge_idxs]
    expected_pos_trg = target_edgeset_data.adjacency[1][seed_edge_idxs]

    np.testing.assert_array_equal(
        edge_neighbor_idxs.pos_src_node_idxs, expected_pos_src
    )
    np.testing.assert_array_equal(
        edge_neighbor_idxs.pos_trg_node_idxs, expected_pos_trg
    )


if __name__ == "__main__":
  absltest.main()

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
import dgf
from dgf.benchmark import mpnn
import jax.numpy as jnp


class MPNNBenchmarkTest(absltest.TestCase):

  def test_base(self):
    scenarios = [
        mpnn.Scenario(
            name="tiny_tree",
            num_nodesets=2,
            num_edgesets=2,
            treeness=1.0,
            batch_size=4,
            num_hops=1,
            hop_width=3,
        ),
        mpnn.Scenario(
            name="tiny_mixed",
            num_nodesets=2,
            num_edgesets=2,
            treeness=0.3,
            batch_size=4,
            num_hops=1,
            hop_width=3,
            edgeset_hop_widths={"e1": 1},
            padding_relative_margin=1.0,
        ),
    ]
    mpnn.mpnn(
        scenarios=scenarios,
        list_dims=[16],
        list_num_layers=[1],
        max_runtime_seconds=0.05,
    )

  def test_describe_relations_regimes(self):
    schema = mpnn.generate_synthetic_schema(
        num_nodesets=2, num_edgesets=1, dims=4, num_hops=1
    )

    def make_graph(num_n0: int, num_n1: int, num_edges: int):
      return dgf.data.JaxInMemoryGraph(
          node_sets={
              "n0": dgf.data.JaxInMemoryNodeSet(
                  num_nodes=num_n0, features={}
              ),
              "n1": dgf.data.JaxInMemoryNodeSet(
                  num_nodes=num_n1, features={}
              ),
          },
          edge_sets={
              "e0": dgf.data.JaxInMemoryEdgeSet(
                  adjacency=jnp.zeros((2, num_edges), dtype=jnp.int32)
              )
          },
      )

    # More nodes than edges: regime 2 in both directions.
    self.assertEqual(
        mpnn.describe_relations(make_graph(100, 100, 50), schema),
        "e0[100->100:50 r2/r2 u16]",
    )
    # More edges than nodes: regime 1 in both directions.
    self.assertEqual(
        mpnn.describe_relations(make_graph(10, 10, 100), schema),
        "e0[10->10:100 r1/r1 u16]",
    )
    # Asymmetric: regime depends on the direction.
    self.assertEqual(
        mpnn.describe_relations(make_graph(10, 200, 100), schema),
        "e0[10->200:100 r2/r1 u16]",
    )
    # Large: sorting enabled, indices do not fit in uint16.
    self.assertEqual(
        mpnn.describe_relations(make_graph(70000, 10, 60000), schema),
        "e0[70000->10:60000 r1/r1 sort]",
    )


if __name__ == "__main__":
  absltest.main()

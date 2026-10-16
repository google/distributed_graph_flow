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

"""Tests for graph canonicalization."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.transform import canonicalize as canonicalize_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np


def _permute_graph(
    graph: in_memory_graph_lib.InMemoryGraph,
    schema: schema_lib.GraphSchema,
    node_orders: dict[str, list[int]],
    edge_orders: dict[str, list[int]],
) -> in_memory_graph_lib.InMemoryGraph:
  """Permutes nodes and edges of `graph` while keeping topology isomorphic."""
  new_node_sets = {}
  inv_node_orders = {}
  for nodeset_name, order in node_orders.items():
    order_arr = np.array(order, dtype=np.int64)
    inv = np.empty_like(order_arr)
    inv[order_arr] = np.arange(len(order_arr), dtype=np.int64)
    inv_node_orders[nodeset_name] = inv

    ns = graph.node_sets[nodeset_name]
    new_node_sets[nodeset_name] = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=ns.num_nodes,
        features={k: v[order_arr] for k, v in ns.features.items()},
    )

  new_edge_sets = {}
  for edgeset_name, edge_order in edge_orders.items():
    edge_order_arr = np.array(edge_order, dtype=np.int64)
    es = graph.edge_sets[edgeset_name]
    es_schema = schema.edge_sets[edgeset_name]
    src_inv = inv_node_orders[es_schema.source]
    dst_inv = inv_node_orders[es_schema.target]
    remapped_adj = np.stack(
        [src_inv[es.adjacency[0]], dst_inv[es.adjacency[1]]], axis=0
    )
    new_edge_sets[edgeset_name] = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=remapped_adj[:, edge_order_arr],
        features={k: v[edge_order_arr] for k, v in es.features.items()},
    )

  return in_memory_graph_lib.InMemoryGraph(
      node_sets=new_node_sets,
      edge_sets=new_edge_sets,
      timestamp=graph.timestamp,
  )


class CanonicalizeTest(parameterized.TestCase):

  @parameterized.parameters(
      (False, True),
      (True, True),
      (False, False),
      (True, False),
  )
  def test_canonicalize_recovers_original_from_permuted_graph(
      self, edge_ids: bool, semantic: bool
  ):
    schema = gen_test_graph.generate_schema(
        node_ids=True,
        edge_ids=edge_ids,
        semantic=semantic,
        variable_length=True,
    )
    expected_graph = gen_test_graph.generate_in_memory_graph(
        node_ids=True,
        edge_ids=edge_ids,
        variable_length=True,
    )
    permuted_graph = _permute_graph(
        expected_graph,
        schema,
        node_orders={"n1": [1, 0], "n2": [1, 0]},
        edge_orders={"e1": [1, 0], "e2": [1, 0]},
    )

    canonical_graph = canonicalize_lib.canonicalize_graph(
        permuted_graph, schema
    )
    test_util.assert_are_equal(self, canonical_graph, expected_graph)

  def test_canonicalize_sorts_edges_by_edge_primary_key_when_present(self):
    schema = gen_test_graph.generate_schema(
        node_ids=True,
        edge_ids=True,
        semantic=True,
        variable_length=False,
    )
    graph = gen_test_graph.generate_in_memory_graph(
        node_ids=True,
        edge_ids=True,
        variable_length=False,
    )
    # Assign edge IDs in reverse order of (source, target) endpoints so edge
    # primary key sorting differs from lexicographic endpoint sorting.
    graph.edge_sets["e1"].features["#id"] = np.array([b"z_edge", b"a_edge"])
    canon = canonicalize_lib.canonicalize_graph(graph, schema)

    np.testing.assert_array_equal(
        canon.edge_sets["e1"].features["#id"],
        np.array([b"a_edge", b"z_edge"]),
    )
    np.testing.assert_array_equal(
        canon.edge_sets["e1"].adjacency,
        np.array([[0, 0], [1, 0]]),
    )

  def test_canonicalize_empty_nodeset_and_edgeset_preserves_timestamp(self):
    schema = gen_test_graph.generate_schema(
        node_ids=True,
        edge_ids=False,
        semantic=True,
        variable_length=False,
    )
    empty_graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=0,
                features={
                    "#id": np.empty((0,), dtype=np.bytes_),
                    "f1": np.empty((0, 1), dtype=np.bytes_),
                    "f2": np.empty((0, 2), dtype=np.float32),
                },
            ),
            "n2": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=0,
                features={
                    "#id": np.empty((0,), dtype=np.int64),
                    "f3": np.empty((0,), dtype=np.int64),
                    "f4": np.empty((0,), dtype=np.int64),
                },
            ),
        },
        edge_sets={
            "e1": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.zeros((2, 0), dtype=np.int64),
                features={},
            ),
            "e2": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.zeros((2, 0), dtype=np.int64),
                features={},
            ),
        },
        timestamp=123456,
    )
    canon = canonicalize_lib.canonicalize_graph(empty_graph, schema)
    test_util.assert_are_equal(self, canon, empty_graph)
    self.assertEqual(canon.timestamp, 123456)

  def test_canonicalize_raises_without_primary_key(self):
    schema = gen_test_graph.generate_schema(
        node_ids=False,
        edge_ids=False,
        semantic=True,
        variable_length=False,
    )
    graph = gen_test_graph.generate_in_memory_graph(
        node_ids=False,
        edge_ids=False,
        variable_length=False,
    )
    with self.assertRaises(ValueError):
      canonicalize_lib.canonicalize_graph(graph, schema)


if __name__ == "__main__":
  absltest.main()

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

"""Tests for connected_components module."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.gbbs import connected_components
from dgf.src.gbbs import loader
import numpy as np


def _make_chain_graph(
    num_nodes: int, symmetric: bool
) -> loader.GbbsGraphHandle:
  """Creates a simple chain graph: 0-1-2-..-(num_nodes-1)."""
  sources = np.arange(num_nodes - 1, dtype=np.int64)
  targets = np.arange(1, num_nodes, dtype=np.int64)
  adjacency = np.stack([sources, targets])
  return loader.create_gbbs_graph_handle(
      num_nodes=num_nodes, adjacency=adjacency, symmetric=symmetric
  )


def _make_two_component_graph(symmetric: bool) -> loader.GbbsGraphHandle:
  """Creates a graph with two disconnected components: {0,1,2} and {3,4}.

  Edges: 0-1, 1-2, 3-4.
  """
  sources = np.array([0, 1, 3], dtype=np.int64)
  targets = np.array([1, 2, 4], dtype=np.int64)
  adjacency = np.stack([sources, targets])
  return loader.create_gbbs_graph_handle(
      num_nodes=5, adjacency=adjacency, symmetric=symmetric
  )


def _make_single_node_graph(symmetric: bool) -> loader.GbbsGraphHandle:
  """Creates a graph with a single node and no edges."""
  adjacency = np.empty((2, 0), dtype=np.int64)
  return loader.create_gbbs_graph_handle(
      num_nodes=1, adjacency=adjacency, symmetric=symmetric
  )


def _make_directed_cycle_with_tail() -> loader.GbbsGraphHandle:
  """Creates a directed graph with one SCC cycle and an outgoing tail.

  Edges: 0→1, 1→2, 2→0, 2→3.

  Strongly connected components:
    SCC 1: {0, 1, 2}  (cycle)
    SCC 2: {3}         (reachable from SCC 1 but cannot reach back)
  """
  sources = np.array([0, 1, 2, 2], dtype=np.int64)
  targets = np.array([1, 2, 0, 3], dtype=np.int64)
  adjacency = np.stack([sources, targets])
  return loader.create_gbbs_graph_handle(
      num_nodes=4, adjacency=adjacency, symmetric=False
  )


class ValidateGraphParamsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    loader.set_num_parlay_workers(1)

  @parameterized.named_parameters(
      dict(
          testcase_name="simple_union_async",
          params=connected_components.SimpleUnionAsyncCCParams(),
      ),
      dict(
          testcase_name="shiloach_vishkin",
          params=connected_components.ShiloachVishkinCCParams(),
      ),
      dict(
          testcase_name="scc",
          params=connected_components.StronglyConnectedComponentsParams(),
      ),
  )
  def test_directed_capable_params_accept_asymmetric_graph(self, params):
    graph = _make_chain_graph(num_nodes=4, symmetric=False)
    # Should not raise — these algorithms work on directed graphs.
    connected_components.validate_graph_params(graph, params)

  @parameterized.named_parameters(
      dict(
          testcase_name="simple_union_async",
          params=connected_components.SimpleUnionAsyncCCParams(),
      ),
      dict(
          testcase_name="shiloach_vishkin",
          params=connected_components.ShiloachVishkinCCParams(),
      ),
      dict(
          testcase_name="scc",
          params=connected_components.StronglyConnectedComponentsParams(),
      ),
  )
  def test_directed_capable_params_accept_symmetric_graph(self, params):
    graph = _make_chain_graph(num_nodes=4, symmetric=True)
    # Should not raise.
    connected_components.validate_graph_params(graph, params)

  @parameterized.named_parameters(
      dict(
          testcase_name="bfs",
          params=connected_components.BfsCCParams(),
      ),
      dict(
          testcase_name="label_propagation",
          params=connected_components.LabelPropagationCCParams(),
      ),
      dict(
          testcase_name="work_efficient",
          params=connected_components.WorkEfficientCCParams(),
      ),
  )
  def test_symmetric_only_params_accept_symmetric_graph(self, params):
    graph = _make_chain_graph(num_nodes=4, symmetric=True)
    # Should not raise.
    connected_components.validate_graph_params(graph, params)

  @parameterized.named_parameters(
      dict(
          testcase_name="bfs",
          params=connected_components.BfsCCParams(),
          expected_name="BfsCCParams",
      ),
      dict(
          testcase_name="label_propagation",
          params=connected_components.LabelPropagationCCParams(),
          expected_name="LabelPropagationCCParams",
      ),
      dict(
          testcase_name="work_efficient",
          params=connected_components.WorkEfficientCCParams(),
          expected_name="WorkEfficientCCParams",
      ),
  )
  def test_symmetric_only_params_reject_asymmetric_graph(
      self, params, expected_name
  ):
    graph = _make_chain_graph(num_nodes=4, symmetric=False)
    with self.assertRaisesRegex(
        ValueError,
        f"{expected_name} requires a symmetric",
    ):
      connected_components.validate_graph_params(graph, params)


class ConnectedComponentsTest(parameterized.TestCase):
  """Tests for the connected_components() dispatch function."""

  def setUp(self):
    super().setUp()
    loader.set_num_parlay_workers(1)

  @parameterized.named_parameters(
      dict(
          testcase_name="default_params",
          params=None,
      ),
      dict(
          testcase_name="simple_union_async",
          params=connected_components.SimpleUnionAsyncCCParams(),
      ),
      dict(
          testcase_name="bfs",
          params=connected_components.BfsCCParams(),
      ),
      dict(
          testcase_name="label_propagation",
          params=connected_components.LabelPropagationCCParams(),
      ),
      dict(
          testcase_name="label_propagation_permuted",
          params=connected_components.LabelPropagationCCParams(
              use_permutation=True
          ),
      ),
      dict(
          testcase_name="work_efficient",
          params=connected_components.WorkEfficientCCParams(),
      ),
      dict(
          testcase_name="shiloach_vishkin",
          params=connected_components.ShiloachVishkinCCParams(),
      ),
  )
  def test_single_component_symmetric(self, params):
    """A connected chain should produce exactly one component."""
    graph = _make_chain_graph(num_nodes=5, symmetric=True)
    result = connected_components.connected_components(
        graph, params=params, progress=False
    )
    self.assertEqual(result.num_components, 1)
    self.assertEqual(result.largest_component_size, 5)
    # All labels should be identical.
    np.testing.assert_array_equal(result.labels, result.labels[0])

  @parameterized.named_parameters(
      dict(
          testcase_name="default_params",
          params=None,
      ),
      dict(
          testcase_name="simple_union_async",
          params=connected_components.SimpleUnionAsyncCCParams(),
      ),
      dict(
          testcase_name="bfs",
          params=connected_components.BfsCCParams(),
      ),
      dict(
          testcase_name="label_propagation",
          params=connected_components.LabelPropagationCCParams(),
      ),
      dict(
          testcase_name="work_efficient",
          params=connected_components.WorkEfficientCCParams(),
      ),
      dict(
          testcase_name="shiloach_vishkin",
          params=connected_components.ShiloachVishkinCCParams(),
      ),
  )
  def test_two_components_symmetric(self, params):
    """Two disconnected components on a symmetric graph."""
    graph = _make_two_component_graph(symmetric=True)
    result = connected_components.connected_components(
        graph, params=params, progress=False
    )
    self.assertEqual(result.num_components, 2)
    self.assertEqual(result.largest_component_size, 3)
    # Nodes 0, 1, 2 share a label; nodes 3, 4 share a different label.
    self.assertEqual(result.labels[0], result.labels[1])
    self.assertEqual(result.labels[0], result.labels[2])
    self.assertEqual(result.labels[3], result.labels[4])
    self.assertNotEqual(result.labels[0], result.labels[3])

  @parameterized.named_parameters(
      dict(
          testcase_name="simple_union_async",
          params=connected_components.SimpleUnionAsyncCCParams(),
      ),
      dict(
          testcase_name="shiloach_vishkin",
          params=connected_components.ShiloachVishkinCCParams(),
      ),
  )
  def test_two_components_asymmetric(self, params):
    """Union-find algorithms should handle directed graphs correctly."""
    graph = _make_two_component_graph(symmetric=False)
    result = connected_components.connected_components(
        graph, params=params, progress=False
    )
    self.assertEqual(result.num_components, 2)
    self.assertEqual(result.largest_component_size, 3)

  def test_single_node_graph(self):
    """A single isolated node is one component."""
    graph = _make_single_node_graph(symmetric=True)
    result = connected_components.connected_components(graph, progress=False)
    self.assertEqual(result.num_components, 1)
    self.assertEqual(result.largest_component_size, 1)
    self.assertLen(result.labels, 1)

  def test_connected_components_rejects_asymmetric_for_bfs(self):
    """connected_components() should raise before dispatching to C++."""
    graph = _make_chain_graph(num_nodes=3, symmetric=False)
    with self.assertRaisesRegex(ValueError, "requires a symmetric"):
      connected_components.connected_components(
          graph, params=connected_components.BfsCCParams(), progress=False
      )

  def test_result_labels_length_matches_num_nodes(self):
    """Labels array length should equal the graph's node count."""
    num_nodes = 10
    graph = _make_chain_graph(num_nodes=num_nodes, symmetric=True)
    result = connected_components.connected_components(graph, progress=False)
    self.assertLen(result.labels, num_nodes)

  def test_result_labels_dtype_uint32(self):
    """Labels should be a uint32 numpy array."""
    graph = _make_chain_graph(num_nodes=3, symmetric=True)
    result = connected_components.connected_components(graph, progress=False)
    self.assertEqual(result.labels.dtype, np.uint32)

  def test_progress_flag_does_not_change_result(self):
    """progress=True and progress=False should produce identical results."""
    graph = _make_two_component_graph(symmetric=True)
    result_no_progress = connected_components.connected_components(
        graph, progress=False
    )
    result_progress = connected_components.connected_components(
        graph, progress=True
    )
    np.testing.assert_array_equal(
        result_no_progress.labels, result_progress.labels
    )
    self.assertEqual(
        result_no_progress.num_components, result_progress.num_components
    )


class StronglyConnectedComponentsTest(parameterized.TestCase):
  """Tests for the SCC algorithm via connected_components()."""

  def setUp(self):
    super().setUp()
    loader.set_num_parlay_workers(1)

  def test_scc_finds_cycle_and_tail(self):
    """Directed cycle 0→1→2→0 with tail 2→3 yields 2 SCCs."""
    graph = _make_directed_cycle_with_tail()
    result = connected_components.connected_components(
        graph,
        params=connected_components.StronglyConnectedComponentsParams(),
        progress=False,
    )
    self.assertEqual(result.num_components, 2)
    # The cycle {0, 1, 2} is the largest SCC.
    self.assertEqual(result.largest_component_size, 3)
    # Nodes in the cycle share a label.
    self.assertEqual(result.labels[0], result.labels[1])
    self.assertEqual(result.labels[0], result.labels[2])
    # Node 3 is in a different SCC.
    self.assertNotEqual(result.labels[0], result.labels[3])

  def test_scc_directed_chain_has_singleton_components(self):
    """A directed chain 0→1→2→3 has no cycles, so each node is its own SCC."""
    graph = _make_chain_graph(num_nodes=4, symmetric=False)
    result = connected_components.connected_components(
        graph,
        params=connected_components.StronglyConnectedComponentsParams(),
        progress=False,
    )
    self.assertEqual(result.num_components, 4)
    self.assertEqual(result.largest_component_size, 1)
    # All labels are distinct.
    unique_labels = set(result.labels.tolist())
    self.assertLen(unique_labels, 4)

  def test_scc_single_node(self):
    """A single isolated node is one SCC."""
    graph = _make_single_node_graph(symmetric=False)
    result = connected_components.connected_components(
        graph,
        params=connected_components.StronglyConnectedComponentsParams(),
        progress=False,
    )
    self.assertEqual(result.num_components, 1)
    self.assertEqual(result.largest_component_size, 1)
    self.assertLen(result.labels, 1)

  def test_scc_on_symmetric_graph(self):
    """On an undirected (symmetric) graph, SCC = WCC — every edge is a cycle."""
    graph = _make_two_component_graph(symmetric=True)
    result = connected_components.connected_components(
        graph,
        params=connected_components.StronglyConnectedComponentsParams(),
        progress=False,
    )
    # Same result as weakly connected components on undirected graph.
    self.assertEqual(result.num_components, 2)
    self.assertEqual(result.largest_component_size, 3)

  def test_scc_custom_beta(self):
    """SCC with a non-default beta parameter still produces correct results."""
    graph = _make_directed_cycle_with_tail()
    result = connected_components.connected_components(
        graph,
        params=connected_components.StronglyConnectedComponentsParams(beta=2.0),
        progress=False,
    )
    self.assertEqual(result.num_components, 2)
    self.assertEqual(result.largest_component_size, 3)

  def test_scc_labels_length_matches_num_nodes(self):
    """Labels array length should match graph node count."""
    graph = _make_directed_cycle_with_tail()
    result = connected_components.connected_components(
        graph,
        params=connected_components.StronglyConnectedComponentsParams(),
        progress=False,
    )
    self.assertLen(result.labels, 4)


class IsSymmetricTest(absltest.TestCase):
  """Tests for the GbbsGraphHandle.is_symmetric() nanobind binding."""

  def setUp(self):
    super().setUp()
    loader.set_num_parlay_workers(1)

  def test_symmetric_graph_reports_symmetric(self):
    graph = _make_chain_graph(num_nodes=3, symmetric=True)
    self.assertTrue(graph.is_symmetric())

  def test_asymmetric_graph_reports_asymmetric(self):
    graph = _make_chain_graph(num_nodes=3, symmetric=False)
    self.assertFalse(graph.is_symmetric())


if __name__ == "__main__":
  absltest.main()

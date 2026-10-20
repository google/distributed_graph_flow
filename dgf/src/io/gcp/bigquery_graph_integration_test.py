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

r"""Manual tests for bigquery_graph.
"""

import os

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_in_memory as gf_graph_in_memory_lib
from dgf.src.io.gcp import bigquery_graph
from dgf.src.transform import canonicalize as canonicalize_lib
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np

_PROJECT = "biggraphs-poc"
# Graphs created by
# `third_party/py/dgf/src/sampling/gcp/testdata/create_sampler_test_graphs.sql`.
_TEST_GRAPHS_DATASET = "dgf_io_tests"
_TEST_GRAPHS_GCS_EXPORT_PATH = "gs://gbm-test/dgf_exporter_test_graphs"


def _object_array(values: list[np.ndarray]) -> np.ndarray:
  """Returns a 1D object array of variable-length arrays."""
  array = np.empty(len(values), dtype=np.object_)
  for index, value in enumerate(values):
    array[index] = value
  return array


def _expected_string_graph() -> in_memory_graph_lib.InMemoryGraph:
  """Returns the content of `dgf_sampler_string_graph`, NULLs as defaults."""
  return in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "person": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=4,
              features={
                  "id": np.array([b"p0", b"p1", b"p2", b"p3"]),
                  "age": np.array([30, 0, 25, 40], dtype=np.int64),
                  "score": np.array([0.5, 0.0, -1.5, 2.0], dtype=np.float64),
                  "name": np.array([
                      "Zoë".encode("utf-8"),
                      b"",
                      "日本".encode("utf-8"),
                      b"isolated",
                  ]),
                  "active": np.array([True, False, False, True]),
                  "emb": _object_array([
                      np.array([1.0, 2.0]),
                      np.array([], dtype=np.float64),
                      np.array([3.0]),
                      np.array([4.0, 5.0, 6.0]),
                  ]),
                  "joined": np.array(
                      [1577836800000000, 0, 1623760200000000, 1551398400000000],
                      dtype=np.int64,
                  ),
              },
          ),
          "item": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=3, features={"id": np.array([b"i0", b"i1", b"i2"])}
          ),
      },
      edge_sets={
          "buys": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.array([[0, 0, 0, 1, 2], [0, 0, 1, 1, 2]]),
              features={
                  "weight": np.array(
                      [1.0, 2.0, 0.0, 3.0, 4.0], dtype=np.float64
                  )
              },
          ),
          "follows": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.array([[0, 1, 2, 2], [1, 2, 0, 2]]), features={}
          ),
      },
  )


def _expected_bytes_graph() -> in_memory_graph_lib.InMemoryGraph:
  """Returns the content of `dgf_sampler_bytes_graph`, NULLs as defaults."""
  return in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "node": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=3,
              features={
                  "id": np.array([b"\x00\xff", b"ascii", b"\xc3\x28"]),
                  "blob": np.array([b"\x01\x02", b"", b"\xfe"]),
              },
          ),
      },
      edge_sets={
          "edge": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.array([[0, 1, 2], [1, 2, 0]]), features={}
          ),
      },
  )


def _expected_int_graph() -> in_memory_graph_lib.InMemoryGraph:
  """Returns the content of `dgf_sampler_int_graph`."""
  return in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "node": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=2, features={"id": np.array([1, 2], dtype=np.int64)}
          ),
      },
      edge_sets={
          "edge": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.array([[0], [1]]), features={}
          ),
      },
  )


def _sorted_edges(
    edgeset: in_memory_graph_lib.InMemoryEdgeSet,
) -> list[tuple[object, ...]]:
  """Returns the edges as sorted `(source, target, *features)` tuples."""
  feature_names = sorted(edgeset.features)
  return sorted(
      (
          int(edgeset.adjacency[0, index]),
          int(edgeset.adjacency[1, index]),
          *(edgeset.features[name][index].item() for name in feature_names),
      )
      for index in range(edgeset.adjacency.shape[1])
  )


def expected_schema() -> schema_lib.GraphSchema:
  return schema_lib.GraphSchema(
      node_sets={
          "nodes": schema_lib.NodeSchema(
              features={
                  "feat": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_64,
                      shape=(None,),
                  ),
                  "id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "labels": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                  ),
                  "year": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                  ),
              }
          )
      },
      edge_sets={
          "edges": schema_lib.EdgeSchema(
              source="nodes",
              target="nodes",
          )
      },
  )


class BigqueryGraphManualTest(absltest.TestCase):

  def test_read_bigquery_graph(self):
    graph, schema = bigquery_graph.read_bigquery_graph(
        project="biggraphs-poc",
        dataset="arxiv",
        graph="arvix_graph",
        work_dir="gs://gbm-test/tmp",
        verbose=2,
    )
    in_memory_graph_validate_lib.validate_graph(
        graph, schema, raise_on_warning=False
    )
    test_util.assert_are_equal(self, schema, expected_schema())

  def test_export_bigquery_graph_to_disk(self):
    bigquery_graph.export_bigquery_graph_to_disk(
        path="gs://gbm-test/graph_export",
        project="biggraphs-poc",
        dataset="arxiv",
        graph="arvix_graph",
        verbose=2,
    )
    graph, schema = gf_graph_in_memory_lib.read_graph(
        "gs://gbm-test/graph_export"
    )
    in_memory_graph_validate_lib.validate_graph(
        graph, schema, raise_on_warning=False
    )
    test_util.assert_are_equal(self, schema, expected_schema())


class ExportCornerCaseGraphsTest(parameterized.TestCase):

  @parameterized.parameters(
      ("dgf_sampler_string_graph", _expected_string_graph),
      ("dgf_sampler_bytes_graph", _expected_bytes_graph),
      ("dgf_sampler_int_graph", _expected_int_graph),
  )
  def test_export_bigquery_graph_to_disk(self, graph_name, expected_graph_fn):
    export_path = os.path.join(_TEST_GRAPHS_GCS_EXPORT_PATH, graph_name)
    bigquery_graph.export_bigquery_graph_to_disk(
        path=export_path,
        project=_PROJECT,
        dataset=_TEST_GRAPHS_DATASET,
        graph=graph_name,
        verbose=0,
    )
    graph, schema = gf_graph_in_memory_lib.read_graph(export_path)
    in_memory_graph_validate_lib.validate_graph(
        graph, schema, raise_on_warning=False
    )

    expected_graph = expected_graph_fn()
    canonical_graph = canonicalize_lib.canonicalize_graph(graph, schema)
    test_util.assert_are_equal(
        self, canonical_graph.node_sets, expected_graph.node_sets
    )
    self.assertEqual(
        canonical_graph.edge_sets.keys(), expected_graph.edge_sets.keys()
    )
    for edgeset_name, expected_edgeset in expected_graph.edge_sets.items():
      self.assertEqual(
          _sorted_edges(canonical_graph.edge_sets[edgeset_name]),
          _sorted_edges(expected_edgeset),
          msg=edgeset_name,
      )

  def test_export_composite_key_raises(self):
    with self.assertRaisesRegex(NotImplementedError, "composite key"):
      bigquery_graph.export_bigquery_graph_to_disk(
          path=os.path.join(_TEST_GRAPHS_GCS_EXPORT_PATH, "composite"),
          project=_PROJECT,
          dataset=_TEST_GRAPHS_DATASET,
          graph="dgf_sampler_composite_graph",
          verbose=0,
      )


if __name__ == "__main__":
  absltest.main()

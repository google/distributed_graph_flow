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

"""Unit tests for combine_graph_snapshots."""

from collections.abc import Mapping, Sequence
import os
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import graph_snapshots_metadata as data_snapshot_metadata
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_constants
from dgf.src.io import graph_in_memory
from dgf.src.io import graph_snapshots as graph_snapshots_io
from dgf.src.io import graph_snapshots_metadata as io_snapshot_metadata
from dgf.src.io import schema as io_schema
from dgf.src.transform import combine_graph_snapshots as combine_lib
from dgf.src.util import filesystem
from dgf.src.util import log
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np

test_util.disable_diff_truncation()


_FORMAT = schema_lib.FeatureFormat
_SEMANTIC = schema_lib.FeatureSemantic


def _occ(
    pairs_list: Sequence[Sequence[tuple[int, int]]],
) -> list[combine_lib._EntityOccurrences]:
  """Constructs a list of _EntityOccurrences from pairs for testing."""
  return [
      combine_lib._EntityOccurrences.from_pairs(p)
      for p in pairs_list
  ]


def _read_and_combine(
    dataset_path: str, verbose: bool = False
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Reads a snapshot dataset from disk and combines it into a temporal graph.

  Mirrors `dgf.io.read_graph_snapshots_as_temporal_graph`, but spelled out here
  so these tests exercise `combine_graph_snapshots` rather than the io wrapper.
  """
  snapshots = graph_snapshots_io.read_graph_snapshots(
      dataset_path, verbose=verbose
  )
  return combine_lib.combine_graph_snapshots(snapshots, verbose=verbose)


# The schema builders below are deliberately kept separate from
# `_create_graph`: the schema is an input under test, not a property of the
# data. Several tests declare a schema the values alone cannot imply, such as a
# timeseries feature over flat values or a reserved feature name.


def _feature(
    fmt: schema_lib.FeatureFormat,
    semantic: schema_lib.FeatureSemantic = schema_lib.FeatureSemantic.UNKNOWN,
    **kwargs: Any,
) -> schema_lib.FeatureSchema:
  """Shorthand for a FeatureSchema, e.g. `_feature(_FORMAT.BYTES)`."""
  return schema_lib.FeatureSchema(format=fmt, semantic=semantic, **kwargs)


def _node_schema(
    primary_id_key: str = "#id", **features: schema_lib.FeatureSchema
) -> schema_lib.NodeSchema:
  """Builds a node schema with a bytes primary id plus the given features."""
  return schema_lib.NodeSchema(
      features={
          primary_id_key: _feature(_FORMAT.BYTES, _SEMANTIC.PRIMARY_ID),
          **features,
      }
  )


def _edge_schema(
    source: str = "n1",
    target: str = "n1",
    features: Mapping[str, schema_lib.FeatureSchema] | None = None,
    **extra_features: schema_lib.FeatureSchema,
) -> schema_lib.EdgeSchema:
  """Builds an edge schema between the `source` and `target` node sets."""
  return schema_lib.EdgeSchema(
      source=source,
      target=target,
      features={**(features or {}), **extra_features},
  )


def _graph_schema(
    node_set: schema_lib.NodeSchema,
    edge_set: schema_lib.EdgeSchema | None = None,
    node_set_name: str = "n1",
    edge_set_name: str = "e1",
) -> schema_lib.GraphSchema:
  """Builds a graph schema with one node set and an optional edge set."""
  return schema_lib.GraphSchema(
      node_sets={node_set_name: node_set},
      edge_sets={edge_set_name: edge_set} if edge_set is not None else {},
  )


def _create_graph(
    ids: Sequence[Any],
    f1: np.ndarray | Sequence[Any] | None = None,
    edges: np.ndarray | Sequence[Sequence[int]] | None = None,
    edge_features: Mapping[str, np.ndarray] | None = None,
    timestamp: int | None = None,
    primary_id_key: str = "#id",
    node_set_name: str = "n1",
    edge_set_name: str = "e1",
    **extra_node_features: Any,
) -> in_memory_graph_lib.InMemoryGraph:
  """Builds a simple in-memory graph for testing."""
  node_features: dict[str, np.ndarray] = {
      primary_id_key: np.array(ids, dtype=np.bytes_)
  }
  if f1 is not None:
    node_features["f1"] = np.array(f1, dtype=np.float32)
  for k, v in extra_node_features.items():
    node_features[k] = v

  edge_sets = {}
  if edges is not None:
    ef: dict[str, np.ndarray] = {}
    if edge_features is not None:
      ef.update(edge_features)
    edge_sets[edge_set_name] = in_memory_graph_lib.InMemoryEdgeSet(
        adjacency=np.array(edges, dtype=np.int64),
        features=ef,
    )

  return in_memory_graph_lib.InMemoryGraph(
      timestamp=timestamp,
      node_sets={
          node_set_name: in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=len(ids), features=node_features
          )
      },
      edge_sets=edge_sets,
  )


class CombineGraphSnapshotsTest(parameterized.TestCase):

  def _write_dataset(
      self,
      root_dir: str,
      schema: schema_lib.GraphSchema,
      snapshots: Sequence[tuple[str, in_memory_graph_lib.InMemoryGraph]],
  ):
    """Writes metadata.json, schema.json and snapshot graphs to root_dir."""
    meta_path = os.path.join(root_dir, graph_constants.FILENAME_METADATA)
    if not filesystem.exists(meta_path):
      metadata = data_snapshot_metadata.GraphSnapshotsMetadata(
          format=data_snapshot_metadata.GraphSnapshotsFormat.GRAPH_SNAPSHOTS,
          version=0,
      )
      io_snapshot_metadata.write_metadata(metadata, meta_path)
    io_schema.write_schema(
        schema, os.path.join(root_dir, graph_constants.FILENAME_SCHEMA)
    )
    for name, g in snapshots:
      graph_in_memory.write_graph(
          g,
          schema,
          path=os.path.join(root_dir, "snapshots", name),
          verbose=False,
      )

  def _create_sample_dataset(
      self,
      root_dir: str,
      with_edge_features: bool = False,
      include_duplicate: bool = False,
  ):
    """Creates a sample Snapshot Graph dataset with 3 snapshots."""
    edge_features = {}
    if with_edge_features:
      edge_features = {
          "f1": _feature(_FORMAT.INTEGER_64, _SEMANTIC.CATEGORICAL),
          "f2": _feature(_FORMAT.FLOAT_32, _SEMANTIC.NUMERICAL),
      }

    schema = _graph_schema(
        _node_schema(
            f1=_feature(_FORMAT.FLOAT_32, _SEMANTIC.NUMERICAL),
            f2=_feature(_FORMAT.BYTES, _SEMANTIC.CATEGORICAL),
        ),
        _edge_schema(features=edge_features),
    )

    snapshots_data = [
        (
            "snap_beta",
            100,
            [b"1", b"2"],
            [1.0, 2.0],
            [b"a", b"b"],
            [[0], [1]],
            [5],
            [1.0],
        ),
        (
            "snap_alpha",
            100 if include_duplicate else 200,
            [b"1", b"3"],
            [1.5, 3.0],
            [b"a", b"c"],
            [[0], [1]],
            [6],
            [2.0],
        ),
        (
            "snap_gamma",
            300,
            [b"1", b"2", b"3"],
            [2.0, 2.5, 3.5],
            [b"a", b"b", b"c"],
            [[0, 0], [1, 2]],
            [5, 6],
            [3.0, 4.0],
        ),
    ]

    snapshots = []
    for (
        name,
        ts,
        ids,
        f1_vals,
        f2_vals,
        edges,
        e_f1_vals,
        e_f2_vals,
    ) in snapshots_data:
      ef = None
      if with_edge_features:
        ef = {
            "f1": np.array(e_f1_vals, dtype=np.int64),
            "f2": np.array(e_f2_vals, dtype=np.float32),
        }
      g = _create_graph(
          ids,
          f1=f1_vals,
          edges=edges,
          edge_features=ef,
          f2=np.array(f2_vals, dtype=np.bytes_),
          timestamp=ts,
          node_set_name="n1",
          edge_set_name="e1",
      )
      snapshots.append((name, g))

    self._write_dataset(root_dir, schema, snapshots)

  def test_build_occurrence_transitions(self):
    occurrences = _occ([
        [(0, 0), (1, 2)],
        [(0, 1)],
        [(0, 2), (1, 0), (2, 1)],
    ])
    transitions = combine_lib._build_occurrence_transitions(occurrences)

    self.assertIn((0, 1), transitions)
    self.assertIn((1, 2), transitions)

    r1_01, r2_01 = transitions[(0, 1)]
    np.testing.assert_array_equal(r1_01, np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(r2_01, np.array([2, 0], dtype=np.int64))

    r1_12, r2_12 = transitions[(1, 2)]
    np.testing.assert_array_equal(r1_12, np.array([0], dtype=np.int64))
    np.testing.assert_array_equal(r2_12, np.array([1], dtype=np.int64))

  @parameterized.named_parameters(
      ("static_int", "static_int", np.int64, True),
      ("dynamic_int", "dynamic_int", np.int64, False),
      ("static_bytes", "static_bytes", np.bytes_, True),
      ("dynamic_bytes", "dynamic_bytes", np.bytes_, False),
      ("static_nan", "static_nan", np.float32, True),
      ("dynamic_nan", "dynamic_nan", np.float32, False),
      ("static_2d", "static_2d", np.int32, True),
      ("dynamic_2d", "dynamic_2d", np.int32, False),
      ("static_2d_bytes", "static_2d_bytes", np.bytes_, True),
      ("dynamic_2d_bytes", "dynamic_2d_bytes", np.bytes_, False),
  )
  def test_is_feature_static_vectorized(
      self, feature_name, feature_dtype, expected_static
  ):
    occurrences = _occ([
        [(0, 0), (1, 1)],
        [(0, 1), (1, 0)],
        [(0, 2)],
    ])
    snapshot_features = [
        {
            "static_int": np.array([10, 20, 30], dtype=np.int64),
            "dynamic_int": np.array([1, 2, 3], dtype=np.int64),
            "static_bytes": np.array([b"x", b"y", b"z"], dtype=np.bytes_),
            "dynamic_bytes": np.array([b"a", b"b", b"c"], dtype=np.bytes_),
            "static_nan": np.array([np.nan, 2.0, 3.0], dtype=np.float32),
            "dynamic_nan": np.array([np.nan, 2.0, 3.0], dtype=np.float32),
            "static_2d": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
            "dynamic_2d": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
            "static_2d_bytes": np.array(
                [[b"p", b"q"], [b"r", b"s"], [b"t", b"u"]], dtype=np.bytes_
            ),
            "dynamic_2d_bytes": np.array(
                [[b"p", b"q"], [b"r", b"s"], [b"t", b"u"]], dtype=np.bytes_
            ),
        },
        {
            "static_int": np.array([20, 10], dtype=np.int64),
            "dynamic_int": np.array([99, 1], dtype=np.int64),
            "static_bytes": np.array([b"y", b"x"], dtype=np.bytes_),
            "dynamic_bytes": np.array([b"b", b"diff"], dtype=np.bytes_),
            "static_nan": np.array([2.0, np.nan], dtype=np.float32),
            "dynamic_nan": np.array([2.0, 5.0], dtype=np.float32),
            "static_2d": np.array([[3, 4], [1, 2]], dtype=np.int32),
            "dynamic_2d": np.array([[3, 99], [1, 2]], dtype=np.int32),
            "static_2d_bytes": np.array(
                [[b"r", b"s"], [b"p", b"q"]], dtype=np.bytes_
            ),
            "dynamic_2d_bytes": np.array(
                [[b"r", b"CHANGED"], [b"p", b"q"]], dtype=np.bytes_
            ),
        },
    ]

    transitions = combine_lib._build_occurrence_transitions(occurrences)

    self.assertEqual(
        combine_lib._is_feature_static(
            feature_name,
            feature_dtype,
            snapshot_features,
            occurrences,
            transitions,
        ),
        expected_static,
    )

  def test_is_feature_static_bytes_ignores_itemsize(self):
    # A snapshot's bytes column is only as wide as its longest value, so the
    # same value can have a different dtype in each snapshot. Comparison must
    # be by value, not by dtype.
    occurrences = _occ([[(0, 0), (1, 0)]])
    snapshot_features = [
        {"f": np.array([b"abc"], dtype="S3")},
        {"f": np.array([b"abc", b"longer"], dtype="S6")},
    ]
    self.assertEqual(snapshot_features[0]["f"].dtype, np.dtype("S3"))
    self.assertEqual(snapshot_features[1]["f"].dtype, np.dtype("S6"))

    self.assertTrue(
        combine_lib._is_feature_static(
            "f", np.bytes_, snapshot_features, occurrences
        )
    )

  def test_is_feature_static_object_array_of_scalars(self):
    occurrences = _occ([[(0, 0), (1, 0)]])
    static_features = []
    for _ in range(2):
      values = np.empty(1, dtype=object)
      values[0] = b"x"
      static_features.append({"f": values})

    dynamic_features = []
    for value in (b"x", b"y"):
      values = np.empty(1, dtype=object)
      values[0] = value
      dynamic_features.append({"f": values})

    self.assertTrue(
        combine_lib._is_feature_static(
            "f", object, static_features, occurrences
        )
    )
    self.assertFalse(
        combine_lib._is_feature_static(
            "f", object, dynamic_features, occurrences
        )
    )

  def test_is_feature_static_object_array_of_subarrays_raises_error(self):
    # `np.array_equal` cannot reduce an object array whose elements are
    # themselves arrays. `combine_graph_snapshots` rejects timeseries input
    # schemas, so this is not reachable through the public API, but it fails
    # loudly rather than silently if it ever becomes reachable.
    occurrences = _occ([[(0, 0), (1, 0)]])
    snapshot_features = []
    for _ in range(2):
      values = np.empty(1, dtype=object)
      values[0] = np.array([1, 2], dtype=np.int64)
      snapshot_features.append({"f": values})

    with self.assertRaisesRegex(ValueError, "truth value of an array"):
      combine_lib._is_feature_static(
          "f", object, snapshot_features, occurrences
      )

  def test_build_static_features_preserves_bytes_width(self):
    features_schema = {"name": _feature(_FORMAT.BYTES)}
    snapshot_features = [
        {"name": np.array([b"alpha", b"beta"], dtype=np.bytes_)}
    ]
    occurrences = _occ([[(0, 0)], [(0, 1)]])

    features, _ = combine_lib._build_static_features(
        2, features_schema, snapshot_features, occurrences
    )

    np.testing.assert_array_equal(
        features["name"], np.array([b"alpha", b"beta"], dtype=np.bytes_)
    )

  def test_build_static_features_multidimensional(self):
    features_schema = {"vec": _feature(_FORMAT.INTEGER_64, shape=(3,))}
    snapshot_features = [
        {"vec": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)}
    ]
    occurrences = _occ([[(0, 0)], [(0, 1)]])

    features, _ = combine_lib._build_static_features(
        2, features_schema, snapshot_features, occurrences
    )

    np.testing.assert_array_equal(
        features["vec"], np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    )

  def test_build_timeseries_features_preserves_bytes_width(self):
    features_schema = {"name": _feature(_FORMAT.BYTES)}
    snapshot_features = [
        {"name": np.array([b"alpha"], dtype=np.bytes_)},
        {"name": np.array([b"gamma"], dtype=np.bytes_)},
    ]
    occurrences = _occ([[(0, 0), (1, 0)]])

    features, _ = combine_lib._build_timeseries_features(
        "n1", 1, features_schema, snapshot_features, occurrences,
        np.array([100, 200], dtype=np.int64), reserved_feature_names=(),
    )

    np.testing.assert_array_equal(
        features["name"][0], np.array([b"alpha", b"gamma"], dtype=np.bytes_)
    )

  def test_build_timeseries_features_multidimensional(self):
    features_schema = {"vec": _feature(_FORMAT.INTEGER_64, shape=(2,))}
    snapshot_features = [
        {"vec": np.array([[1, 2]], dtype=np.int64)},
        {"vec": np.array([[3, 4]], dtype=np.int64)},
    ]
    occurrences = _occ([[(0, 0), (1, 0)]])

    features, _ = combine_lib._build_timeseries_features(
        "n1", 1, features_schema, snapshot_features, occurrences,
        np.array([100, 200], dtype=np.int64), reserved_feature_names=(),
    )

    np.testing.assert_array_equal(
        features["vec"][0], np.array([[1, 2], [3, 4]], dtype=np.int64)
    )

  def test_build_timeseries_features_no_occurrences(self):
    features_schema = {"vec": _feature(_FORMAT.INTEGER_64, shape=(2,))}

    features, _ = combine_lib._build_timeseries_features(
        "n1", 1, features_schema, [], _occ([[]]),
        np.array([], dtype=np.int64), reserved_feature_names=(),
    )

    self.assertEqual(features["vec"][0].shape, (0, 2))

  def test_build_timeseries_features_unexpected_element_shape_raises_error(
      self,
  ):
    # A scalar-declared feature whose stored values are vectors must not be
    # silently widened into an extra dimension.
    features_schema = {"f": _feature(_FORMAT.INTEGER_64)}
    snapshot_features = [
        {"f": np.array([[1, 2]], dtype=np.int64)},
        {"f": np.array([[3, 4]], dtype=np.int64)},
    ]
    occurrences = _occ([[(0, 0), (1, 0)]])

    with self.assertRaises(ValueError):
      combine_lib._build_timeseries_features(
          "n1", 1, features_schema, snapshot_features, occurrences,
          np.array([100, 200], dtype=np.int64), reserved_feature_names=(),
      )

  def test_process_node_set(self):
    ns_schema = _node_schema(
        f1=_feature(_FORMAT.FLOAT_32, _SEMANTIC.NUMERICAL),
        f2=_feature(_FORMAT.BYTES, _SEMANTIC.CATEGORICAL),
    )
    timestamps = np.array([100, 200, 300], dtype=np.int64)
    g1 = _create_graph(
        [b"1", b"2", b"4"],
        f1=[1.0, 2.0, 4.0],
        f2=np.array([b"a", b"b", b"d"], dtype=np.bytes_),
    )
    g2 = _create_graph(
        [b"1", b"3"],
        f1=[1.5, 3.0],
        f2=np.array([b"a", b"c"], dtype=np.bytes_),
    )
    g3 = _create_graph(
        [b"1", b"2", b"3"],
        f1=[2.0, 2.5, 3.5],
        f2=np.array([b"a", b"b", b"c"], dtype=np.bytes_),
    )

    node_set, node_schema = combine_lib._process_node_set(
        "n1", ns_schema, [g1, g2, g3], timestamps
    )

    self.assertEqual(node_set.num_nodes, 4)
    node_ids = list(node_set.features["#id"])
    n1_idx = node_ids.index(b"1")
    n2_idx = node_ids.index(b"2")
    n3_idx = node_ids.index(b"3")
    n4_idx = node_ids.index(b"4")

    # Lifespan intervals
    self.assertEqual(node_set.features["creation_time"][n1_idx], 100)
    np.testing.assert_array_equal(
        node_set.features["deletion_time"][n1_idx], np.array([], dtype=np.int64)
    )

    # 2 reappears at 300: coalesced into single timeseries node, not deleted.
    self.assertEqual(node_set.features["creation_time"][n2_idx], 100)
    np.testing.assert_array_equal(
        node_set.features["deletion_time"][n2_idx],
        np.array([], dtype=np.int64),
    )

    self.assertEqual(node_set.features["creation_time"][n3_idx], 200)
    np.testing.assert_array_equal(
        node_set.features["deletion_time"][n3_idx], np.array([], dtype=np.int64)
    )

    # 4 only appears at 100: deleted at 200.
    self.assertEqual(node_set.features["creation_time"][n4_idx], 100)
    np.testing.assert_array_equal(
        node_set.features["deletion_time"][n4_idx],
        np.array([200], dtype=np.int64),
    )

    # Static feature assertions
    self.assertFalse(node_schema.features["f2"].is_timeseries)
    self.assertIsNone(node_schema.features["f2"].group)
    self.assertIsNone(node_schema.features["f2"].shape)
    self.assertEqual(node_set.features["f2"][n1_idx], b"a")
    self.assertEqual(node_set.features["f2"][n2_idx], b"b")
    self.assertEqual(node_set.features["f2"][n3_idx], b"c")
    self.assertEqual(node_set.features["f2"][n4_idx], b"d")

    # f1 timeseries
    np.testing.assert_allclose(
        node_set.features["f1"][n1_idx],
        np.array([1.0, 1.5, 2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        node_set.features["f1"][n2_idx],
        np.array([2.0, 2.5], dtype=np.float32),
    )
    np.testing.assert_allclose(
        node_set.features["f1"][n3_idx],
        np.array([3.0, 3.5], dtype=np.float32),
    )
    np.testing.assert_allclose(
        node_set.features["f1"][n4_idx],
        np.array([4.0], dtype=np.float32),
    )

    # Timestamps
    np.testing.assert_array_equal(
        node_set.features["n1_group_timestamp"][n1_idx],
        np.array([100, 200, 300], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        node_set.features["n1_group_timestamp"][n2_idx],
        np.array([100, 300], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        node_set.features["n1_group_timestamp"][n3_idx],
        np.array([200, 300], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        node_set.features["n1_group_timestamp"][n4_idx],
        np.array([100], dtype=np.int64),
    )

    self.assertTrue(node_schema.features["creation_time"].is_creation_time)
    self.assertTrue(node_schema.features["deletion_time"].is_timeseries)
    self.assertTrue(node_schema.features["deletion_time"].is_creation_time)
    self.assertEqual(
        node_schema.features["deletion_time"].group, "deletion_time"
    )
    self.assertTrue(node_schema.features["f1"].is_timeseries)
    self.assertEqual(node_schema.features["f1"].shape, (None,))
    self.assertEqual(node_schema.features["f1"].group, "n1_group")
    self.assertTrue(
        node_schema.features["n1_group_timestamp"].is_timeseries
    )
    self.assertTrue(
        node_schema.features["n1_group_timestamp"].is_creation_time
    )
    self.assertEqual(
        node_schema.features["n1_group_timestamp"].semantic,
        schema_lib.FeatureSemantic.TIMESTAMP,
    )
    self.assertEqual(
        node_schema.features["n1_group_timestamp"].format,
        schema_lib.FeatureFormat.INTEGER_64,
    )
    self.assertEqual(
        node_schema.features["n1_group_timestamp"].group, "n1_group"
    )

  def test_process_node_set_multiple_features(self):
    ns_schema = _node_schema(
        f1=_feature(_FORMAT.FLOAT_32, _SEMANTIC.NUMERICAL),
        f2=_feature(_FORMAT.FLOAT_32, _SEMANTIC.NUMERICAL),
    )
    timestamps = np.array([100, 200, 300], dtype=np.int64)
    g1 = _create_graph(
        [b"1", b"2"],
        f1=np.array([1.0, 2.0], dtype=np.float32),
        f2=np.array([10.0, 20.0], dtype=np.float32),
    )
    g2 = _create_graph(
        [b"1", b"2"],
        f1=np.array([1.5, 2.5], dtype=np.float32),
        f2=np.array([15.0, 25.0], dtype=np.float32),
    )
    g3 = _create_graph(
        [b"1", b"2"],
        f1=np.array([2.0, 3.0], dtype=np.float32),
        f2=np.array([20.0, 30.0], dtype=np.float32),
    )

    node_set, node_schema = combine_lib._process_node_set(
        "n1", ns_schema, [g1, g2, g3], timestamps
    )

    self.assertEqual(node_set.num_nodes, 2)
    np.testing.assert_allclose(
        node_set.features["f1"][0],
        np.array([1.0, 1.5, 2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        node_set.features["f2"][0],
        np.array([10.0, 15.0, 20.0], dtype=np.float32),
    )

    self.assertIn("n1_group_timestamp", node_schema.features)
    self.assertEqual(node_schema.features["f1"].group, "n1_group")
    self.assertEqual(node_schema.features["f2"].group, "n1_group")
    self.assertEqual(
        node_schema.features["n1_group_timestamp"].group, "n1_group"
    )

    np.testing.assert_array_equal(
        node_set.features["n1_group_timestamp"][0],
        np.array([100, 200, 300], dtype=np.int64),
    )

  def test_process_node_set_timestamp_name_collision(self):
    ns_schema = _node_schema(
        f1=_feature(_FORMAT.FLOAT_32, _SEMANTIC.NUMERICAL),
        n1_group_timestamp=_feature(_FORMAT.FLOAT_32, _SEMANTIC.NUMERICAL),
    )
    timestamps = np.array([100, 200], dtype=np.int64)
    g1 = _create_graph(
        [b"1"],
        f1=[1.0],
        n1_group_timestamp=np.array([5.0], dtype=np.float32),
    )
    g2 = _create_graph(
        [b"1"],
        f1=[2.0],
        n1_group_timestamp=np.array([6.0], dtype=np.float32),
    )

    with self.assertRaisesRegex(
        ValueError,
        "Timestamp feature name 'n1_group_timestamp' conflicts with an"
        " existing",
    ):
      combine_lib._process_node_set(
          "n1", ns_schema, [g1, g2], timestamps
      )

  def test_process_node_set_custom_primary_id_name(self):
    ns_schema = _node_schema(
        primary_id_key="custom_id",
        f1=_feature(_FORMAT.FLOAT_32, _SEMANTIC.NUMERICAL),
    )
    timestamps = np.array([100, 200], dtype=np.int64)
    g1 = _create_graph([b"1"], f1=[1.0], primary_id_key="custom_id")
    g2 = _create_graph([b"1"], f1=[2.0], primary_id_key="custom_id")

    node_set, _ = combine_lib._process_node_set(
        "n1", ns_schema, [g1, g2], timestamps
    )

    self.assertEqual(node_set.num_nodes, 1)
    self.assertIn("custom_id", node_set.features)
    self.assertNotIn("#id", node_set.features)
    self.assertEqual(node_set.features["custom_id"][0], b"1")
    np.testing.assert_allclose(
        node_set.features["f1"][0], np.array([1.0, 2.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        node_set.features["n1_group_timestamp"][0],
        np.array([100, 200], dtype=np.int64),
    )

  def test_process_node_set_multidimensional_feature(self):
    ns_schema = _node_schema(
        f1=_feature(_FORMAT.FLOAT_32, _SEMANTIC.EMBEDDING, shape=(2,)),
        f2=_feature(_FORMAT.FLOAT_32, _SEMANTIC.EMBEDDING, shape=(2,)),
    )
    timestamps = np.array([100, 200], dtype=np.int64)
    g1 = _create_graph(
        [b"1"],
        f1=np.array([[1.0, 2.0]], dtype=np.float32),
        f2=np.array([[1.0, 2.0]], dtype=np.float32),
    )
    g2 = _create_graph(
        [b"1"],
        f1=np.array([[1.0, 2.0]], dtype=np.float32),
        f2=np.array([[3.0, 4.0]], dtype=np.float32),
    )

    node_set, node_schema = combine_lib._process_node_set(
        "n1", ns_schema, [g1, g2], timestamps
    )
    self.assertFalse(node_schema.features["f1"].is_timeseries)
    self.assertEqual(node_schema.features["f1"].shape, (2,))
    np.testing.assert_allclose(
        node_set.features["f1"],
        np.array([[1.0, 2.0]], dtype=np.float32),
    )
    self.assertTrue(node_schema.features["f2"].is_timeseries)
    self.assertEqual(node_schema.features["f2"].shape, (None, 2))
    np.testing.assert_allclose(
        node_set.features["f2"][0],
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )

  def test_extract_edge_intervals_gap_splits_into_new_edge(self):
    # Locks in the headline semantic: an edge that disappears and reappears
    # becomes a *new* edge rather than one edge with a hole in it.
    timestamps = np.array([100, 200, 300], dtype=np.int64)
    pair_timeline = {
        (0, 1): {0: 0, 2: 1},  # gapped: at t0 and t2, missing at t1
        (2, 3): {0: 1, 1: 0, 2: 0},  # present in every snapshot
        (4, 5): {0: 2},  # present only at t0, never returns
    }

    adj, creation, deletion, occurrences = combine_lib._extract_edge_intervals(
        pair_timeline, timestamps
    )

    # The gapped pair yields two intervals; the other pairs yield one each.
    test_util.assert_are_equal(
        self, adj, np.array([[0, 0, 2, 4], [1, 1, 3, 5]], dtype=np.int64)
    )
    test_util.assert_are_equal(
        self, creation, np.array([100, 300, 100, 100], dtype=np.int64)
    )

    # deletion_time is a length-0-or-1 timeseries: the gapped edge's first
    # interval dies at t1 (the snapshot it went missing), its second interval
    # survives to the end and so has no deletion time at all.
    self.assertEqual(
        [d.tolist() for d in deletion], [[200], [], [], [200]]
    )

    # Each interval keeps only the occurrences belonging to that interval.
    self.assertEqual(
        [list(zip(o.snapshot_indices, o.row_indices)) for o in occurrences],
        [[(0, 0)], [(2, 1)], [(0, 1), (1, 0), (2, 0)], [(0, 2)]],
    )

  def test_process_edge_set(self):
    edge_set_schema = _edge_schema(
        f1=_feature(_FORMAT.INTEGER_64, _SEMANTIC.CATEGORICAL),
        f2=_feature(_FORMAT.FLOAT_32, _SEMANTIC.NUMERICAL),
    )
    timestamps = np.array([100, 200, 300], dtype=np.int64)
    temp_n1_ns = in_memory_graph_lib.InMemoryNodeSet(
        num_nodes=3,
        features={"#id": np.array([b"1", b"2", b"3"], dtype=np.bytes_)},
    )
    temporal_node_sets = {"n1": temp_n1_ns}
    node_schemas = {"n1": _node_schema()}

    g1 = _create_graph(
        [b"1", b"2"],
        edges=[[0], [1]],
        edge_features={
            "f1": np.array([5], dtype=np.int64),
            "f2": np.array([1.0], dtype=np.float32),
        },
    )
    g2 = _create_graph(
        [b"1", b"3"],
        edges=[[0], [1]],
        edge_features={
            "f1": np.array([6], dtype=np.int64),
            "f2": np.array([2.0], dtype=np.float32),
        },
    )
    g3 = _create_graph(
        [b"1", b"2", b"3"],
        edges=[[0, 0], [1, 2]],
        edge_features={
            "f1": np.array([5, 6], dtype=np.int64),
            "f2": np.array([3.0, 4.0], dtype=np.float32),
        },
    )

    edge_set, edge_schema = combine_lib._process_edge_set(
        "e1",
        edge_set_schema,
        [g1, g2, g3],
        temporal_node_sets,
        timestamps,
        node_schemas=node_schemas,
    )

    # Edge 1->2 is present at t0 and t2 but not t1, so it splits into two
    # intervals; edge 1->3 spans t1-t2 contiguously for one. Hence 3.
    self.assertEqual(edge_set.adjacency.shape[1], 3)
    self.assertTrue(edge_schema.features["creation_time"].is_creation_time)
    self.assertTrue(edge_schema.features["deletion_time"].is_timeseries)
    self.assertTrue(edge_schema.features["deletion_time"].is_creation_time)
    self.assertEqual(
        edge_schema.features["deletion_time"].group, "deletion_time"
    )

    # f1 is static
    self.assertFalse(edge_schema.features["f1"].is_timeseries)
    self.assertIsNone(edge_schema.features["f1"].group)
    self.assertEqual(edge_set.features["f1"].ndim, 1)

    # f2 is dynamic
    self.assertTrue(edge_schema.features["f2"].is_timeseries)
    self.assertEqual(edge_schema.features["f2"].group, "e1_group")
    self.assertTrue(
        edge_schema.features["e1_group_timestamp"].is_timeseries
    )

  def test_process_edge_set_heterogeneous(self):
    edge_set_schema = _edge_schema(
        source="n1",
        target="n2",
        f1=_feature(_FORMAT.INTEGER_64, _SEMANTIC.CATEGORICAL),
    )
    timestamps = np.array([100, 200], dtype=np.int64)
    temporal_node_sets = {
        "n1": in_memory_graph_lib.InMemoryNodeSet(
            num_nodes=2,
            features={"#id": np.array([b"src_a", b"src_b"], dtype=np.bytes_)},
        ),
        "n2": in_memory_graph_lib.InMemoryNodeSet(
            num_nodes=2,
            features={"#id": np.array([b"dst_x", b"dst_y"], dtype=np.bytes_)},
        ),
    }
    node_schemas = {"n1": _node_schema(), "n2": _node_schema()}

    g1 = in_memory_graph_lib.InMemoryGraph(
        timestamp=100,
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "#id": np.array([b"src_a", b"src_b"], dtype=np.bytes_)
                },
            ),
            "n2": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=2,
                features={
                    "#id": np.array([b"dst_y", b"dst_x"], dtype=np.bytes_)
                },
            ),
        },
        edge_sets={
            "e1": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[1], [0]], dtype=np.int64),
                features={"f1": np.array([42], dtype=np.int64)},
            )
        },
    )
    g2 = in_memory_graph_lib.InMemoryGraph(
        timestamp=200,
        node_sets={
            "n1": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=1,
                features={"#id": np.array([b"src_b"], dtype=np.bytes_)},
            ),
            "n2": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=1,
                features={"#id": np.array([b"dst_y"], dtype=np.bytes_)},
            ),
        },
        edge_sets={
            "e1": in_memory_graph_lib.InMemoryEdgeSet(
                adjacency=np.array([[0], [0]], dtype=np.int64),
                features={"f1": np.array([42], dtype=np.int64)},
            )
        },
    )

    edge_set, edge_schema = combine_lib._process_edge_set(
        "e1",
        edge_set_schema,
        [g1, g2],
        temporal_node_sets,
        timestamps,
        node_schemas=node_schemas,
    )

    np.testing.assert_array_equal(
        edge_set.adjacency, np.array([[1], [1]], dtype=np.int64)
    )
    self.assertEqual(edge_schema.source, "n1")
    self.assertEqual(edge_schema.target, "n2")
    self.assertTrue(edge_schema.features["creation_time"].is_creation_time)
    np.testing.assert_array_equal(
        edge_set.features["f1"], np.array([42], dtype=np.int64)
    )

  def test_combine_graph_snapshots(self):
    dataset_dir = self.create_tempdir().full_path
    self._create_sample_dataset(dataset_dir)

    graph, schema = _read_and_combine(dataset_dir)

    n1_ns = graph.node_sets["n1"]
    self.assertEqual(n1_ns.num_nodes, 3)

    node_ids = list(n1_ns.features["#id"])
    self.assertCountEqual(node_ids, [b"1", b"2", b"3"])

    n1_idx = node_ids.index(b"1")
    n2_idx = node_ids.index(b"2")
    n3_idx = node_ids.index(b"3")

    # Node creation_time & deletion_time
    self.assertEqual(n1_ns.features["creation_time"][n1_idx], 100)
    np.testing.assert_array_equal(
        n1_ns.features["deletion_time"][n1_idx], np.array([], dtype=np.int64)
    )

    self.assertEqual(n1_ns.features["creation_time"][n2_idx], 100)
    np.testing.assert_array_equal(
        n1_ns.features["deletion_time"][n2_idx],
        np.array([], dtype=np.int64),
    )

    self.assertEqual(n1_ns.features["creation_time"][n3_idx], 200)
    np.testing.assert_array_equal(
        n1_ns.features["deletion_time"][n3_idx], np.array([], dtype=np.int64)
    )

    # Node f2 static feature
    self.assertFalse(schema.node_sets["n1"].features["f2"].is_timeseries)
    self.assertIsNone(schema.node_sets["n1"].features["f2"].group)
    self.assertIsNone(schema.node_sets["n1"].features["f2"].shape)
    self.assertEqual(n1_ns.features["f2"][n1_idx], b"a")
    self.assertEqual(n1_ns.features["f2"][n2_idx], b"b")
    self.assertEqual(n1_ns.features["f2"][n3_idx], b"c")

    # Node f1 timeseries
    np.testing.assert_allclose(
        n1_ns.features["f1"][n1_idx],
        np.array([1.0, 1.5, 2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        n1_ns.features["f1"][n2_idx],
        np.array([2.0, 2.5], dtype=np.float32),
    )
    np.testing.assert_allclose(
        n1_ns.features["f1"][n3_idx],
        np.array([3.0, 3.5], dtype=np.float32),
    )

    # Interval semantics are covered by
    # test_extract_edge_intervals_gap_splits_into_new_edge, and the per-set
    # schema flags by test_process_node_set / test_process_edge_set. Here we
    # only check the pieces are wired together and the result validates.
    self.assertIn("creation_time", schema.node_sets["n1"].features)
    self.assertIn("deletion_time", schema.node_sets["n1"].features)
    self.assertIn("creation_time", schema.edge_sets["e1"].features)
    self.assertIn("deletion_time", schema.edge_sets["e1"].features)
    self.assertEmpty(in_memory_graph_validate_lib.issues(graph, schema))

  def test_combine_graph_snapshots_verbose_logging(self):
    dataset_dir = self.create_tempdir().full_path
    self._create_sample_dataset(dataset_dir)

    with log.capture_logs(log_info=True) as logs:
      graph, schema = _read_and_combine(
          dataset_dir, verbose=True
      )

    self.assertNotEmpty(logs)
    log_texts = [msg.text for msg in logs]
    self.assertTrue(any("Discovering snapshots" in t for t in log_texts))
    self.assertTrue(any("Discovered 3 snapshot(s)" in t for t in log_texts))
    self.assertTrue(any("Successfully converted" in t for t in log_texts))
    self.assertEmpty(in_memory_graph_validate_lib.issues(graph, schema))

  def test_combine_graph_snapshots_with_dynamic_edge_features(self):
    dataset_dir = self.create_tempdir().full_path
    self._create_sample_dataset(dataset_dir, with_edge_features=True)

    graph, schema = _read_and_combine(dataset_dir)

    e1_es = graph.edge_sets["e1"]
    edge_set_schema = schema.edge_sets["e1"]

    self.assertIn("f1", e1_es.features)
    self.assertIn("f2", e1_es.features)
    self.assertIn("e1_group_timestamp", e1_es.features)
    self.assertIn("creation_time", e1_es.features)
    self.assertIn("deletion_time", e1_es.features)

    self.assertFalse(edge_set_schema.features["f1"].is_timeseries)
    self.assertIsNone(edge_set_schema.features["f1"].group)
    self.assertEqual(e1_es.features["f1"].ndim, 1)

    self.assertTrue(edge_set_schema.features["f2"].is_timeseries)
    self.assertEqual(edge_set_schema.features["f2"].group, "e1_group")
    self.assertEqual(
        edge_set_schema.features["e1_group_timestamp"].group, "e1_group"
    )
    self.assertEmpty(in_memory_graph_validate_lib.issues(graph, schema))

  def test_add_interval_schemas_reserved_collision_raises_error(self):
    features = {
        "creation_time": _feature(_FORMAT.INTEGER_64, _SEMANTIC.TIMESTAMP)
    }
    with self.assertRaisesRegex(
        ValueError,
        "Feature name 'creation_time' is reserved for temporal interval"
        " tracking",
    ):
      combine_lib._add_interval_schemas(features)

    features = {
        "deletion_time": _feature(_FORMAT.INTEGER_64, _SEMANTIC.TIMESTAMP)
    }
    with self.assertRaisesRegex(
        ValueError,
        "Feature name 'deletion_time' is reserved for temporal interval"
        " tracking",
    ):
      combine_lib._add_interval_schemas(features)

    features = {
        "existing_ts": _feature(
            _FORMAT.INTEGER_64, _SEMANTIC.TIMESTAMP, is_creation_time=True
        )
    }
    with self.assertRaisesRegex(
        ValueError,
        "already has is_creation_time=True",
    ):
      combine_lib._add_interval_schemas(features)

    # A reserved name that also carries the flag is reported as a flag
    # conflict, since that check runs first.
    features = {
        "creation_time": _feature(
            _FORMAT.INTEGER_64, _SEMANTIC.TIMESTAMP, is_creation_time=True
        )
    }
    with self.assertRaisesRegex(
        ValueError,
        "already has is_creation_time=True",
    ):
      combine_lib._add_interval_schemas(features)

  @parameterized.named_parameters(
      (
          "reserved_node_feature",
          _graph_schema(
              _node_schema(
                  creation_time=_feature(
                      _FORMAT.INTEGER_64, _SEMANTIC.TIMESTAMP
                  )
              )
          ),
          _create_graph([b"1"], timestamp=100, creation_time=np.array([100])),
          "Feature name 'creation_time' is reserved for temporal interval"
          " tracking",
      ),
      (
          "multigraph_in_snapshot",
          _graph_schema(_node_schema(), _edge_schema()),
          # Parallel edges between 1 (idx 0) and 2 (idx 1) in one snapshot.
          _create_graph([b"1", b"2"], edges=[[0, 0], [1, 1]], timestamp=100),
          "Multigraph detected in edge set 'e1'.*Multigraphs are not yet "
          "supported",
      ),
      (
          "timeseries_node_feature",
          _graph_schema(
              _node_schema(
                  sensor_ts=_feature(_FORMAT.FLOAT_32, is_timeseries=True)
              )
          ),
          _create_graph(
              [b"1"],
              sensor_ts=np.array([1.0], dtype=np.float32),
              timestamp=100,
          ),
          "Snapshot graphs containing a timeseries are not supported yet",
      ),
      (
          "timeseries_edge_feature",
          _graph_schema(
              _node_schema(),
              _edge_schema(
                  weight_ts=_feature(_FORMAT.FLOAT_32, is_timeseries=True)
              ),
          ),
          _create_graph(
              [b"1", b"2"],
              edges=[[0], [1]],
              edge_features={"weight_ts": np.array([0.5], dtype=np.float32)},
              timestamp=100,
          ),
          "Snapshot graphs containing a timeseries are not supported yet",
      ),
  )
  def test_combine_raises_error(self, schema, graph, expected_error):
    dataset_dir = self.create_tempdir().full_path
    self._write_dataset(dataset_dir, schema, [("s1", graph)])
    with self.assertRaisesRegex(ValueError, expected_error):
      _read_and_combine(dataset_dir)


if __name__ == "__main__":
  absltest.main()

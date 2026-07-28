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

"""Unit tests for graph_snapshots."""

import os

from absl.testing import absltest
from dgf.src.data import graph_snapshots as data_graph_snapshots
from dgf.src.data import graph_snapshots_metadata as data_snapshot_metadata
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_constants
from dgf.src.io import graph_in_memory
from dgf.src.io import graph_snapshots
from dgf.src.io import graph_snapshots_metadata as io_snapshot_metadata
from dgf.src.io import schema as io_schema
from dgf.src.transform import combine_graph_snapshots as combine_lib
from dgf.src.util import filesystem
from dgf.src.util import test_util
import numpy as np


def _create_schema() -> schema_lib.GraphSchema:
  """Builds the schema shared by all the test snapshots."""
  return schema_lib.GraphSchema(
      node_sets={
          "n1": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "f1": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
              }
          )
      },
      edge_sets={},
  )


def _create_graph(
    ids: list[bytes], f1: list[float], timestamp: int
) -> in_memory_graph_lib.InMemoryGraph:
  """Builds a single-node-set in-memory graph for testing."""
  return in_memory_graph_lib.InMemoryGraph(
      timestamp=timestamp,
      node_sets={
          "n1": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=len(ids),
              features={
                  "#id": np.array(ids, dtype=np.bytes_),
                  "f1": np.array(f1, dtype=np.float32),
              },
          )
      },
      edge_sets={},
  )


class GraphSnapshotsTest(absltest.TestCase):

  def _write_dataset(
      self,
      root_dir: str,
      schema: schema_lib.GraphSchema,
      snapshots: list[tuple[str, in_memory_graph_lib.InMemoryGraph]],
  ) -> None:
    """Writes metadata.json, schema.json and snapshot graphs to root_dir."""
    metadata = data_snapshot_metadata.GraphSnapshotsMetadata(
        format=data_snapshot_metadata.GraphSnapshotsFormat.GRAPH_SNAPSHOTS,
        version=0,
    )
    io_snapshot_metadata.write_metadata(
        metadata, os.path.join(root_dir, graph_constants.FILENAME_METADATA)
    )
    io_schema.write_schema(
        schema, os.path.join(root_dir, graph_constants.FILENAME_SCHEMA)
    )
    for snapshot_id, graph in snapshots:
      graph_in_memory.write_graph(
          graph,
          schema,
          path=os.path.join(
              root_dir, graph_constants.DIRNAME_SNAPSHOTS, snapshot_id
          ),
          verbose=False,
      )

  def _create_sample_dataset(
      self, root_dir: str, include_duplicate: bool = False
  ) -> schema_lib.GraphSchema:
    """Creates a sample dataset whose 3 snapshots are not sorted by name."""
    schema = _create_schema()
    # The snapshot ids are deliberately not in chronological order, so that
    # sorting by timestamp differs from sorting by id.
    snapshots = [
        ("snap_beta", _create_graph([b"1", b"2"], [1.0, 2.0], timestamp=100)),
        (
            "snap_alpha",
            _create_graph(
                [b"1", b"3"],
                [1.5, 3.0],
                timestamp=100 if include_duplicate else 200,
            ),
        ),
        (
            "snap_gamma",
            _create_graph([b"1", b"2", b"3"], [2.0, 2.5, 3.5], timestamp=300),
        ),
    ]
    self._write_dataset(root_dir, schema, snapshots)
    return schema

  def test_read_graph_snapshots_sorts_by_timestamp(self):
    dataset_dir = self.create_tempdir().full_path
    self._create_sample_dataset(dataset_dir)

    snapshots = graph_snapshots.read_graph_snapshots(dataset_dir)

    self.assertLen(snapshots, 3)
    test_util.assert_are_equal(
        self, snapshots.timestamps, np.array([100, 200, 300], dtype=np.int64)
    )
    self.assertEqual(
        list(snapshots.snapshot_ids),
        ["snap_beta", "snap_alpha", "snap_gamma"],
    )

  def test_read_graph_snapshots_returns_graphs_in_timestamp_order(self):
    dataset_dir = self.create_tempdir().full_path
    self._create_sample_dataset(dataset_dir)

    snapshots = graph_snapshots.read_graph_snapshots(dataset_dir)

    self.assertLen(snapshots.graphs, 3)
    num_nodes = [g.node_sets["n1"].num_nodes for g in snapshots.graphs]
    self.assertEqual(num_nodes, [2, 2, 3])
    test_util.assert_are_equal(
        self,
        snapshots.graphs[0].node_sets["n1"].features["#id"],
        np.array([b"1", b"2"], dtype=np.bytes_),
    )
    test_util.assert_are_equal(
        self,
        snapshots.graphs[2].node_sets["n1"].features["f1"],
        np.array([2.0, 2.5, 3.5], dtype=np.float32),
    )

  def test_read_graph_snapshots_returns_root_schema(self):
    dataset_dir = self.create_tempdir().full_path
    expected_schema = self._create_sample_dataset(dataset_dir)

    snapshots = graph_snapshots.read_graph_snapshots(dataset_dir)

    test_util.assert_are_equal(self, snapshots.schema, expected_schema)

  def test_read_graph_snapshots_single_worker(self):
    dataset_dir = self.create_tempdir().full_path
    self._create_sample_dataset(dataset_dir)

    snapshots = graph_snapshots.read_graph_snapshots(
        dataset_dir, num_workers=1
    )

    test_util.assert_are_equal(
        self, snapshots.timestamps, np.array([100, 200, 300], dtype=np.int64)
    )
    self.assertEqual(
        list(snapshots.snapshot_ids),
        ["snap_beta", "snap_alpha", "snap_gamma"],
    )

  def test_read_graph_snapshots_missing_metadata_raises_error(self):
    dataset_dir = self.create_tempdir().full_path
    self._create_sample_dataset(dataset_dir)
    bad_snapshot_dir = os.path.join(
        dataset_dir, graph_constants.DIRNAME_SNAPSHOTS, "snap_corrupt"
    )
    filesystem.makedirs(bad_snapshot_dir)

    # `validate_snapshots` runs before the snapshots are discovered, so it is
    # the one reporting the incomplete snapshot.
    with self.assertRaisesRegex(ValueError, "errors found"):
      graph_snapshots.read_graph_snapshots(dataset_dir)

  def test_read_graph_snapshots_duplicate_timestamps_raises_error(self):
    dataset_dir = self.create_tempdir().full_path
    self._create_sample_dataset(dataset_dir, include_duplicate=True)

    with self.assertRaisesRegex(ValueError, "errors found"):
      graph_snapshots.read_graph_snapshots(dataset_dir)

  def test_read_graph_snapshots_missing_root_schema_raises_error(self):
    dataset_dir = self.create_tempdir().full_path
    snapshot_dir = os.path.join(
        dataset_dir, graph_constants.DIRNAME_SNAPSHOTS, "s1"
    )
    filesystem.makedirs(snapshot_dir)
    with filesystem.open_write(
        os.path.join(snapshot_dir, graph_constants.FILENAME_METADATA)
    ) as f:
      f.write('{"version": 0, "timestamp": 100}')

    with self.assertRaisesRegex(ValueError, "Root schema.json missing"):
      graph_snapshots.read_graph_snapshots(dataset_dir)

  def test_read_graph_snapshots_missing_snapshots_dir_raises_error(self):
    dataset_dir = self.create_tempdir().full_path
    self._create_sample_dataset(dataset_dir)
    filesystem.rmtree(
        os.path.join(dataset_dir, graph_constants.DIRNAME_SNAPSHOTS)
    )

    with self.assertRaisesRegex(ValueError, "errors found"):
      graph_snapshots.read_graph_snapshots(dataset_dir)

  def test_read_graph_snapshots_empty_snapshots_dir_raises_error(self):
    dataset_dir = self.create_tempdir().full_path
    self._create_sample_dataset(dataset_dir)
    filesystem.rmtree(
        os.path.join(dataset_dir, graph_constants.DIRNAME_SNAPSHOTS)
    )
    filesystem.makedirs(
        os.path.join(dataset_dir, graph_constants.DIRNAME_SNAPSHOTS)
    )

    with self.assertRaisesRegex(ValueError, "No snapshot directories found"):
      graph_snapshots.read_graph_snapshots(dataset_dir)

  def test_graph_snapshots_invariants(self):
    schema = _create_schema()
    g1 = _create_graph([b"1"], [1.0], timestamp=100)
    g2 = _create_graph([b"1"], [2.0], timestamp=200)

    with self.assertRaisesRegex(
        ValueError, "timestamps, snapshot_ids and graphs must have the same"
    ):
      data_graph_snapshots.GraphSnapshots(
          timestamps=np.array([100], dtype=np.int64),
          snapshot_ids=["s1", "s2"],
          schema=schema,
          graphs=[g1, g2],
      )

    with self.assertRaisesRegex(
        ValueError, "timestamps must be strictly increasing"
    ):
      data_graph_snapshots.GraphSnapshots(
          timestamps=np.array([200, 100], dtype=np.int64),
          snapshot_ids=["s2", "s1"],
          schema=schema,
          graphs=[g2, g1],
      )

    with self.assertRaisesRegex(
        ValueError, "timestamps must be strictly increasing"
    ):
      data_graph_snapshots.GraphSnapshots(
          timestamps=np.array([100, 100], dtype=np.int64),
          snapshot_ids=["s1", "s1_dup"],
          schema=schema,
          graphs=[g1, g1],
      )

  def test_read_graph_snapshots_as_temporal_graph(self):
    dataset_dir = self.create_tempdir().full_path
    self._create_sample_dataset(dataset_dir)

    graph, schema = graph_snapshots.read_graph_snapshots_as_temporal_graph(
        dataset_dir
    )

    # The wrapper must be exactly equivalent to the explicit two-step.
    snapshots = graph_snapshots.read_graph_snapshots(dataset_dir)
    expected_graph, expected_schema = (
        combine_lib.combine_graph_snapshots(snapshots)
    )
    test_util.assert_are_equal(self, graph, expected_graph)
    test_util.assert_are_equal(self, schema, expected_schema)

    # Nodes b"1", b"2", b"3" are merged across the three snapshots.
    self.assertEqual(graph.node_sets["n1"].num_nodes, 3)
    self.assertIn("creation_time", schema.node_sets["n1"].features)


if __name__ == "__main__":
  absltest.main()

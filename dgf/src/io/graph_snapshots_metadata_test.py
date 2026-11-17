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

"""Tests for GraphSnapshotsMetadata serialization and deserialization."""

import os
import tempfile
from absl.testing import absltest
from dgf.src.data import graph_snapshots_metadata as data_snapshot_metadata
from dgf.src.io import graph_snapshots_metadata as io_snapshot_metadata


class GraphSnapshotsMetadataTest(absltest.TestCase):

  def test_snapshots_metadata(self):
    meta = data_snapshot_metadata.GraphSnapshotsMetadata(
        format=data_snapshot_metadata.GraphSnapshotsFormat.GRAPH_SNAPSHOTS,
        version=0,
    )
    self.assertEqual(
        meta.format, data_snapshot_metadata.GraphSnapshotsFormat.GRAPH_SNAPSHOTS
    )
    self.assertEqual(meta.format, "graph_snapshots")
    self.assertEqual(meta.version, 0)

  def test_write_and_read_metadata(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      meta_path = os.path.join(tmpdir, "metadata.json")
      meta = data_snapshot_metadata.GraphSnapshotsMetadata(
          format=data_snapshot_metadata.GraphSnapshotsFormat.GRAPH_SNAPSHOTS,
          version=1,
      )
      io_snapshot_metadata.write_metadata(meta, meta_path)

      self.assertTrue(os.path.exists(meta_path))

      loaded_meta = io_snapshot_metadata.read_metadata(meta_path)
      self.assertEqual(
          loaded_meta.format,
          data_snapshot_metadata.GraphSnapshotsFormat.GRAPH_SNAPSHOTS,
      )
      self.assertEqual(loaded_meta.format, "graph_snapshots")
      self.assertEqual(loaded_meta.version, 1)
      self.assertIsNone(loaded_meta.name)

  def test_write_and_read_metadata_with_name(self):
    meta_path = os.path.join(self.create_tempdir().full_path, "metadata.json")
    io_snapshot_metadata.write_metadata(
        data_snapshot_metadata.GraphSnapshotsMetadata(
            format=data_snapshot_metadata.GraphSnapshotsFormat.GRAPH_SNAPSHOTS,
            version=0,
            name="my_graph",
        ),
        meta_path,
    )
    self.assertEqual(
        io_snapshot_metadata.read_metadata(meta_path).name, "my_graph"
    )

  def test_read_metadata_written_without_name(self):
    meta_path = os.path.join(self.create_tempdir().full_path, "metadata.json")
    with open(meta_path, "w") as f:
      f.write('{"format": "graph_snapshots", "version": 0}')
    self.assertIsNone(io_snapshot_metadata.read_metadata(meta_path).name)


if __name__ == "__main__":
  absltest.main()

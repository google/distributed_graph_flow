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

"""Tests for snapshots dataset validation."""

import dataclasses
import os
import tempfile
from absl.testing import absltest
from dgf.src.data import graph_snapshots_metadata as data_snapshot_metadata
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_in_memory
from dgf.src.io import graph_snapshots_metadata as io_snapshot_metadata
from dgf.src.io import schema as io_schema
from dgf.src.util import filesystem
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
from dgf.src.validate import graph_snapshots as snapshots_validate_lib
from dgf.src.validate import validate as validate_lib

Issue = validate_lib.Issue

test_util.disable_diff_truncation()


def _add_test_snapshot(dataset_path, snapshot_id, timestamp, graph, schema):
  filesystem.makedirs(dataset_path)
  meta_path = os.path.join(dataset_path, "metadata.json")
  if not filesystem.exists(meta_path):
    metadata = data_snapshot_metadata.GraphSnapshotsMetadata(
        format="graph_snapshots", version=0
    )
    io_snapshot_metadata.write_metadata(metadata, meta_path)
  schema_path = os.path.join(dataset_path, "schema.json")
  if not filesystem.exists(schema_path):
    io_schema.write_schema(schema, schema_path)

  snap_dir = os.path.join(dataset_path, "snapshots", snapshot_id)
  graph_with_ts = dataclasses.replace(graph, timestamp=timestamp)
  graph_in_memory.write_graph(
      graph_with_ts, schema, path=snap_dir, verbose=False
  )


class GraphSnapshotsTest(absltest.TestCase):

  def test_valid(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      dataset_path = os.path.join(tmpdir, "snapshot_dataset")
      schema = gen_test_graph.generate_schema(node_ids=True, semantic=True)
      graph1 = gen_test_graph.generate_in_memory_graph(node_ids=True)
      graph2 = gen_test_graph.generate_in_memory_graph(node_ids=True)

      _add_test_snapshot(dataset_path, "snap_001", 1000, graph1, schema)
      _add_test_snapshot(dataset_path, "snap_002", 2000, graph2, schema)

      issues = snapshots_validate_lib.snapshots_issues(dataset_path)
      self.assertEqual(issues, [])

      snapshots_validate_lib.validate_snapshots(dataset_path)

  def test_missing_root_metadata(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      dataset_path = os.path.join(tmpdir, "non_existent_dataset")
      metadata_path = os.path.join(dataset_path, "metadata.json")
      schema_path = os.path.join(dataset_path, "schema.json")
      snaps_dir = os.path.join(dataset_path, "snapshots")
      issues = snapshots_validate_lib.snapshots_issues(dataset_path)
      self.assertEqual(
          issues,
          [
              Issue.error(f"Missing root metadata.json at {metadata_path}"),
              Issue.error(f"Missing root schema.json at {schema_path}"),
              Issue.error(f"Snapshots directory missing at {snaps_dir}"),
          ],
      )

      with self.assertRaisesRegex(ValueError, "3 errors found"):
        snapshots_validate_lib.validate_snapshots(dataset_path)

  def test_invalid_root_metadata_format(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      dataset_path = os.path.join(tmpdir, "dataset")
      schema = gen_test_graph.generate_schema(node_ids=True, semantic=True)
      graph = gen_test_graph.generate_in_memory_graph(node_ids=True)
      _add_test_snapshot(dataset_path, "snap_001", 1000, graph, schema)

      metadata = data_snapshot_metadata.GraphSnapshotsMetadata(
          format="invalid_format", version=0
      )
      io_snapshot_metadata.write_metadata(
          metadata, os.path.join(dataset_path, "metadata.json")
      )

      issues = snapshots_validate_lib.snapshots_issues(dataset_path)
      self.assertEqual(
          issues,
          [
              Issue.error(
                  "Expected root metadata format='graph_snapshots', found"
                  " 'invalid_format'"
              )
          ],
      )

  def test_missing_root_schema(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      dataset_path = os.path.join(tmpdir, "dataset")
      schema = gen_test_graph.generate_schema(node_ids=True, semantic=True)
      graph = gen_test_graph.generate_in_memory_graph(node_ids=True)
      _add_test_snapshot(dataset_path, "snap_001", 1000, graph, schema)

      filesystem.remove_paths([os.path.join(dataset_path, "schema.json")])

      schema_path = os.path.join(dataset_path, "schema.json")
      issues = snapshots_validate_lib.snapshots_issues(dataset_path)
      self.assertEqual(
          issues, [Issue.error(f"Missing root schema.json at {schema_path}")]
      )

  def test_missing_snapshot_metadata(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      dataset_path = os.path.join(tmpdir, "dataset")
      schema = gen_test_graph.generate_schema(node_ids=True, semantic=True)
      graph = gen_test_graph.generate_in_memory_graph(node_ids=True)

      _add_test_snapshot(dataset_path, "snap_001", 1000, graph, schema)

      snap_meta_path = os.path.join(
          dataset_path, "snapshots", "snap_001", "metadata.json"
      )
      filesystem.remove_paths([snap_meta_path])

      issues = snapshots_validate_lib.snapshots_issues(dataset_path)
      self.assertEqual(
          issues,
          [
              Issue.error(
                  "Snapshot 'snap_001' missing metadata.json at"
                  f" {snap_meta_path}"
              )
          ],
      )

  def test_missing_snapshot_timestamp(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      dataset_path = os.path.join(tmpdir, "dataset")
      schema = gen_test_graph.generate_schema(node_ids=True, semantic=True)
      graph = gen_test_graph.generate_in_memory_graph(node_ids=True)

      _add_test_snapshot(dataset_path, "snap_001", 1000, graph, schema)

      snap_meta_path = os.path.join(
          dataset_path, "snapshots", "snap_001", "metadata.json"
      )
      with filesystem.open_write(snap_meta_path) as f:
        f.write('{"format": "gf_graph"}')

      issues = snapshots_validate_lib.snapshots_issues(dataset_path)
      self.assertEqual(
          issues,
          [
              Issue.error(
                  "Snapshot 'snap_001' metadata.json missing 'timestamp'"
              )
          ],
      )

  def test_duplicate_timestamp(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      dataset_path = os.path.join(tmpdir, "dataset")
      schema = gen_test_graph.generate_schema(node_ids=True, semantic=True)
      graph1 = gen_test_graph.generate_in_memory_graph(node_ids=True)
      graph2 = gen_test_graph.generate_in_memory_graph(node_ids=True)

      _add_test_snapshot(dataset_path, "snap_001", 1000, graph1, schema)
      _add_test_snapshot(dataset_path, "snap_002", 1000, graph2, schema)

      issues = snapshots_validate_lib.snapshots_issues(dataset_path)
      self.assertEqual(
          issues,
          [
              Issue.error(
                  "Duplicate timestamp 1000 found in snapshot 'snap_002'"
                  " (already defined in 'snap_001')"
              )
          ],
      )

  def test_validate_contents_false(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      dataset_path = os.path.join(tmpdir, "snapshot_dataset")
      schema = gen_test_graph.generate_schema(node_ids=True, semantic=True)
      graph = gen_test_graph.generate_in_memory_graph(node_ids=True)

      _add_test_snapshot(dataset_path, "snap_001", 1000, graph, schema)

      issues = snapshots_validate_lib.snapshots_issues(
          dataset_path, validate_contents=False
      )
      self.assertEqual(issues, [])

  def test_snapshot_invalid_content(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      dataset_path = os.path.join(tmpdir, "snapshot_dataset")
      schema = gen_test_graph.generate_schema(node_ids=True, semantic=True)
      graph = gen_test_graph.generate_in_memory_graph(node_ids=True)
      _add_test_snapshot(dataset_path, "snap_001", 1000, graph, schema)

      target_nodeset = list(schema.node_sets.keys())[0]
      modified_schema = dataclasses.replace(
          schema,
          node_sets={
              **schema.node_sets,
              target_nodeset: dataclasses.replace(
                  schema.node_sets[target_nodeset],
                  features={
                      **schema.node_sets[target_nodeset].features,
                      "missing_feat": schema_lib.FeatureSchema(
                          format=schema_lib.FeatureFormat.FLOAT_32, shape=(4,)
                      ),
                  },
              ),
          },
      )
      io_schema.write_schema(
          modified_schema, os.path.join(dataset_path, "schema.json")
      )

      issues = snapshots_validate_lib.snapshots_issues(dataset_path)
      self.assertEqual(
          issues,
          [
              Issue.error(
                  "Snapshot 'snap_001': Missing feature 'missing_feat' in"
                  f" nodeset {target_nodeset!r}."
              )
          ],
      )

  def test_empty_snapshots_dir(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      dataset_path = os.path.join(tmpdir, "dataset")
      metadata = data_snapshot_metadata.GraphSnapshotsMetadata(
          format="graph_snapshots", version=0
      )
      filesystem.makedirs(dataset_path)
      io_snapshot_metadata.write_metadata(
          metadata, os.path.join(dataset_path, "metadata.json")
      )
      io_schema.write_schema(
          gen_test_graph.generate_schema(node_ids=True),
          os.path.join(dataset_path, "schema.json"),
      )
      snaps_dir = os.path.join(dataset_path, "snapshots")
      filesystem.makedirs(snaps_dir)

      issues = snapshots_validate_lib.snapshots_issues(dataset_path)
      self.assertEqual(
          issues,
          [Issue.warning(f"No snapshot directories found in {snaps_dir}")],
      )


if __name__ == "__main__":
  absltest.main()

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

import os
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.util import shard


class ShardTest(parameterized.TestCase):

  def test_shard_pattern_to_glob(self):
    self.assertEqual(
        shard.shard_pattern_to_glob("my_file", ".csv"),
        "my_file-*.csv",
    )
    self.assertEqual(
        shard.shard_pattern_to_glob("my_file", ""),
        "my_file-*",
    )

  def test_sharded_filename(self):
    self.assertEqual(
        shard.sharded_filename("my_file", 0, 10, ".csv"),
        "my_file-00000-of-00010.csv",
    )
    self.assertEqual(
        shard.sharded_filename("my_file", 9, 10, ".csv"),
        "my_file-00009-of-00010.csv",
    )
    self.assertEqual(
        shard.sharded_filename("my_file", 0, 10, ""),
        "my_file-00000-of-00010",
    )

  def test_shard_path_to_glob(self):
    self.assertEqual(
        shard.shard_path_to_glob("my_file@*"),
        "my_file-?????-of-?????",
    )
    self.assertEqual(
        shard.shard_path_to_glob("my_file@*.csv"),
        "my_file-?????-of-?????.csv",
    )
    self.assertEqual(
        shard.shard_path_to_glob("my_file@10"),
        "my_file-?????-of-00010",
    )
    self.assertEqual(
        shard.shard_path_to_glob("my_file@10.csv"),
        "my_file-?????-of-00010.csv",
    )
    self.assertEqual(
        shard.shard_path_to_glob("my_file"),
        "my_file",
    )

  def test_parse_sharded_filename(self):
    self.assertEqual(
        shard.parse_sharded_filename("my_file@*"),
        ("my_file", None, ""),
    )
    self.assertEqual(
        shard.parse_sharded_filename("my_file@*.csv"),
        ("my_file", None, ".csv"),
    )
    self.assertEqual(
        shard.parse_sharded_filename("my_file@10"),
        ("my_file", 10, ""),
    )
    self.assertEqual(
        shard.parse_sharded_filename("my_file@10.csv"),
        ("my_file", 10, ".csv"),
    )
    self.assertEqual(
        shard.parse_sharded_filename("my_file"),
        ("my_file", None, ""),
    )

  def test_list_paths(self):
    tmpdir = self.create_tempdir().full_path
    base_name = os.path.join(tmpdir, "my_sharded_file")
    num_shards = 4
    expected_paths = []
    for i in range(4):
      path = shard.sharded_filename(base_name, i, num_shards, ".txt")
      # Create an empty file for each shard.
      self.create_tempfile(path)
      expected_paths.append(path)

    paths = shard.list_paths(base_name, ".txt")
    self.assertEqual(paths, expected_paths)

  def test_expand_input_paths(self):
    tmpdir = self.create_tempdir().full_path
    for i in range(2):
      path = shard.sharded_filename(
          os.path.join(tmpdir, "my_file"), i, 2, ".ext"
      )
      self.create_tempfile(path)

    self.create_tempfile(
        shard.sharded_filename(os.path.join(tmpdir, "other_file"), 0, 2, ".ext")
    )

    # Note: the tmpdir part is only necessary when the directory needs to be
    # scanned.
    self.assertEqual(
        shard.expand_input_paths("my_file"),
        ["my_file"],
    )
    self.assertEqual(
        shard.expand_input_paths("my_file@2"),
        ["my_file-00000-of-00002", "my_file-00001-of-00002"],
    )
    self.assertEqual(
        shard.expand_input_paths("my_file@2.ext"),
        ["my_file-00000-of-00002.ext", "my_file-00001-of-00002.ext"],
    )
    self.assertEqual(
        shard.expand_input_paths(tmpdir + "/my_file@*.ext"),
        [
            tmpdir + "/my_file-00000-of-00002.ext",
            tmpdir + "/my_file-00001-of-00002.ext",
        ],
    )
    self.assertEqual(
        shard.expand_input_paths(tmpdir + "/m*.ext"),
        [
            tmpdir + "/my_file-00000-of-00002.ext",
            tmpdir + "/my_file-00001-of-00002.ext",
        ],
    )
    self.assertEqual(
        shard.expand_input_paths("my_file-00001-of-00002.ext"),
        ["my_file-00001-of-00002.ext"],
    )
    self.assertEqual(
        shard.expand_input_paths(tmpdir + "/my_file-0000?-of-00002.ext"),
        [
            tmpdir + "/my_file-00000-of-00002.ext",
            tmpdir + "/my_file-00001-of-00002.ext",
        ],
    )

  def test_expand_output_paths(self):

    self.assertEqual(
        shard.expand_output_paths("my_file", num_shards=2),
        ["my_file"],
    )
    self.assertEqual(
        shard.expand_output_paths("my_file@2", num_shards=2),
        ["my_file-00000-of-00002", "my_file-00001-of-00002"],
    )
    self.assertEqual(
        shard.expand_output_paths("my_file@2.ext", num_shards=2),
        ["my_file-00000-of-00002.ext", "my_file-00001-of-00002.ext"],
    )
    self.assertEqual(
        shard.expand_output_paths("my_file@*.ext", num_shards=2),
        [
            "my_file-00000-of-00002.ext",
            "my_file-00001-of-00002.ext",
        ],
    )

    with self.assertRaises(ValueError):
      shard.expand_output_paths("m*.ext", num_shards=2)

  @parameterized.parameters(
      (100, 1, 1000),
      (1000, 1, 1000),
      (1001, 2, 1000),
      (1000 * 100, 100, 1000),
      (1000 * 100 + 1, 100, 1001),
      (1000 * 200, 100, 2000),
      (1000 * 200 + 1, 100, 2001),
  )
  def test_estimate_num_node_shards(
      self, num_nodes, expected_num_shards, expected_num_nodes_per_shard
  ):
    self.assertEqual(
        shard.estimate_num_node_shards(num_nodes),
        (expected_num_shards, expected_num_nodes_per_shard),
    )

  @parameterized.parameters(
      (100, 1, 1000),
      (1000, 1, 1000),
      (1001, 2, 1000),
      (1000 * 100, 100, 1000),
      (1000 * 100 + 1, 100, 1001),
      (1000 * 200, 100, 2000),
      (1000 * 200 + 1, 100, 2001),
  )
  def test_estimate_num_edge_shards(
      self, num_edges, expected_num_shards, expected_num_edeg_per_shard
  ):
    self.assertEqual(
        shard.estimate_num_edge_shards(num_edges),
        (expected_num_shards, expected_num_edeg_per_shard),
    )


if __name__ == "__main__":
  absltest.main()

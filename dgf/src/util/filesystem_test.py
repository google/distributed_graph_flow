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
import tempfile
from absl.testing import absltest
from dgf.src.util import filesystem as fs


class FilesystemGcsTest(absltest.TestCase):

  def test_base(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "data.txt")
      with fs.open_write(path, binary=False) as f:
        f.write("hello")
      with fs.open_read(path, binary=False) as f:
        self.assertEqual(f.read(), "hello")

  def test_is_gcs_path(self):
    self.assertTrue(fs.is_gcs_path("gs://bucket/path"))
    self.assertFalse(fs.is_gcs_path("/tmp/path"))
    self.assertFalse(fs.is_gcs_path("local/path"))

  def test_glob(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create some dummy files
      with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
        f.write("1")
      with open(os.path.join(tmpdir, "file2.txt"), "w") as f:
        f.write("2")
      with open(os.path.join(tmpdir, "other.csv"), "w") as f:
        f.write("3")

      # Test glob with a specific pattern
      txt_files = fs.glob(os.path.join(tmpdir, "*.txt"))
      self.assertCountEqual(
          txt_files,
          [
              os.path.join(tmpdir, "file1.txt"),
              os.path.join(tmpdir, "file2.txt"),
          ],
      )

      # Test glob with a more general pattern
      all_files = fs.glob(os.path.join(tmpdir, "*"))
      self.assertCountEqual(
          all_files,
          [
              os.path.join(tmpdir, "file1.txt"),
              os.path.join(tmpdir, "file2.txt"),
              os.path.join(tmpdir, "other.csv"),
          ],
      )

  def test_exists(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      existing_path = os.path.join(tmpdir, "existing.txt")
      non_existing_path = os.path.join(tmpdir, "non_existing.txt")

      with open(existing_path, "w") as f:
        f.write("test")

      self.assertTrue(fs.exists(existing_path))
      self.assertFalse(fs.exists(non_existing_path))

  def test_makedirs_rmtree(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      subdir_path = os.path.join(tmpdir, "a", "b", "c")
      fs.makedirs(subdir_path)
      self.assertTrue(os.path.isdir(subdir_path))

      # Test rmtree on a parent directory
      parent_dir = os.path.join(tmpdir, "a")
      fs.rmtree(parent_dir)
      self.assertFalse(os.path.exists(parent_dir))

  def test_remove_paths(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      file1 = os.path.join(tmpdir, "file1.txt")
      file2 = os.path.join(tmpdir, "file2.txt")
      file3 = os.path.join(tmpdir, "file3.txt")
      non_existent = os.path.join(tmpdir, "non_existent.txt")

      for f in [file1, file2, file3]:
        with open(f, "w") as opened_file:
          opened_file.write("test")

      self.assertTrue(fs.exists(file1))
      self.assertTrue(fs.exists(file2))
      self.assertTrue(fs.exists(file3))

      # Remove file1 and file2
      fs.remove_paths([file1, file2])
      self.assertFalse(fs.exists(file1))
      self.assertFalse(fs.exists(file2))
      self.assertTrue(fs.exists(file3))

      # Test fail_if_absent=False
      fs.remove_paths([file3, non_existent], fail_if_absent=False)
      self.assertFalse(fs.exists(file3))
      # No error should be raised for non_existent

      # Test fail_if_absent=True with non-existent file
      with self.assertRaises(Exception):  # tf.errors.NotFoundError
        fs.remove_paths([non_existent], fail_if_absent=True)

  def test_rename(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      old_path = os.path.join(tmpdir, "old.txt")
      new_path = os.path.join(tmpdir, "new.txt")

      with open(old_path, "w") as f:
        f.write("content")

      self.assertTrue(fs.exists(old_path))
      self.assertFalse(fs.exists(new_path))

      fs.rename(old_path, new_path)

      self.assertFalse(fs.exists(old_path))
      self.assertTrue(fs.exists(new_path))
      with open(new_path, "r") as f:
        self.assertEqual(f.read(), "content")


if __name__ == "__main__":
  absltest.main()

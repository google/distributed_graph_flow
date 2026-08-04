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

"""Tests for weak_dep."""

from __future__ import annotations

from absl.testing import absltest
from dgf.src.util.weak_dep.base import LazyModule


class WeakDepTest(absltest.TestCase):

  def test_import_weak_dependency_success(self):
    math_module = LazyModule(
        local_name="math",
        import_path="math",
        library_name="Math",
        pip="math",
        bazel_rule="//math",
    )
    self.assertEqual(math_module.cos(0), 1.0)
    self.assertTrue(math_module.is_available())

  def test_import_weak_dependency_failure(self):
    non_existent = LazyModule(
        local_name="nonexistent",
        import_path="non_existent_module_xyz",
        library_name="NonExistent",
        pip="non_existent",
        bazel_rule="//non/existent",
    )
    with self.assertRaisesRegex(
        RuntimeError,
        "This feature requires the NonExistent library to be installed\\n"
        "manually.*Install it with\\n.*pip install"
        " non_existent.*//non/existent",
    ):
      _ = non_existent.foo

    self.assertFalse(non_existent.is_available())

if __name__ == "__main__":
  absltest.main()

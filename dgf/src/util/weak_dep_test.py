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

from absl.testing import absltest
from dgf.src.util import weak_dep


class WeakDepTest(absltest.TestCase):

  def test_import_weak_dependency_success(self):
    # Use a standard library module that is guaranteed to be present.
    math_module = weak_dep._import_weak_dependency(
        import_path="math",
        library_name="Math",
        pip="math",
        bazel_rule="//math",
    )
    self.assertEqual(math_module.cos(0), 1.0)

  def test_import_weak_dependency_failure(self):
    with self.assertRaisesRegex(
        RuntimeError,
        "This feature requires the NonExistent library to be installed"
        " manually.*pip install non_existent.*//non/existent",
    ):
      weak_dep._import_weak_dependency(
          import_path="non_existent_module_xyz",
          library_name="NonExistent",
          pip="non_existent",
          bazel_rule="//non/existent",
      )

  def test_import_weak_dependency_attribute_success(self):
    # Test importing with attribute
    cos_func = weak_dep._import_weak_dependency(
        import_path="math",
        library_name="Math",
        pip="math",
        bazel_rule="//math",
        attribute_name="cos",
    )
    self.assertEqual(cos_func(0), 1.0)

  def test_import_weak_dependency_attribute_failure(self):
    with self.assertRaisesRegex(
        RuntimeError,
        "This feature requires the Math library to be installed manually",
    ):
      # math exists, but non_existent_attr does not.
      weak_dep._import_weak_dependency(
          import_path="math",
          library_name="Math",
          pip="math",
          bazel_rule="//math",
          attribute_name="non_existent_attr",
      )


if __name__ == "__main__":

  absltest.main()

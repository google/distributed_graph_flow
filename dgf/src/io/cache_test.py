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
from unittest import mock
from absl.testing import absltest
from dgf.src.io import cache as cache_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util


class CacheTest(absltest.TestCase):

  def test_variable(self):

    def create():
      return "hello"

    mock_create = mock.Mock(side_effect=create)

    tmp_path = os.path.join(self.create_tempdir().full_path, "a.pkl")

    for _ in range(3):
      self.assertEqual(cache_lib.cache(tmp_path, mock_create), "hello")
      mock_create.assert_called_once()

  def test_multi_variable(self):

    def create():
      return "hello", "world"

    mock_create = mock.Mock(side_effect=create)

    tmp_path = os.path.join(self.create_tempdir().full_path, "a.pkl")

    for _ in range(3):
      a, b = cache_lib.cache(tmp_path, mock_create)
      mock_create.assert_called_once()
      self.assertEqual(a, "hello")
      self.assertEqual(b, "world")

  def test_variable_with_name(self):
    def create():
      return "hello"

    mock_create = mock.Mock(side_effect=create)

    tmp_path = os.path.join(self.create_tempdir().full_path, "a.pkl")

    self.assertNotIn("a", locals())
    for i in range(3):
      a = cache_lib.cache(tmp_path, mock_create, variable_names="a")
      if i == 0:
        os.remove(tmp_path)
      self.assertEqual(a, "hello")
      mock_create.assert_called_once()

  def test_multi_variable_with_name(self):
    def create():
      return "hello", "world"

    mock_create = mock.Mock(side_effect=create)

    tmp_path = os.path.join(self.create_tempdir().full_path, "a.pkl")

    self.assertNotIn("a", locals())
    self.assertNotIn("b", locals())
    for i in range(3):
      a, b = cache_lib.cache(tmp_path, mock_create, variable_names=("a", "b"))
      if i == 0:
        os.remove(tmp_path)
      self.assertEqual(a, "hello")
      self.assertEqual(b, "world")
      mock_create.assert_called_once()

  def test_in_memory_graph(self):

    def create():
      return (
          gen_test_graph.generate_in_memory_graph(),
          gen_test_graph.generate_schema(variable_length=False),
      )

    tmp_path = os.path.join(self.create_tempdir().full_path, "a.pkl")

    for _ in range(3):
      graph, schema = cache_lib.cache(
          tmp_path, create, variable_names=("graph", "schema")
      )

      test_util.assert_are_equal(
          self, graph, gen_test_graph.generate_in_memory_graph()
      )
      test_util.assert_are_equal(
          self, schema, gen_test_graph.generate_schema(variable_length=False)
      )


if __name__ == "__main__":
  absltest.main()

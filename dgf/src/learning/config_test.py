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

"""Test for the basic buildable config class."""

import dataclasses
import tempfile
from absl.testing import absltest
from dgf.src.learning import config


# Trivial example.
class MyObject:

  def __init__(self, x: int, y: int):
    self.x = x
    self.y = y

  def __call__(self):
    return self.x + self.y


@dataclasses.dataclass(frozen=True, kw_only=True)
class MyConfig(config.Config[MyObject]):
  x: int
  y: int

  def make(self):
    return MyObject(x=self.x, y=self.y)

  def name(self):
    return "MyObject"


class ConfigTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.cfg = MyConfig(x=1, y=2)

  def test_basic(self):
    obj = self.cfg.make()
    self.assertEqual(obj(), 3)
    self.assertEqual(self.cfg.name(), "MyObject")

  def test_to_dict(self):
    expected_dict = {"x": 1, "y": 2, "name": "MyObject"}
    self.assertEqual(self.cfg.to_dict(), expected_dict)

  def test_save_load(self):
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
      self.cfg.json_save(tmp.name)
      loaded_cfg = MyConfig.json_load(tmp.name)
      self.assertEqual(self.cfg, loaded_cfg)

      # Loaded object is makeable.
      assert loaded_cfg is not None
      obj = loaded_cfg.make()
      self.assertEqual(obj(), 3)


if __name__ == "__main__":
  absltest.main()

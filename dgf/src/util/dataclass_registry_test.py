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

import dataclasses
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import dataclasses_json
from dgf.src.util import dataclass_registry
from dgf.src.util import test_util

registry = dataclass_registry.create_registry('my_registry')


@registry.register
@dataclasses_json.dataclass_json(undefined=dataclasses_json.Undefined.RAISE)
@dataclasses.dataclass
class A:
  x: int


@registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class B:
  a: Any = registry.field()


class RegisterTest(parameterized.TestCase):

  def test_base(self):
    b = B(a=A(2))
    b_json = b.to_json()  # pyrefly: ignore[missing-attribute]
    self.assertEqual(b_json, '{"a": {"x": 2, "__type": "my_registry.A"}}')
    new_b = B.from_json(b_json)  # pyrefly: ignore[missing-attribute]
    test_util.assert_are_equal(self, b, new_b)

  def test_none(self):
    b = B(a=None)
    new_b = B.from_json(b.to_json())  # pyrefly: ignore[missing-attribute]
    test_util.assert_are_equal(self, b, new_b)

  def test_double_registration(self):
    with self.assertRaises(ValueError):

      @registry.register
      @dataclasses_json.dataclass_json
      @dataclasses.dataclass
      class A:
        x: int

  def test_unregistered_class(self):
    @dataclasses_json.dataclass_json
    @dataclasses.dataclass
    class C:
      x: int

    b = B(a=C(2))
    with self.assertRaises(ValueError):
      b.to_json()  # pyrefly: ignore[missing-attribute]

  def test_invalid_json_missing_field(self):
    with self.assertRaises(KeyError):
      _ = B.from_json('{"a": {"__type": "my_registry.A"}}')  # pyrefly: ignore[missing-attribute]

  def test_invalid_json_extra_field(self):
    with self.assertRaises(dataclasses_json.undefined.UndefinedParameterError):
      _ = B.from_json('{"a": {"x": 2, "y": 3, "__type": "my_registry.A"}}')  # pyrefly: ignore[missing-attribute]

  def test_invalid_json_wrong_field_type(self):
    with self.assertRaises(ValueError):
      _ = B.from_json('{"a": {"x": "hello", "__type": "my_registry.A"}}')  # pyrefly: ignore[missing-attribute]


if __name__ == '__main__':
  absltest.main()

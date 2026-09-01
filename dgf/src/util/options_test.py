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

from collections.abc import Mapping, MutableMapping
import concurrent.futures
import time

from absl.testing import absltest
from dgf.src.util import options as options_lib


class OptionsTest(absltest.TestCase):

  def test_context_basic(self):
    ctx = options_lib.Context({"a": 1, "b": 2})
    self.assertEqual(ctx.get_option("a"), 1)
    self.assertEqual(ctx.get_option("b"), 2)
    self.assertEqual(ctx.get("a"), 1)
    self.assertEqual(ctx["a"], 1)
    self.assertIsNone(ctx.get_option("c"))
    self.assertEqual(ctx.get_option("c", default=10), 10)
    self.assertEqual(ctx.get("c", 10), 10)
    self.assertIn("a", ctx)
    self.assertNotIn("c", ctx)
    self.assertLen(ctx, 2)
    self.assertEqual(set(ctx), {"a", "b"})
    self.assertEqual(ctx.to_dict(), {"a": 1, "b": 2})

    with self.assertRaises(KeyError):
      _ = ctx["c"]

    self.assertIsInstance(ctx, Mapping)
    self.assertNotIsInstance(ctx, MutableMapping)
    self.assertFalse(hasattr(ctx, "set_option"))
    self.assertFalse(hasattr(ctx, "reset_option"))
    self.assertFalse(hasattr(ctx, "__setitem__"))

    derived = options_lib.Context(ctx)
    self.assertEqual(derived.to_dict(), {"a": 1, "b": 2})

  def test_context_none_values(self):
    ctx = options_lib.Context({"seed": None})
    self.assertIn("seed", ctx)
    self.assertIsNone(ctx.get_option("seed", default=42))
    self.assertIsNone(ctx["seed"])

    parent = options_lib.Context({"seed": 42})
    child = options_lib.Context({"seed": None}, parent=parent)
    self.assertIn("seed", child)
    self.assertIsNone(child.get_option("seed", default=99))
    self.assertIsNone(child["seed"])

  def test_defensive_dict_copying(self):
    orig = {"a": 1, "b": 2}
    ctx = options_lib.Context(orig)
    orig["a"] = 999
    orig["c"] = 3
    self.assertEqual(ctx.get_option("a"), 1)
    self.assertNotIn("c", ctx)

  def test_context_hierarchy(self):
    parent = options_lib.Context({"a": 1, "b": 2})
    child = options_lib.Context({"b": 20, "c": 30}, parent=parent)

    self.assertEqual(child.get_option("a"), 1)  # From parent
    self.assertEqual(child.get_option("b"), 20)  # Overridden in child
    self.assertEqual(child.get_option("c"), 30)  # From child
    self.assertEqual(child["a"], 1)

    self.assertEqual(child.to_dict(), {"a": 1, "b": 20, "c": 30})

  def test_context_copy_on_write_derivation(self):
    ctx1 = options_lib.Context({"a": 1, "b": 2})

    ctx2 = ctx1.with_option("b", 20).with_option("c", 30)
    self.assertEqual(ctx1.to_dict(), {"a": 1, "b": 2})  # ctx1 untouched
    self.assertEqual(ctx2.to_dict(), {"a": 1, "b": 20, "c": 30})

    ctx3 = ctx1.with_options(b=200, d=400)
    self.assertEqual(ctx1.to_dict(), {"a": 1, "b": 2})  # ctx1 untouched
    self.assertEqual(ctx3.to_dict(), {"a": 1, "b": 200, "d": 400})

    ctx4 = ctx2.without_option("b")
    self.assertEqual(ctx4.to_dict(), {"a": 1, "c": 30})

    ctx5 = ctx2.without_option()
    self.assertEqual(ctx5.to_dict(), {})

  def test_manager_context(self):
    manager = options_lib.Manager("test_manager", defaults={"width": 100})
    self.assertEqual(manager.get("width"), 100)
    self.assertIsNone(manager.get("height"))

    with manager.context(width=200, height=50) as ctx:
      self.assertEqual(manager.get("width"), 200)
      self.assertEqual(manager.get("height"), 50)
      self.assertEqual(ctx.get_option("width"), 200)
      self.assertEqual(manager.to_dict(), {"width": 200, "height": 50})

      with manager(height=80, color="blue", layout=None):
        self.assertEqual(manager.get("width"), 200)  # Inherited from outer
        self.assertEqual(manager.get("height"), 80)  # Overridden in inner
        self.assertEqual(manager.get("color"), "blue")
        self.assertIsNone(manager.get("layout", default="force"))

      self.assertEqual(manager.get("height"), 50)
      self.assertIsNone(manager.get("color"))

    self.assertEqual(manager.get("width"), 100)
    self.assertIsNone(manager.get("height"))

  def test_manager_mapping_protocol(self):
    manager = options_lib.Manager("test_mapping", defaults={"a": 1, "b": 2})
    self.assertIsInstance(manager, MutableMapping)
    self.assertLen(manager, 2)
    self.assertEqual(set(manager), {"a", "b"})
    self.assertIn("a", manager)
    self.assertNotIn("c", manager)
    self.assertEqual(manager["a"], 1)

    manager["c"] = 3
    self.assertLen(manager, 3)
    self.assertEqual(manager["c"], 3)
    self.assertIn("c", manager)

    del manager["c"]
    self.assertNotIn("c", manager)
    self.assertLen(manager, 2)

    with self.assertRaises(KeyError):
      del manager["non_existent"]

    with self.assertRaises(KeyError):
      _ = manager["missing"]

    with manager(a=10, d=40):
      self.assertLen(manager, 3)
      self.assertEqual(set(manager), {"a", "b", "d"})
      self.assertEqual(manager["a"], 10)
      self.assertEqual(manager["d"], 40)

    self.assertLen(manager, 2)
    self.assertEqual(manager["a"], 1)
    self.assertNotIn("d", manager)

  def test_cow_thread_isolation(self):
    manager = options_lib.Manager("thread_cow", defaults={"counter": 0})

    def worker(val: int) -> int:
      manager["counter"] = val
      time.sleep(0.01)
      return manager["counter"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      f1 = executor.submit(worker, 100)
      f2 = executor.submit(worker, 200)
      self.assertEqual(f1.result(), 100)
      self.assertEqual(f2.result(), 200)

    self.assertEqual(manager["counter"], 0)


if __name__ == "__main__":
  absltest.main()

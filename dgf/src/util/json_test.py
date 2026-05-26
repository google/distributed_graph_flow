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

import tempfile
from absl.testing import absltest
from dgf.src.util import json


class JsonTest(absltest.TestCase):

  def test_read_write_json(self):
    with tempfile.NamedTemporaryFile(delete=True) as f:
      path = f.name
      data = {"a": 1, "b": [2, 3], "c": {"d": 4}}
      json.write_json(path, data)
      loaded_data = json.read_json(path)
      self.assertEqual(data, loaded_data)


if __name__ == "__main__":
  absltest.main()

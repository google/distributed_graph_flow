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

"""IO of schema."""

from dgf.src.data import schema as schema_lib
from dgf.src.util import filesystem


def read_schema(path: str) -> schema_lib.GraphSchema:
  """Loads graph schema from disk in a json format.

  Usage example:

  ```python
    schema = gf.io.read_schema("path/to/schema.json")
  ```

  Args:
    path: Input path.

  Returns:
    The loaded graph schema.
  """
  with filesystem.open_read(path) as f:
    return schema_lib.GraphSchema.from_json(f.read())


def write_schema(schema: schema_lib.GraphSchema, path: str):
  """Saves graph schema to disk in a json format.

  Usage example:

  ```python
    schema = gf.io.read_schema("path/to/output.json")
    gf.io.write_schema(schema, "path/to/output_again.json")
  ```

  Args:
    schema: The schema to save.
    path: Output path.
  """
  with filesystem.open_write(path) as f:
    f.write(schema.to_json(indent=2))

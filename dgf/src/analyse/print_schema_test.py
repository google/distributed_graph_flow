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

import logging
from absl.testing import absltest
from dgf.src.analyse import print_schema as print_schema_lib
from dgf.src.util import gen_test_graph


class PrintSchema(absltest.TestCase):

  def test_basic(self):
    schema = gen_test_graph.generate_schema(
        node_ids=True, semantic=True, variable_length=True
    )
    str_schema = print_schema_lib.print_schema(schema, return_output=True)
    logging.info("str_schema:\n%s", str_schema)
    self.assertEqual(
        str_schema,
        """\
Graph Schema:

Node Sets:
  n1:
    | Feature   | Format   | Semantic    | Shape   | Num cat. vals   |
    |-----------|----------|-------------|---------|-----------------|
    | #id       | BYTES    | PRIMARY_ID  | None    | None            |
    | f1        | BYTES    | CATEGORICAL | (1,)    | None            |
    | f2        | FLOAT_32 | EMBEDDING   | (2,)    | None            |

  n2:
    | Feature   | Format     | Semantic   | Shape   | Num cat. vals   |
    |-----------|------------|------------|---------|-----------------|
    | #id       | INTEGER_64 | PRIMARY_ID | None    | None            |
    | f3        | INTEGER_64 | NUMERICAL  | None    | None            |
    | f4        | INTEGER_64 | NUMERICAL  | ()      | None            |
    | f5        | INTEGER_64 | NUMERICAL  | (None,) | None            |


Edge Sets:
  e1: (Source: n1, Target: n1)
    (No features)

  e2: (Source: n1, Target: n2)
    (No features)
""",
    )


if __name__ == "__main__":
  absltest.main()

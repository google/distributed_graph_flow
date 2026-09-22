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
from absl.testing import parameterized
from dgf.src.analyse import print_schema as print_schema_lib
from dgf.src.data import schema as schema_lib
from dgf.src.util import gen_test_graph


class FormatDetail(parameterized.TestCase):

  @parameterized.named_parameters(
      ("empty", {}, ""),
      ("num_categorical_values", {"num_categorical_values": 30}, "#num.cat:30"),
      ("zero_categorical_values", {"num_categorical_values": 0}, "#num.cat:0"),
      ("is_utf8_string", {"is_utf8_string": True}, "utf8"),
      ("is_timeseries", {"is_timeseries": True}, "timeseries"),
      ("is_creation_time", {"is_creation_time": True}, "creation"),
      # The group is reported in a dedicated column, not in the details.
      ("group", {"group": "g1"}, ""),
      (
          "all",
          {
              "num_categorical_values": 30,
              "is_utf8_string": True,
              "is_timeseries": True,
              "is_creation_time": True,
              "group": "g1",
          },
          "#num.cat:30, utf8, timeseries, creation",
      ),
  )
  def test_format_detail(self, kwargs, expected):
    feature_schema = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64, **kwargs
    )
    self.assertEqual(print_schema_lib._format_detail(feature_schema), expected)


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
    | Feature   | Format   | Semantic    | Shape   | Detail   |
    |-----------|----------|-------------|---------|----------|
    | #id       | BYTES    | PRIMARY_ID  | None    |          |
    | f1        | BYTES    | CATEGORICAL | (1,)    |          |
    | f2        | FLOAT_32 | EMBEDDING   | (2,)    |          |

  n2:
    | Feature   | Format     | Semantic   | Shape     | Detail   |
    |-----------|------------|------------|-----------|----------|
    | #id       | INTEGER_64 | PRIMARY_ID | None      |          |
    | f3        | INTEGER_64 | NUMERICAL  | None      |          |
    | f4        | INTEGER_64 | NUMERICAL  | ()        |          |
    | f5        | INTEGER_64 | NUMERICAL  | (None,)   |          |
    | f6        | INTEGER_64 | NUMERICAL  | (None, 2) |          |


Edge Sets:
  e1: (Source: n1, Target: n1)
    (No features)

  e2: (Source: n1, Target: n2)
    (No features)
""",
    )

  def test_details(self):
    schema = gen_test_graph.generate_schema(
        node_ids=True, semantic=True, variable_length=True
    )
    n1_features = schema.node_sets["n1"].features
    n2_features = schema.node_sets["n2"].features
    n1_features["f1"].num_categorical_values = 30
    n1_features["f1"].is_utf8_string = True
    n2_features["f3"].is_timeseries = True
    n2_features["f3"].group = "g1"
    n2_features["f4"].is_creation_time = True
    n2_features["f5"].num_categorical_values = 5
    n2_features["f5"].is_utf8_string = True
    n2_features["f5"].is_timeseries = True
    n2_features["f5"].is_creation_time = True
    n2_features["f5"].group = "g2"

    str_schema = print_schema_lib.print_schema(schema, return_output=True)
    logging.info("str_schema:\n%s", str_schema)
    self.assertEqual(
        str_schema,
        """\
Graph Schema:

Node Sets:
  n1:
    | Feature   | Format   | Semantic    | Shape   | Detail            |
    |-----------|----------|-------------|---------|-------------------|
    | #id       | BYTES    | PRIMARY_ID  | None    |                   |
    | f1        | BYTES    | CATEGORICAL | (1,)    | #num.cat:30, utf8 |
    | f2        | FLOAT_32 | EMBEDDING   | (2,)    |                   |

  n2:
    | Feature   | Format     | Semantic   | Shape     | Group   | Detail                                 |
    |-----------|------------|------------|-----------|---------|----------------------------------------|
    | #id       | INTEGER_64 | PRIMARY_ID | None      |         |                                        |
    | f3        | INTEGER_64 | NUMERICAL  | None      | g1      | timeseries                             |
    | f4        | INTEGER_64 | NUMERICAL  | ()        |         | creation                               |
    | f5        | INTEGER_64 | NUMERICAL  | (None,)   | g2      | #num.cat:5, utf8, timeseries, creation |
    | f6        | INTEGER_64 | NUMERICAL  | (None, 2) |         |                                        |


Edge Sets:
  e1: (Source: n1, Target: n1)
    (No features)

  e2: (Source: n1, Target: n2)
    (No features)
""",
    )


if __name__ == "__main__":
  absltest.main()

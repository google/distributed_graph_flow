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

from absl.testing import absltest
from dgf.src.data import schema_ext as lib
from dgf.src.util import gen_test_graph


class SchemaCC(absltest.TestCase):

  def test_parse_schema(self):
    schema = gen_test_graph.generate_schema(node_ids=True)
    self.assertEqual(
        lib.ParseAndDebugPrintSchema(schema),
        """\
GraphSchema(nodesets=[
  Nodeset(name='n1', features=[
    Feature(name='#id', shape=[], format=BYTES),
    Feature(name='f1', shape=[1], format=BYTES),
    Feature(name='f2', shape=[2], format=FLOAT_32)
  ]),
  Nodeset(name='n2', features=[
    Feature(name='#id', shape=[], format=INTEGER_64),
    Feature(name='f3', shape=[], format=INTEGER_64),
    Feature(name='f4', shape=[], format=INTEGER_64),
    Feature(name='f5', shape=[None], format=INTEGER_64)
  ])
], edgesets=[
  Edgeset(name='e1', source_nodeset=0, target_nodeset=0, features=[]),
  Edgeset(name='e2', source_nodeset=0, target_nodeset=1, features=[])
])""",
    )


if __name__ == "__main__":
  absltest.main()

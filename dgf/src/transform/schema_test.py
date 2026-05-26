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
from absl.testing import parameterized
from dgf.src.data import schema as schema_lib
from dgf.src.transform import schema as schema_filter_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util


_SRC_SCHEMA = gen_test_graph.generate_schema(
    node_ids=True, edge_ids=True, semantic=True, variable_length=True
)


class SchemaTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_filter',
          graph_schema_filter=schema_lib.GraphSchemaFilter(),
          expected_schema=_SRC_SCHEMA,
      ),
      dict(
          testcase_name='filter_n1',
          graph_schema_filter=schema_lib.GraphSchemaFilter(
              nodeset_fn=lambda name, _: name == 'n1'
          ),
          expected_schema=schema_lib.GraphSchema(
              node_sets={'n1': _SRC_SCHEMA.node_sets['n1']},
              edge_sets={'e1': _SRC_SCHEMA.edge_sets['e1']},
          ),
      ),
      dict(
          testcase_name='filter_e1',
          graph_schema_filter=schema_lib.GraphSchemaFilter(
              edgeset_fn=lambda name, _: name == 'e1'
          ),
          expected_schema=schema_lib.GraphSchema(
              node_sets=_SRC_SCHEMA.node_sets,
              edge_sets={'e1': _SRC_SCHEMA.edge_sets['e1']},
          ),
      ),
      dict(
          testcase_name='filter_n1_and_e2',
          graph_schema_filter=schema_lib.GraphSchemaFilter(
              nodeset_fn=lambda name, _: name == 'n1',
              edgeset_fn=lambda name, _: name == 'e2',
          ),
          expected_schema=schema_lib.GraphSchema(
              node_sets={'n1': _SRC_SCHEMA.node_sets['n1']},
              edge_sets={},
          ),
      ),
      dict(
          testcase_name='filter_no_match',
          graph_schema_filter=schema_lib.GraphSchemaFilter(
              nodeset_fn=lambda name, _: name == 'non_existent'
          ),
          expected_schema=schema_lib.GraphSchema(node_sets={}, edge_sets={}),
      ),
  )
  def test_basic(
      self,
      graph_schema_filter: schema_lib.GraphSchemaFilter,
      expected_schema: schema_lib.GraphSchema,
  ):
    test_util.assert_are_equal(
        self,
        schema_filter_lib.filter_schema(_SRC_SCHEMA, graph_schema_filter),
        expected_schema,
    )

  def test_drop_edge_features(self):
    test_util.assert_are_equal(
        self,
        schema_filter_lib.drop_edge_features_from_schema(_SRC_SCHEMA),
        schema_lib.GraphSchema(
            node_sets=_SRC_SCHEMA.node_sets,
            edge_sets={
                'e1': schema_lib.EdgeSchema(
                    source='n1',
                    target='n1',
                    features={},
                ),
                'e2': schema_lib.EdgeSchema(
                    source='n1',
                    target='n2',
                    features={},
                ),
            },
        ),
    )


if __name__ == '__main__':
  absltest.main()

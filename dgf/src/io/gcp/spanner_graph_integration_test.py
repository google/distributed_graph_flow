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

r"""Manual tests for spanner_graph.
"""

from absl.testing import absltest
from dgf.src.data import schema as schema_lib
from dgf.src.io.gcp import spanner_graph
from dgf.src.util import test_util
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib


def expected_schema() -> schema_lib.GraphSchema:
  return schema_lib.GraphSchema(
      node_sets={
          "nodes": schema_lib.NodeSchema(
              features={
                  "id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                      is_utf8_string=True,
                  ),
                  "feat": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_64,
                      shape=(None,),
                  ),
                  "labels": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                  ),
                  "year": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                  ),
              }
          )
      },
      edge_sets={
          "edges": schema_lib.EdgeSchema(
              source="nodes",
              target="nodes",
          )
      },
  )


class SpannerGraphManualTest(absltest.TestCase):

  def test_read_spanner_graph(self):
    graph, schema = spanner_graph.read_spanner_graph(
        project="biggraphs-poc",
        instance="gcp-gnns",
        database="ogbn_arxiv",
        graph="ogbn_arxiv",
        verbose=2,
    )
    in_memory_graph_validate_lib.validate_graph(
        graph, schema, raise_on_warning=False
    )
    test_util.assert_are_equal(self, schema, expected_schema())


if __name__ == "__main__":
  absltest.main()

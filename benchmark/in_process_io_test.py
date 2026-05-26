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

import os
import tempfile
from unittest import mock

from absl.testing import absltest
import dgf
from dgf.benchmark import in_process_io
from dgf.src.util import gen_test_graph


class InProcessIOTest(absltest.TestCase):

  def test_base(self):
    with tempfile.TemporaryDirectory() as tmpdir:

      # Prepare some data
      work_dir = os.path.join(tmpdir, "workdir")
      os.makedirs(work_dir, exist_ok=True)

      hgraph_path = os.path.join(tmpdir, "hgraph")
      gen_test_graph.generate_hgraph(
          hgraph_path, node_id=True, edge_id=False, variable_length=False
      )

      gf_graph_path = os.path.join(tmpdir, "gf_graph")
      gen_test_graph.generate_gf_graph(
          gf_graph_path, edge_ids=True, variable_length=False
      )

      tf_graph_samples_path = os.path.join(tmpdir, "tfgraph_sample")
      os.makedirs(tf_graph_samples_path, exist_ok=True)

      graph = gen_test_graph.generate_in_memory_graph(
          node_ids=True, edge_ids=False, variable_length=False
      )
      schema = gen_test_graph.generate_schema(
          node_ids=True, edge_ids=False, variable_length=False
      )

      def in_mem_graphs():
        yield graph
        yield graph
        yield graph

      dgf.io.write_tfgnn_graphs(
          in_mem_graphs(),
          os.path.join(tf_graph_samples_path, "data@20.rio"),
          schema=schema,
      )
      schema = gen_test_graph.generate_schema(variable_length=False)
      dgf.io.write_schema(
          schema, os.path.join(tf_graph_samples_path, "schema.json")
      )

      with mock.patch(
          "dgf.io.read_spanner_graph"
      ) as mock_read_spanner:

        mock_read_spanner.return_value = (graph, schema)

        # Run benchmark
        in_process_io.io_in_memory_dataset_in_process(
            work_dir=work_dir,
            hgraph_path=hgraph_path,
            gf_graph_path=gf_graph_path,
            tf_graph_samples_path=tf_graph_samples_path,
            spanner_config=in_process_io.SpannerGraphConfig(
                project_id="test_project",
                instance_id="test_instance",
                database_id="test_database",
                graph_id="SpannerGraph",
            ),
            spanner_write_config=in_process_io.SpannerGraphConfig(
                project_id="test_project_2",
                instance_id="test_instance_2",
                database_id="test_database_2",
                graph_id="SpannerGraph2",
            ),
        )

      self.assertEqual(mock_read_spanner.call_count, 1)

      mock_read_spanner.assert_called_with(
          project="test_project",
          instance="test_instance",
          database="test_database",
          graph="SpannerGraph",
          verbose=True,
      )


if __name__ == "__main__":
  absltest.main()

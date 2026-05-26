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
from dgf.src.io.gcp import bigquery_graph_metadata as bqgm_lib
from dgf.src.io.gcp import common
from dgf.src.io.gcp import parquet_export


class ParquetExportTest(parameterized.TestCase):

  def test_create_nodeset_export_sql(self):
    node_table = bqgm_lib.NodeTable(
        data_source_table=bqgm_lib.DataSourceTable(
            project_id="project", dataset_id="dataset", table_id="table"
        ),
        key_columns=["k1", "k2"],
        label_and_properties=[
            bqgm_lib.LabelAndProperties(
                label="Label1",
                properties=[
                    bqgm_lib.Property(
                        data_type=bqgm_lib.DataType(type_kind="STRING"),
                        expression="col1",
                        name="prop1",
                    )
                ],
            )
        ],
        name="NodeTable1",
    )
    gcs_prefix = "gs://bucket/path"
    sql = parquet_export.create_export_sql(
        parquet_export.create_nodeset_sql(node_table),
        gcs_prefix,
        common.GCS_PREFIX_NODESETS,
        node_table.name,
    )

    self.assertIn("EXPORT DATA", sql)
    self.assertIn("URI = 'gs://bucket/path/nodesets/NodeTable1-*.parquet'", sql)
    self.assertIn("FROM `project.dataset.table`", sql)
    self.assertIn("CONCAT(`k1`, `k2`) AS `#id`", sql)
    self.assertIn("`col1` AS `prop1`", sql)


if __name__ == "__main__":
  absltest.main()

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
        key_columns=["k1"],
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
    self.assertNotIn("#id", sql)
    self.assertIn("COALESCE(`col1`, '') AS `prop1`", sql)

  @parameterized.parameters(
      ("INT64", None, "COALESCE(`col`, 0) AS `prop`"),
      ("FLOAT64", None, "COALESCE(`col`, 0.0) AS `prop`"),
      ("BOOL", None, "COALESCE(`col`, FALSE) AS `prop`"),
      ("BYTES", None, "COALESCE(`col`, b'') AS `prop`"),
      ("TIMESTAMP", None, "COALESCE(UNIX_MICROS(`col`), 0) AS `prop`"),
      ("ARRAY", "FLOAT64", "`col` AS `prop`"),
      (
          "ARRAY",
          "TIMESTAMP",
          "ARRAY(SELECT UNIX_MICROS(x) FROM UNNEST(`col`) AS x) AS `prop`",
      ),
  )
  def test_property_projection(
      self, type_kind: str, element_type_kind: str | None, expected: str
  ):
    data_type = bqgm_lib.DataType(
        type_kind=type_kind,
        array_element_type=(
            bqgm_lib.ArrayElementType(type_kind=element_type_kind)
            if element_type_kind
            else None
        ),
    )
    edge_table = bqgm_lib.EdgeTable(
        data_source_table=bqgm_lib.DataSourceTable(
            project_id="project", dataset_id="dataset", table_id="table"
        ),
        source_node_reference=bqgm_lib.NodeReference(
            edge_table_columns=["source"], node_table="n", node_table_columns=["id"]
        ),
        destination_node_reference=bqgm_lib.NodeReference(
            edge_table_columns=["target"], node_table="n", node_table_columns=["id"]
        ),
        key_columns=["source", "target"],
        label_and_properties=[
            bqgm_lib.LabelAndProperties(
                label="e",
                properties=[
                    bqgm_lib.Property(
                        data_type=data_type, expression="col", name="prop"
                    )
                ],
            )
        ],
        name="e",
    )
    self.assertIn(expected, parquet_export.create_edgeset_sql(edge_table))

  def test_composite_edge_endpoint_raises(self):
    edge_table = bqgm_lib.EdgeTable(
        data_source_table=bqgm_lib.DataSourceTable(
            project_id="project", dataset_id="dataset", table_id="table"
        ),
        source_node_reference=bqgm_lib.NodeReference(
            edge_table_columns=["source_a", "source_b"],
            node_table="n",
            node_table_columns=["a", "b"],
        ),
        destination_node_reference=bqgm_lib.NodeReference(
            edge_table_columns=["target"], node_table="n", node_table_columns=["id"]
        ),
        key_columns=["source_a", "source_b", "target"],
        label_and_properties=[],
        name="e",
    )
    with self.assertRaisesRegex(NotImplementedError, "Composite keys"):
      parquet_export.create_edgeset_sql(edge_table)


if __name__ == "__main__":
  absltest.main()

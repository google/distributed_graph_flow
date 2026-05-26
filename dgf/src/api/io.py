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

"""Read and write graph, schema, and other related data."""

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top,g-bad-import-order,reimported,disable=attribute-error

from dgf.src.io.dataset_loader import fetch_ogb_graph
from dgf.src.io.dataset_loader import fetch_graphland_graph

from dgf.src.io.hgraph_in_memory import read_graphai_hgraph

from dgf.src.io.tf_graph_sample import read_tfgnn_graphs
from dgf.src.io.tf_graph_sample import write_tfgnn_graphs


from dgf.src.io.statistics import read_feature_statistics
from dgf.src.io.statistics import write_feature_statistics

from dgf.src.io.schema import read_schema
from dgf.src.io.schema import write_schema

from dgf.src.io.gcp.spanner_graph import read_spanner_graph
from dgf.src.io.gcp.spanner_graph import read_spanner_graph_schema
from dgf.src.io.gcp.bigquery_graph import read_bigquery_graph_schema

from dgf.src.io.gcp.bigquery_graph import read_bigquery_graph
from dgf.src.io.gcp.bigquery_graph import read_bigquery_graph_schema
from dgf.src.io.gcp.bigquery_graph import export_bigquery_to_disk

from dgf.src.io.spanner import create_spanner_tables_from_graph_schema

from dgf.src.io.graph_in_memory import read_graph
from dgf.src.io.graph_in_memory import write_graph

from dgf.src.util.proto import read_text_proto
from dgf.src.util.proto import write_text_proto

from dgf.src.io.cache import cache

from dgf.src.util import filesystem

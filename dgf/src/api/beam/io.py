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

"""Functions to read and write graphs, schemas, and related data using Beam."""

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top,g-bad-import-order,reimported,disable=attribute-error


from dgf.src.io.hgraph_in_beam import read_graphai_hgraph
from dgf.src.io.hgraph_in_beam import write_graphai_hgraph

from dgf.src.io.tf_graph_sample import read_tfgnn_graphs_beam as read_tfgnn_graphs
from dgf.src.io.tf_graph_sample import write_tfgnn_graphs_beam as write_tfgnn_graphs

from dgf.src.io.statistics import write_feature_statistics_beam as write_feature_statistics

from dgf.src.io.spanner import CreateSpannerTables
from dgf.src.io.spanner import write_spanner
from dgf.src.io.spanner import write_node_set_to_spanner  # TODO(gbm): Needed?
from dgf.src.io.spanner import write_edge_set_to_spanner  # TODO(gbm): Needed?

from dgf.src.io.graph_in_beam import read_graph
from dgf.src.io.graph_in_beam import write_graph

from dgf.src.io.gcp.bigquery_graph_beam import distributed_read_beam as read_bigquery_graph
from dgf.src.io.gcp.spanner_graph_beam import distributed_read_beam as read_spanner_graph

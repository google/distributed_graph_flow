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

"""Convert object formats e.g. convert a graph to a Sparse Deferred struct."""

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top,g-bad-import-order,reimported,disable=attribute-error

from dgf.src.io.hgraph_in_memory import tfgnn_schema_to_schema
from dgf.src.io.hgraph_in_memory import schema_to_tfgnn_schema

from dgf.src.io.tf_graph_sample import tfgnn_graph_to_graph
from dgf.src.io.tf_graph_sample import graph_dict_to_graph

from dgf.src.io.tf_graph_sample import graph_to_tfgnn_graph
from dgf.src.io.tf_graph_sample import graph_to_tfgnn_graph_dict

from dgf.src.io.tf_graph_sample import graph_to_serialized_tfgnn_graph
from dgf.src.io.tf_graph_sample import graphs_to_serialized_tfgnn_graphs

from dgf.src.io.tf import graph_to_tf_graph
from dgf.src.io.tf import tf_graph_to_tf_graph_dict
from dgf.src.io.tf import tf_graph_dict_to_tf_graph

from dgf.src.io.spanner import schema_to_spanner_ddl

from dgf.src.io.networkx import graph_to_networkx
from dgf.src.io.networkx import networkx_to_graph

from dgf.src.io.sparse_deferred import sparse_deferred_struct_to_graph
from dgf.src.io.sparse_deferred import graph_to_sparse_deferred_struct
from dgf.src.io.sparse_deferred import schema_to_sparse_deferred_schema

from dgf.src.io.jax import graph_to_jax_graph

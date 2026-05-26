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

"""Transform graph data into other graphs data."""

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top,g-bad-import-order,reimported,disable=attribute-error


from dgf.src.transform.merge import merge_graphs
from dgf.src.transform.merge import remove_padding_sentinels

from dgf.src.transform.normalize import GraphNormalizer
from dgf.src.transform.normalize import GraphNormalizerConfig
from dgf.src.transform.normalize import auto_normalize
from dgf.src.transform.normalize import AutoNormalizeConfig
from dgf.src.transform.normalize import DictionaryIndexNormalizer
from dgf.src.transform.normalize import IdentityNormalizer
from dgf.src.transform.normalize import SoftQuantileNormalizer

from dgf.src.transform.extract import filter_schema
from dgf.src.transform.extract import filter_graph
from dgf.src.transform.extract import drop_edge_features

from dgf.src.transform.in_memory_graph_filter import filter_graphs
from dgf.src.transform.in_memory_graph_filter import NumNodesPredicate
from dgf.src.transform.in_memory_graph_filter import ContainsLabelPredicate

from dgf.src.util.util import batch_indices_generator
from dgf.src.learning.ten_lines.node_prediction_dataset import GNNDatasetPreparator

from dgf.src.transform.schema import filter_schema
from dgf.src.transform.schema import drop_edge_features_from_schema

from dgf.src.transform.nx import homogeneous_graph_piece_to_nx

from dgf.src.transform.homogenize import homogenize
from dgf.src.transform.homogenize import apply_feature

from dgf.src.transform.temporal import propagate_timestamp_to_edges
from dgf.src.transform.table_2_graph import table2graph

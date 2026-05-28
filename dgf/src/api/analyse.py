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

"""Utilities to analyze graphs, e.g., feature and graph statistics."""

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top,g-bad-import-order,reimported,disable=attribute-error


from dgf.src.analyse.padding import padding_from_graph_generator
from dgf.src.analyse.in_process_feature_statistics import feature_statistics_from_graphs
from dgf.src.analyse.in_process_feature_statistics import feature_statistics

# TODO: Use third_party/py/dgf/src/api/print.py version instead.
from dgf.src.analyse.print_schema import print_schema

from dgf.src.analyse.reports import data_model as reports_data_model
from dgf.src.analyse.reports import reporter
from dgf.src.analyse.topology import global_graph_topology
from dgf.src.analyse.topology import node_degree

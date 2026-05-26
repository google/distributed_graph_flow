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

"""Tools to generate synthetic data."""

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top,g-bad-import-order,reimported,disable=attribute-error

from dgf.src.generate.edge_neighbor_generator import EdgeNeighborGenerator
from dgf.src.generate.edge_neighbor_generator import RandomNegativeSampler
from dgf.src.generate.edge_neighbor_generator import RandomWalkNegativeSampler

from dgf.src.generate.graphs import SyntheticGraphSampleConfig
from dgf.src.generate.graphs import generate_synthetic_graph_sample
from dgf.src.generate.graphs import write_synthetic_graph_sample_as_tfgnn_graphs

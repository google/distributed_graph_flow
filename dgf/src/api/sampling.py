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

"""Extracts subsets of graphs for GNN training."""

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top,g-bad-import-order,reimported,disable=attribute-error


from dgf.src.sampling.in_memory_sampler import create_sampler
from dgf.src.sampling.in_memory_sampler import Sampler

from dgf.src.sampling.config import SimpleSamplingConfig
from dgf.src.sampling.config import SamplingPlan
from dgf.src.sampling.config import simple_sampling_config_to_sampling_plan

# TODO: Use third_party/py/dgf/src/api/beam/sampling.py instead.
from dgf.src.sampling.beam_semi_distributed_sampler import extract_beam_nodes_ids
from dgf.src.sampling.beam_semi_distributed_sampler_v1 import sample_with_beam_semi_distributed_sampler
from dgf.src.sampling.beam_semi_distributed_sampler_v2 import sample_with_beam_semi_distributed_sampler_v2

# TODO(gbm): Move to a separate namespace.

from dgf.src.sampling.gcp.spanner_graph_sampler import create_graph_spanner_sampler
from dgf.src.sampling.gcp.spanner_graph_sampler import SpannerGraphSampler

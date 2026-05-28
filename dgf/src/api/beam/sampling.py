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

"""Functions to extract subsets of graphs for GNN training using Beam."""

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top,g-bad-import-order,reimported,disable=attribute-error


from dgf.src.sampling.beam_semi_distributed_sampler import extract_beam_nodes_ids as extract_nodes_ids
from dgf.src.sampling.beam_semi_distributed_sampler_v1 import sample_with_beam_semi_distributed_sampler as semi_distributed_sampler_v1
from dgf.src.sampling.beam_semi_distributed_sampler_v2 import sample_with_beam_semi_distributed_sampler_v2 as semi_distributed_sampler_v2

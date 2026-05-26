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

"""Classes to represent graph data. Contains no functions / algorithms."""

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top,g-bad-import-order,reimported,disable=attribute-error


from dgf.src.data.schema import GraphSchema

GraphSchemaV2 = GraphSchema  # Retro-compatiblity


from dgf.src.data.schema import EdgeSchema
from dgf.src.data.schema import NodeSchema
from dgf.src.data.schema import FeatureSchema
from dgf.src.data.schema import FeatureFormat
from dgf.src.data.schema import FeatureSemantic
from dgf.src.data.schema import GraphSchemaFilter

from dgf.src.data.in_memory_graph import InMemoryNodeSet
from dgf.src.data.in_memory_graph import InMemoryEdgeSet
from dgf.src.data.in_memory_graph import InMemoryGraph

from dgf.src.data.jax_in_memory_graph import JaxInMemoryGraph
from dgf.src.data.jax_in_memory_graph import JaxInMemoryNodeSet
from dgf.src.data.jax_in_memory_graph import JaxInMemoryEdgeSet

from dgf.src.data.tf_in_memory_graph import TFInMemoryGraph
from dgf.src.data.tf_in_memory_graph import TFInMemoryNodeSet
from dgf.src.data.tf_in_memory_graph import TFInMemoryEdgeSet
from dgf.src.data.tf_in_memory_graph import TFInMemoryGraphDict

from dgf.src.data.statistics import GraphFeatureStatistics
from dgf.src.data.statistics import FeatureStatistics
from dgf.src.data.statistics import FeatureSetStatistics

from dgf.src.data.padding import Padding
from dgf.src.data.padding import EdgeSetPadding
from dgf.src.data.padding import NodeSetPadding

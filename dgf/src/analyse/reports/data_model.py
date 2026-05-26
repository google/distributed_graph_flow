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

"""Data models for the reports module."""

import dataclasses
import datetime
from typing import Any, Dict, List, Optional, Union

from dgf.src.analyse.topology import global_graph_topology as global_graph_topology_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
import networkx as nx


@dataclasses.dataclass
class GraphStatsPayload:
  """Payload containing all statistics required for report generation.

  Attributes:
    dataset_name: Name of the dataset.
    task_type: Type of task (e.g., "Node Classification").
    feature_dimensionality: Dimensionality of node features.
    num_classes: Number of unique classes (if applicable).
    feature_stats: Optional detailed feature statistics.
    global_graph_topology: Global graph topology object.
    subgraphs: Optional list of NetworkX subgraphs to visualize.
    subgraphs: Optional list of NetworkX subgraphs to visualize.
    graph_schema: Optional Graph Schema (contains edge definitions for
      heterogeneous graphs).
    color_by_attribute: Attribute to use for node coloring (e.g., "label",
      "class", "gender").
    node_label_attribute: Attribute to use for node labeling (default: "id").
    visual_gallery_data: Derived data for PyVis visualization (list of dicts
      with 'pyvis_data', etc.)
    generated_at: Timestamp when the payload was created.
  """

  dataset_name: str
  task_type: Optional[str] = None
  feature_dimensionality: Optional[int] = None
  num_classes: Optional[int] = None
  feature_stats: Optional[statistics_lib.GraphFeatureStatistics] = None

  global_graph_topology: Optional[
      global_graph_topology_lib.GlobalGraphTopology
  ] = None

  # Visual Inspection
  # Optional list of NetworkX subgraphs or InMemoryGraphs to visualize
  subgraphs: Optional[
      List[Union[nx.Graph, in_memory_graph_lib.InMemoryGraph]]
  ] = None

  # Graph Schema (contains edge definitions for heterogeneous graphs)
  graph_schema: Optional[schema_lib.GraphSchema] = None

  ## TODO(tewariy): Add support for heterogeneous graph attributes.
  # Attribute to use for node coloring in Homogeneous graphs
  # (e.g., "label", "class", "gender")
  color_by_attribute: Optional[str] = None

  # Attribute to use for node labeling (default: "id")
  node_label_attribute: Optional[str] = None

  # Derived data for PyVis visualization (list of dicts with 'pyvis_data', etc.)
  visual_gallery_data: Optional[List[Dict[str, Any]]] = None

  generated_at: datetime.datetime = dataclasses.field(
      default_factory=datetime.datetime.now
  )

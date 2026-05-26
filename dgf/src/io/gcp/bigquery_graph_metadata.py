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

"""Dataclasses for BigQuery Graph metadata."""

import dataclasses
import datetime
from typing import Dict, List, Optional

import dataclasses_json


dataclass_json = dataclasses_json.dataclass_json
config = dataclasses_json.config
LetterCase = dataclasses_json.LetterCase
field = dataclasses.field
dataclass = dataclasses.dataclass


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class DataSourceTable:
  dataset_id: str
  project_id: str
  table_id: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class NodeReference:
  edge_table_columns: List[str]
  node_table: str
  node_table_columns: List[str]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ArrayElementType:
  type_kind: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class DataType:
  type_kind: str
  array_element_type: Optional[ArrayElementType] = None

  def resolved_data_type(self) -> str:
    """Returns the resolved data type of the DataType."""
    if self.array_element_type:
      return f"ARRAY<{self.array_element_type.type_kind}>"
    else:
      return self.type_kind


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Property:
  data_type: DataType
  expression: str
  name: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class LabelAndProperties:
  label: str
  properties: Optional[List[Property]] = None


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class EdgeTable:
  data_source_table: DataSourceTable
  destination_node_reference: NodeReference
  key_columns: List[str]
  label_and_properties: List[LabelAndProperties]
  name: str
  source_node_reference: NodeReference


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class NodeTable:
  data_source_table: DataSourceTable
  key_columns: List[str]
  label_and_properties: List[LabelAndProperties]
  name: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class PropertyGraphReference:
  dataset_id: str
  project_id: str
  property_graph_id: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class BigQueryGraphMetadata:
  """Dataclass for BigQuery Graph metadata."""

  creation_time: str
  edge_tables: List[EdgeTable]
  etag: str
  last_modified_time: str
  node_tables: List[NodeTable]
  property_graph_reference: PropertyGraphReference
  metadata_load_time: datetime.datetime = field(
      default_factory=datetime.datetime.now
  )
  nodeset: Optional[Dict[str, NodeTable]] = None
  edgeset: Optional[Dict[str, EdgeTable]] = None
  has_duplicate_labels: Optional[bool] = False

  def __post_init__(self):
    """To simplify node table look up during parquet export."""
    self.nodeset = {}
    self.edgeset = {}
    for node_table in self.node_tables:
      self.nodeset[node_table.name] = node_table
    for edge_table in self.edge_tables:
      self.edgeset[edge_table.name] = edge_table

    all_labels = set()
    for ge_table in self.node_tables + self.edge_tables:
      for label_and_properties in ge_table.label_and_properties:
        if label_and_properties.label in all_labels:
          self.has_duplicate_labels = True
          break
        else:
          all_labels.add(label_and_properties.label)
      if self.has_duplicate_labels:
        break

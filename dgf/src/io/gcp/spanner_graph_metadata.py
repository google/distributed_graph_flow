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

"""Dataclasses for Spanner Graph metadata."""

from collections import defaultdict
import dataclasses
from typing import Dict, List, Optional
import dataclasses_json


dataclass_json = dataclasses_json.dataclass_json
config = dataclasses_json.config
LetterCase = dataclasses_json.LetterCase
field = dataclasses.field
dataclass = dataclasses.dataclass


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class PropertyDefinition:
  property_declaration_name: str
  value_expression_sql: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class NodeTableRef:
  edge_table_columns: List[str]
  node_table_columns: List[str]
  node_table_name: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class EdgeTable:
  base_catalog_name: str
  base_schema_name: str
  base_table_name: str
  destination_node_table: NodeTableRef
  key_columns: List[str]
  kind: str
  label_names: List[str]
  name: str
  source_node_table: NodeTableRef
  property_definitions: Optional[List[PropertyDefinition]] = None


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Label:
  name: str
  property_declaration_names: Optional[List[str]] = None


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class NodeTable:
  base_catalog_name: str
  base_schema_name: str
  base_table_name: str
  key_columns: List[str]
  kind: str
  label_names: List[str]
  name: str
  property_definitions: Optional[List[PropertyDefinition]] = None


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class PropertyDeclaration:
  name: str
  type: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class SpannerGraphMetadata:
  """Dataclass for Spanner Graph metadata."""

  catalog: str
  edge_tables: List[EdgeTable]
  labels: List[Label]
  name: str
  node_tables: List[NodeTable]
  property_declarations: List[PropertyDeclaration]
  # The key 'schema' can conflict, so we map it to 'schema_field'
  # letter_case doesn't apply to keys with explicit field_name/data_key
  schema_field: str = field(metadata=config(field_name='schema'))
  property_types: Optional[Dict[str, str]] = None
  has_duplicate_labels: Optional[bool] = False

  def __post_init__(self):
    """To simplify property type look up."""

    self.property_types = defaultdict(str)
    for property_declaration in self.property_declarations:
      self.property_types[property_declaration.name] = property_declaration.type

    all_labels = set()
    for ge_table in self.node_tables + self.edge_tables:
      for label in ge_table.label_names:
        if label in all_labels:
          self.has_duplicate_labels = True
          break
        else:
          all_labels.add(label)
      if self.has_duplicate_labels:
        break

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

"""Generates a human-readable string representation of the graph schema."""

from dgf.src.data import schema as schema_lib
import tabulate


def _format_detail(feature_schema: schema_lib.FeatureSchema) -> str:
  """Formats the secondary properties of a feature into a compact string.

  Args:
    feature_schema: The schema of the feature.

  Returns:
    A comma separated list of details (e.g., "#num.cat:30, utf8"). Empty if the
    feature does not have any of those details.
  """
  details = []
  if feature_schema.num_categorical_values is not None:
    details.append(f"#num.cat:{feature_schema.num_categorical_values}")
  if feature_schema.is_utf8_string:
    details.append("utf8")
  if feature_schema.is_timeseries:
    details.append("timeseries")
  if feature_schema.is_creation_time:
    details.append("creation")
  return ", ".join(details)


def _format_feature_table(features: schema_lib.FeatureSetSchema) -> list[str]:
  """Generates an ASCII table for a set of features using tabulate."""
  if not features:
    return ["    (No features)"]

  # The "Group" column is only shown if at least one feature has a group.
  with_group = any(
      feature_schema.group is not None for feature_schema in features.values()
  )

  headers = ["Feature", "Format", "Semantic", "Shape"]
  if with_group:
    headers.append("Group")
  headers.append("Detail")

  data = []
  sorted_features = sorted(features.items())
  for name, feature_schema in sorted_features:
    shape_str = (
        str(feature_schema.shape)
        if feature_schema.shape is not None
        else "None"
    )
    row = [
        name,
        feature_schema.format.name,
        feature_schema.semantic.name,
        shape_str,
    ]
    if with_group:
      row.append(feature_schema.group or "")
    row.append(_format_detail(feature_schema))
    data.append(row)

  table = tabulate.tabulate(data, headers=headers, tablefmt="github")
  return ["    " + line for line in table.splitlines()]


def print_schema(
    schema: schema_lib.GraphSchema,
    return_output: bool = False,
    header: bool = True,
) -> str | None:
  """Generates a human-readable string representation of a graph schema.

  Usage example:

  ```python
    graph, schema = dgf.io.read_graph("/path/to/graph")
    dgf.analysis.print_schema(schema)
  ```

  Args:
    schema: The graph schema to print.
    return_output: If true, returns the output text instead of printing it.
    header: If true, print the "Graph Schema" header.

  Returns:
    A string containing the human-readable representation of the schema.
  """
  lines = []

  if header:
    lines.append("Graph Schema:\n")

  # Node Sets
  lines.append("Node Sets:")
  if not schema.node_sets:
    lines.append("  (No node sets)")
  else:
    for node_name in sorted(schema.node_sets.keys()):
      node_schema = schema.node_sets[node_name]
      lines.append(f"  {node_name}:")
      lines.extend(_format_feature_table(node_schema.features))
      lines.append("")

  # Edge Sets
  lines.append("\nEdge Sets:")
  if not schema.edge_sets:
    lines.append("  (No edge sets)")
  else:
    for edge_name in sorted(schema.edge_sets.keys()):
      edge_schema = schema.edge_sets[edge_name]
      lines.append(
          f"  {edge_name}: (Source: {edge_schema.source}, Target:"
          f" {edge_schema.target})"
      )
      lines.extend(_format_feature_table(edge_schema.features))
      lines.append("")

  text_content = "\n".join(lines)

  if return_output:
    return text_content
  else:
    print(text_content)

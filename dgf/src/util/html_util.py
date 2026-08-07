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

"""Utilities for HTML report generation."""

from typing import Optional


def get_table_style(component_id: Optional[str] = None) -> str:
  """CSS style string for a DGF table.

  Args:
    component_id: Optional string for the DOM ID. If provided, constraints the
      style exclusively to elements beneath this ID.

  Returns:
    A CSS styles block as a string.
  """
  prefix = f"#{component_id} " if component_id else ""
  return f"""
  {prefix}.dgf-table {{
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 20px;
    font-size: inherit;
    font-family: 'Roboto', sans-serif;
  }}
  {prefix}.dgf-table td {{
    padding: 4px 8px;
    text-align: left;
  }}
  {prefix}.dgf-table td:first-child {{
    white-space: nowrap;
  }}
  {prefix}.dgf-table td:last-child {{
    width: 100%;
  }}
  {prefix}.dgf-table tr:nth-child(odd) {{
    background-color: #f8f9fa;
  }}
"""

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

from typing import List, Optional
from dgf.src.sampling import config as config_lib


def print_sampling_plan(
    plan: config_lib.SamplingPlan,
    return_output: bool = False,
    header: bool = True,
) -> Optional[str]:
  """Generates a human-readable tree representation of a sampling plan.

  Args:
    plan: The sampling plan to print.
    return_output: If true, returns the output text instead of printing it.
    header: If true, print the "Sampling Plan" header.

  Returns:
    A string containing the human-readable representation of the sampling plan.
  """
  lines = []

  if header:
    lines.append("Sampling Plan:\n")

  extra_info = ""
  if plan.with_replacement:
    extra_info = " (with replacement)"
  lines.append(f"Root: {plan.root.nodeset}{extra_info}")

  _append_plan_node_str(plan.root, lines, prefix="")

  text_content = "\n".join(lines)

  if return_output:
    return text_content
  else:
    print(text_content)
    return None


def _append_plan_node_str(
    node: config_lib.PlanNode, lines: List[str], prefix: str
):
  """Recursively appends the string representation of the plan tree."""
  if not node.children:
    return

  for i, edge in enumerate(node.children):
    is_last = i == len(node.children) - 1
    marker = "└── " if is_last else "├── "

    edge_desc = f"{edge.edgeset}"
    if edge.reversed:
      edge_desc += " (reversed)"
    edge_desc += f" [width={edge.hop_width}] ➔ {edge.node.nodeset}"

    lines.append(f"{prefix}{marker}{edge_desc}")

    new_prefix = prefix + ("    " if is_last else "│   ")
    _append_plan_node_str(edge.node, lines, new_prefix)

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

r"""Generates a graph sampling plan from a graph schema.

This binary is a simple wrapper on top of:
  dgf.sampling.simple_sampling_config_to_sampling_plan

Usage example:

blaze run //third_party/py/dgf/src/bin:create_graph_sampling_plan -- \
  --input_graph_schema=/cns/iz-d/home/research-graph/public/graphflow_datasets/fetch_repo/ogb_mag/schema.json \
  --output_sampling_plan=/tmp/sampling_config.json \
  --seed_nodeset=paper \
  --num_hops=2 \
  --hop_width=10
"""

from absl import app
from absl import flags
import dgf

_INPUT_GRAPH_SCHEMA = flags.DEFINE_string(
    "input_graph_schema",
    None,
    "Path to the input JSON graph schema.",
    required=True,
)
_OUTPUT_SAMPLING_PLAN = flags.DEFINE_string(
    "output_sampling_plan",
    None,
    "Path to the output JSON sampling plan.",
    required=True,
)
_SEED_NODESET = flags.DEFINE_string(
    "seed_nodeset",
    "paper",
    "The seed nodeset to start sampling from.",
)
_NUM_HOPS = flags.DEFINE_integer(
    "num_hops",
    2,
    "Number of sampling hops.",
)
_HOP_WIDTH = flags.DEFINE_integer(
    "hop_width",
    10,
    "Width of sampling per hop.",
)
_REVERSE = flags.DEFINE_bool(
    "reverse",
    True,
    "Should edges be traversed in both directions.",
)
_PRINT_PLAN = flags.DEFINE_bool(
    "print_plan",
    True,
    "Printt a nice display of the sampling_plan.",
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Read the graph schema
  schema = dgf.io.read_schema(_INPUT_GRAPH_SCHEMA.value)

  # Create the simple sampling config
  simple_config = dgf.sampling.SimpleSamplingConfig(
      seed_nodeset=_SEED_NODESET.value,
      num_hops=_NUM_HOPS.value,
      hop_width=_HOP_WIDTH.value,
      reverse=_REVERSE.value,
  )

  # Convert the config to a full sampling plan
  plan = dgf.sampling.simple_sampling_config_to_sampling_plan(
      simple_config, schema=schema
  )

  if _PRINT_PLAN.value:
    dgf.print.sampling_plan(plan)

  # Write the plan as indented JSON using gfile
  with dgf.filesystem.open_write(_OUTPUT_SAMPLING_PLAN.value) as f:
    f.write(plan.to_json(indent=2))  # pyrefly: ignore[missing-attribute]


if __name__ == "__main__":
  app.run(main)

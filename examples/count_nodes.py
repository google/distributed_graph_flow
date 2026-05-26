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

r"""This example shows how to count the number of nodes in an HGraph.
"""

from collections.abc import Sequence

from absl import app
from absl import flags
import apache_beam as beam
import dgf

_INPUT = flags.DEFINE_string(
    "input",
    None,
    "Input dataset (path to a GZip TF.Record of TF.Examples).",
    required=True,
)
_INPUT_FORMAT = flags.DEFINE_string(
    "input_format",
    "TF_RECORD",
    "Format of the HGraph.",
)
_NODESET = flags.DEFINE_string(
    "nodeset",
    None,
    "Nodeset to count notes in.",
    required=True,
)

_RUNNER = flags.DEFINE_string(
    "runner", "DirectRunner", "The apache beam runner."
)


def run(input_path: str, input_format: str, target_nodeset: str) -> None:

  def pipeline(root: beam.Pipeline):
    graph = dgf.beam.io.read_graphai_hgraph(root, input_path, input_format)
    num_nodes = (
        graph.node_sets[target_nodeset]
        | "Count nodes" >> beam.combiners.Count.Globally()
    )
    _ = num_nodes | "Number of nodes: " >> beam.Map(print)

  # TODO(gbm): Make this example OSS compatible.
  runner = dgf.beam.runner_from_options({"runner": _RUNNER.value})
  runner.run(pipeline)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  dgf.beam.program_started(_RUNNER.value)
  run(_INPUT.value, _INPUT_FORMAT.value, _NODESET.value)


if __name__ == "__main__":
  app.run(main)

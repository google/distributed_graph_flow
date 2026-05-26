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

r"""Converts between HGraph formats (e.g., SSTable or TFRecord based HGraph).
"""

from collections.abc import Sequence

from absl import app
from absl import flags
import apache_beam as beam
import dgf

_INPUT = flags.DEFINE_string(
    "input",
    None,
    "Input HGraph dataset.",
    required=True,
)
_INPUT_FORMAT = flags.DEFINE_string(
    "input_format",
    "TF_RECORD",
    "Format of the input HGraph.",
)

_OUTPUT = flags.DEFINE_string(
    "output",
    None,
    "Output HGraph dataset.",
    required=True,
)
_OUTPUT_FORMAT = flags.DEFINE_string(
    "output_format",
    "TF_RECORD",
    "Format of the output HGraph.",
)

_RUNNER = flags.DEFINE_string("runner", "DirectRunner", "Apache beam runner.")


def run(
    input_path: str, input_format: str, output_path: str, output_format: str
) -> None:

  def pipeline(root: beam.Pipeline):
    # Read the graph
    graph = dgf.beam.io.read_graphai_hgraph(root, input_path, input_format)
    # Write the graph
    dgf.beam.io.write_graphai_hgraph(graph, output_path, output_format)

  runner = dgf.beam.runner_from_name(_RUNNER.value)
  runner.run(pipeline)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  dgf.beam.program_started(_RUNNER.value)
  run(_INPUT.value, _INPUT_FORMAT.value, _OUTPUT.value, _OUTPUT_FORMAT.value)


if __name__ == "__main__":
  app.run(main)

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

r"""This example shows how to compute feature statistics on graph samples.
"""

# TODO(gbm): Convert as package binary.

from collections.abc import Sequence

from absl import app
from absl import flags
import apache_beam as beam
import dgf
from tensorflow_gnn import proto as tf_gnn_proto


_INPUT = flags.DEFINE_string(
    "input",
    None,
    "Input graph sample path.",
    required=True,
)
_INPUT_FORMAT = flags.DEFINE_string(
    "input_format",
    "TF_RECORD",
    "Format of the graph samples.",
)
_SCHEMA = flags.DEFINE_string(
    "schema",
    None,
    "Path to Graph Schema V2. Either tf_schema or schema should be provided.",
)

_TF_SCHEMA = flags.DEFINE_string(
    "tf_schema",
    None,
    "Path to TF Graph Schema. Either tf_schema or schema should be provided.",
)
_OUTPUT = flags.DEFINE_string(
    "output",
    None,
    "Path to json files where feature statistics will be exported",
    required=True,
)

_RUNNER = flags.DEFINE_string("runner", "DirectRunner", "Apache beam runner.")


def run():

  def pipeline(root: beam.Pipeline):

    # Get the graph schema
    if _SCHEMA.value is not None:
      schema = dgf.io.read_schema(_SCHEMA.value)
    elif _TF_SCHEMA.value is not None:
      tf_schema = dgf.io.read_text_proto(
          _TF_SCHEMA.value, tf_gnn_proto.GraphSchema
      )
      schema = dgf.convert.tfgnn_schema_to_schema(tf_schema)
    else:
      raise ValueError("Either --schema or --tf_schema should be provided.")

    # Read the graph
    graph = dgf.beam.io.read_tfgnn_graphs(
        root, _INPUT.value, schema=schema, container_type=_INPUT_FORMAT.value
    )
    # Compute the statistics
    stats = dgf.beam.analyse.feature_statistics_from_graphs(
        graph, schema=schema
    )

    # Save the statistics to a json file
    dgf.beam.io.write_feature_statistics(stats, _OUTPUT.value)

  dgf.beam.runner_from_name(_RUNNER.value).run(pipeline)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  dgf.beam.program_started(_RUNNER.value)
  run()


if __name__ == "__main__":
  app.run(main)

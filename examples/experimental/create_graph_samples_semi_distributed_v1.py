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

r"""Creates a set of graph samples and save them as tf graph samples.
"""

from collections.abc import Sequence

from absl import app
from absl import flags
import apache_beam as beam
import dgf

_INPUT_HGRAPH = flags.DEFINE_string(
    "input_hgraph",
    None,
    "Path to the input heterogeneous graph data.",
    required=True,
)
_INPUT_FORMAT = flags.DEFINE_string(
    "input_format",
    "TF_RECORD",
    "Format of the input HGraph.",
)
_INPUT_SCHEMA = flags.DEFINE_string(
    "input_schema",
    None,
    "Path to the input GraphFlow schema. If not provided, the schema is loaded"
    " from the HGraph. Specifying the schema allows to filter features /"
    " nodesets / edgesets e.g. remove the reverse edges.",
)
_OUTPUT_TF_GRAPH_SAMPLES = flags.DEFINE_string(
    "output_tf_gaph_samples",
    None,
    "Path to save the output TF graph samples.",
    required=True,
)
_SEED_NODESET = flags.DEFINE_string(
    "seed_nodeset", None, "Seed nodeset.", required=True
)
_NUM_HOPS = flags.DEFINE_integer(
    "num_hops", None, "Number of sampling hops.", required=True
)
_HOP_WIDTH = flags.DEFINE_integer(
    "hop_width", None, "Width of sampling.", required=True
)
_REVERSE = flags.DEFINE_boolean(
    "reverse", None, "If true, also sample the reverse edges.", required=True
)
_RUNNER = flags.DEFINE_string("runner", "DirectRunner", "Apache beam runner.")


def run():

  def pipeline(root: beam.Pipeline):
    # Read the graph
    if _INPUT_SCHEMA.value is not None:
      override_schema = dgf.io.read_schema(_INPUT_SCHEMA.value)
    else:
      override_schema = None
    graph = dgf.beam.io.read_graphai_hgraph(
        root, _INPUT_HGRAPH.value, _INPUT_FORMAT.value, override_schema  # pyrefly: ignore[bad-argument-type]
    )

    # Create sampling config
    sampling_config = dgf.sampling.SimpleSamplingConfig(
        seed_nodeset=_SEED_NODESET.value,
        num_hops=_NUM_HOPS.value,
        hop_width=_HOP_WIDTH.value,
        reverse=_REVERSE.value,
    )
    sampling_plan = dgf.sampling.simple_sampling_config_to_sampling_plan(
        sampling_config,
        graph.schema,
    )

    # Generate samples
    seeds = dgf.sampling.extract_beam_nodes_ids(
        graph, sampling_plan.root.nodeset
    )
    samples = dgf.sampling.sample_with_beam_semi_distributed_sampler(
        graph, sampling_plan, seeds=seeds, debug_sampling=True
    )

    # Save samples
    dgf.beam.io.write_tfgnn_graphs(
        samples, _OUTPUT_TF_GRAPH_SAMPLES.value, schema=graph.schema
    )

  dgf.beam.runner_from_name(_RUNNER.value).run(pipeline)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  dgf.beam.program_started(_RUNNER.value)
  run()


if __name__ == "__main__":
  app.run(main)

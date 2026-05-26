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
import random

from absl import app
from absl import flags
import apache_beam as beam
import dgf

_INPUT_GRAPH = flags.DEFINE_string(
    "input_graph",
    None,
    "Path to the input gf graph.",
    required=True,
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
_BEAM_FEATURE_COLLECTION = flags.DEFINE_bool(
    "beam_feature_collection",
    True,
    "If true, collects the feature values using a Beam join. This approach"
    " requires less RAM per worker and scales better, but can be overall slower"
    " than having the sampler fetch feature values directly.",
)
_RUNNER = flags.DEFINE_string("runner", "DirectRunner", "Apache beam runner.")
_NUM_SEEDS = flags.DEFINE_integer(
    "num_seeds",
    None,
    "If specified, randomly sample this many seed nodes instead of using all.",
)


def run():

  def pipeline(root: beam.Pipeline):

    # Read all the nodes as seed nodes.
    # TODO(gbm): Create read node ids utility to replace this block.
    graph = dgf.beam.io.read_graph(
        root,
        _INPUT_GRAPH.value,
        schema_filter=dgf.data.GraphSchemaFilter(
            nodeset_fn=lambda key, sch: key == _SEED_NODESET.value,
            edgeset_fn=lambda key, sch: False,
            feature_fn=lambda key, sch: key == "#id",
        ),
    )
    seeds = dgf.sampling.extract_beam_nodes_ids(graph, _SEED_NODESET.value)
    if _NUM_SEEDS.value is not None:
      total_count = seeds | "Count seeds" >> beam.combiners.Count.Globally()

      def filter_fn(element, total, target):
        if total == 0:
          return False
        prob = min(1.0, target / total)
        return random.random() < prob

      seeds = seeds | "Filter seeds" >> beam.Filter(
          filter_fn,
          total=beam.pvalue.AsSingleton(total_count),
          target=_NUM_SEEDS.value,
      )
    seeds = seeds | "Reshuffle seeds" >> beam.Reshuffle()

    # Create sampling config
    sampling_config = dgf.sampling.SimpleSamplingConfig(
        seed_nodeset=_SEED_NODESET.value,
        num_hops=_NUM_HOPS.value,
        hop_width=_HOP_WIDTH.value,
    )

    # Generate samples
    samples, schema = dgf.sampling.sample_with_beam_semi_distributed_sampler_v2(
        _INPUT_GRAPH.value,
        sampling_config,
        seeds=seeds,
        beam_feature_collection=_BEAM_FEATURE_COLLECTION.value,
    )

    # Save samples
    dgf.beam.io.write_tfgnn_graphs(
        samples, _OUTPUT_TF_GRAPH_SAMPLES.value, schema=schema
    )

  dgf.beam.runner_from_name(_RUNNER.value).run(pipeline)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  dgf.beam.program_started(_RUNNER.value)
  run()


if __name__ == "__main__":
  app.run(main)

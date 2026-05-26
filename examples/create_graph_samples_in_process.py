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

The sampling is done in process.
"""

import time

from absl import app
from absl import flags
import dgf
import numpy as np
import tqdm

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
_OUTPUT_SCHEMA = flags.DEFINE_string(
    "output_schema",
    None,
    "Optional path to save the output graph-flow schema.",
)

_SEED_NODESET = flags.DEFINE_string(
    "seed_nodeset", None, "Seed nodeset.", required=True
)
_NUM_SAMPLES = flags.DEFINE_integer(
    "num_samples",
    None,
    "Number of samples. If 0, creates one sample for each nodeset. If >=0,"
    " randomly sample `num_samples` nodes.",
    required=True,
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

# TODO(gbm): Add feature stats


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if _INPUT_SCHEMA.value is not None:
    schema = dgf.io.read_schema(_INPUT_SCHEMA.value)
  else:
    schema = None

  print("Load graph in memory")
  graph, schema = dgf.io.read_graphai_hgraph(
      _INPUT_HGRAPH.value,
      override_schema=schema,
      container_type=_INPUT_FORMAT.value,
  )

  num_nodes = graph.node_sets[_SEED_NODESET.value].num_nodes
  print(f"Found {num_nodes} in the seed nodeset")
  assert num_nodes is not None

  if _OUTPUT_SCHEMA.value:
    print("Save schema")
    dgf.io.write_schema(schema, _OUTPUT_SCHEMA.value)

  if _NUM_SAMPLES.value >= 0:
    selected_nodes = np.random.choice(
        num_nodes, size=_NUM_SAMPLES.value, replace=False
    )
    print(
        f"Will generate {len(selected_nodes)} samples from random nodes"
        " (without replacement)"
    )
  else:
    selected_nodes = np.arange(num_nodes, dtype=np.int64)
    print(
        f"Will generate one sample for each of the {len(selected_nodes)} input"
        " nodes."
    )

  print("Create sampler")
  batch_size = 32
  sampling_config = dgf.sampling.SimpleSamplingConfig(
      seed_nodeset=_SEED_NODESET.value,
      num_hops=_NUM_HOPS.value,
      hop_width=_HOP_WIDTH.value,
      reverse=_REVERSE.value,
  )
  sampling_plan = dgf.sampling.simple_sampling_config_to_sampling_plan(
      sampling_config,
      schema,
  )
  sampler = dgf.sampling.create_sampler(
      graph, sampling_plan, schema, batch_size=batch_size
  )

  def sample_generator():
    num_samples = len(selected_nodes)
    num_batches = (num_samples + batch_size - 1) // batch_size
    for begin in tqdm.tqdm(
        range(0, num_samples, batch_size),
        total=num_batches,
    ):
      end = min(begin + batch_size, num_samples)
      graph_samples = sampler.sample(selected_nodes[begin:end].tolist())
      for graph_sample in graph_samples:
        yield graph_sample

  # Generate samples and save them
  print("Start sampling")
  start_time = time.monotonic()
  dgf.io.write_tfgnn_graphs(
      sample_generator(),
      _OUTPUT_TF_GRAPH_SAMPLES.value,
      num_shards=min(
          100, len(selected_nodes) // 1000
      ),  # 1000 samples per shards
      schema=schema,
  )
  end_time = time.monotonic()
  print(
      f"{len(selected_nodes)} samples generated and exported in"
      f" {end_time - start_time:.2f} seconds"
  )


if __name__ == "__main__":
  app.run(main)

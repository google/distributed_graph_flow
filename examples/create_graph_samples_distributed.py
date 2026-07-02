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

r"""Creates TF graph samples using the MPI distributed sampler in batch mode.

For more control, use the distributed sampler in online mode.

Alternatively, you can also use the MapReduce/Flume distributed sampler.

"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
import dgf

_INPUT_GRAPH = flags.DEFINE_string(
    "input_graph",
    None,
    "Path to the input graph data. Should be a GF graph. Only for the manager.",
)
_OUTPUT_TF_GRAPH_SAMPLES = flags.DEFINE_string(
    "output_samples",
    None,
    "Path to save the output TF graph samples. Only for the manager.",
)
_WORKING_DIR = flags.DEFINE_string(
    "working_dir",
    None,
    "Working directory for temporary files, enabling resume of interrupted"
    " sampling. Do not reuse for different datasets or parameters.",
    required=True,
)
_SEED_NODESET = flags.DEFINE_string(
    "seed_nodeset", None, "Seed nodeset. Only for the manager."
)
_NUM_HOPS = flags.DEFINE_integer(
    "num_hops", None, "Number of sampling hops. Only for the manager."
)
_HOP_WIDTH = flags.DEFINE_integer(
    "hop_width", None, "Width of sampling. Only for the manager."
)
_NUM_SAMPLES = flags.DEFINE_integer(
    "num_samples",
    None,
    "The total number of graph samples to generate. If not specified, a sample"
    " will be generated for every node in the seed nodeset. Only for the"
    " manager.",
)
_NODE_SPEC = flags.DEFINE_string(
    "node_spec",
    None,
    "Distribution configuration. If not provided, extract the node spec from"
    " the environment.",
)


def run(node_spec: dgf.data.ComputeNodeSpec):

  if node_spec.is_worker():
    logging.info("Start worker #%d", node_spec.worker_idx)
    dgf.sampling.start_worker(
        working_directory=_WORKING_DIR.value,
        node_spec=node_spec,
    )
    return

  logging.info("Start manager")
  dgf.sampling.sample_with_distributed_batching(
      graph_path=_INPUT_GRAPH.value,  # pyrefly: ignore[bad-argument-type]
      plan=dgf.sampling.SimpleSamplingConfig(
          seed_nodeset=_SEED_NODESET.value,  # pyrefly: ignore[bad-argument-type]
          num_hops=_NUM_HOPS.value,  # pyrefly: ignore[bad-argument-type]
          hop_width=_HOP_WIDTH.value,  # pyrefly: ignore[bad-argument-type]
      ),
      working_directory=_WORKING_DIR.value,
      samples_path=_OUTPUT_TF_GRAPH_SAMPLES.value,  # pyrefly: ignore[bad-argument-type]
      node_spec=node_spec,
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if _NODE_SPEC.value:
    # The distribution spec is passed manually (used in manual Borg).
    node_spec = dgf.data.ComputeNodeSpec.from_json(_NODE_SPEC.value)  # pyrefly: ignore[missing-attribute]
  else:
    # The distribution node spec is obtained from the env (used in VertexAI and
    # TF distribution enviroments).
    node_spec = dgf.data.ComputeNodeSpec.from_vertex_ai_env()

  run(node_spec)


if __name__ == "__main__":
  app.run(main)

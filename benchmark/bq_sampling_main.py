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

r"""Binary to run BigQuery Graph sampling benchmarks.

Example invocation:

blaze run -c opt //third_party/py/dgf/benchmark:bq_sampling_main -- \
  --project=biggraphs-poc \
  --dataset=mag \
  --graph_id=mag_graph \
  --seed_nodeset=paper \
  --list_num_hops=1,2
"""

from absl import app
from absl import flags
from dgf.benchmark import bq_sampling

_PROJECT = flags.DEFINE_string(
    "project", "biggraphs-poc", "GCP Project ID for BigQuery Graph."
)
_DATASET = flags.DEFINE_string("dataset", "mag", "BigQuery dataset ID.")
_GRAPH_ID = flags.DEFINE_string(
    "graph_id", "mag_graph", "The name of the graph in BigQuery."
)
_SEED_NODESET = flags.DEFINE_string(
    "seed_nodeset", "paper", "Name of the seed nodeset."
)
_LIST_NUM_HOPS = flags.DEFINE_list(
    "list_num_hops",
    ["1", "2"],
    "Comma-separated list of hop counts to benchmark.",
)
_LIST_BATCH_SIZES = flags.DEFINE_list(
    "list_batch_sizes",
    ["1", "8"],
    "Comma-separated list of batch sizes to benchmark.",
)
_HOP_WIDTH = flags.DEFINE_integer(
    "hop_width", 5, "Number of neighbors sampled per hop."
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  config = bq_sampling.BigQuerySamplingBenchmarkConfig(
      project_id=_PROJECT.value,
      dataset_id=_DATASET.value,
      graph_id=_GRAPH_ID.value,
      seed_nodeset=_SEED_NODESET.value,
  )
  bq_sampling.run_bq_sampling_benchmark(
      config=config,
      list_num_hops=[int(x) for x in _LIST_NUM_HOPS.value],
      list_batch_sizes=[int(x) for x in _LIST_BATCH_SIZES.value],
      hop_width=_HOP_WIDTH.value,
  )


if __name__ == "__main__":
  app.run(main)

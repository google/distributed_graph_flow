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

r"""Binary to run In process sampling benchmarks.

Usage example:

sudo apt install linux-cpupower
sudo cpupower frequency-set --governor performance

Note: Use local copies instead of CNS for faster execution (no impact on the
results).

blaze run -c opt --cpu=haswell //third_party/py/dgf/benchmark:in_process_sampling_main -- \
  --work_dir=/tmp/gf_benchmark \
  --graph_path=/cns/iz-d/home/research-graph/public/graphflow_datasets/fetch_repo/ogb_arxiv\
  --seed_nodeset=nodes



# Note: For very large datasets like `papers100` (1.6B edges), avoid using
# `--work_dir` to prevent local caching, as the dataset is very large.
# Note: You can also manually make a local copy of the graph for faster
# iterations.
blaze run -c opt --cpu=haswell //third_party/py/dgf/benchmark:in_process_sampling_main -- \
  --graph_path=/cns/iz-d/home/research-graph/public/graphflow_datasets/ogbn_papers100m/raw_gf_graph\
  --seed_nodeset=node \
  --list_num_hops=2
Results:
GFGraph read in memory in 311.86 seconds.
"""

from absl import app
from absl import flags
from dgf.benchmark import in_process_sampling

_WORK_DIR_PATH = flags.DEFINE_string(
    "work_dir",
    None,
    "Working directory with read and write access. Needs to exist already.",
)
_GRAPH_PATH = flags.DEFINE_string(
    "graph_path", None, "Optional path to the graph dataset."
)
_SEED_NODESET = flags.DEFINE_string(
    "seed_nodeset", None, "Name of the seed nodeset.", required=True
)
_LIST_NUM_HOPS = flags.DEFINE_list(
    "list_num_hops",
    "2,3,4",
    "A comma-separated list of integers representing the number of hops to"
    " sample.",
)
_BENCHMARK_OUTPUT_FORMATS = flags.DEFINE_bool(
    "benchmark_output_formats",
    False,
    "Whether to benchmark different output formats.",
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  in_process_sampling.in_process_sampling(
      work_dir=_WORK_DIR_PATH.value,
      list_num_hops=[int(x) for x in _LIST_NUM_HOPS.value],
      gf_graph_path=_GRAPH_PATH.value,
      seed_nodeset=_SEED_NODESET.value,
      benchmark_output_formats=_BENCHMARK_OUTPUT_FORMATS.value,
  )


if __name__ == "__main__":
  app.run(main)

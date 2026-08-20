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

r"""Binary to run in-process IO benchmarks.

Usage example:

sudo apt install linux-cpupower
sudo cpupower frequency-set --governor performance
blaze run -c opt //third_party/py/dgf/benchmark:in_process_io_main -- \
    --work_dir=/usr/local/google/home/gbm/project/graphflow_2/runs/io_benchmark

Last run:

=================================================================================================================================
Wall time (s)      CPU time (s)       Wall time (s)/unit    units/s            Name
=================================================================================================================================
       34.05342            2.84482            0.00020         4972.86307    Read DGF Graph from CNS; num_nodes=169343 num_edges=1166243 (169343)
        0.56409            0.66995            0.00000       300204.38709    Read DGF Graph from local; num_nodes=169343 num_edges=1166243 (169343)
        0.32633            2.32004            0.00000       518926.13800    Write DGF Graph; num_nodes=169343 num_edges=1166243 (169343)
       43.08198           11.68357            0.00431          232.11560    Read TF-GNN Graph samples from CNS; num_samples=10000 (10000)
        1.86787            4.37955            0.00019         5353.68895    Read TF-GNN Graph samples from local; num_samples=10000 (10000)
        8.85905            8.90962            0.00089         1128.78878    Write TF-GNN Graph Samples; num_samples=10000 (10000)
        7.56402            5.88479            0.00004        22387.95834    Read GraphAI Graph from CNS; num_nodes=169343 num_edges=1166243 (169343)
        1.69344            4.69786            0.00001        99999.28070    Read GraphAI Graph from local; num_nodes=169343 num_edges=1166243 (169343)
        0.01274            0.01177            0.00000     13296115.73024    Read Pickled DGF Graph; num_nodes=169343 num_edges=1166243 (169343)
=================================================================================================================================
"""  # fmt: skip

import os
import subprocess
from absl import app
from absl import flags
import dgf
from dgf.benchmark import in_process_io as lib
from dgf.benchmark import utils as benchmark_utils

_WORK_DIR = flags.DEFINE_string(
    "work_dir",
    None,
    "Working directory with read and write access. Should be local since the"
    " benchmark tests both local and cns reading.",
    required=True,
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Configure dataset used for the benchmark
  graph_repo = "/cns/iz-d/home/research-graph/public/graphflow_datasets"
  graph_name = "ogbn_arxiv"  # Possible values: ogbn_arxiv, ogbn_mag_v2

  graphai_graph_path = f"{graph_repo}/{graph_name}/raw_hgraph"
  dgf_graph_path = f"{graph_repo}/{graph_name}/raw_gf_graph"
  tfgnn_graph_samples = f"{graph_repo}/{graph_name}/raw_gnn_samples_d2_w2_sfull"

  # Create local workdir
  work_dir = _WORK_DIR.value
  dgf.filesystem.makedirs(work_dir)

  # Copy the CNS data locally to test local reading.
  local_cache = os.path.join(work_dir, "cache")
  dgf.filesystem.makedirs(local_cache, exist_ok=True)
  local_graphai_graph_path = os.path.join(local_cache, "graphai")
  local_dgf_graph_path = os.path.join(local_cache, "dgf")
  local_tfgnn_graph_samples = os.path.join(local_cache, "tfgnn_samples")
  for src, dst in [
      (graphai_graph_path, local_graphai_graph_path),
      (dgf_graph_path, local_dgf_graph_path),
      (tfgnn_graph_samples, local_tfgnn_graph_samples),
  ]:
    if not dgf.filesystem.exists(dst):
      print(f"Copy {src} to {dst}")
      subprocess.check_call(["fileutil", "cp", "-R", src, dst])

  # Configure benchmarks
  # Note: You can comment-out benchmarks you don't care.
  benchmarks = [
      # DGF
      lib.ReadGFGraphInMemory(gf_graph_path=dgf_graph_path).set_extra_name(
          "from CNS (PARQUET)"
      ),  # TODO(gbm): This is slow. Improve speed.
      lib.ReadGFGraphInMemory(
          gf_graph_path=local_dgf_graph_path
      ).set_extra_name("from local (PARQUET)"),
      lib.WriteGFGraphInMemory(
          work_dir=work_dir, gf_graph_path=local_dgf_graph_path
      ),
      # Graph samples
      lib.ReadTFGraphSamplesInMemory(
          tf_graph_samples_path=tfgnn_graph_samples
      ).set_extra_name("from CNS"),
      lib.ReadTFGraphSamplesInMemory(
          tf_graph_samples_path=local_tfgnn_graph_samples,
      ).set_extra_name("from local"),
      lib.WriteTFGraphSamplesInMemory(
          work_dir=work_dir,
          tf_graph_samples_path=local_tfgnn_graph_samples,
      ),
      # GraphAI Graph
      lib.ReadGraphAIGraphInMemory(
          hgraph_path=graphai_graph_path
      ).set_extra_name("from CNS"),
      lib.ReadGraphAIGraphInMemory(
          hgraph_path=local_graphai_graph_path
      ).set_extra_name("from local"),
      # Pickle
      lib.ReadPickleInMemoryGraph(
          work_dir=work_dir, hgraph_path=graphai_graph_path
      ),
  ]

  common_kwargs = {
      "repetitions": 1,
      "warmup_repetitions": 0,
  }
  benchmarker = benchmark_utils.Benchmarker()
  for benchmark_idx, benchmark in enumerate(benchmarks):
    print(f"Running benchmark {benchmark_idx+1}/{len(benchmarks)}")
    benchmarker.run(benchmark, **common_kwargs)
  benchmarker.print_results()


if __name__ == "__main__":
  app.run(main)

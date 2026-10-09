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

r"""Binary to run message passing neural network (MPNN) benchmarks.

Usage example:

blaze run -c opt //third_party/py/dgf/benchmark:mpnn_main -- \
  --scenarios=small_tree,small_loops,many_edges_many_nodes \
  --layer_types=GRAPHSAGE --layer_types=GCN \
  --list_dims=64,128 \
  --list_num_layers=1,3

Run without `--scenarios` / `--layer_types` to benchmark all
`mpnn.DEFAULT_SCENARIOS` / `mpnn.LayerType`.
"""

from absl import app
from absl import flags
from dgf.benchmark import mpnn

_SCENARIOS = flags.DEFINE_list(
    "scenarios",
    [s.name for s in mpnn.DEFAULT_SCENARIOS],
    "Comma-separated list of scenario names to benchmark. Available:"
    f" {', '.join(s.name for s in mpnn.DEFAULT_SCENARIOS)}.",
)
_LAYER_TYPES = flags.DEFINE_multi_enum_class(
    "layer_types",
    list(mpnn.LayerType),
    mpnn.LayerType,
    "Layer types to benchmark (repeat the flag to select several).",
)
_LIST_DIMS = flags.DEFINE_list(
    "list_dims",
    ["128"],
    "Comma-separated list of embedding dimensions to benchmark.",
)
_LIST_NUM_LAYERS = flags.DEFINE_list(
    "list_num_layers",
    ["1", "3"],
    "Comma-separated list of MPNN layer depths to benchmark.",
)
_MAX_RUNTIME_SECONDS = flags.DEFINE_float(
    "max_runtime_seconds",
    2.0,
    "Target duration of each measured benchmark run.",
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  scenarios_by_name = {s.name: s for s in mpnn.DEFAULT_SCENARIOS}
  unknown = [name for name in _SCENARIOS.value if name not in scenarios_by_name]
  if unknown:
    raise app.UsageError(
        f"Unknown scenarios {unknown}. Available:"
        f" {sorted(scenarios_by_name)}"
    )

  mpnn.mpnn(
      scenarios=[scenarios_by_name[name] for name in _SCENARIOS.value],
      layer_types=_LAYER_TYPES.value,
      list_dims=[int(x) for x in _LIST_DIMS.value],
      list_num_layers=[int(x) for x in _LIST_NUM_LAYERS.value],
      max_runtime_seconds=_MAX_RUNTIME_SECONDS.value,
  )


if __name__ == "__main__":
  app.run(main)

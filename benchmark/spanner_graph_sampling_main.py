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

r"""Binary to run Spanner Graph sampling benchmarks.

Example invocation on Cloud Spanner:

blaze run -c opt //third_party/py/dgf/benchmark:spanner_graph_sampling_main -- \
  --backend=GCP \
  --project=biggraphs-poc \
  --instance=gcp-gnns \
  --database=ogbn_mag \
  --graph=ogbn_mag \
  --seed_nodeset=paper \
  --seed_id_format="paper{}"
"""

import enum

from absl import app
from absl import flags
import dgf
from dgf.benchmark import spanner_graph_sampling
from dgf.src.sampling import spanner_graph_sampler as spanner_graph_sampler_lib
from dgf.src.sampling.gcp import spanner_graph_sampler as spanner_graph_sampler_gcp_lib
from google.cloud import spanner


class Backend(enum.Enum):
  GCP = "GCP"


_BACKEND = flags.DEFINE_enum_class(
    "backend", Backend.GCP, Backend, "Spanner flavor of the database."
)
_PROJECT = flags.DEFINE_string("project", None, "GCP project (GCP only).")
_INSTANCE = flags.DEFINE_string(
    "instance", None, "Spanner instance (GCP only)."
)
_DATABASE = flags.DEFINE_string(
    "database",
    None,
    "Database name (GCP) or database path (Borg).",
    required=True,
)
_GRAPH = flags.DEFINE_string(
    "graph", None, "Name of the property graph.", required=True
)
_SEED_NODESET = flags.DEFINE_string(
    "seed_nodeset",
    None,
    "Name of the element table of the seed nodes.",
    required=True,
)
_SEED_ID_FORMAT = flags.DEFINE_string(
    "seed_id_format",
    None,
    "Python format of the id of the i-th seed node e.g. 'paper{}'.",
    required=True,
)
_NUM_SEED_IDS = flags.DEFINE_integer(
    "num_seed_ids", 1000, "Number of seed nodes (the ids 0 to num_seed_ids-1)."
)
_LIST_QUERY_ENGINES = flags.DEFINE_list(
    "list_query_engines",
    ["SQL", "GQL"],
    "Comma-separated list of query engines to benchmark.",
)
_LIST_NUM_HOPS = flags.DEFINE_list(
    "list_num_hops",
    ["0", "1", "2"],
    "Comma-separated list of hop counts to benchmark.",
)
_LIST_BATCH_SIZES = flags.DEFINE_list(
    "list_batch_sizes",
    ["1", "32"],
    "Comma-separated list of batch sizes to benchmark.",
)
_HOP_WIDTH = flags.DEFINE_integer(
    "hop_width", 10, "Number of neighbors sampled per hop."
)
_REVERSE = flags.DEFINE_bool(
    "reverse",
    True,
    "Whether the edges are also traversed backward. This requires an index on"
    " the target key of the edge tables to be fast.",
)
_REPETITIONS = flags.DEFINE_integer(
    "repetitions", 5, "Number of timed queries per setup."
)
_PRINT_QUERY = flags.DEFINE_bool(
    "print_query", False, "Whether to print the sampling queries."
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if _BACKEND.value == Backend.GCP:
    if _PROJECT.value is None or _INSTANCE.value is None:
      raise app.UsageError("--project and --instance are required on GCP.")
    project: str = _PROJECT.value
    instance: str = _INSTANCE.value
    metadata = spanner_graph_sampler_gcp_lib.read_metadata(
        spanner.Client(project=project)
        .instance(instance)
        .database(_DATABASE.value),
        _GRAPH.value,
    )

    def create_sampler(
        plan: dgf.sampling.SimpleSamplingConfig,
        schema: dgf.data.GraphSchema,
        query_engine: str,
    ) -> spanner_graph_sampler_lib.SpannerGraphSampler:
      return dgf.sampling.create_spanner_graph_sampler_gcp(
          project=project,
          instance=instance,
          database=_DATABASE.value,
          graph=_GRAPH.value,
          plan=plan,
          schema=schema,
          query_engine=query_engine,
      )

  else:
    raise ValueError(f"Unsupported backend: {_BACKEND.value}")

  spanner_graph_sampling.run_spanner_graph_sampling_benchmark(
      create_sampler=create_sampler,
      sampler_name=_BACKEND.value.value,
      schema=spanner_graph_sampling.schema_from_metadata(metadata),
      seed_nodeset=_SEED_NODESET.value,
      seed_ids=[
          _SEED_ID_FORMAT.value.format(seed_idx).encode("utf-8")
          for seed_idx in range(_NUM_SEED_IDS.value)
      ],
      list_query_engines=_LIST_QUERY_ENGINES.value,
      list_num_hops=[int(x) for x in _LIST_NUM_HOPS.value],
      list_batch_sizes=[int(x) for x in _LIST_BATCH_SIZES.value],
      hop_width=_HOP_WIDTH.value,
      reverse=_REVERSE.value,
      repetitions=_REPETITIONS.value,
      print_query=_PRINT_QUERY.value,
  )


if __name__ == "__main__":
  app.run(main)

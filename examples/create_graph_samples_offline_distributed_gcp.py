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

r"""Creates a set of graph samples using the offline distributed sampler on GCP.

The sampling pipeline runs on Google Cloud using Vertex AI and Dataflow.

Usage example:

```shell
blaze run -c opt //third_party/py/dgf/examples:create_graph_samples_offline_distributed_gcp -- \
  --input_graph=gs://gf-experiment-gbm-test/fetch_repo/ogb_mag \
  --output_samples=gs://gf-experiment-gbm-test/examples/ogb_mag_samples \
  --project=graphflow-experiments-49784 \
  --seed_nodeset=paper \
  --num_hops=2 \
  --hop_width=10 \
  --num_workers=5 \
  --num_seeds=1000 \
  --alsologtostderr
```
"""

import datetime
import os
from typing import Sequence

from absl import app
from absl import flags
import dgf

_INPUT_GRAPH = flags.DEFINE_string(
    "input_graph",
    "gs://gf-experiment-gbm-test/fetch_repo/ogb_mag",
    "Path to the input GraphFlow graph directory on GCS.",
)
_OUTPUT_SAMPLES = flags.DEFINE_string(
    "output_samples",
    "gs://gf-experiment-gbm-test/examples/ogb_mag_samples",
    "Base GCS output directory path for graph samples.",
)
_SEED_NODESET = flags.DEFINE_string(
    "seed_nodeset",
    "paper",
    "Seed nodeset name to sample around.",
)
_NUM_HOPS = flags.DEFINE_integer(
    "num_hops",
    2,
    "Number of hops in the sampling plan.",
)
_HOP_WIDTH = flags.DEFINE_integer(
    "hop_width",
    10,
    "Number of neighbors to sample per hop.",
)
_NUM_WORKERS = flags.DEFINE_integer(
    "num_workers",
    5,
    "Number of Dataflow workers.",
)
_NUM_SEEDS = flags.DEFINE_integer(
    "num_seeds",
    1000,
    "Number of seeds to sample (0 means all nodes).",
)
_PROJECT = flags.DEFINE_string(
    "project",
    None,
    "GCP Project ID. If None, auto-detected from environment.",
)
_REGION = flags.DEFINE_string(
    "region",
    "us-central1",
    "GCP Region to run Vertex AI CustomJob and Dataflow.",
)
_BLOCKING = flags.DEFINE_boolean(
    "blocking",
    True,
    "If True, wait for the job to complete while showing progress.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  output_path = os.path.join(
      _OUTPUT_SAMPLES.value, f"samples_{timestamp}"
  )

  print(f"Starting offline distributed sampling from {_INPUT_GRAPH.value} to {output_path}...")

  plan = dgf.sampling.SimpleSamplingConfig(
      seed_nodeset=_SEED_NODESET.value,
      num_hops=_NUM_HOPS.value,
      hop_width=_HOP_WIDTH.value,
  )

  job = dgf.sampling.offline_distributed_sampler_gcp(
      input_path=_INPUT_GRAPH.value,
      output_path=output_path,
      plan=plan,
      project=_PROJECT.value,
      region=_REGION.value,
      num_workers=_NUM_WORKERS.value,
      num_seeds=_NUM_SEEDS.value,
      blocking=_BLOCKING.value,
  )

  print(f"Sampling job finished. Vertex AI Job Resource: {job.resource_name}")
  print(f"Generated samples available at: {output_path}")


if __name__ == "__main__":
  app.run(main)

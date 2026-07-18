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

r"""Binary to run BigQuery Parquet IO benchmarks.

Example invocation:

blaze run -c opt //third_party/py/dgf/benchmark:bq_parquet_io_main -- \
  --project=biggraphs-poc \
  --dataset=graphflow_demo \
  --graph_id=ieee_graph \
  --gcs_prefix=gs://jfoniok-testbucket/bq2gf1
"""

from absl import app
from absl import flags
from dgf.benchmark import bq_parquet_io


_PROJECT = flags.DEFINE_string(
    "project", None, "GCP Project ID for BigQuery Graph."
)
_DATASET = flags.DEFINE_string("dataset", None, "BigQuery dataset ID.")
_GRAPH_ID = flags.DEFINE_string(
    "graph_id", None, "The name of the graph in BigQuery."
)
_GCS_PREFIX = flags.DEFINE_string(
    "gcs_prefix", None, "The GCS prefix to use for the parquet export."
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  bigquery_config = None
  if _PROJECT.value and _DATASET.value and _GRAPH_ID.value:
    bigquery_config = bq_parquet_io.BigQueryGraphConfig(
        project_id=_PROJECT.value,
        dataset_id=_DATASET.value,
        graph_id=_GRAPH_ID.value,
        gcs_prefix=_GCS_PREFIX.value,  # pyrefly: ignore[bad-argument-type]
    )

  if bigquery_config:
    bq_parquet_io.run_bq_parquet_benchmark(bigquery_config)
  else:
    print("Missing required flags for BigQuery Graph Export Benchmark.")
    print("Please specify --project, --dataset, and --graph_id.")


if __name__ == "__main__":
  app.run(main)

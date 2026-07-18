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

blaze run //third_party/py/dgf/benchmark:in_process_io_main -- \
  --work_dir=/cns/is-d/home/gbm/tmp/ttl=15d \
  --hgraph_path=/cns/iz-d/home/research-graph/public/hgraph_datasets/tfrecord_based/ogbn_arxiv\
  --tf_graph_samples_path=/cns/iz-d/home/research-graph/public/graph_samples_datasets/tfrecord_based/ogbn_arxiv \
  --spanner_project=biggraphs-poc \
  --spanner_instance=gcp-gnns \
  --spanner_database=ogbn_arxiv \
  --spanner_graph_id=ogbn_arxiv \
  --spanner_database_2=ogbn_arxiv_2 \
  --spanner_graph_id_2=ogbn_arxiv_2

Preparation for Spanner Graph Benchmark:
----------------------------------------
To run the Spanner benchmark, you must have an existing Spanner database pre-loaded
with OGBN-MAG data and a Property Graph schema defined.

1.  **Resources**: Ensure you have access to the Spanner instance.
    - `spanner_project`: Your Google Cloud Project ID (e.g., 'dgf-demo').
    - `spanner_instance`: The Spanner Instance ID (e.g., 'gcp-gnns').
    - `spanner_database`: The Database ID containing the tables (e.g., 'ogbn_mag').

2.  **Schema Setup (Step A - Create Tables)**:
    Before creating the graph view, the base tables must exist. Run this DDL first:

    CREATE TABLE author (id STRING(MAX) NOT NULL) PRIMARY KEY(id);
    CREATE TABLE field_of_study (id STRING(MAX) NOT NULL) PRIMARY KEY(id);
    CREATE TABLE institution (id STRING(MAX) NOT NULL) PRIMARY KEY(id);
    CREATE TABLE paper (
      id STRING(MAX) NOT NULL,
      year INT64,
      labels INT64,
      feat ARRAY<FLOAT64>
    ) PRIMARY KEY(id);

    CREATE TABLE affiliated_with (
      id STRING(MAX) NOT NULL,
      target_id STRING(MAX) NOT NULL,
      CONSTRAINT FK_affiliated_author FOREIGN KEY(id) REFERENCES author(id),
      CONSTRAINT FK_affiliated_inst FOREIGN KEY(target_id) REFERENCES institution(id)
    ) PRIMARY KEY(id, target_id);

    CREATE TABLE cites (
      id STRING(MAX) NOT NULL,
      target_id STRING(MAX) NOT NULL,
      CONSTRAINT FK_cites_source FOREIGN KEY(id) REFERENCES paper(id),
      CONSTRAINT FK_cites_target FOREIGN KEY(target_id) REFERENCES paper(id)
    ) PRIMARY KEY(id, target_id);

    CREATE TABLE has_topic (
      id STRING(MAX) NOT NULL,
      target_id STRING(MAX) NOT NULL,
      CONSTRAINT FK_topic_paper FOREIGN KEY(id) REFERENCES paper(id),
      CONSTRAINT FK_topic_field FOREIGN KEY(target_id) REFERENCES field_of_study(id)
    ) PRIMARY KEY(id, target_id);

    CREATE TABLE writes (
      id STRING(MAX) NOT NULL,
      target_id STRING(MAX) NOT NULL,
      CONSTRAINT FK_writes_author FOREIGN KEY(id) REFERENCES author(id),
      CONSTRAINT FK_writes_paper FOREIGN KEY(target_id) REFERENCES paper(id)
    ) PRIMARY KEY(id, target_id);

3.  **Schema Setup (Step B - Create Graph View)**:
    Once tables are created, run this DDL to define the Property Graph 'ogbn_mag':

    CREATE OR REPLACE PROPERTY GRAPH ogbn_mag
      NODE TABLES(
        author KEY(id) LABEL author PROPERTIES(id),
        field_of_study KEY(id) LABEL field_of_study PROPERTIES(id),
        institution KEY(id) LABEL institution PROPERTIES(id),
        paper KEY(id) LABEL paper PROPERTIES(id, labels, year, feat)
      )
      EDGE TABLES(
        affiliated_with
          KEY(id, target_id)
          SOURCE KEY(id) REFERENCES author(id)
          DESTINATION KEY(target_id) REFERENCES institution(id)
          LABEL affiliated_with PROPERTIES(id, target_id),
        cites
          KEY(id, target_id)
          SOURCE KEY(id) REFERENCES paper(id)
          DESTINATION KEY(target_id) REFERENCES paper(id)
          LABEL cites PROPERTIES(id, target_id),
        has_topic
          KEY(id, target_id)
          SOURCE KEY(id) REFERENCES paper(id)
          DESTINATION KEY(target_id) REFERENCES field_of_study(id)
          LABEL has_topic PROPERTIES(id, target_id),
        writes
          KEY(id, target_id)
          SOURCE KEY(id) REFERENCES author(id)
          DESTINATION KEY(target_id) REFERENCES paper(id)
          LABEL writes PROPERTIES(id, target_id)
      );

4.  **Data Loading**: Ensure you import or insert the OGBN-MAG dataset into these
    tables before benchmarking.

5.  **Authentication**: Ensure you have Application Default Credentials active
    (e.g., via `gcloud auth application-default login`).
"""

from absl import app
from absl import flags
from dgf.benchmark import in_process_io

_WORK_DIR_PATH = flags.DEFINE_string(
    "work_dir",
    None,
    "Working directory with read and write access. Needs to exist already.",
)
_HGRAPH_PATH = flags.DEFINE_string(
    "hgraph_path", None, "Optional path to the hgraph dataset."
)
_GF_GRAPH_PATH = flags.DEFINE_string(
    "gf_graph_path", None, "Optional path to the GF graph dataset."
)
_TF_GRAPH_SAMPLES_PATH = flags.DEFINE_string(
    "tf_graph_samples_path",
    None,
    "Directory containing TF graph samples dataset in data@*.rio and a schema"
    " in schema.json.",
)
_SPANNER_PROJECT = flags.DEFINE_string(
    "spanner_project", None, "GCP Project ID for Spanner Graph."
)
_SPANNER_INSTANCE = flags.DEFINE_string(
    "spanner_instance", None, "Spanner instance ID."
)
_SPANNER_DATABASE = flags.DEFINE_string(
    "spanner_database", None, "Spanner database ID."
)
_SPANNER_GRAPH_ID = flags.DEFINE_string(
    "spanner_graph_id", None, "The name of the graph in Spanner."
)
_SPANNER_DATABASE_2 = flags.DEFINE_string(
    "spanner_database_2", None, "Spanner database ID for write."
)
_SPANNER_GRAPH_ID_2 = flags.DEFINE_string(
    "spanner_graph_id_2", None, "The name of the graph in Spanner for write."
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  spanner_config = None
  if _SPANNER_PROJECT.value and _SPANNER_INSTANCE.value:
    spanner_config = in_process_io.SpannerGraphConfig(
        project_id=_SPANNER_PROJECT.value,
        instance_id=_SPANNER_INSTANCE.value,
        database_id=_SPANNER_DATABASE.value,  # pyrefly: ignore[bad-argument-type]
        graph_id=_SPANNER_GRAPH_ID.value,  # pyrefly: ignore[bad-argument-type]
    )

  spanner_write_config = None
  if _SPANNER_PROJECT.value and _SPANNER_INSTANCE.value:
    spanner_write_config = in_process_io.SpannerGraphConfig(
        project_id=_SPANNER_PROJECT.value,
        instance_id=_SPANNER_INSTANCE.value,
        database_id=_SPANNER_DATABASE_2.value,  # pyrefly: ignore[bad-argument-type]
        graph_id=_SPANNER_GRAPH_ID_2.value,  # pyrefly: ignore[bad-argument-type]
    )

  in_process_io.io_in_memory_dataset_in_process(
      work_dir=_WORK_DIR_PATH.value,  # pyrefly: ignore[bad-argument-type]
      hgraph_path=_HGRAPH_PATH.value,
      gf_graph_path=_GF_GRAPH_PATH.value,
      tf_graph_samples_path=_TF_GRAPH_SAMPLES_PATH.value,
      spanner_config=spanner_config,
      spanner_write_config=spanner_write_config,
  )


if __name__ == "__main__":
  app.run(main)

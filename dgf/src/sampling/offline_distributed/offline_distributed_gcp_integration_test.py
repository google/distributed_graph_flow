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

r"""Integration test for offline_distributed_sampler_gcp on GCP.

Usage example:

   blaze build -c opt //third_party/py/dgf/src/sampling/offline_distributed:offline_distributed_gcp_integration_test && \
   blaze-bin/third_party/py/dgf/src/sampling/offline_distributed/offline_distributed_gcp_integration_test \
     --test_dir=gs://gf-experiment-gbm-test/integration_test --alsologtostderr
"""

import datetime
import os
from absl import flags
from absl.testing import absltest
from dgf.src.io import schema as schema_io
from dgf.src.io import tf_graph_sample as tf_graph_sample_io
from dgf.src.sampling import config as config_lib
from dgf.src.sampling.offline_distributed import offline_distributed_gcp
from dgf.src.util import filesystem
from dgf.src.util import log

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "test_dir",
    "gs://gf-experiment-gbm-test/integration_test",
    "Base GCS directory path for test outputs.",
)
flags.DEFINE_string(
    "input_graph",
    "gs://gf-experiment-gbm-test/fetch_repo/ogb_mag",
    "Input GCS graph path.",
)
flags.DEFINE_string(
    "project",
    "graphflow-experiments-49784",
    "GCP Project ID to run Vertex AI and Dataflow jobs.",
)
flags.DEFINE_string(
    "region",
    "us-central1",
    "GCP Region.",
)
flags.DEFINE_integer(
    "num_workers",
    5,
    "Number of Dataflow workers.",
)
flags.DEFINE_integer(
    "num_seeds",
    1000,
    "Number of seeds to sample.",
)


class OfflineDistributedGcpIntegrationTest(absltest.TestCase):

  def test_offline_distributed_sampler_gcp(self):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_samples = os.path.join(
        FLAGS.test_dir, f"dgf_samples_integration_test_{timestamp}"
    )

    log.info("Running integration test with output at %s", output_samples)

    plan = config_lib.SimpleSamplingConfig(
        seed_nodeset="paper",
        num_hops=2,
        hop_width=10,
    )

    job = offline_distributed_gcp.offline_distributed_sampler_gcp(
        input_path=FLAGS.input_graph,
        output_path=output_samples,
        plan=plan,
        project=FLAGS.project,
        region=FLAGS.region,
        num_workers=FLAGS.num_workers,
        num_seeds=FLAGS.num_seeds,
        blocking=True,
    )

    self.assertIsNotNone(job.resource_name)

    # Check the structure of the output directory.
    sampling_config_file = os.path.join(output_samples, "sampling_config.json")
    self.assertTrue(
        filesystem.exists(sampling_config_file),
        f"Expected {sampling_config_file} to exist on GCS.",
    )
    schema_file = os.path.join(output_samples, "schema.json")
    self.assertTrue(
        filesystem.exists(schema_file),
        f"Expected {schema_file} to exist on GCS.",
    )
    sample_shards = filesystem.glob(
        os.path.join(output_samples, "samples-*.tfrecord.gz")
    )
    self.assertNotEmpty(
        sample_shards,
        f"Expected sample shard files in {output_samples}.",
    )

    # Read the graph samples
    num_read_graphs = 0
    for graph in tf_graph_sample_io.read_tfgnn_graphs(
        path=os.path.join(output_samples, "samples-*.tfrecord.gz"),
        schema=schema_io.read_schema(schema_file),
    ):
      num_read_graphs += 1

    # The distributed sampler performs scalable Bernoulli sampling of seed
    # nodes, so the number of generated graphs is approximately FLAGS.num_seeds.
    self.assertAlmostEqual(
        num_read_graphs, FLAGS.num_seeds, delta=int(FLAGS.num_seeds * 0.1)
    )

    log.info(
        "Integration test verified %d sample shards in %s",
        len(sample_shards),
        output_samples,
    )
    log.info("Integration test finished successfully!")


if __name__ == "__main__":
  absltest.main()

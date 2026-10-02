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

import os

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.deploy import vertex
from dgf.src.learning.ten_lines import node_prediction_train
from dgf.src.util import gen_test_graph


class VertexIntegrationTest(parameterized.TestCase):

  def test_deploy_model(self):
    graph, schema = gen_test_graph.gen_toy_classification_dataset()

    # Extract training config
    model = node_prediction_train.train_node_model(
        graph=graph,
        schema=schema,
        target_nodeset="N1",
        target_column="label",
        num_train_steps=10,
        batch_size=8,
    )

    test_dir = self.create_tempdir().full_path
    endpoint = vertex.to_vertex_ai(
        model=model,
        model_dir_on_gcs=os.path.join(test_dir, "models"),
        display_name="test-integration-model",
        location="us-central1",
        machine_type="n1-standard-4",
        blocking=True,
    )
    self.assertIsNotNone(endpoint)

    # Clean up (undeploy)
    endpoint.undeploy_all()
    endpoint.delete()


if __name__ == "__main__":
  absltest.main()

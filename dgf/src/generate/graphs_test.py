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
import tempfile
from absl.testing import absltest
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.generate import graphs
from dgf.src.io import tf_graph_sample
from dgf.src.sampling import config as config_lib
from dgf.src.util import gen_test_graph
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib


class SyntheticTest(absltest.TestCase):

  def _test_sample(
      self,
      sample: in_memory_graph.InMemoryGraph,
      schema: schema_lib.GraphSchema,
  ):
    in_memory_graph_validate_lib.validate_graph(sample, schema=schema)

  def test_synthetic_graph_sample_generator(self):

    schema = gen_test_graph.generate_schema(node_ids=True, semantic=True)
    schema.node_sets["n1"].features["f6"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        shape=(None,),
        semantic=schema_lib.FeatureSemantic.TIMESERIES,
    )
    schema.node_sets["n1"].features["f7"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        shape=(10,),
        semantic=schema_lib.FeatureSemantic.TIMESERIES,
    )
    sample = graphs.generate_synthetic_graph_sample(
        schema,
        plan=config_lib.SimpleSamplingConfig(
            seed_nodeset="n1",
            num_hops=2,
            hop_width=1,
        ),
    )
    self._test_sample(sample, schema=schema)

  def test_write_synthetic_graph_sample_as_tfgnn_graphs(self):

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "sample@*.tfrecord.gz")
      schema = gen_test_graph.generate_schema(node_ids=True, semantic=True)
      graphs.write_synthetic_graph_sample_as_tfgnn_graphs(
          schema=schema,
          plan=config_lib.SimpleSamplingConfig(
              seed_nodeset="n1",
              num_hops=2,
              hop_width=1,
          ),
          path=path,
          num_samples=100,
          verbose=True,
          validate=True,
      )
      generator = tf_graph_sample.read_tfgnn_graphs(path, schema)
      for sample in generator:
        self._test_sample(sample, schema=schema)


if __name__ == "__main__":
  absltest.main()

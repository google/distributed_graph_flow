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
from dgf.src.io import tf_graph_sample
from dgf.src.learning.ten_lines import dataset
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.util import gen_test_graph
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib


class EvaluationTest(absltest.TestCase):

  def test_in_memory_graph(self):
    graph = gen_test_graph.generate_in_memory_graph()
    schema = gen_test_graph.generate_schema()
    generator = dataset.SampleGeneratorFromAnything(
        graph=graph,
        schema=schema,
        batch_size=2,
        seed_node_idxs=None,
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1"
        ),
        format=dataset.GraphFormat.AUTO,
        drop_remainder=False,
        shuffle=False,
    )
    self.assertEqual(generator.num_seed_nodes, 2)
    num_batches = 0
    num_graphs = 0
    for sample, offsets in generator.iterator():
      in_memory_graph_validate_lib.validate_graph(
          sample, schema, raise_on_warning=False
      )
      num_batches += 1
      num_graphs += len(offsets["n1"]) - 1
    self.assertEqual(num_batches, 1)
    self.assertEqual(num_graphs, 2)

  def test_sampler_returns_node_idxs_only(self):
    graph = gen_test_graph.generate_in_memory_graph()
    schema = gen_test_graph.generate_schema()
    generator = dataset.SampleGeneratorFromAnything(
        graph=graph,
        schema=schema,
        batch_size=2,
        seed_node_idxs=None,
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1"
        ),
        format=dataset.GraphFormat.AUTO,
        drop_remainder=False,
        shuffle=False,
    )
    generator.set_sampler_returns_node_idxs_only(True)
    self.assertEqual(generator.num_seed_nodes, 2)
    num_batches = 0
    num_graphs = 0
    merge_schema = generator._get_merge_schema()
    for sample, offsets in generator.iterator():
      self.assertEqual(list(sample.node_sets["n1"].features.keys()), ["#idx"])
      num_batches += 1
      num_graphs += len(offsets["n1"]) - 1
      in_memory_graph_validate_lib.validate_graph(
          sample, merge_schema, raise_on_warning=False
      )
    self.assertEqual(num_batches, 1)
    self.assertEqual(num_graphs, 2)

  def test_tf_gnn_samples_tfrecord(self):
    tmpdir = self.create_tempdir().full_path
    path = os.path.join(tmpdir, "samples@5.tfrecord")
    schema = gen_test_graph.generate_schema(variable_length=True)
    subgraph = gen_test_graph.generate_in_memory_graph(variable_length=True)

    def in_mem_graphs():
      for _ in range(21):
        yield subgraph

    tf_graph_sample.write_tfgnn_graphs(
        in_mem_graphs(),
        path,
        schema=schema,
        container_type="TF_RECORD",
    )

    generator = dataset.SampleGeneratorFromAnything(
        graph=path,
        schema=schema,
        batch_size=2,
        seed_node_idxs=None,
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1"
        ),
        format=dataset.GraphFormat.PATH_TF_SAMPLE_TF_RECORD,
        drop_remainder=False,
        shuffle=True,
    )

    self.assertIsNone(generator.num_seed_nodes, 21)
    num_batches = 0
    num_graphs = 0
    for sample, offsets in generator.iterator():
      in_memory_graph_validate_lib.validate_graph(
          sample, schema, raise_on_warning=False
      )
      num_batches += 1
      num_graphs += len(offsets["n1"]) - 1
    self.assertEqual(num_batches, 11)
    self.assertEqual(num_graphs, 21)

  def test_not_supported(self):
    schema = gen_test_graph.generate_schema()
    with self.assertRaises(ValueError):
      dataset.SampleGeneratorFromAnything(
          graph=[1, 2, 3],  # pytype: disable=wrong-arg-types
          schema=schema,
          batch_size=2,
          seed_node_idxs=None,
          sampling_config=sampling_config_lib.SimpleSamplingConfig(
              seed_nodeset="N1"
          ),
          format=dataset.GraphFormat.AUTO,
          drop_remainder=False,
          shuffle=False,
      )


if __name__ == "__main__":
  absltest.main()

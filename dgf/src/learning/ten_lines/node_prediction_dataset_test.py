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

"""Test converting between heterogeneous graph edge types."""

import os
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import schema as schema_lib
from dgf.src.io import jax as jax_io_lib
from dgf.src.io import tf_graph_sample
from dgf.src.learning.ten_lines import node_prediction_dataset
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.util import gen_test_graph
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np


class GNNDatasetPreparatorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("CacheHost", True, "host"),
      ("CacheDevice", True, "device"),
      ("NoCache", False, "host"),
  )
  def test_in_memory_graph(self, cache_features, cache_device):
    graph = gen_test_graph.generate_in_memory_graph(True, False)
    schema = gen_test_graph.generate_schema(True, False, True, False)
    print("schema:\n", schema)
    preparator = node_prediction_dataset.GNNDatasetPreparator(
        graph=graph,
        schema=schema,
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1",
            num_hops=2,
            hop_width=3,
            reverse=True,
        ),
        batch_size=2,
        drop_remainder=True,
        shuffle=True,
        cache_normalized_features=cache_features,
        cache_normalized_features_device=cache_device,
    )
    self.assertFalse(preparator.is_prepared())
    preparator.prepare()
    self.assertTrue(preparator.is_prepared())

    num_batches = 0
    num_graphs = 0
    normalized_schema = preparator.get_live().normalizer.output_schema()

    if cache_device == "host":
      generator = preparator.generate()
    else:

      def sanitize(g):
        ng = jax_io_lib.jax_graph_to_graph(g)
        for node_set in ng.node_sets.values():
          for k, v in node_set.features.items():
            if v.dtype == np.int32:
              node_set.features[k] = v.astype(np.int64)
        return ng

      generator = (
          (sanitize(g), offsets) for g, offsets in preparator.generate_jax()
      )

    for graph_sample, merge_offset in generator:
      in_memory_graph_validate_lib.validate_graph(
          graph_sample, normalized_schema, raise_on_warning=False
      )
      num_batches += 1
      num_graphs += len(merge_offset["n1"]) - 1
    self.assertEqual(num_batches, 1)
    self.assertEqual(num_graphs, 2)

  def test_in_memory_temporal_graph(self):
    graph, schema = gen_test_graph.generate_temporal_in_memory_graph(False)
    preparator = node_prediction_dataset.GNNDatasetPreparator(
        graph=graph,
        schema=schema,
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1",
            num_hops=2,
            hop_width=3,
            reverse=True,
        ),
        batch_size=2,
        drop_remainder=True,
        shuffle=True,
        temporal_sampling=True,
        nodeset_timestamp_features={"n1": "timestamp"},
        edgeset_timestamp_features={"e1": "timestamp"},
        cache_normalized_features_device="host",
    )
    self.assertFalse(preparator.is_prepared())
    preparator.prepare()
    self.assertTrue(preparator.is_prepared())

    num_batches = 0
    num_graphs = 0
    normalized_schema = preparator.get_live().normalizer.output_schema()
    for graph_sample, merge_offset in preparator.generate():
      in_memory_graph_validate_lib.validate_graph(
          graph_sample, normalized_schema, raise_on_warning=False
      )
      num_batches += 1
      num_graphs += len(merge_offset["n1"]) - 1
    self.assertEqual(num_batches, 2)
    self.assertEqual(num_graphs, 4)

  def test_tf_gnn_samples(self):
    tmpdir = self.create_tempdir().full_path
    path = os.path.join(tmpdir, "samples@5.tfrecord")
    schema = gen_test_graph.generate_schema(
        variable_length=False, semantic=True
    )
    subgraph = gen_test_graph.generate_in_memory_graph(variable_length=False)

    def in_mem_graphs():
      for _ in range(21):
        yield subgraph

    tf_graph_sample.write_tfgnn_graphs(
        in_mem_graphs(),
        path,
        schema=schema,
        container_type="TF_RECORD",
    )

    preparator = node_prediction_dataset.GNNDatasetPreparator(
        graph=path,
        schema=schema,
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1",
            num_hops=2,
            hop_width=3,
            reverse=True,
        ),
        batch_size=2,
        drop_remainder=True,
        shuffle=True,
    )
    self.assertFalse(preparator.is_prepared())
    preparator.prepare()
    self.assertTrue(preparator.is_prepared())

    num_batches = 0
    num_graphs = 0
    normalized_schema = preparator.get_live().normalizer.output_schema()
    for graph_sample, merge_offset in preparator.generate():
      in_memory_graph_validate_lib.validate_graph(
          graph_sample, normalized_schema, raise_on_warning=False
      )
      num_batches += 1
      num_graphs += len(merge_offset["n1"]) - 1
    self.assertEqual(num_batches, 10)
    self.assertEqual(num_graphs, 20)


if __name__ == "__main__":
  absltest.main()

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
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import tf_graph_sample
from dgf.src.learning.ten_lines import dataset
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.transform import timeseries as timeseries_transform
from dgf.src.util import gen_test_graph
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np


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
    for sample, offsets in generator.batch_iterator():
      in_memory_graph_validate_lib.validate_graph(
          sample, schema, raise_on_warning=False
      )
      num_batches += 1
      num_graphs += len(offsets["n1"]) - 1
    self.assertEqual(num_batches, 1)
    self.assertEqual(num_graphs, 2)

  def test_in_memory_graph_with_list_seed_node_idxs(self):
    graph = gen_test_graph.generate_in_memory_graph()
    schema = gen_test_graph.generate_schema()
    generator = dataset.SampleGeneratorFromAnything(
        graph=graph,
        schema=schema,
        batch_size=2,
        seed_node_idxs=np.array([0, 1]),
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1"
        ),
        format=dataset.GraphFormat.AUTO,
        drop_remainder=False,
        shuffle=False,
    )
    self.assertEqual(generator.num_seed_nodes, 2)
    self.assertIsInstance(generator.seed_node_idxs, np.ndarray)
    np.testing.assert_array_equal(generator.seed_node_idxs, np.array([0, 1]))
    num_batches = 0
    num_graphs = 0
    for sample, offsets in generator.batch_iterator():
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
    for sample, offsets in generator.batch_iterator():
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
    for sample, offsets in generator.batch_iterator():
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

  def _create_temporal_test_graph_and_schema(self):
    schema = schema_lib.GraphSchema(
        node_sets={
            "alerts": schema_lib.NodeSchema(
                features={
                    "creation_time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_creation_time=True,
                    ),
                    "label": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                    ),
                }
            ),
            "hardware": schema_lib.NodeSchema(
                features={
                    "time": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.INTEGER_64,
                        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                        is_timeseries=True,
                        is_creation_time=True,
                        shape=(None,),
                    ),
                    "signal": schema_lib.FeatureSchema(
                        format=schema_lib.FeatureFormat.FLOAT_32,
                        semantic=schema_lib.FeatureSemantic.NUMERICAL,
                        is_timeseries=True,
                        shape=(None,),
                    ),
                }
            ),
        },
        edge_sets={
            "alert_to_hw": schema_lib.EdgeSchema(
                source="alerts",
                target="hardware",
                features={},
            ),
            "hw_to_alert": schema_lib.EdgeSchema(
                source="hardware",
                target="alerts",
                features={},
            ),
        },
    )

    alerts_nodes = in_memory_graph.InMemoryNodeSet(
        num_nodes=4,
        features={
            "creation_time": np.array([100, 200, 300, 400], dtype=np.int64),
            "label": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        },
    )
    hardware_nodes = in_memory_graph.InMemoryNodeSet(
        num_nodes=2,
        features={
            "time": np.array(
                [
                    np.array([50, 80, 120], dtype=np.int64),
                    np.array([150, 250], dtype=np.int64),
                ],
                dtype=object,
            ),
            "signal": np.array(
                [
                    np.array([1.5, 2.5, 3.5], dtype=np.float32),
                    np.array([4.5, 5.5], dtype=np.float32),
                ],
                dtype=object,
            ),
        },
    )
    alert_to_hw = in_memory_graph.InMemoryEdgeSet(
        adjacency=np.array([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=np.int64),
        features={},
    )
    hw_to_alert = in_memory_graph.InMemoryEdgeSet(
        adjacency=np.array([[0, 0, 1, 1], [0, 1, 2, 3]], dtype=np.int64),
        features={},
    )
    graph = in_memory_graph.InMemoryGraph(
        node_sets={"alerts": alerts_nodes, "hardware": hardware_nodes},
        edge_sets={"alert_to_hw": alert_to_hw, "hw_to_alert": hw_to_alert},
    )
    return graph, schema

  def test_per_sample_transformations(self):
    graph, schema = self._create_temporal_test_graph_and_schema()

    pad_and_cap_config = timeseries_transform.PadAndCapTimeseriesConfig(
        sequence_length=5
    )

    generator = dataset.SampleGeneratorFromAnything(
        graph=graph,
        schema=schema,
        batch_size=2,
        seed_node_idxs=None,
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="alerts",
            num_hops=1,
            hop_width=2,
            temporal_sampling=True,
        ),
        temporal=True,
        per_sample_transforms=timeseries_transform.PerSampleTransformConfig(
            timeseries_pad_and_cap=pad_and_cap_config,
            timedelta_extraction=timeseries_transform.TimestampFeatureExtractorConfig(),
        ),
        drop_remainder=False,
        shuffle=False,
    )

    self.assertIn(
        "time_mask", generator.output_schema().node_sets["hardware"].features
    )
    self.assertIn(
        "creation_time_seed_delta",
        generator.output_schema().node_sets["alerts"].features,
    )
    self.assertIn(
        "time_seed_delta",
        generator.output_schema().node_sets["hardware"].features,
    )

    num_batches = 0
    for sample, _ in generator.batch_iterator():
      in_memory_graph_validate_lib.validate_graph(
          sample, generator.output_schema(), raise_on_warning=False
      )
      self.assertEqual(
          sample.node_sets["hardware"].features["signal"].shape[1], 5
      )
      self.assertEqual(
          sample.node_sets["hardware"].features["time_mask"].shape[1], 5
      )
      self.assertIn("time_seed_delta", sample.node_sets["hardware"].features)
      self.assertIn(
          "creation_time_seed_delta", sample.node_sets["alerts"].features
      )
      self.assertFalse(
          np.all(sample.node_sets["hardware"].features["time_mask"] == 0)
      )
      num_batches += 1

    self.assertEqual(num_batches, 2)

  def test_default_no_per_sample_transforms(self):
    graph, schema = self._create_temporal_test_graph_and_schema()
    schema.node_sets["hardware"].features["time"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_timeseries=True,
        is_creation_time=True,
        shape=(3,),
    )
    schema.node_sets["hardware"].features["signal"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        is_timeseries=True,
        shape=(3,),
    )
    graph.node_sets["hardware"].features["time"] = np.array(
        [[50, 80, 120], [150, 250, 300]], dtype=np.int64
    )
    graph.node_sets["hardware"].features["signal"] = np.array(
        [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32
    )

    generator = dataset.SampleGeneratorFromAnything(
        graph=graph,
        schema=schema,
        batch_size=2,
        seed_node_idxs=None,
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="alerts",
            num_hops=1,
            hop_width=2,
            temporal_sampling=True,
        ),
        temporal=True,
        drop_remainder=False,
        shuffle=False,
    )

    self.assertNotIn(
        "time_mask", generator.output_schema().node_sets["hardware"].features
    )
    self.assertNotIn(
        "creation_time_seed_delta",
        generator.output_schema().node_sets["alerts"].features,
    )
    self.assertNotIn(
        "time_seed_delta",
        generator.output_schema().node_sets["hardware"].features,
    )

  def test_temporal_sampling_requires_seed_timestamps(self):
    graph = gen_test_graph.generate_in_memory_graph(True, False)
    schema = gen_test_graph.generate_schema(True, False, True, False)
    with self.assertRaisesRegex(
        ValueError, "no creation timestamp feature was found"
    ):
      dataset.SampleGeneratorFromAnything(
          graph=graph,
          schema=schema,
          batch_size=2,
          seed_node_idxs=None,
          sampling_config=sampling_config_lib.SimpleSamplingConfig(
              seed_nodeset="n1",
              num_hops=1,
              hop_width=2,
          ),
          temporal=True,
          drop_remainder=False,
          shuffle=False,
      )

  def test_sampler_returns_node_idxs_only_with_transforms_raises(self):
    graph, schema = self._create_temporal_test_graph_and_schema()
    with self.assertRaisesRegex(
        ValueError, "cannot be used when `sampler_returns_node_idxs_only=True`"
    ):
      dataset.SampleGeneratorFromAnything(
          graph=graph,
          schema=schema,
          batch_size=2,
          seed_node_idxs=None,
          sampling_config=sampling_config_lib.SimpleSamplingConfig(
              seed_nodeset="alerts",
              num_hops=1,
              hop_width=2,
              temporal_sampling=True,
          ),
          temporal=True,
          per_sample_transforms=timeseries_transform.PerSampleTransformConfig(
              timeseries_pad_and_cap=timeseries_transform.PadAndCapTimeseriesConfig()
          ),
          sampler_returns_node_idxs_only=True,
          drop_remainder=False,
          shuffle=False,
      )

  def test_non_in_memory_format_with_transforms_raises(self):
    _, schema = self._create_temporal_test_graph_and_schema()
    with self.assertRaisesRegex(
        NotImplementedError,
        "only supported for GraphFormat.IN_MEMORY_GRAPH",
    ):
      dataset.SampleGeneratorFromAnything(
          graph="dummy.bagz",
          schema=schema,
          batch_size=2,
          seed_node_idxs=None,
          sampling_config=sampling_config_lib.SimpleSamplingConfig(
              seed_nodeset="alerts",
              num_hops=1,
              hop_width=2,
              temporal_sampling=True,
          ),
          temporal=True,
          drop_remainder=False,
          shuffle=False,
      )

  def test_timedelta_extraction_with_temporal_false(self):
    graph, schema = self._create_temporal_test_graph_and_schema()
    generator = dataset.SampleGeneratorFromAnything(
        graph=graph,
        schema=schema,
        batch_size=2,
        seed_node_idxs=np.array([0, 1, 2, 3], dtype=np.int64),
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="alerts",
            num_hops=1,
            hop_width=2,
        ),
        per_sample_transforms=timeseries_transform.PerSampleTransformConfig(
            timeseries_pad_and_cap=timeseries_transform.PadAndCapTimeseriesConfig(
                sequence_length=5
            ),
            timedelta_extraction=timeseries_transform.TimestampFeatureExtractorConfig(),
        ),
        temporal=False,
        drop_remainder=False,
        shuffle=False,
    )

    self.assertIn(
        "creation_time_seed_delta",
        generator.output_schema().node_sets["alerts"].features,
    )
    self.assertIn(
        "time_seed_delta",
        generator.output_schema().node_sets["hardware"].features,
    )
    for sample, _ in generator.batch_iterator():
      self.assertIn("time_seed_delta", sample.node_sets["hardware"].features)
      self.assertIn(
          "creation_time_seed_delta", sample.node_sets["alerts"].features
      )
      break

  def test_dynamic_set_sampler_returns_node_idxs_only_raises(self):
    graph, schema = self._create_temporal_test_graph_and_schema()
    generator = dataset.SampleGeneratorFromAnything(
        graph=graph,
        schema=schema,
        batch_size=2,
        seed_node_idxs=None,
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="alerts",
            num_hops=1,
            hop_width=2,
            temporal_sampling=True,
        ),
        temporal=True,
        per_sample_transforms=timeseries_transform.PerSampleTransformConfig(
            timeseries_pad_and_cap=timeseries_transform.PadAndCapTimeseriesConfig()
        ),
        drop_remainder=False,
        shuffle=False,
    )
    with self.assertRaisesRegex(
        ValueError, "cannot be used when `sampler_returns_node_idxs_only=True`"
    ):
      generator.set_sampler_returns_node_idxs_only(True)

  def test_timedelta_extraction_without_pad_and_cap_dynamic_ts_raises(self):
    graph, schema = self._create_temporal_test_graph_and_schema()
    # Add a dynamic shape timeseries feature
    schema.node_sets["hardware"].features["ts_dynamic"] = (
        schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            shape=(None,),
            is_timeseries=True,
        )
    )
    with self.assertRaisesRegex(
        ValueError,
        "Dynamic shape timeseries features were detected in the schema",
    ):
      dataset.SampleGeneratorFromAnything(
          graph=graph,
          schema=schema,
          batch_size=2,
          seed_node_idxs=None,
          sampling_config=sampling_config_lib.SimpleSamplingConfig(
              seed_nodeset="alerts",
              num_hops=1,
              hop_width=2,
              temporal_sampling=True,
          ),
          temporal=True,
          per_sample_transforms=timeseries_transform.PerSampleTransformConfig(
              timedelta_extraction=timeseries_transform.TimestampFeatureExtractorConfig()
          ),
          drop_remainder=False,
          shuffle=False,
      )


if __name__ == "__main__":
  absltest.main()

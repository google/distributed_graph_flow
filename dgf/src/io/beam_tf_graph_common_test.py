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
import absl.testing.absltest as absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as beam_test_util
from dgf.src.beam import runners
from dgf.src.data import distributed_graph
from dgf.src.data import schema as schema_lib
from dgf.src.io import beam_tf_graph_common
from dgf.src.io import tf_graph_common
import numpy as np
import tensorflow as tf


def _create_pipeline():
  return test_pipeline.TestPipeline(
      runner=runners.runner_from_name("FlumePython")
  )


class BeamTfGraphCommonTest(absltest.TestCase):

  def test_node_to_tf_example(self):
    node = distributed_graph.Node(id=b"n1", features={"f1": np.array([1, 2])})
    schema = {
        "f1": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64, shape=(2,)
        ),
        tf_graph_common.DEFAULT_KEY_ID: schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BYTES, shape=()
        ),
    }
    nodeset_schema = schema_lib.NodeSchema(features=schema)
    example = beam_tf_graph_common.node_to_tf_example(
        node, tf_graph_common.DEFAULT_KEY_ID, nodeset_schema
    )
    self.assertIn("f1", example.features.feature)
    self.assertIn(tf_graph_common.DEFAULT_KEY_ID, example.features.feature)
    self.assertEqual(example.features.feature["f1"].int64_list.value[:], [1, 2])
    self.assertEqual(
        example.features.feature[
            tf_graph_common.DEFAULT_KEY_ID
        ].bytes_list.value[0],
        b"n1",
    )

  def test_edge_to_tf_example(self):
    edge = distributed_graph.Edge(
        id=b"e1",
        source=b"s1",
        target=10,
        features={"w": np.array([0.5], dtype=np.float32)},
    )
    edge_schema = schema_lib.EdgeSchema(
        source="n1",
        target="n2",
        features={
            "w": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32, shape=(1,)
            )
        },
    )
    example = beam_tf_graph_common.edge_to_tf_example(
        edge=edge,
        edge_id_column="#id",
        edge_schema=edge_schema,
        source_format=schema_lib.FeatureFormat.BYTES,
        target_format=schema_lib.FeatureFormat.INTEGER_64,
    )
    self.assertEqual(
        example.features.feature["#source"].bytes_list.value[0], b"s1"
    )
    self.assertEqual(
        example.features.feature["#target"].int64_list.value[0], 10
    )
    self.assertEqual(example.features.feature["#id"].bytes_list.value[0], b"e1")
    self.assertAlmostEqual(
        example.features.feature["w"].float_list.value[0], 0.5
    )

  def test_tf_example_to_edge(self):
    example = tf.train.Example()
    example.features.feature["#source"].bytes_list.value.append(b"s1")
    example.features.feature["#target"].bytes_list.value.append(b"t1")
    example.features.feature["#id"].bytes_list.value.append(b"e1")
    example.features.feature["w"].float_list.value.append(0.5)

    edge_schema = schema_lib.EdgeSchema(
        source="n1",
        target="n2",
        features={
            "w": schema_lib.FeatureSchema(
                format=schema_lib.FeatureFormat.FLOAT_32, shape=(1,)
            )
        },
    )
    edge = beam_tf_graph_common.tf_example_to_edge(
        example=example,
        edge_id_column="#id",
        schema=edge_schema,
        ignore_keys=(
            tf_graph_common.KEY_SOURCE,
            tf_graph_common.KEY_TARGET,
        ),
    )
    self.assertEqual(edge.source, b"s1")
    self.assertEqual(edge.target, b"t1")
    self.assertEqual(edge.id, b"e1")
    self.assertIsNotNone(edge.features)
    assert edge.features is not None
    self.assertIn("w", edge.features)
    self.assertAlmostEqual(edge.features["w"][0], 0.5)

  def test_recordio_container_read_write(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      prefix = os.path.join(tmpdir, "examples")
      example1 = tf.train.Example()
      example1.features.feature["f"].int64_list.value.append(42)
      example2 = tf.train.Example()
      example2.features.feature["f"].int64_list.value.append(43)

      with _create_pipeline() as p_write:
        examples_pcoll = p_write | beam.Create([example1, example2])
        _ = examples_pcoll | beam_tf_graph_common.WriteTfExampleContainer(
            file_path_prefix=prefix,
            extension=".recordio",
            container_type=tf_graph_common.TfExampleContainer.RECORDIO,
            num_shards=1,
        )

      with _create_pipeline() as p_read:
        read_examples = p_read | beam_tf_graph_common.ReadTfExampleContainer(
            file_pattern=f"{prefix}*.recordio",
            container_type=tf_graph_common.TfExampleContainer.RECORDIO,
        )
        beam_test_util.assert_that(
            read_examples,
            beam_test_util.equal_to([example1, example2]),
        )


if __name__ == "__main__":
  absltest.main()

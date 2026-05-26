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

import copy
import functools
import logging
import os
import tempfile
from typing import List

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util
from dgf.src.data import distributed_graph as distributed_graph_lib
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_in_beam as gf_graph_in_beam_lib
from dgf.src.io import tf_graph_sample as tf_graph_sample_lib
from dgf.src.sampling import beam_semi_distributed_sampler
from dgf.src.sampling import beam_semi_distributed_sampler_v2
from dgf.src.sampling import config as config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util
import numpy as np

test_util.disable_diff_truncation()


def _are_equal_unordered(obj1, obj2):
  if isinstance(obj1, tuple) and len(obj1) == 2 and isinstance(obj1[1], dict):
    key1, dict1 = obj1
    key2, dict2 = obj2
    if key1 != key2:
      return False

    dict1_copy = copy.deepcopy(dict1)
    dict2_copy = copy.deepcopy(dict2)

    for k in ["n1", "n2"]:
      if k in dict1_copy:
        dict1_copy[k].sort(key=lambda x: x[0])
      if k in dict2_copy:
        dict2_copy[k].sort(key=lambda x: x[0])

    return test_util.are_equal(dict1_copy, dict2_copy)
  return test_util.are_equal(obj1, obj2)


class BeamSemiDistributedSamplerV2Test(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="empty_values",
          values=[],
          schema=schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.INTEGER_64, shape=()
          ),
          expected=np.array([], dtype=np.int64),
      ),
      dict(
          testcase_name="empty_shaped",
          values=[],
          schema=schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.INTEGER_64, shape=(4,)
          ),
          expected=np.zeros(
              dtype=np.int64,
              shape=(0, 4),
          ),
      ),
      dict(
          testcase_name="scalar_int",
          values=[np.array(1), np.array(2)],
          schema=schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.INTEGER_64, shape=()
          ),
          expected=np.array([1, 2], dtype=np.int64),
      ),
      dict(
          testcase_name="fixed_shape_float",
          values=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
          schema=schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.FLOAT_32, shape=(2,)
          ),
          expected=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
      ),
      dict(
          testcase_name="ragged_shape_int",
          values=[np.array([1]), np.array([2, 3])],
          schema=schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.INTEGER_64, shape=(None,)
          ),
          expected=np.array(
              [np.array([1]), np.array([2, 3])], dtype=np.object_
          ),
      ),
      dict(
          testcase_name="ragged_shape_bytes",
          values=[np.array([b"1"]), np.array([b"2", b"3"])],
          schema=schema_lib.FeatureSchema(
              format=schema_lib.FeatureFormat.BYTES, shape=(None,)
          ),
          expected=np.array(
              [np.array([b"1"]), np.array([b"2", b"3"])], dtype=np.object_
          ),
      ),
  )
  def test_safe_stack(
      self,
      values: List[np.ndarray],
      schema: schema_lib.FeatureSchema,
      expected: np.ndarray,
  ):
    test_util.assert_are_equal(
        self,
        beam_semi_distributed_sampler_v2.safe_stack(values, schema),
        expected,
    )

  def test_add_features_to_graph_samples(self):
    with test_pipeline.TestPipeline() as p:
      raw_samples = p | "CreateRawSamples" >> beam.Create([
          distributed_graph_lib.KeyedInMemoryGraph(
              b"a",
              in_memory_graph_lib.InMemoryGraph(
                  node_sets={
                      "n1": in_memory_graph_lib.InMemoryNodeSet(
                          num_nodes=2,
                          features={
                              "#id": np.array([b"n11", b"n12"], dtype=np.bytes_)
                          },
                      ),
                      "n2": in_memory_graph_lib.InMemoryNodeSet(
                          num_nodes=2,
                          features={
                              "#id": np.array([b"n21", b"n22"], dtype=np.bytes_)
                          },
                      ),
                  },
                  edge_sets={},
              ),
          ),
          distributed_graph_lib.KeyedInMemoryGraph(
              b"b",
              in_memory_graph_lib.InMemoryGraph(
                  node_sets={
                      "n1": in_memory_graph_lib.InMemoryNodeSet(
                          num_nodes=2,
                          features={
                              "#id": np.array([b"n11", b"n13"], dtype=np.bytes_)
                          },
                      ),
                      "n2": in_memory_graph_lib.InMemoryNodeSet(
                          num_nodes=1,
                          features={"#id": np.array([b"n23"], dtype=np.bytes_)},
                      ),
                  },
                  edge_sets={},
              ),
          ),
      ])
      feature_graph = distributed_graph_lib.Graph(
          schema=schema_lib.GraphSchema(
              node_sets={
                  "n1": schema_lib.NodeSchema(
                      features={
                          "#id": schema_lib.FeatureSchema(
                              format=schema_lib.FeatureFormat.BYTES
                          ),
                          "f1": schema_lib.FeatureSchema(
                              format=schema_lib.FeatureFormat.FLOAT_64, shape=()
                          ),
                          "f2": schema_lib.FeatureSchema(
                              format=schema_lib.FeatureFormat.INTEGER_64,
                              shape=(2,),
                          ),
                          "f3": schema_lib.FeatureSchema(
                              format=schema_lib.FeatureFormat.BYTES, shape=(2,)
                          ),
                          "f4": schema_lib.FeatureSchema(
                              format=schema_lib.FeatureFormat.INTEGER_64,
                              shape=(2, 2),
                          ),
                      }
                  ),
                  "n2": schema_lib.NodeSchema(
                      features={
                          "#id": schema_lib.FeatureSchema(
                              format=schema_lib.FeatureFormat.BYTES
                          ),
                          "f5": schema_lib.FeatureSchema(
                              format=schema_lib.FeatureFormat.INTEGER_64,
                              shape=(),
                          ),
                      }
                  ),
              },
              edge_sets={},
          ),
          node_sets={
              "n1": (
                  p
                  | "CreateN1Features"
                  >> beam.Create([
                      distributed_graph_lib.Node(
                          b"n11",
                          features={
                              "f1": np.array(1.0),
                              "f2": np.array([2, 3]),
                              "f3": np.array([b"x", b"y"], dtype=np.bytes_),
                              "f4": np.array([[4, 5], [6, 7]]),
                          },
                      ),
                      distributed_graph_lib.Node(
                          b"n12",
                          features={
                              "f1": np.array(1.1),
                              "f2": np.array([22, 33]),
                              "f3": np.array([b"xx", b"yy"], dtype=np.bytes_),
                              "f4": np.array([[44, 55], [66, 77]]),
                          },
                      ),
                      distributed_graph_lib.Node(
                          b"n13",
                          features={
                              "f1": np.array(1.2),
                              "f2": np.array([222, 333]),
                              "f3": np.array([b"xxx", b"yyy"], dtype=np.bytes_),
                              "f4": np.array([[444, 555], [666, 777]]),
                          },
                      ),
                      distributed_graph_lib.Node(
                          b"n14",
                          features={
                              "f1": np.array(1.3),
                              "f2": np.array([2223, 3334]),
                              "f3": np.array(
                                  [b"xxxx", b"yyyy"], dtype=np.bytes_
                              ),
                              "f4": np.array([[4444, 5555], [6666, 7777]]),
                          },
                      ),
                  ])
              ),
              "n2": (
                  p
                  | "CreateN2Features"
                  >> beam.Create([
                      distributed_graph_lib.Node(
                          b"n21", features={"f5": np.array(10)}
                      ),
                      distributed_graph_lib.Node(
                          b"n22", features={"f5": np.array(11)}
                      ),
                      distributed_graph_lib.Node(
                          b"n23", features={"f5": np.array(12)}
                      ),
                      distributed_graph_lib.Node(
                          b"n24", features={"f5": np.array(13)}
                      ),
                  ])
              ),
          },
          edge_sets={},
      )

      probe_stages = {}
      augmented_samples = (
          beam_semi_distributed_sampler_v2.add_features_to_graph_samples(
              raw_samples, feature_graph, probe_stages=probe_stages
          )
      )
      array = np.array

      util.assert_that(
          probe_stages["stage_3"]["n1"],
          util.equal_to([
              (b"n11", (b"a", 0)),
              (b"n12", (b"a", 1)),
              (b"n11", (b"b", 0)),
              (b"n13", (b"b", 1)),
          ]),
      )
      util.assert_that(
          probe_stages["stage_3"]["n2"],
          util.equal_to([
              (b"n21", (b"a", 0)),
              (b"n22", (b"a", 1)),
              (b"n23", (b"b", 0)),
          ]),
      )

      util.assert_that(
          probe_stages["stage_4"]["n1"],
          util.equal_to(
              [
                  (
                      b"n11",
                      {
                          "f1": np.array(1.0),
                          "f2": np.array([2, 3]),
                          "f3": np.array([b"x", b"y"], dtype=np.bytes_),
                          "f4": np.array([[4, 5], [6, 7]]),
                      },
                  ),
                  (
                      b"n12",
                      {
                          "f1": np.array(1.1),
                          "f2": np.array([22, 33]),
                          "f3": np.array([b"xx", b"yy"], dtype=np.bytes_),
                          "f4": np.array([[44, 55], [66, 77]]),
                      },
                  ),
                  (
                      b"n13",
                      {
                          "f1": np.array(1.2),
                          "f2": np.array([222, 333]),
                          "f3": np.array([b"xxx", b"yyy"], dtype=np.bytes_),
                          "f4": np.array([[444, 555], [666, 777]]),
                      },
                  ),
                  (
                      b"n14",
                      {
                          "f1": np.array(1.3),
                          "f2": np.array([2223, 3334]),
                          "f3": np.array([b"xxxx", b"yyyy"], dtype=np.bytes_),
                          "f4": np.array([[4444, 5555], [6666, 7777]]),
                      },
                  ),
              ],
              equals_fn=functools.partial(test_util.are_equal),
          ),
      )
      util.assert_that(
          probe_stages["stage_4"]["n2"],
          util.equal_to(
              [
                  (b"n21", {"f5": np.array(10)}),
                  (b"n22", {"f5": np.array(11)}),
                  (b"n23", {"f5": np.array(12)}),
                  (b"n24", {"f5": np.array(13)}),
              ],
              equals_fn=functools.partial(test_util.are_equal),
          ),
      )

      util.assert_that(
          probe_stages["stage_5"]["n1"],
          util.equal_to(
              [
                  (
                      b"n11",
                      {
                          "s": [(b"a", 0), (b"b", 0)],
                          "f": [{
                              "f1": array(1.0),
                              "f2": array([2, 3]),
                              "f3": array([b"x", b"y"], dtype=np.bytes_),
                              "f4": array([[4, 5], [6, 7]]),
                          }],
                      },
                  ),
                  (
                      b"n12",
                      {
                          "s": [(b"a", 1)],
                          "f": [{
                              "f1": array(1.1),
                              "f2": array([22, 33]),
                              "f3": array([b"xx", b"yy"], dtype=np.bytes_),
                              "f4": array([[44, 55], [66, 77]]),
                          }],
                      },
                  ),
                  (
                      b"n13",
                      {
                          "s": [(b"b", 1)],
                          "f": [{
                              "f1": array(1.2),
                              "f2": array([222, 333]),
                              "f3": array([b"xxx", b"yyy"], dtype=np.bytes_),
                              "f4": array([[444, 555], [666, 777]]),
                          }],
                      },
                  ),
              ],
              equals_fn=functools.partial(test_util.are_equal),
          ),
      )
      util.assert_that(
          probe_stages["stage_5"]["n2"],
          util.equal_to(
              [
                  (
                      b"n21",
                      {"s": [(b"a", 0)], "f": [{"f5": array(10)}]},
                  ),
                  (
                      b"n22",
                      {"s": [(b"a", 1)], "f": [{"f5": array(11)}]},
                  ),
                  (
                      b"n23",
                      {"s": [(b"b", 0)], "f": [{"f5": array(12)}]},
                  ),
              ],
              equals_fn=functools.partial(test_util.are_equal),
          ),
      )
      util.assert_that(
          probe_stages["stage_6"]["n1"],
          util.equal_to(
              [
                  (
                      b"a",
                      (
                          0,
                          {
                              "f1": array(1.0),
                              "f2": array([2, 3]),
                              "f3": array([b"x", b"y"], dtype="|S1"),
                              "f4": array([[4, 5], [6, 7]]),
                          },
                      ),
                  ),
                  (
                      b"b",
                      (
                          0,
                          {
                              "f1": array(1.0),
                              "f2": array([2, 3]),
                              "f3": array([b"x", b"y"], dtype="|S1"),
                              "f4": array([[4, 5], [6, 7]]),
                          },
                      ),
                  ),
                  (
                      b"a",
                      (
                          1,
                          {
                              "f1": array(1.1),
                              "f2": array([22, 33]),
                              "f3": array([b"xx", b"yy"], dtype="|S2"),
                              "f4": array([[44, 55], [66, 77]]),
                          },
                      ),
                  ),
                  (
                      b"b",
                      (
                          1,
                          {
                              "f1": array(1.2),
                              "f2": array([222, 333]),
                              "f3": array([b"xxx", b"yyy"], dtype="|S3"),
                              "f4": array([[444, 555], [666, 777]]),
                          },
                      ),
                  ),
              ],
              equals_fn=functools.partial(test_util.are_equal),
          ),
      )
      util.assert_that(
          probe_stages["stage_6"]["n2"],
          util.equal_to(
              [
                  (b"a", (0, {"f5": array(10)})),
                  (b"a", (1, {"f5": array(11)})),
                  (b"b", (0, {"f5": array(12)})),
              ],
              equals_fn=functools.partial(test_util.are_equal),
          ),
      )

      InMemoryGraph = in_memory_graph_lib.InMemoryGraph
      InMemoryNodeSet = in_memory_graph_lib.InMemoryNodeSet
      util.assert_that(
          probe_stages["stage_7"],
          util.equal_to(
              [
                  (
                      b"a",
                      {
                          "n1": [
                              (
                                  0,
                                  {
                                      "f1": array(1.0),
                                      "f2": array([2, 3]),
                                      "f3": array([b"x", b"y"], dtype="|S1"),
                                      "f4": array([[4, 5], [6, 7]]),
                                  },
                              ),
                              (
                                  1,
                                  {
                                      "f1": array(1.1),
                                      "f2": array([22, 33]),
                                      "f3": array([b"xx", b"yy"], dtype="|S2"),
                                      "f4": array([[44, 55], [66, 77]]),
                                  },
                              ),
                          ],
                          "n2": [
                              (0, {"f5": array(10)}),
                              (1, {"f5": array(11)}),
                          ],
                          "__gfsample__": [
                              InMemoryGraph(
                                  node_sets={
                                      "n1": InMemoryNodeSet(
                                          num_nodes=2,
                                          features={
                                              "#id": array(
                                                  [b"n11", b"n12"], dtype="|S3"
                                              )
                                          },
                                      ),
                                      "n2": InMemoryNodeSet(
                                          num_nodes=2,
                                          features={
                                              "#id": array(
                                                  [b"n21", b"n22"], dtype="|S3"
                                              )
                                          },
                                      ),
                                  },
                                  edge_sets={},
                              )
                          ],
                      },
                  ),
                  (
                      b"b",
                      {
                          "n1": [
                              (
                                  0,
                                  {
                                      "f1": array(1.0),
                                      "f2": array([2, 3]),
                                      "f3": array([b"x", b"y"], dtype="|S1"),
                                      "f4": array([[4, 5], [6, 7]]),
                                  },
                              ),
                              (
                                  1,
                                  {
                                      "f1": array(1.2),
                                      "f2": array([222, 333]),
                                      "f3": array(
                                          [b"xxx", b"yyy"], dtype="|S3"
                                      ),
                                      "f4": array([[444, 555], [666, 777]]),
                                  },
                              ),
                          ],
                          "n2": [(0, {"f5": array(12)})],
                          "__gfsample__": [
                              InMemoryGraph(
                                  node_sets={
                                      "n1": InMemoryNodeSet(
                                          num_nodes=2,
                                          features={
                                              "#id": array(
                                                  [b"n11", b"n13"], dtype="|S3"
                                              )
                                          },
                                      ),
                                      "n2": InMemoryNodeSet(
                                          num_nodes=1,
                                          features={
                                              "#id": array(
                                                  [b"n23"], dtype="|S3"
                                              )
                                          },
                                      ),
                                  },
                                  edge_sets={},
                              )
                          ],
                      },
                  ),
              ],
              equals_fn=functools.partial(_are_equal_unordered),
          ),
      )

      util.assert_that(
          augmented_samples,
          util.equal_to(
              [
                  distributed_graph_lib.KeyedInMemoryGraph(
                      key=b"a",
                      graph=InMemoryGraph(
                          node_sets={
                              "n1": InMemoryNodeSet(
                                  num_nodes=2,
                                  features={
                                      "f1": array([1.0, 1.1]),
                                      "f2": array([[2, 3], [22, 33]]),
                                      "f3": array(
                                          [[b"x", b"y"], [b"xx", b"yy"]],
                                          dtype="|S2",
                                      ),
                                      "f4": array([
                                          [[4, 5], [6, 7]],
                                          [[44, 55], [66, 77]],
                                      ]),
                                      "#id": array(
                                          [b"n11", b"n12"], dtype="|S3"
                                      ),
                                  },
                              ),
                              "n2": InMemoryNodeSet(
                                  num_nodes=2,
                                  features={
                                      "f5": array([10, 11]),
                                      "#id": array(
                                          [b"n21", b"n22"], dtype="|S3"
                                      ),
                                  },
                              ),
                          },
                          edge_sets={},
                      ),
                  ),
                  distributed_graph_lib.KeyedInMemoryGraph(
                      key=b"b",
                      graph=InMemoryGraph(
                          node_sets={
                              "n1": InMemoryNodeSet(
                                  num_nodes=2,
                                  features={
                                      "f1": array([1.0, 1.2]),
                                      "f2": array([[2, 3], [222, 333]]),
                                      "f3": array(
                                          [[b"x", b"y"], [b"xxx", b"yyy"]],
                                          dtype="|S3",
                                      ),
                                      "f4": array([
                                          [[4, 5], [6, 7]],
                                          [[444, 555], [666, 777]],
                                      ]),
                                      "#id": array(
                                          [b"n11", b"n13"], dtype="|S3"
                                      ),
                                  },
                              ),
                              "n2": InMemoryNodeSet(
                                  num_nodes=1,
                                  features={
                                      "f5": array([12]),
                                      "#id": array([b"n23"], dtype="|S3"),
                                  },
                              ),
                          },
                          edge_sets={},
                      ),
                  ),
              ],
              equals_fn=functools.partial(test_util.are_equal),
          ),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="collect_feature_in_sampler",
          beam_feature_collection=False,
      ),
      dict(
          testcase_name="collect_feature_with_beam",
          beam_feature_collection=True,
      ),
  )
  def test_sample_with_beam_semi_distributed_sampler_v2(
      self, beam_feature_collection: bool
  ):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Generate some toy data
      graph_path = os.path.join(tmpdir, "graph")
      gen_test_graph.generate_gf_graph(graph_path, edge_ids=False)

      # Sampling configuration
      sampling_config = config_lib.SimpleSamplingConfig(
          seed_nodeset="n1", num_hops=2
      )

      with test_pipeline.TestPipeline() as p:
        # Generate some samples (with the beam sampler).
        graph = gf_graph_in_beam_lib.read_graph(
            p,
            graph_path,
            schema_filter=schema_lib.GraphSchemaFilter(
                nodeset_fn=lambda key, sch: key == "n1",
                edgeset_fn=lambda key, sch: False,
                feature_fn=lambda key, sch: key == "#id",
            ),
        )
        seeds = beam_semi_distributed_sampler.extract_beam_nodes_ids(
            graph, sampling_config.seed_nodeset
        )
        samples, schema = (
            beam_semi_distributed_sampler_v2.sample_with_beam_semi_distributed_sampler_v2(
                graph_path,
                sampling_config,
                seeds=seeds,
                debug_sampling=True,
                beam_feature_collection=beam_feature_collection,
            )
        )
        _ = samples | "Generated samples" >> beam.Map(print)

        # Write the samples to file (not checked; just to make sure the
        # signature is correct).
        sample_path = os.path.join(tmpdir, "samples@*")

        tf_graph_sample_lib.write_tfgnn_graphs_beam(
            samples, sample_path, schema
        )

      # Generate some samples with the in-process sampler.
      in_memory_sampler = in_memory_sampler_lib.create_sampler(
          gen_test_graph.generate_in_memory_graph(node_ids=True),
          sampling_config,
          schema=schema,
          batch_size=1,
          debug_sampling=True,
          return_features=True,
          return_node_idxs=False,
      )

      loaded_samples = list(
          tf_graph_sample_lib.read_tfgnn_graphs(sample_path, schema)
      )

      expected_samples = [
          in_memory_sampler.sample(0),
          in_memory_sampler.sample(1),
      ]
      logging.info("Expected samples:\n%s", expected_samples)
      test_util.assert_are_equal(self, loaded_samples, expected_samples)


if __name__ == "__main__":
  absltest.main()

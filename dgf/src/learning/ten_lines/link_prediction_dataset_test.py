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

import collections
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.generate import edge_neighbor_generator as edge_neighbor_generator_lib
from dgf.src.io import jax as jax_io_lib
from dgf.src.learning.ten_lines import link_prediction_dataset
from dgf.src.plot import network as network_lib
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.transform import normalize as normalize_lib
from dgf.src.util import gen_test_graph
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np

Counter = collections.Counter


class GNNLinkDatasetPreparatorTest(parameterized.TestCase):
  preparator: link_prediction_dataset.GNNLinkDatasetPreparator
  preparator_mask_edge: link_prediction_dataset.GNNLinkDatasetPreparator

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.graph, cls.schema = (
        gen_test_graph.generate_recommender_like_in_memory_graph()
    )

    common_kwargs = {
        "graph": cls.graph,
        "schema": cls.schema,
        "sampling_config": sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1",
            num_hops=1,
            hop_width=2,
            reverse=True,
        ),
        "batch_size": 2,
        "drop_remainder": False,
        "shuffle": True,
        "target_edgeset": "e2",
        "num_negative_nodes": 3,
        "seed_edge_idxs": None,
        "auto_normalize_config": normalize_lib.AutoNormalizeConfig(
            keep_raw_features={"#id"}
        ),
        "edge_neighbor_generator": (
            edge_neighbor_generator_lib.RandomEdgeNeighborGeneratorConfig()
        ),
    }

    cls.preparator = link_prediction_dataset.GNNLinkDatasetPreparator(
        **common_kwargs,
    )
    cls.preparator.prepare()

    cls.preparator_mask_edge = link_prediction_dataset.GNNLinkDatasetPreparator(
        **common_kwargs,
        mask_seed_edge=True,
    )
    cls.preparator_mask_edge.prepare()

  def test_generate_node_ids(self):
    visited_pos_src = Counter()
    visited_pos_target = Counter()
    visited_edge = Counter()
    num_batches = 0
    for batch in self.preparator._in_memory_generate_node_ids(
        self.preparator._build_neighbor_generator(None)
    ):
      num_batches += 1
      self.assertEqual(batch.pos_src_node_idxs.shape, (2,))
      self.assertEqual(batch.pos_trg_node_idxs.shape, (2,))
      self.assertEqual(batch.neg_trg_node_idxs.shape, (2, 3))
      self.assertEqual(batch.edge_idxs.shape, (2,))
      visited_pos_src.update(batch.pos_src_node_idxs)
      visited_pos_target.update(batch.pos_trg_node_idxs)
      visited_edge.update(batch.edge_idxs)
    self.assertEqual(num_batches, 2)
    self.assertEqual(visited_pos_src, Counter({0: 2, 1: 2}))
    self.assertEqual(visited_pos_target, Counter({0: 1, 1: 2, 2: 1}))
    self.assertEqual(visited_edge, Counter({0: 1, 1: 1, 2: 1, 3: 1}))

  def test_statistics(self):
    logging.info("%s", self.preparator.get_live().source_feature_stats)
    logging.info("%s", self.preparator.get_live().target_feature_stats)

    source_stats = self.preparator.get_live().source_feature_stats
    target_stats = self.preparator.get_live().target_feature_stats

    def _assert_feature_stats(
        stats, node_set, feature_name, expected_min, expected_max
    ):
      feature_stats = stats.node_sets[node_set].features[feature_name]
      self.assertEqual(feature_stats.minimum, expected_min)
      self.assertEqual(feature_stats.maximum, expected_max)

    _assert_feature_stats(source_stats, "n1", "f1", 0, 1)
    _assert_feature_stats(source_stats, "n2", "f2", 0, 2)

    _assert_feature_stats(target_stats, "n1", "f1", 0, 1)
    _assert_feature_stats(target_stats, "n2", "f2", 0, 5)

  def test_padding(self):
    pos_src_padding = self.preparator.get_live().positive_source_padding
    pos_trg_padding = self.preparator.get_live().positive_target_padding
    neg_trg_padding = self.preparator.get_live().negative_target_padding

    logging.info("%s", pos_src_padding)
    logging.info("%s", pos_trg_padding)
    logging.info("%s", neg_trg_padding)

    # Assertions for positive_source_padding
    self.assertEqual(pos_src_padding.node_sets["n1"].num_nodes, 4)
    self.assertEqual(pos_src_padding.node_sets["n2"].num_nodes, 6)
    self.assertEqual(pos_src_padding.edge_sets["e2"].num_edges, 6)

    # Assertions for positive_target_padding
    self.assertEqual(pos_trg_padding.node_sets["n2"].num_nodes, 4)
    self.assertEqual(pos_trg_padding.edge_sets["e2"].num_edges, 6)

    # Assertions for negative_target_padding
    # Note: Negative edge padding is very stocastic

  def test_sampling_plan(self):
    self.assertEqual(
        self.preparator.get_live().source_sampling_plan.root.nodeset, "n1"
    )
    self.assertEqual(
        self.preparator.get_live().target_sampling_plan.root.nodeset, "n2"
    )

  @parameterized.named_parameters(
      ("default", False),
      ("mask_edge", True),
  )
  def test_generate(self, use_mask_edge):
    preparator = self.preparator_mask_edge if use_mask_edge else self.preparator

    # Note: Disabling the padding in link_prediction_dataset.py (i.e.
    # padding=None) make the plots more readable.
    plot_to_tmp = False

    if plot_to_tmp:
      output_dir = "/tmp/gf"
      network_lib.plot_graph(self.graph, self.schema).render(
          f"{output_dir}/original", format="png", cleanup=True
      )

    live = preparator.get_live()
    num_batches = 0
    for sample in preparator.generate():
      in_memory_graph_validate_lib.validate_graph(
          sample.positive_source_graph,
          live.source_normalizer.output_schema(),
          raise_on_warning=False,
      )
      in_memory_graph_validate_lib.validate_graph(
          sample.positive_target_graph,
          live.target_normalizer.output_schema(),
          raise_on_warning=False,
      )
      in_memory_graph_validate_lib.validate_graph(
          sample.negative_target_graph,
          live.target_normalizer.output_schema(),
          raise_on_warning=False,
      )

      if plot_to_tmp:
        # Plot sampled graph. For debug.
        output_dir = "/tmp/gf"
        test_name = "mask_edge" if use_mask_edge else "default"
        network_lib.plot_graph(
            sample.positive_source_graph, live.source_normalizer.output_schema()
        ).render(
            f"{output_dir}/positive_source_graph_{test_name}_{num_batches}",
            format="png",
            cleanup=True,
        )
        network_lib.plot_graph(
            sample.positive_target_graph, live.target_normalizer.output_schema()
        ).render(
            f"{output_dir}/positive_target_graph_{test_name}_{num_batches}",
            format="png",
            cleanup=True,
        )
        network_lib.plot_graph(
            sample.negative_target_graph, live.target_normalizer.output_schema()
        ).render(
            f"{output_dir}/negative_target_graph_{test_name}_{num_batches}",
            format="png",
            cleanup=True,
        )

      num_batches += 1
    self.assertEqual(num_batches, 2)

  def test_generate_one_default(self):
    live = self.preparator.get_live()
    batch_seed = link_prediction_dataset.NodeIdsBatch(
        pos_src_node_idxs=np.array([0], dtype=np.int64),
        pos_trg_node_idxs=np.array([1], dtype=np.int64),
        neg_trg_node_idxs=np.array([[2, 3, 4]], dtype=np.int64),
        edge_idxs=np.array([1], dtype=np.int64),
    )
    sample = self.preparator._generate_one(live, batch_seed, padding=False)
    in_memory_graph_validate_lib.validate_graph(
        sample.positive_source_graph,
        live.source_normalizer.output_schema(),
        raise_on_warning=False,
    )
    in_memory_graph_validate_lib.validate_graph(
        sample.positive_target_graph,
        live.target_normalizer.output_schema(),
        raise_on_warning=False,
    )
    in_memory_graph_validate_lib.validate_graph(
        sample.negative_target_graph,
        live.target_normalizer.output_schema(),
        raise_on_warning=False,
    )
    self.assertIsNotNone(sample.positive_source_graph)
    self.assertIsNotNone(sample.positive_target_graph)
    self.assertIsNotNone(sample.negative_target_graph)

    # Assertions for positive_source_graph
    self.assertEqual(
        set(sample.positive_source_graph.node_sets["n1"].features["#id"]), {0}
    )
    self.assertEqual(
        set(sample.positive_source_graph.node_sets["n2"].features["#id"]),
        {0, 1},
    )
    self.assertEqual(
        sample.positive_source_graph.edge_sets["e2"].num_edges(), 2
    )

    # Assertions for positive_target_graph
    self.assertEqual(
        set(sample.positive_target_graph.node_sets["n2"].features["#id"]), {1}
    )
    self.assertEqual(
        set(sample.positive_target_graph.node_sets["n1"].features["#id"]),
        {0, 1},
    )
    self.assertEqual(
        sample.positive_target_graph.edge_sets["e2"].num_edges(), 2
    )

    # Assertions for negative_target_graph
    self.assertEqual(
        set(sample.negative_target_graph.node_sets["n2"].features["#id"]),
        {2, 3, 4},
    )
    self.assertEqual(
        set(sample.negative_target_graph.node_sets["n1"].features["#id"]), {1}
    )
    self.assertEqual(
        sample.negative_target_graph.edge_sets["e2"].num_edges(), 1
    )

  def test_generate_one_mask_edge(self):
    live = self.preparator_mask_edge.get_live()
    batch_seed = link_prediction_dataset.NodeIdsBatch(
        pos_src_node_idxs=np.array([0], dtype=np.int64),
        pos_trg_node_idxs=np.array([1], dtype=np.int64),
        neg_trg_node_idxs=np.array([[2, 3, 4]], dtype=np.int64),
        edge_idxs=np.array([1], dtype=np.int64),
    )
    sample = self.preparator_mask_edge._generate_one(
        live, batch_seed, padding=False
    )
    in_memory_graph_validate_lib.validate_graph(
        sample.positive_source_graph,
        live.source_normalizer.output_schema(),
        raise_on_warning=False,
    )
    in_memory_graph_validate_lib.validate_graph(
        sample.positive_target_graph,
        live.target_normalizer.output_schema(),
        raise_on_warning=False,
    )
    in_memory_graph_validate_lib.validate_graph(
        sample.negative_target_graph,
        live.target_normalizer.output_schema(),
        raise_on_warning=False,
    )
    self.assertIsNotNone(sample.positive_source_graph)
    self.assertIsNotNone(sample.positive_target_graph)
    self.assertIsNotNone(sample.negative_target_graph)

    # Assertions for positive_source_graph
    self.assertEqual(
        set(sample.positive_source_graph.node_sets["n1"].features["#id"]), {0}
    )
    self.assertEqual(
        set(sample.positive_source_graph.node_sets["n2"].features["#id"]), {0}
    )
    self.assertEqual(
        sample.positive_source_graph.edge_sets["e2"].num_edges(), 1
    )

    # Assertions for positive_target_graph
    self.assertEqual(
        set(sample.positive_target_graph.node_sets["n2"].features["#id"]), {1}
    )
    self.assertEqual(
        set(sample.positive_target_graph.node_sets["n1"].features["#id"]), {1}
    )
    self.assertEqual(
        sample.positive_target_graph.edge_sets["e2"].num_edges(), 1
    )

    # Assertions for negative_target_graph
    self.assertEqual(
        set(sample.negative_target_graph.node_sets["n2"].features["#id"]),
        {2, 3, 4},
    )
    self.assertEqual(
        set(sample.negative_target_graph.node_sets["n1"].features["#id"]), {1}
    )
    self.assertEqual(
        sample.negative_target_graph.edge_sets["e2"].num_edges(), 1
    )

  def test_prepare_from_existing_one(self):
    preparator2 = link_prediction_dataset.GNNLinkDatasetPreparator(
        graph=self.graph,
        schema=self.schema,
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1",
            num_hops=1,
            hop_width=2,
            reverse=True,
        ),
        batch_size=2,
        drop_remainder=False,
        shuffle=True,
        target_edgeset="e2",
        num_negative_nodes=3,
        seed_edge_idxs=None,
        edge_neighbor_generator=edge_neighbor_generator_lib.RandomEdgeNeighborGeneratorConfig(),
    )
    preparator2.prepare_from_existing_one(self.preparator)

    self.assertTrue(preparator2.is_prepared())

    # Verify shared data
    self.assertIs(
        preparator2.get_live().source_feature_stats,
        self.preparator.get_live().source_feature_stats,
    )
    self.assertIs(
        preparator2.get_live().target_feature_stats,
        self.preparator.get_live().target_feature_stats,
    )
    self.assertIs(
        preparator2.get_live().source_normalizer,
        self.preparator.get_live().source_normalizer,
    )
    self.assertIs(
        preparator2.get_live().target_normalizer,
        self.preparator.get_live().target_normalizer,
    )
    self.assertIs(
        preparator2.get_live().positive_source_padding,
        self.preparator.get_live().positive_source_padding,
    )
    self.assertIs(
        preparator2.get_live().positive_target_padding,
        self.preparator.get_live().positive_target_padding,
    )
    self.assertIs(
        preparator2.get_live().negative_target_padding,
        self.preparator.get_live().negative_target_padding,
    )

    # Verify generated samples
    num_batches = 0
    for sample in preparator2.generate():
      num_batches += 1
      self.assertIsInstance(
          sample, link_prediction_dataset.GNNLinkDatasetPreparatorSample
      )
    self.assertEqual(num_batches, 2)

  def test_prepare_from_existing_one_caching(self):
    preparator1 = link_prediction_dataset.GNNLinkDatasetPreparator(
        graph=self.graph,
        schema=self.schema,
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1",
            num_hops=1,
            hop_width=2,
            reverse=True,
        ),
        batch_size=2,
        drop_remainder=False,
        shuffle=True,
        target_edgeset="e2",
        num_negative_nodes=3,
        cache_normalized_features=True,
        cache_normalized_features_device="device",
        edge_neighbor_generator=edge_neighbor_generator_lib.RandomEdgeNeighborGeneratorConfig(),
    )
    preparator1.prepare()

    preparator2 = link_prediction_dataset.GNNLinkDatasetPreparator(
        graph=self.graph,
        schema=self.schema,
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1",
            num_hops=1,
            hop_width=2,
            reverse=True,
        ),
        batch_size=2,
        drop_remainder=False,
        shuffle=True,
        target_edgeset="e2",
        num_negative_nodes=3,
        cache_normalized_features=True,
        cache_normalized_features_device="device",
        edge_neighbor_generator=edge_neighbor_generator_lib.RandomEdgeNeighborGeneratorConfig(),
    )
    preparator2.prepare_from_existing_one(preparator1)

    self.assertTrue(preparator2.is_prepared())
    self.assertIsNotNone(preparator2.get_live().normalized_jax_source_graph)
    self.assertIsNotNone(preparator2.get_live().normalized_jax_target_graph)

  @parameterized.named_parameters(
      ("CacheHost", True, "host"),
      ("CacheDevice", True, "device"),
      ("NoCache", False, "host"),
  )
  def test_caching(self, cache_features, cache_device):
    preparator = link_prediction_dataset.GNNLinkDatasetPreparator(
        graph=self.graph,
        schema=self.schema,
        sampling_config=sampling_config_lib.SimpleSamplingConfig(
            seed_nodeset="n1",
            num_hops=1,
            hop_width=2,
            reverse=True,
        ),
        batch_size=2,
        drop_remainder=False,
        shuffle=True,
        target_edgeset="e2",
        num_negative_nodes=3,
        cache_normalized_features=cache_features,
        cache_normalized_features_device=cache_device,
        edge_neighbor_generator=edge_neighbor_generator_lib.RandomEdgeNeighborGeneratorConfig(),
    )
    preparator.prepare()
    self.assertTrue(preparator.is_prepared())

    live = preparator.get_live()
    num_batches = 0

    if cache_device == "host":
      generator = preparator.generate()
    else:
      generator = (
          link_prediction_dataset.GNNLinkDatasetPreparatorSample(
              positive_source_graph=jax_io_lib.jax_graph_to_graph(
                  sample.positive_source_graph
              ),
              positive_target_graph=jax_io_lib.jax_graph_to_graph(
                  sample.positive_target_graph
              ),
              negative_target_graph=jax_io_lib.jax_graph_to_graph(
                  sample.negative_target_graph
              ),
              positive_source_offsets={
                  k: np.asarray(v)
                  for k, v in sample.positive_source_offsets.items()
              },
              positive_target_offsets={
                  k: np.asarray(v)
                  for k, v in sample.positive_target_offsets.items()
              },
              negative_target_offsets={
                  k: np.asarray(v)
                  for k, v in sample.negative_target_offsets.items()
              },
          )
          for sample in preparator.generate_jax()
      )

    for sample in generator:
      in_memory_graph_validate_lib.validate_graph(
          sample.positive_source_graph,
          live.source_normalizer.output_schema(),
          raise_on_warning=False,
      )
      in_memory_graph_validate_lib.validate_graph(
          sample.positive_target_graph,
          live.target_normalizer.output_schema(),
          raise_on_warning=False,
      )
      in_memory_graph_validate_lib.validate_graph(
          sample.negative_target_graph,
          live.target_normalizer.output_schema(),
          raise_on_warning=False,
      )
      num_batches += 1
    self.assertEqual(num_batches, 2)


if __name__ == "__main__":
  absltest.main()

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

"""Tests for shared batched inference and normalization helpers."""

from absl.testing import absltest
from dgf.src.data import in_memory_graph
from dgf.src.data import padding as padding_data_lib
from dgf.src.data import schema as schema_lib
from dgf.src.learning.ten_lines import inference
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.transform import merge as merge_lib
from dgf.src.util import gen_test_graph
import numpy as np


def _make_simple_graph_and_schema(
    *, with_timestamp: bool = False
) -> tuple[in_memory_graph.InMemoryGraph, schema_lib.GraphSchema]:
  features: dict[str, np.ndarray] = {
      "f": np.array(
          [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32
      ),
  }
  feature_schemas: dict[str, schema_lib.FeatureSchema] = {
      "f": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
          shape=(2,),
      ),
  }
  if with_timestamp:
    features["t"] = np.array([100, 200, 300, 400], dtype=np.int64)
    feature_schemas["t"] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.INTEGER_64,
        semantic=schema_lib.FeatureSemantic.TIMESTAMP,
        is_creation_time=True,
    )

  graph = in_memory_graph.InMemoryGraph(
      node_sets={
          "n": in_memory_graph.InMemoryNodeSet(features=features, num_nodes=4)
      },
      edge_sets={
          "e": in_memory_graph.InMemoryEdgeSet(
              adjacency=np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64),
              features={},
          )
      },
  )
  schema = schema_lib.GraphSchema(
      node_sets={"n": schema_lib.NodeSchema(features=feature_schemas)},
      edge_sets={
          "e": schema_lib.EdgeSchema(
              source="n",
              target="n",
              features={},
          )
      },
  )
  return graph, schema


def _make_padding(
    node_counts: dict[str, int],
    edge_counts: dict[str, int],
) -> padding_data_lib.Padding:
  return padding_data_lib.Padding(
      node_sets={
          k: padding_data_lib.NodeSetPadding(num_nodes=v)
          for k, v in node_counts.items()
      },
      edge_sets={
          k: padding_data_lib.EdgeSetPadding(num_edges=v)
          for k, v in edge_counts.items()
      },
  )


def _make_sampling_plan(
    seed_nodeset: str,
    schema: schema_lib.GraphSchema,
    *,
    num_hops: int = 1,
    hop_width: int = 2,
) -> sampling_config_lib.SamplingPlan:
  return sampling_config_lib.simple_sampling_config_to_sampling_plan(
      sampling_config_lib.SimpleSamplingConfig(
          seed_nodeset=seed_nodeset,
          num_hops=num_hops,
          hop_width=hop_width,
      ),
      schema,
  )


class _RecordingNormalizer:
  """Minimal normalizer stub that records kwargs passed to `normalize_numpy`."""

  def __init__(self, accepted_kwargs: set[str]):
    self.accepted_kwargs = accepted_kwargs
    self.received_kwargs_list: list[dict[str, object]] = []

  def normalize_numpy(
      self, sample: in_memory_graph.InMemoryGraph, **kwargs
  ) -> in_memory_graph.InMemoryGraph:
    self.received_kwargs_list.append(kwargs)
    return sample


class InferenceTest(absltest.TestCase):

  def test_normalize_merged_graph_with_and_without_seed_timestamps(self):
    _, schema = _make_simple_graph_and_schema(with_timestamp=True)
    merged_graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n": in_memory_graph.InMemoryNodeSet(
                features={
                    "f": np.zeros((2, 2), dtype=np.float32),
                    "t": np.array([10, 20], dtype=np.int64),
                },
                num_nodes=2,
            ),
        },
        edge_sets={
            "e": in_memory_graph.InMemoryEdgeSet(
                adjacency=np.zeros((2, 0), dtype=np.int64),
                features={},
            ),
        },
    )
    merge_offsets = {
        "n": np.array([0, 1, 2], dtype=np.int32),
        "e": np.array([0, 0, 0], dtype=np.int32),
    }

    normalizer_with_ts = _RecordingNormalizer(
        accepted_kwargs={"seed_timestamps"}
    )
    out = inference.normalize_merged_graph(
        merged_graph,
        merge_offsets,
        schema=schema,
        normalizer=normalizer_with_ts,  # pyrefly: ignore[bad-argument-type]
        target_nodeset="n",
    )
    self.assertIs(out, merged_graph)
    self.assertLen(normalizer_with_ts.received_kwargs_list, 1)
    received = normalizer_with_ts.received_kwargs_list[0]
    self.assertIn("seed_timestamps", received)
    np.testing.assert_array_equal(
        received["seed_timestamps"]["n"],  # pyrefly: ignore[bad-index]
        np.array([10, 20]),
    )

    normalizer_without_ts = _RecordingNormalizer(accepted_kwargs=set())
    out_no_ts = inference.normalize_merged_graph(
        merged_graph,
        merge_offsets,
        schema=schema,
        normalizer=normalizer_without_ts,  # pyrefly: ignore[bad-argument-type]
        target_nodeset="n",
    )
    self.assertIs(out_no_ts, merged_graph)
    self.assertEqual(normalizer_without_ts.received_kwargs_list, [{}])

  def test_normalize_merged_graph_raises_when_missing_creation_time(self):
    _, schema_no_ts = _make_simple_graph_and_schema(with_timestamp=False)
    merged_graph = in_memory_graph.InMemoryGraph(
        node_sets={
            "n": in_memory_graph.InMemoryNodeSet(
                features={"f": np.zeros((2, 2), dtype=np.float32)},
                num_nodes=2,
            ),
        },
        edge_sets={},
    )
    merge_offsets = {"n": np.array([0, 1, 2], dtype=np.int32)}
    normalizer_with_ts = _RecordingNormalizer(
        accepted_kwargs={"seed_timestamps"}
    )

    with self.assertRaisesRegex(
        ValueError,
        "The target nodeset 'n' must have a creation time feature",
    ):
      inference.normalize_merged_graph(
          merged_graph,
          merge_offsets,
          schema=schema_no_ts,
          normalizer=normalizer_with_ts,  # pyrefly: ignore[bad-argument-type]
          target_nodeset="n",
      )

  def test_generate_normalized_batches_splits_on_padding_overflow(self):
    graph, schema = _make_simple_graph_and_schema(with_timestamp=False)
    sampling_plan = _make_sampling_plan("n", schema)
    normalizer = _RecordingNormalizer(accepted_kwargs=set())

    # Tight padding that fits 1 sample (up to 4 nodes including sentinel, 3
    # edges) but overflows on a merged batch of 2 samples (requiring 7 nodes),
    # forcing `sample_merged_sub_batches` to split.
    tight_padding = _make_padding({"n": 4}, {"e": 3})
    seed_idxs = np.array([0, 1, 2, 3], dtype=np.int32)
    batches = list(
        inference.generate_normalized_batches(
            graph,
            seed_idxs,
            schema=schema,
            sampling_plan=sampling_plan,
            padding=tight_padding,
            normalizer=normalizer,  # pyrefly: ignore[bad-argument-type]
            target_nodeset="n",
            batch_size=2,
            verbose=0,
        )
    )
    # Every seed node is yielded in order without being skipped.
    self.assertLen(batches, 4)
    concatenated_seed_idxs = np.concatenate(
        [
            b.batch_seed_node_idxs
            for b in batches
            if b.batch_seed_node_idxs is not None
        ],
        axis=0,
    )
    np.testing.assert_array_equal(concatenated_seed_idxs, seed_idxs)
    for b in batches:
      self.assertEqual(b.merged_seed_node_idxs.shape, (1,))
      self.assertEqual(b.jax_batch[1].shape, (1,))
      self.assertEqual(b.jax_batch[0].node_sets["n"].num_nodes, 4)

  def test_generate_normalized_batches_temporal_sampling(self):
    graph, schema = gen_test_graph.generate_temporal_in_memory_graph(False)
    sampling_plan = _make_sampling_plan("n1", schema)
    normalizer = _RecordingNormalizer(accepted_kwargs={"seed_timestamps"})
    padding = _make_padding(
        {name: 32 for name in schema.node_sets},
        {name: 32 for name in schema.edge_sets},
    )
    seed_idxs = np.array([0, 1, 2], dtype=np.int32)
    batches = list(
        inference.generate_normalized_batches(
            graph,
            seed_idxs,
            schema=schema,
            sampling_plan=sampling_plan,
            padding=padding,
            normalizer=normalizer,  # pyrefly: ignore[bad-argument-type]
            target_nodeset="n1",
            batch_size=2,
            temporal_sampling=True,
            verbose=0,
        )
    )
    self.assertLen(batches, 2)
    np.testing.assert_array_equal(
        np.concatenate(
            [
                b.batch_seed_node_idxs
                for b in batches
                if b.batch_seed_node_idxs is not None
            ],
            axis=0,
        ),
        seed_idxs,
    )
    self.assertLen(normalizer.received_kwargs_list, 2)
    for kwargs in normalizer.received_kwargs_list:
      self.assertIn("seed_timestamps", kwargs)

  def test_generate_normalized_batches_temporal_raises_without_timestamp(self):
    graph, schema = _make_simple_graph_and_schema(with_timestamp=False)
    sampling_plan = _make_sampling_plan("n", schema)
    normalizer = _RecordingNormalizer(accepted_kwargs=set())
    padding = _make_padding({"n": 16}, {"e": 16})

    with self.assertRaisesRegex(
        ValueError,
        "The target nodeset 'n' must have a creation time feature when"
        " temporal_sampling=True.",
    ):
      list(
          inference.generate_normalized_batches(
              graph,
              np.array([0, 1], dtype=np.int32),
              schema=schema,
              sampling_plan=sampling_plan,
              padding=padding,
              normalizer=normalizer,  # pyrefly: ignore[bad-argument-type]
              target_nodeset="n",
              batch_size=2,
              temporal_sampling=True,
              verbose=0,
          )
      )

  def test_generate_normalized_batches_from_samples(self):
    graph, schema = _make_simple_graph_and_schema(with_timestamp=False)
    sampling_plan = _make_sampling_plan("n", schema)
    sampler = in_memory_sampler_lib.create_sampler(
        graph=graph,
        plan=sampling_plan,
        schema=schema,
        batch_size=1,
    )
    samples = [sampler.sample(i) for i in range(4)]
    normalizer = _RecordingNormalizer(accepted_kwargs=set())

    # Tight padding forces sub-batch splitting when merging 2 samples at a time.
    tight_padding = _make_padding({"n": 4}, {"e": 3})
    batches = list(
        inference.generate_normalized_batches_from_samples(
            iter(samples),
            schema=schema,
            padding=tight_padding,
            normalizer=normalizer,  # pyrefly: ignore[bad-argument-type]
            target_nodeset="n",
            batch_size=2,
            verbose=0,
        )
    )
    self.assertLen(batches, 4)
    total_seeds = sum(b.merged_seed_node_idxs.shape[0] for b in batches)
    self.assertEqual(total_seeds, 4)

    # Comfortable padding merges 2 samples per batch without splitting.
    wide_padding = _make_padding({"n": 16}, {"e": 16})
    wide_batches = list(
        inference.generate_normalized_batches_from_samples(
            iter(samples),
            schema=schema,
            padding=wide_padding,
            normalizer=normalizer,  # pyrefly: ignore[bad-argument-type]
            target_nodeset="n",
            batch_size=2,
            verbose=0,
        )
    )
    self.assertLen(wide_batches, 2)
    for b in wide_batches:
      self.assertEqual(b.merged_seed_node_idxs.shape, (2,))
      self.assertIsNone(b.batch_seed_node_idxs)

  def test_merged_graph_to_normalized_batch(self):
    graph, schema = _make_simple_graph_and_schema(with_timestamp=False)
    padding = _make_padding({"n": 8}, {"e": 8})
    merged_graph, merge_offsets = merge_lib.GraphMerger(
        schema=schema,
        padding=padding,
    )([graph])
    normalizer = _RecordingNormalizer(accepted_kwargs=set())
    batch = inference.merged_graph_to_normalized_batch(
        merged_graph,
        merge_offsets,
        schema=schema,
        normalizer=normalizer,  # pyrefly: ignore[bad-argument-type]
        target_nodeset="n",
        batch_seed_node_idxs=np.array([0], dtype=np.int32),
    )
    assert batch.batch_seed_node_idxs is not None
    np.testing.assert_array_equal(batch.batch_seed_node_idxs, np.array([0]))
    np.testing.assert_array_equal(batch.merged_seed_node_idxs, np.array([0]))
    self.assertEqual(batch.jax_batch[0].node_sets["n"].num_nodes, 8)


if __name__ == "__main__":
  absltest.main()

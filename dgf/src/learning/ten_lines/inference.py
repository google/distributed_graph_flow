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

"""Shared batched subgraph sampling and normalization helpers for inference."""

from collections.abc import Iterator
import dataclasses

from dgf.src.data import in_memory_graph
from dgf.src.data import jax_in_memory_graph
from dgf.src.data import padding as padding_data_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import jax as jax_lib
from dgf.src.learning.ten_lines import common
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.transform import merge as merge_lib
from dgf.src.transform import normalize as normalize_lib
from dgf.src.util import temporal as temporal_util
from dgf.src.util import util
import jax
import jax.numpy as jnp
import numpy as np
import tqdm


@dataclasses.dataclass(frozen=True, kw_only=True)
class NormalizedBatch:
  """A normalized merged (sub-)batch ready for core model evaluation.

  Attributes:
    normalized_merged_graph: The merged and normalized `InMemoryGraph` in numpy
      format.
    merged_seed_node_idxs: The indices of the seed nodes inside
      `normalized_merged_graph`.
    jax_batch: Tuple `(jax_normalized_merged_graph, jax_merged_seed_node_idxs)`
      ready to pass to `apply_core_model`.
    batch_seed_node_idxs: Indices of the seed nodes in the input graph, or
      `None` when normalizing pre-sampled subgraphs where input-graph indices
      are unknown.
  """

  normalized_merged_graph: in_memory_graph.InMemoryGraph
  merged_seed_node_idxs: np.ndarray
  jax_batch: tuple[jax_in_memory_graph.JaxInMemoryGraph, jax.Array]
  batch_seed_node_idxs: np.ndarray | None = None


def resolve_normalizer_timestamp_feature(
    normalizer: normalize_lib.GraphNormalizer,
    schema: schema_lib.GraphSchema,
    target_nodeset: str,
) -> str | None:
  """Returns the creation-time feature name if `normalizer` requires `seed_timestamps`."""
  if "seed_timestamps" not in normalizer.accepted_kwargs:
    # Normalizer does not use seed timestamps, nothing to resolve.
    return None
  timestamp_feature = temporal_util.creation_time_feature_name(
      schema.node_sets[target_nodeset].features
  )
  if timestamp_feature is None:
    raise ValueError(
        f"The target nodeset '{target_nodeset}' must have a creation time"
        " feature when the normalizer requires seed_timestamps."
    )
  return timestamp_feature


def normalize_merged_graph(
    merged_graph: in_memory_graph.InMemoryGraph,
    merge_offsets: dict[str, np.ndarray],
    *,
    schema: schema_lib.GraphSchema,
    normalizer: normalize_lib.GraphNormalizer,
    target_nodeset: str,
    timestamp_feature: str | None = None,
) -> in_memory_graph.InMemoryGraph:
  """Normalizes a merged batch sample, expanding seed timestamps if required."""
  normalizer_kwargs = {}
  if "seed_timestamps" in normalizer.accepted_kwargs:
    if timestamp_feature is None:
      timestamp_feature = resolve_normalizer_timestamp_feature(
          normalizer=normalizer,
          schema=schema,
          target_nodeset=target_nodeset,
      )
    assert timestamp_feature is not None
    # sentinel_offset=True produces N+1 offsets; omit trailing sentinel offset.
    seed_node_idxs = merge_offsets[target_nodeset][:-1]
    seed_timestamps = np.asarray(
        merged_graph.node_sets[target_nodeset].features[timestamp_feature][
            seed_node_idxs
        ]
    ).reshape(-1)
    normalizer_kwargs["seed_timestamps"] = (
        temporal_util.expand_batch_seed_timestamps(
            sample=merged_graph,
            merge_offsets=merge_offsets,
            schema=schema,
            seed_timestamps=seed_timestamps,
        )
    )
  return normalizer.normalize_numpy(merged_graph, **normalizer_kwargs)


def merged_graph_to_normalized_batch(
    merged_graph: in_memory_graph.InMemoryGraph,
    merge_offsets: dict[str, np.ndarray],
    *,
    schema: schema_lib.GraphSchema,
    normalizer: normalize_lib.GraphNormalizer,
    target_nodeset: str,
    batch_seed_node_idxs: np.ndarray | None = None,
    timestamp_feature: str | None = None,
) -> NormalizedBatch:
  """Normalizes an already-merged (sub-)batch and converts it to a `NormalizedBatch`."""
  normalized_merged = normalize_merged_graph(
      merged_graph,
      merge_offsets,
      schema=schema,
      normalizer=normalizer,
      target_nodeset=target_nodeset,
      timestamp_feature=timestamp_feature,
  )
  return _to_normalized_batch(
      normalized_merged,
      merge_offsets,
      target_nodeset=target_nodeset,
      batch_seed_node_idxs=batch_seed_node_idxs,
  )


def _to_normalized_batch(
    normalized_merged: in_memory_graph.InMemoryGraph,
    merge_offsets: dict[str, np.ndarray],
    *,
    target_nodeset: str,
    batch_seed_node_idxs: np.ndarray | None,
) -> NormalizedBatch:
  """Converts a merged and normalized (sub-)batch to a `NormalizedBatch`."""
  # sentinel_offset=True produces N+1 offsets; omit trailing sentinel offset.
  merged_seed_node_idxs = merge_offsets[target_nodeset][:-1]
  normalized_merged_jax = jax_lib.graph_to_jax_graph(normalized_merged)
  jax_seed_node_idxs = jnp.asarray(merged_seed_node_idxs)
  return NormalizedBatch(
      normalized_merged_graph=normalized_merged,
      merged_seed_node_idxs=merged_seed_node_idxs,
      jax_batch=(normalized_merged_jax, jax_seed_node_idxs),
      batch_seed_node_idxs=batch_seed_node_idxs,
  )


def generate_normalized_batches(
    graph: in_memory_graph.InMemoryGraph,
    seed_node_idxs: common.SeedNodeIdxs,
    *,
    schema: schema_lib.GraphSchema,
    sampling_plan: sampling_config_lib.SamplingPlan,
    padding: padding_data_lib.Padding,
    normalizer: normalize_lib.GraphNormalizer,
    target_nodeset: str,
    batch_size: int,
    temporal_sampling: bool = False,
    sampler_schema: schema_lib.GraphSchema | None = None,
    verbose: int = 2,
    progress_bar_desc: str = "Inference",
    is_normalized: bool = False,
) -> Iterator[NormalizedBatch]:
  """Samples, merges, and normalizes sub-batches from an `InMemoryGraph`.

  Splits batches that overflow `padding` into smaller sub-batches so that no
  seed node is dropped during inference.

  Args:
    graph: The full input graph.
    seed_node_idxs: Seed node indices in `target_nodeset`.
    schema: The full graph schema (used for timestamp resolution and seed
      timestamp expansion).
    sampling_plan: The neighborhood sampling plan.
    padding: The padding specification for merged batches.
    normalizer: The fitted `GraphNormalizer`.
    target_nodeset: The target nodeset name.
    batch_size: Maximum number of seed nodes per merged batch.
    temporal_sampling: Whether temporal neighborhood sampling is enabled.
    sampler_schema: Optional schema passed to `create_sampler` (e.g. without the
      target label column during label-free inference). Defaults to `schema`.
    verbose: Verbosity level (`>= 2` shows a tqdm progress bar).
    progress_bar_desc: Description shown on the tqdm progress bar.
    is_normalized: If true, the features of `graph` are already normalized
      (and `sampler_schema` is the normalized schema).

  Yields:
    `NormalizedBatch` instances in seed-node order.
  """
  if sampler_schema is None:
    sampler_schema = schema

  sampler = in_memory_sampler_lib.create_sampler(
      graph=graph,
      plan=sampling_plan,
      schema=sampler_schema,
      batch_size=batch_size,
      padding=padding,
  )

  np_seed_node_idxs = np.asarray(seed_node_idxs)
  batch_seed_node_idxs_generator = util.batch_indices_generator(
      np_seed_node_idxs,
      batch_size=batch_size,
      drop_remainder=False,
      shuffle=False,
  )

  if verbose >= 2:
    batch_seed_node_idxs_generator = tqdm.tqdm(
        batch_seed_node_idxs_generator,
        desc=progress_bar_desc,
        total=util.num_batches(
            np_seed_node_idxs,
            batch_size=batch_size,
            drop_remainder=False,
        ),
    )

  timestamp_feature = None
  if temporal_sampling:
    timestamp_feature = temporal_util.creation_time_feature_name(
        schema.node_sets[target_nodeset].features
    )
    if timestamp_feature is None:
      raise ValueError(
          f"The target nodeset '{target_nodeset}' must have a creation time"
          " feature when temporal_sampling=True."
      )

  normalizer_timestamp_feature = resolve_normalizer_timestamp_feature(
      normalizer=normalizer,
      schema=schema,
      target_nodeset=target_nodeset,
  )

  for batch_seed_node_idxs in batch_seed_node_idxs_generator:
    seed_timestamps = None
    if temporal_sampling:
      assert timestamp_feature is not None
      timestamps = graph.node_sets[target_nodeset].features[timestamp_feature]
      seed_timestamps = np.asarray(timestamps[batch_seed_node_idxs]).reshape(-1)

    for (
        merged_graph,
        merge_offsets,
        sub_slice,
    ) in sampler.sample_merged_sub_batches(
        batch_seed_node_idxs,
        seed_timestamps=seed_timestamps,
        skip_overflow_padding_error=False,
        split_overflow_padding_error=True,
    ):
      if is_normalized:
        yield _to_normalized_batch(
            merged_graph,
            merge_offsets,
            target_nodeset=target_nodeset,
            batch_seed_node_idxs=batch_seed_node_idxs[sub_slice],
        )
      else:
        yield merged_graph_to_normalized_batch(
            merged_graph,
            merge_offsets,
            schema=schema,
            normalizer=normalizer,
            target_nodeset=target_nodeset,
            batch_seed_node_idxs=batch_seed_node_idxs[sub_slice],
            timestamp_feature=normalizer_timestamp_feature,
        )


def generate_normalized_batches_from_samples(
    graph_samples: Iterator[in_memory_graph.InMemoryGraph],
    *,
    schema: schema_lib.GraphSchema,
    padding: padding_data_lib.Padding,
    normalizer: normalize_lib.GraphNormalizer,
    target_nodeset: str,
    batch_size: int,
    num_steps: int | None = None,
    verbose: int = 0,
    progress_bar_desc: str = "Evaluation",
) -> Iterator[NormalizedBatch]:
  """Merges and normalizes batches from an iterator of pre-sampled subgraphs.

  Args:
    graph_samples: An iterator of individual subgraph samples.
    schema: The graph schema.
    padding: The padding specification for merged batches.
    normalizer: The fitted `GraphNormalizer`.
    target_nodeset: The target nodeset name.
    batch_size: Maximum number of samples per merged batch.
    num_steps: Optional maximum number of samples to consume from
      `graph_samples` (passed as `total` to tqdm when `verbose >= 2`).
    verbose: Verbosity level (`>= 2` shows a tqdm progress bar).
    progress_bar_desc: Description shown on the tqdm progress bar.

  Yields:
    `NormalizedBatch` instances in sample order.
  """
  graph_merger = merge_lib.GraphMerger(
      schema=schema,
      padding=padding,
  )
  iterator = graph_samples
  if verbose >= 2:
    iterator = tqdm.tqdm(iterator, desc=progress_bar_desc, total=num_steps)

  normalizer_timestamp_feature = resolve_normalizer_timestamp_feature(
      normalizer=normalizer,
      schema=schema,
      target_nodeset=target_nodeset,
  )

  def _merge_and_normalize(
      samples: list[in_memory_graph.InMemoryGraph],
  ) -> Iterator[NormalizedBatch]:
    for (
        merged_graph,
        merge_offsets,
        _,
    ) in graph_merger.merge_sub_batches(
        samples,
        skip_overflow_padding_error=False,
        split_overflow_padding_error=True,
    ):
      yield merged_graph_to_normalized_batch(
          merged_graph,
          merge_offsets,
          schema=schema,
          normalizer=normalizer,
          target_nodeset=target_nodeset,
          timestamp_feature=normalizer_timestamp_feature,
      )

  batch = []
  for sample in iterator:
    batch.append(sample)
    if len(batch) == batch_size:
      yield from _merge_and_normalize(batch)
      batch = []
  if batch:
    yield from _merge_and_normalize(batch)

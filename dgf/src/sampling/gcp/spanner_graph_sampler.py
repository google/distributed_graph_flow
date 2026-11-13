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

"""Samples subgraphs from a Cloud Spanner Graph."""

import base64
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
import warnings

from dgf.src.data import schema as schema_lib
from dgf.src.io.gcp import spanner_graph as spanner_graph_io_lib
from dgf.src.io.gcp import spanner_graph_metadata as spanner_graph_metadata_lib
from dgf.src.sampling import config as config_lib
from dgf.src.sampling import spanner_graph_sampler as spanner_graph_sampler_lib
from dgf.src.util import log
from google.cloud import spanner

_PARAM_TYPES: dict[str, Any] = {
    spanner_graph_sampler_lib.PARAM_TYPE_BOOL: spanner.param_types.BOOL,
    spanner_graph_sampler_lib.PARAM_TYPE_BYTES: spanner.param_types.BYTES,
    spanner_graph_sampler_lib.PARAM_TYPE_INT64: spanner.param_types.INT64,
    spanner_graph_sampler_lib.PARAM_TYPE_STRING: spanner.param_types.STRING,
}


def read_metadata(
    database: Any, graph: str
) -> spanner_graph_metadata_lib.SpannerGraphMetadata:
  """Reads the definition of a property graph of a Cloud Spanner database."""
  with database.snapshot() as snapshot:
    return spanner_graph_io_lib.get_metadata(snapshot, graph)


def _read_column_spanner_types(database: Any) -> dict[tuple[str, str], str]:
  """Reads the Spanner type of each (table_name, column_name)."""
  with database.snapshot() as snapshot:
    return spanner_graph_io_lib.read_column_spanner_types_gcp(snapshot)


def _decode_ids(row: Sequence[Any]) -> list[Any]:
  """Decodes the id columns of a row of the sampling query.

  The query returns all the ids as BYTES (STRING keys are cast to BYTES), and
  the Cloud client returns BYTES values in base64.
  """
  values = list(row)
  for column in spanner_graph_sampler_lib.ID_COLUMN_INDICES:
    if values[column] is not None:
      values[column] = base64.b64decode(values[column])
  return values


class SpannerGraphSamplerGcp(spanner_graph_sampler_lib.SpannerGraphSampler):
  """Samples subgraphs from a Cloud Spanner Graph, with one query per batch.

  Usage example:

  ```python
  sampler = dgf.sampling.create_spanner_graph_sampler_gcp(
      project="my-project",
      instance="my-instance",
      database="my-database",
      graph="my_graph",
      plan=dgf.sampling.SimpleSamplingConfig(
          seed_nodeset="paper", num_hops=2, hop_width=10, multi_visit=False
      ),
      schema=schema,
  )
  samples = sampler.sample([b"paper_1", b"paper_2"])
  ```
  """

  def __init__(
      self,
      database: Any,
      graph: str,
      plan: config_lib.SimpleSamplingConfig | config_lib.SamplingPlan,
      schema: schema_lib.GraphSchema,
      *,
      debug_sampling: bool,
      query_engine: spanner_graph_sampler_lib.QueryEngine | str,
  ):
    """Creates the sampler from a google.cloud.spanner database client."""
    super().__init__(
        metadata=read_metadata(database, graph),
        column_spanner_types=_read_column_spanner_types(database),
        graph=graph,
        plan=plan,
        schema=schema,
        debug_sampling=debug_sampling,
        query_engine=query_engine,
        sampler_name="Cloud Spanner Graph sampler",
    )
    self._database = database
    self._param_types = {
        name: spanner.param_types.Array(_PARAM_TYPES[param_type])
        for name, param_type in self._seed_parameter_types.items()
    }
    if not debug_sampling:
      self._param_types[spanner_graph_sampler_lib.SAMPLING_SEED_PARAMETER] = (
          spanner.param_types.BYTES
      )

  def _execute(
      self,
      seed_parameters: Mapping[str, list[Any]],
      sampling_seed: bytes | None,
  ) -> Iterable[Sequence[Any]]:
    # The Cloud client expects the BYTES parameters in base64.
    params: dict[str, Any] = {}
    for name, values in seed_parameters.items():
      if (
          self._seed_parameter_types[name]
          == spanner_graph_sampler_lib.PARAM_TYPE_BYTES
      ):
        values = [base64.b64encode(value) for value in values]
      params[name] = values
    if sampling_seed is not None:
      params[spanner_graph_sampler_lib.SAMPLING_SEED_PARAMETER] = (
          base64.b64encode(sampling_seed)
      )
    with self._database.snapshot() as snapshot:
      rows = snapshot.execute_sql(
          self.query, params=params, param_types=self._param_types
      )
      return [_decode_ids(row) for row in rows]


def create_spanner_graph_sampler_gcp(
    project: str,
    instance: str,
    database: str,
    graph: str,
    plan: config_lib.SimpleSamplingConfig | config_lib.SamplingPlan,
    schema: schema_lib.GraphSchema,
    *,
    debug_sampling: bool = False,
    query_engine: spanner_graph_sampler_lib.QueryEngine | str = "AUTO",
) -> SpannerGraphSamplerGcp:
  """Creates a sampler of subgraphs from a Cloud Spanner Graph.

  Each call to sample fetches the subgraphs of a batch of seed nodes, with
  their features, with a single Spanner query.

  The DGF schema defines what is sampled: Nodeset and edgeset names are the
  labels of the Spanner property graph (or the names of its element tables),
  and feature names are its properties. The property graph definition is used
  to locate the tables and columns.

  Limitations:

  - The primary key feature of a nodeset is a BYTES feature holding the key of
    its node table. With a composite key, it is a feature that is not a
    property (e.g. "#id") holding the key columns joined with "/".
  - Features are scalar or 1D properties defined by a plain column (not by an
    expression).
  - Parallel edges (same source and target) of an edgeset are merged into a
    single edge.
  - Duplicate seeds in a call to sample get the same subgraph.
  - With query_engine "GQL", a label cannot be shared by several node tables
    or by several edge tables.

  Recommendations for fast sampling:

  - Key the edge tables by (source key, target key) and interleave them in
    their source node table (INTERLEAVE IN PARENT), so that the edges of a node
    are stored next to it.
  - If the sampling plan traverses edges backward (e.g. reverse=True), create
    an index on the edge table by (target key, source key), with the edge
    features in STORING.
  - Sampling the neighbors of a node reads all its edges in the edgeset: Avoid
    sampling plans that traverse nodes with many (e.g. thousands of) edges.

  Usage example:

  ```python
  sampler = dgf.sampling.create_spanner_graph_sampler_gcp(
      project="my-project",
      instance="my-instance",
      database="my-database",
      graph="my_graph",
      plan=dgf.sampling.SimpleSamplingConfig(
          seed_nodeset="paper", num_hops=2, hop_width=10, multi_visit=False
      ),
      schema=schema,
  )
  samples = sampler.sample([b"paper_1", b"paper_2"])
  ```

  Args:
    project: GCP project of the Spanner instance.
    instance: Spanner instance.
    database: Spanner database.
    graph: Name of the Spanner property graph.
    plan: The sampling plan. Must have multi_visit=False.
    schema: The DGF schema of the graph.
    debug_sampling: If true, the sampled neighbors are the ones with the
      smallest ids (deterministic).
    query_engine: Implementation of the sampling query. "SQL" reads the tables
      of the property graph directly. "GQL" matches the property graph. "AUTO"
      selects "SQL".

  Returns:
    The sampler.
  """
  return SpannerGraphSamplerGcp(
      database=spanner.Client(project=project)
      .instance(instance)
      .database(database),
      graph=graph,
      plan=plan,
      schema=schema,
      debug_sampling=debug_sampling,
      query_engine=query_engine,
  )


def create_graph_spanner_sampler(
    project: str,
    instance: str,
    database: str,
    graph: str,
    plan: config_lib.SimpleSamplingConfig | config_lib.SamplingPlan,
    schema: schema_lib.GraphSchema,
    debug_sampling: bool = False,
) -> SpannerGraphSamplerGcp:
  """Creates a Cloud Spanner Graph sampler (deprecated).

  Deprecated: Use dgf.sampling.create_spanner_graph_sampler_gcp instead.

  Args:
    project: GCP project of the Spanner instance.
    instance: Spanner instance.
    database: Spanner database.
    graph: Name of the Spanner property graph.
    plan: The sampling plan. Must have multi_visit=False.
    schema: The DGF schema of the graph.
    debug_sampling: If true, the sampled neighbors are the ones with the
      smallest ids (deterministic).

  Returns:
    The sampler.
  """
  message = (
      "create_graph_spanner_sampler is deprecated; use"
      " create_spanner_graph_sampler_gcp instead."
  )
  warnings.warn(message, DeprecationWarning, stacklevel=2)
  log.warning("%s", message)
  return create_spanner_graph_sampler_gcp(
      project=project,
      instance=instance,
      database=database,
      graph=graph,
      plan=plan,
      schema=schema,
      debug_sampling=debug_sampling,
  )

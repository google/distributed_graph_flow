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

"""Library for working with Google Cloud Spanner Graphs via Apache Beam."""

import json
from typing import Any, Dict, Iterator, List, NamedTuple, Tuple

import apache_beam as beam
import apache_beam.io.gcp.spanner as beam_spanner_io
from dgf.src.data import distributed_graph as distributed_graph_lib
from dgf.src.io.gcp import common as gcp_common_lib
from dgf.src.io.gcp import common_beam as gcp_common_beam_lib
from dgf.src.io.gcp import spanner_graph
from dgf.src.io.gcp import spanner_graph_metadata as spanner_graph_metadata_lib
from dgf.src.util import log
from google.cloud import spanner as gcp_spanner


class SpannerGraphNodeRow(NamedTuple):
  """NamedTuple representing the structure of the Spanner graph node row for Beam reads."""

  id: str
  graph_element: str


class SpannerGraphEdgeRow(NamedTuple):
  """NamedTuple representing the structure of the Spanner graph edge row for Beam reads."""

  id: str
  source_id: str
  target_id: str
  graph_element: str


def _generate_read_partitions(
    project: str,
    instance: str,
    database: str,
    query: str,
) -> Tuple[Iterator[dict[str, Any]] | None, bool]:
  """Returns a list of Spanner graph read partitions."""
  spanner_client = gcp_spanner.Client(project=project)
  database = spanner_client.instance(instance).database(database)
  snapshot = database.batch_snapshot()
  try:
    partitions = snapshot.generate_query_batches(sql=query)
    is_root_partitionable = any(partitions)
    print(f"Query is root partitionable:\n {query}")
  except Exception as e:
    raise ValueError(
        "Edge table with non PK-FK aligned source and destination node tables"
        " are not supported."
    ) from e

  return partitions, is_root_partitionable


def check_metadata(
    project: str,
    instance: str,
    database: str,
    graph: str,
    spanner_graph_metadata: spanner_graph_metadata_lib.SpannerGraphMetadata,
) -> None:
  """Raises an error if the graph is not PK-FK aligned."""
  for edge_table in spanner_graph_metadata.edge_tables:
    spanner_graph_query_string = spanner_graph.graph_data_read_query(
        graph,
        gcp_common_lib.GRAPH_ELEMENT_TYPE_EDGE,
        edge_table,
    )
    _generate_read_partitions(
        project,
        instance,
        database,
        spanner_graph_query_string,
    )


def distributed_read_beam(
    project: str,
    instance: str,
    database: str,
    graph: str,
    p: beam.pvalue.PBegin,
    combine_as_json: bool = False,
    **kwargs,
) -> distributed_graph_lib.Graph:
  """Read Spanner Graph via Beam and return a distributed GraphFlow graph.

  Args:
    project: The GCP project ID of the Spanner Graph.
    instance: The Spanner instance ID of the Spanner Graph.
    database: The Spanner database ID of the Spanner Graph.
    graph: The ID of the Spanner Graph.
    p: The Beam pipeline.
    combine_as_json: Whether to combine the features as JSON.
    **kwargs: Additional arguments to pass to the ReadFromSpanner transform.

  Returns:
    A distributed GraphFlow graph.
  """
  spanner_client = gcp_spanner.Client(project=project)
  db = spanner_client.instance(instance).database(database)
  spanner_graph_metadata = spanner_graph.get_metadata(db, graph)
  graph_schema = spanner_graph.graph_schema(
      spanner_graph_metadata, combine_as_json
  )
  log.info("Spanner graph metadata: %s", spanner_graph_metadata.to_json())  # pyrefly: ignore[missing-attribute]
  log.info("Graph schema: %s", graph_schema)

  node_sets = {}
  for node_table in spanner_graph_metadata.node_tables:
    spanner_graph_query_string = spanner_graph.graph_data_read_query(
        graph,
        gcp_common_lib.GRAPH_ELEMENT_TYPE_NODE,
        node_table,
    )
    _, is_root_partitionable = _generate_read_partitions(
        project,
        instance,
        database,
        spanner_graph_query_string,
    )

    node_pcollection = (
        p
        | f"SpannerGraphRead_{node_table.name}_{node_table.kind}"
        >> beam_spanner_io.ReadFromSpanner(
            project=project,
            instance=instance,
            database=database,
            sql=spanner_graph_query_string,
            row_type=SpannerGraphNodeRow,
            batching=is_root_partitionable,
            **kwargs,
        )
        | f"Dict from DGFNodes_{node_table.name}_{node_table.kind}"
        >> beam.Map(
            lambda named_tuple: {
                gcp_common_lib.GRAPH_ELEMENT_ID_KEY: named_tuple.id,
                gcp_common_lib.GRAPH_ELEMENT_JSON_KEY: json.loads(
                    named_tuple.graph_element
                ),
            }
        )
        | f"DGFNodes_{node_table.name}_{node_table.kind}"
        >> beam.Map(
            gcp_common_beam_lib.create_distributed_node_set,
            graph_element_name=node_table.name,
            graph_schema=graph_schema,
            combine_as_json=combine_as_json,
        )
    )
    node_sets[node_table.name] = node_pcollection

  edge_sets = {}
  for edge_table in spanner_graph_metadata.edge_tables:
    spanner_graph_query_string = spanner_graph.graph_data_read_query(
        graph,
        gcp_common_lib.GRAPH_ELEMENT_TYPE_EDGE,
        edge_table,
    )
    _, is_root_partitionable = _generate_read_partitions(
        project,
        instance,
        database,
        spanner_graph_query_string,
    )
    edge_pcollection = (
        p
        | f"SpannerGraphRead_{edge_table.name}_{edge_table.kind}"
        >> beam_spanner_io.ReadFromSpanner(
            project=project,
            instance=instance,
            database=database,
            sql=spanner_graph_query_string,
            row_type=SpannerGraphEdgeRow,
            batching=is_root_partitionable,
            **kwargs,
        )
        | f"Dict from DGFNodes_{edge_table.name}_{edge_table.kind}"
        >> beam.Map(
            lambda named_tuple: {
                gcp_common_lib.GRAPH_ELEMENT_ID_KEY: named_tuple.id,
                gcp_common_lib.GRAPH_ELEMENT_SOURCE_ID_KEY: (
                    named_tuple.source_id
                ),
                gcp_common_lib.GRAPH_ELEMENT_TARGET_ID_KEY: (
                    named_tuple.target_id
                ),
                gcp_common_lib.GRAPH_ELEMENT_JSON_KEY: json.loads(
                    named_tuple.graph_element
                ),
            }
        )
        | f"DGFEdges_{edge_table.name}_{edge_table.kind}"
        >> beam.Map(
            gcp_common_beam_lib.create_distributed_edge_set,
            graph_element_name=edge_table.name,
            graph_schema=graph_schema,
            combine_as_json=combine_as_json,
        )
    )
    edge_sets[edge_table.name] = edge_pcollection

  return distributed_graph_lib.Graph(
      schema=graph_schema,
      node_sets=node_sets,
      edge_sets=edge_sets,
  )

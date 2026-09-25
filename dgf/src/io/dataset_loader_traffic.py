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

"""IO to load a traffic speed forecasting dataset into an in-memory graph.

METR-LA and PEMS-BAY are the reference spatio-temporal forecasting benchmarks
introduced by DCRNN (Li et al., 2018, https://arxiv.org/abs/1707.01926). Both
combine a graph (the road network) with one speed time series per node.
"""

from __future__ import annotations

import dataclasses
import functools
import io
import os
import tempfile
from typing import Any
import urllib.request

from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import cache as cache_lib
from dgf.src.io import dataset_loader
from dgf.src.io.dataset_loader import Repo
from dgf.src.util import log
import dgf.src.util.filesystem as fs
import numpy as np
import pandas as pd


# The reference copies of the METR-LA and PEMS-BAY speed recordings are the HDF5
# frames of the Google Drive folders linked from the DCRNN repository
# (https://github.com/liyaguang/DCRNN). The URLs below point at the Zenodo
# record https://doi.org/10.5281/zenodo.5724362, which serves a csv export of
# those same frames.
METR_LA_SPEEDS_URL = (
    "https://zenodo.org/records/5724362/files/METR-LA.csv?download=1"
)
PEMS_BAY_SPEEDS_URL = (
    "https://zenodo.org/records/5724362/files/PEMS-BAY.csv?download=1"
)

# Road network distances and sensor coordinates, from the DCRNN repository.
DCRNN_SENSOR_GRAPH_URL = (
    "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph"
)

# Both datasets encode a missing speed recording with a zero.
TRAFFIC_MISSING_SPEED = 0.0


@dataclasses.dataclass(frozen=True)
class TrafficDatasetSpec:
  """Location and layout of the raw files of a traffic speed dataset.

  Attributes:
    speeds_url: URL of the csv with the speed recordings.
    distances_url: URL of the `from,to,cost` road network distances csv.
    distances_have_header: Whether the distances csv has a header row.
    locations_url: URL of the sensor coordinates csv.
    locations_have_header: Whether the locations csv has a header row.
  """

  speeds_url: str
  distances_url: str
  distances_have_header: bool
  locations_url: str
  locations_have_header: bool


TRAFFIC_DATASETS: dict[str, TrafficDatasetSpec] = {
    "metr_la": TrafficDatasetSpec(
        speeds_url=METR_LA_SPEEDS_URL,
        distances_url=f"{DCRNN_SENSOR_GRAPH_URL}/distances_la_2012.csv",
        distances_have_header=True,
        locations_url=f"{DCRNN_SENSOR_GRAPH_URL}/graph_sensor_locations.csv",
        locations_have_header=True,
    ),
    "pems_bay": TrafficDatasetSpec(
        speeds_url=PEMS_BAY_SPEEDS_URL,
        distances_url=f"{DCRNN_SENSOR_GRAPH_URL}/distances_bay_2017.csv",
        distances_have_header=False,
        locations_url=(
            f"{DCRNN_SENSOR_GRAPH_URL}/graph_sensor_locations_bay.csv"
        ),
        locations_have_header=False,
    ),
}


def traffic_dataset_spec(dataset: str) -> TrafficDatasetSpec:
  """Returns the spec of the traffic dataset called `dataset`."""
  if dataset not in TRAFFIC_DATASETS:
    raise ValueError(
        f"Unknown traffic dataset {dataset!r}. Available datasets:"
        f" {sorted(TRAFFIC_DATASETS)}."
    )
  return TRAFFIC_DATASETS[dataset]


# Forecasting horizons cached on CNS and their name suffix.
CACHED_FORECAST_HORIZONS_SECONDS: dict[int, str] = {
    900: "15m",
    1800: "30m",
    3600: "60m",
}


def traffic_cns_name(dataset: str, forecast_horizon_seconds: int) -> str:
  """Returns the name of the cached CNS graph for `dataset` and `forecast_horizon_seconds`.

  Args:
    dataset: Name of the traffic dataset ("metr_la" or "pems_bay").
    forecast_horizon_seconds: Forecasting horizon in seconds (must be 900, 1800,
      or 3600).

  Returns:
    The dataset name under the CNS repository (e.g. "metr_la_15m").

  Raises:
    ValueError: If `dataset` is unknown or `forecast_horizon_seconds` is not
      cached on CNS.
  """
  traffic_dataset_spec(dataset)  # Validates dataset name.
  if forecast_horizon_seconds not in CACHED_FORECAST_HORIZONS_SECONDS:
    raise ValueError(
        f"Forecast horizon {forecast_horizon_seconds}s is not cached on CNS for"
        f" dataset {dataset!r}. Cached horizons are:"
        f" {sorted(CACHED_FORECAST_HORIZONS_SECONDS)} (seconds, corresponding"
        f" to {[CACHED_FORECAST_HORIZONS_SECONDS[h] for h in sorted(CACHED_FORECAST_HORIZONS_SECONDS)]})."
        " To build a graph with an arbitrary horizon, pass"
        " repo=dgf.io.Repo.WEB."
    )
  suffix = CACHED_FORECAST_HORIZONS_SECONDS[forecast_horizon_seconds]
  return f"{dataset}_{suffix}"


def _download_bytes(url: str) -> bytes:
  """Returns the content served at `url`.

  The default `urllib` user agent is used on purpose: Zenodo answers 403 to
  requests that claim to come from a browser.

  Args:
    url: The URL to download.
  """
  with urllib.request.urlopen(url) as response:
    return response.read()


def _decode_sensor_id(sensor_id: Any) -> str:
  """Returns the sensor id as a string.

  Sensor ids are stored as bytes in METR-LA and as integers in PEMS-BAY.

  Args:
    sensor_id: The raw sensor id.
  """
  if isinstance(sensor_id, bytes):
    return sensor_id.decode("utf-8")
  return str(sensor_id)


def download_traffic_speeds(
    dataset: str, source: str | None = None
) -> pd.DataFrame:
  """Downloads and parses the speed recordings of a traffic dataset.

  Args:
    dataset: Name of the dataset, e.g. "metr_la" or "pems_bay".
    source: Optional URL or file path of the csv. If None, downloads from the
      public mirror of the dataset.

  Returns:
    A frame of speeds in miles per hour, indexed by unix timestamp in seconds,
    with one column per sensor named after the sensor id.
  """
  spec = traffic_dataset_spec(dataset)
  url_or_path = source or spec.speeds_url
  csv_source: Any = url_or_path
  if fs.is_url(url_or_path):
    log.info("Downloading the %s speeds from %s", dataset, url_or_path)
    csv_source = io.BytesIO(_download_bytes(url_or_path))

  # The first column holds the recording time, as a "%Y-%m-%d %H:%M:%S" string,
  # and every other column holds the speeds of one sensor.
  speeds = pd.read_csv(csv_source, index_col=0)
  timestamps = (
      pd.to_datetime(speeds.index)
      .to_numpy()
      .astype("datetime64[s]")
      .astype(np.int64)
  )
  return (
      speeds.set_axis(timestamps, axis="index")
      .set_axis(
          [_decode_sensor_id(sensor_id) for sensor_id in speeds.columns],
          axis="columns",
      )
      .astype(np.float32)
  )


def _read_traffic_csv(
    url_or_path: str, names: list[str], has_header: bool
) -> pd.DataFrame:
  """Reads a sensor graph csv, with or without a header row.

  Args:
    url_or_path: URL or file path of the csv.
    names: Names of the columns to return. When the csv has no header row, the
      columns are assumed to be in this order.
    has_header: Whether the csv has a header row.

  Returns:
    A frame with exactly the `names` columns.
  """
  source: Any = url_or_path
  if fs.is_url(url_or_path):
    log.info("Downloading %s", url_or_path)
    source = io.BytesIO(_download_bytes(url_or_path))

  if not has_header:
    return pd.read_csv(source, header=None, names=names)

  frame = pd.read_csv(source)
  missing_columns = [name for name in names if name not in frame.columns]
  if missing_columns:
    raise ValueError(
        f"The csv {url_or_path} is missing the columns {missing_columns}."
    )
  return frame[names]


def download_traffic_sensor_graph(
    dataset: str,
    distances_source: str | None = None,
    locations_source: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Downloads the road network distances and the coordinates of the sensors.

  Args:
    dataset: Name of the dataset, e.g. "metr_la" or "pems_bay".
    distances_source: Optional URL or file path of the distances csv.
    locations_source: Optional URL or file path of the sensor locations csv.

  Returns:
    A tuple of (distances, locations). `distances` has the columns `from`, `to`
    and `cost`, where the cost is the driving distance in meters between two
    sensors. `locations` has the columns `sensor_id`, `latitude` and
    `longitude`. Sensor ids are strings in both frames.
  """
  spec = traffic_dataset_spec(dataset)
  distances = _read_traffic_csv(
      distances_source or spec.distances_url,
      ["from", "to", "cost"],
      spec.distances_have_header,
  )
  locations = _read_traffic_csv(
      locations_source or spec.locations_url,
      ["sensor_id", "latitude", "longitude"],
      spec.locations_have_header,
  )
  distances = distances.assign(**{
      "from": [_decode_sensor_id(value) for value in distances["from"]],
      "to": [_decode_sensor_id(value) for value in distances["to"]],
      "cost": distances["cost"].astype(np.float32),
  })
  locations = locations.assign(
      sensor_id=[_decode_sensor_id(value) for value in locations["sensor_id"]],
      latitude=locations["latitude"].astype(np.float32),
      longitude=locations["longitude"].astype(np.float32),
  )
  return distances, locations


def build_sensor_adjacency(
    distances: pd.DataFrame,
    sensor_ids: list[str],
    threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Builds the sensor graph from the pairwise road network distances.

  The edges follow the construction of DCRNN (Li et al., 2018): the weight of
  the edge between two sensors is `exp(-(distance / std)**2)`, where `std` is
  the standard deviation of the observed distances, and weights below
  `threshold` are dropped to make the graph sparse. Edges are directed: `cost`
  is the driving distance from the `from` sensor to the `to` sensor. Self loops
  are excluded, since each query node is already connected to its own sensor.

  Args:
    distances: Frame with the `from`, `to` and `cost` columns.
    sensor_ids: Sensor ids, in the node order of the sensor node set.
    threshold: Weights strictly below this value are dropped.

  Returns:
    A tuple of (sources, targets, weights) arrays of equal length, where
    sources and targets index into `sensor_ids`.
  """
  num_sensors = len(sensor_ids)
  sensor_index = {
      sensor_id: index for index, sensor_id in enumerate(sensor_ids)
  }

  pairwise_distances = np.full(
      (num_sensors, num_sensors), np.inf, dtype=np.float32
  )
  known_pairs = distances["from"].isin(sensor_index) & distances["to"].isin(
      sensor_index
  )
  known_distances = distances[known_pairs]
  if known_distances.empty:
    raise ValueError(
        "None of the distances relate two sensors of the dataset. The"
        " distances and the speed recordings are likely inconsistent."
    )
  sources = known_distances["from"].map(sensor_index).to_numpy(dtype=np.int64)
  targets = known_distances["to"].map(sensor_index).to_numpy(dtype=np.int64)
  pairwise_distances[sources, targets] = known_distances["cost"].to_numpy(
      dtype=np.float32
  )

  finite_distances = pairwise_distances[np.isfinite(pairwise_distances)]
  distance_std = finite_distances.std()
  if distance_std == 0.0:
    raise ValueError(
        "All the road network distances are identical, the Gaussian kernel is"
        " undefined."
    )
  weights = np.where(
      np.isfinite(pairwise_distances),
      np.exp(-np.square(pairwise_distances / distance_std)),
      0.0,
  )
  weights[weights < threshold] = 0.0
  np.fill_diagonal(weights, 0.0)

  edge_sources, edge_targets = np.nonzero(weights)
  return (
      edge_sources.astype(np.int64),
      edge_targets.astype(np.int64),
      weights[edge_sources, edge_targets].astype(np.float32),
  )


def build_traffic_graph(
    speeds: pd.DataFrame,
    distances: pd.DataFrame,
    locations: pd.DataFrame,
    forecast_horizon_seconds: int = 900,
    query_step: int = 1,
    adjacency_threshold: float = 0.1,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Constructs a DGF InMemoryGraph and GraphSchema from traffic recordings.

  Graph structure:
    - One sensor node per loop detector, holding the full speed time series and
      the coordinates of the detector.
    - Directed sensor to sensor edges, weighted by road network proximity.
    - Query nodes, one per (sensor, query time) pair, with `creation_time` and
      the regression label `speed` (the speed of the sensor at `creation_time +
      horizon`). Each query node is connected to its own sensor, so a model has
      to walk the sensor graph to see the neighboring detectors. Query times
      without a recording exactly one horizon ahead are dropped, and so are the
      queries whose target speed is missing (missing recordings are encoded as
      zeros).

  Args:
    speeds: Speed recordings, as returned by `download_traffic_speeds`.
    distances: Road network distances, as returned by
      `download_traffic_sensor_graph`.
    locations: Sensor coordinates, as returned by
      `download_traffic_sensor_graph`.
    forecast_horizon_seconds: Horizon into the future to predict (default: 900s
      = 15 minutes). Must be a multiple of the sampling period.
    query_step: Stride, in sampling periods, between two query times (default:
      1, i.e. one query per sensor and per recording, as in DCRNN).
    adjacency_threshold: Sensor edges with a weight below this value are
      dropped.

  Returns:
    A tuple of (InMemoryGraph, GraphSchema).
  """
  sensor_ids = [_decode_sensor_id(sensor_id) for sensor_id in speeds.columns]
  num_sensors = len(sensor_ids)
  timestamps = speeds.index.to_numpy(dtype=np.int64)
  values = speeds.to_numpy(dtype=np.float32)

  if len(timestamps) < 2:
    raise ValueError(
        f"The speed recordings only contain {len(timestamps)} time steps."
    )
  sampling_periods = np.diff(timestamps)
  if np.any(sampling_periods <= 0):
    raise ValueError(
        "The speed recordings are not sorted by strictly increasing time."
    )
  observed_periods, period_counts = np.unique(
      sampling_periods, return_counts=True
  )
  sampling_period_seconds = int(observed_periods[np.argmax(period_counts)])
  if forecast_horizon_seconds % sampling_period_seconds != 0:
    raise ValueError(
        f"The forecast horizon {forecast_horizon_seconds}s is not a multiple of"
        f" the {sampling_period_seconds}s sampling period."
    )
  num_gaps = int(np.sum(sampling_periods != sampling_period_seconds))
  if num_gaps:
    # PEMS-BAY is recorded in local time, so the daylight saving time change
    # leaves a one hour hole in the recordings.
    log.info(
        "The speed recordings have %d gap(s) in their %ds sampling period.",
        num_gaps,
        sampling_period_seconds,
    )

  # Sensor nodes.
  location_index = {
      sensor_id: index
      for index, sensor_id in enumerate(locations["sensor_id"].to_numpy())
  }
  sensors_without_location = [
      sensor_id for sensor_id in sensor_ids if sensor_id not in location_index
  ]
  if sensors_without_location:
    raise ValueError(
        f"{len(sensors_without_location)} sensors of the speed recordings are"
        f" missing from the locations, e.g. {sensors_without_location[:5]}."
    )
  location_positions = np.array(
      [location_index[sensor_id] for sensor_id in sensor_ids], dtype=np.int64
  )

  sensor_times = np.empty(num_sensors, dtype=object)
  sensor_speeds = np.empty(num_sensors, dtype=object)
  for index in range(num_sensors):
    sensor_times[index] = timestamps
    sensor_speeds[index] = np.ascontiguousarray(values[:, index])

  sensor_node_features: dict[str, np.ndarray] = {
      "#id": np.array(
          [f"sensor_{sensor_id}".encode("utf-8") for sensor_id in sensor_ids],
          dtype=np.bytes_,
      ),
      "time": sensor_times,
      "speed": sensor_speeds,
      "latitude": (
          locations["latitude"].to_numpy(dtype=np.float32)[location_positions]
      ),
      "longitude": (
          locations["longitude"].to_numpy(dtype=np.float32)[location_positions]
      ),
  }
  sensor_schema_features: dict[str, schema_lib.FeatureSchema] = {
      "#id": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BYTES,
          semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
      ),
      "time": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.TIMESTAMP,
          is_timeseries=True,
          is_creation_time=True,
          group="speed_ts",
          shape=(None,),
      ),
      "speed": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
          is_timeseries=True,
          group="speed_ts",
          shape=(None,),
      ),
      "latitude": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
      ),
      "longitude": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
      ),
  }

  # Query nodes. Queries are generated time major, i.e. all the sensors of a
  # query time are consecutive, so that a chronological split never splits the
  # queries of a same time between two splits.
  #
  # The target of a query is resolved by timestamp rather than by index, so
  # that a query right before a gap in the recordings is dropped instead of
  # being paired with a target further ahead than the forecast horizon.
  index_by_timestamp = {
      int(timestamp): index for index, timestamp in enumerate(timestamps)
  }
  candidate_time_indices = np.arange(0, len(timestamps), query_step)
  target_time_indices = np.array(
      [
          index_by_timestamp.get(
              int(timestamps[index]) + forecast_horizon_seconds, -1
          )
          for index in candidate_time_indices
      ],
      dtype=np.int64,
  )
  has_target = target_time_indices >= 0
  num_times_without_target = int(np.sum(~has_target))
  if num_times_without_target:
    log.info(
        "Dropping %d of the %d query times which have no recording %ds ahead.",
        num_times_without_target,
        len(candidate_time_indices),
        forecast_horizon_seconds,
    )
  query_time_indices = candidate_time_indices[has_target]
  target_time_indices = target_time_indices[has_target]
  num_query_times = len(query_time_indices)
  if num_query_times == 0:
    raise ValueError(
        f"No query time is left with a horizon of {forecast_horizon_seconds}s"
        f" and a query step of {query_step}."
    )

  query_times = np.repeat(timestamps[query_time_indices], num_sensors)
  query_sensors = np.tile(
      np.arange(num_sensors, dtype=np.int64), num_query_times
  )
  target_speeds = values[target_time_indices].reshape(-1)

  # Chronological 70/10/20 split, matching the canonical METR-LA and PEMS-BAY
  # protocol introduced by DCRNN (https://arxiv.org/abs/1707.01926), so that
  # test metrics are comparable to published baselines.
  num_train_times = int(num_query_times * 0.7)
  num_valid_times = int(num_query_times * 0.1)
  time_splits = np.full(num_query_times, "n/a", dtype="S5")
  time_splits[:num_train_times] = b"train"
  time_splits[num_train_times : num_train_times + num_valid_times] = b"valid"
  time_splits[num_train_times + num_valid_times :] = b"test"
  split_labels = np.repeat(time_splits, num_sensors)

  # A missing recording is indistinguishable from a genuine zero speed, so a
  # query pointing at one cannot be labelled. This removes around 10% of the
  # METR-LA queries, and none of the PEMS-BAY ones.
  observed = target_speeds != TRAFFIC_MISSING_SPEED
  num_missing_targets = int(np.sum(~observed))
  if num_missing_targets:
    log.info(
        "Dropping %d of the %d queries (%.1f%%) whose target speed is missing.",
        num_missing_targets,
        len(target_speeds),
        100.0 * num_missing_targets / len(target_speeds),
    )
  query_times = query_times[observed]
  query_sensors = query_sensors[observed]
  target_speeds = target_speeds[observed]
  split_labels = split_labels[observed]

  num_queries = len(query_times)
  if num_queries == 0:
    raise ValueError("All the target speeds are missing.")

  query_node_features: dict[str, np.ndarray] = {
      "#id": dataset_loader.generate_ids("query_", num_queries),
      "creation_time": query_times,
      "speed": target_speeds,
      "#split": split_labels,
  }
  query_schema_features: dict[str, schema_lib.FeatureSchema] = {
      "#id": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BYTES,
          semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
      ),
      "creation_time": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.TIMESTAMP,
          is_creation_time=True,
      ),
      "speed": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
      ),
      "#split": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BYTES,
          semantic=schema_lib.FeatureSemantic.CATEGORICAL,
      ),
  }

  # Edges.
  query_indices = np.arange(num_queries, dtype=np.int64)
  sensor_sources, sensor_targets, sensor_weights = build_sensor_adjacency(
      distances, sensor_ids, threshold=adjacency_threshold
  )

  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "queries": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=num_queries,
              features=query_node_features,
          ),
          "sensors": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=num_sensors,
              features=sensor_node_features,
          ),
      },
      edge_sets={
          "query_to_sensor": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.stack([query_indices, query_sensors], axis=0),
              features={},
          ),
          "sensor_to_query": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.stack([query_sensors, query_indices], axis=0),
              features={},
          ),
          "sensor_to_sensor": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.stack([sensor_sources, sensor_targets], axis=0),
              features={"weight": sensor_weights},
          ),
      },
  )

  schema = schema_lib.GraphSchema(
      node_sets={
          "queries": schema_lib.NodeSchema(features=query_schema_features),
          "sensors": schema_lib.NodeSchema(features=sensor_schema_features),
      },
      edge_sets={
          "query_to_sensor": schema_lib.EdgeSchema(
              source="queries",
              target="sensors",
              features={},
          ),
          "sensor_to_query": schema_lib.EdgeSchema(
              source="sensors",
              target="queries",
              features={},
          ),
          "sensor_to_sensor": schema_lib.EdgeSchema(
              source="sensors",
              target="sensors",
              features={
                  "weight": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  )
              },
          ),
      },
  )

  return graph, schema


def fetch_traffic_graph(
    dataset: str,
    forecast_horizon_seconds: int = 900,
    cache_dir: str | None = "AUTO",
    verbose: bool = True,
    query_step: int = 1,
    adjacency_threshold: float = 0.1,
    repo: Repo | str = Repo.AUTO,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Gets the METR-LA and PEMS-BAY traffic speed forecasting datasets.

  Both supported datasets record the speed of highway loop detectors every five
  minutes: METR-LA covers 207 detectors of the Los Angeles county highways over
  four months of 2012, and PEMS-BAY covers 325 detectors of the Bay Area over
  six months of 2017. They are the reference benchmarks for spatio-temporal
  forecasting, and combine a graph (the road network) with one time series per
  node. See Li et al., 2018 (https://arxiv.org/abs/1707.01926).

  Usage example:

  ```python
  graph, schema = dgf.io.fetch_traffic_graph("metr_la")
  dgf.analyse.print_schema(schema)
  ```

  See `build_traffic_graph` for the structure of the returned graph.

  Args:
    dataset: Name of the dataset, e.g. "metr_la" or "pems_bay".
    forecast_horizon_seconds: Forecasting horizon in seconds (default: 900 for
      15 minutes ahead). For CNS, must be one of 900 (15m), 1800 (30m), or 3600
      (60m).
    cache_dir: Optional. Directory to cache the graph in order to avoid
      re-downloading/re-parsing it each time. If "AUTO", uses the OS default
      temporary directory. If None, does not cache the graph.
    verbose: Optional. Whether to print cache and download progress.
    query_step: Stride, in sampling periods, between two query times (default:
      1, i.e. one query per sensor and per recording, as in DCRNN).
    adjacency_threshold: Sensor edges with a weight below this value are
      dropped.
    repo: Define the source of the data (Repo.AUTO, Repo.CNS, Repo.WEB).

  Returns:
    An InMemoryGraph instance and its GraphSchema.
  """
  spec = traffic_dataset_spec(dataset)

  if cache_dir == "AUTO":
    cache_dir = os.path.join(tempfile.gettempdir(), "gf_fetch_traffic")

  if cache_dir is not None:
    fs.makedirs(cache_dir)
    cache_key = (
        f"{dataset}_h{forecast_horizon_seconds}_qs{query_step}_"
        f"at{adjacency_threshold}.cache"
    )
    cache_graph_path = os.path.join(cache_dir, cache_key)
    if verbose:
      log.info("Caching the %s graph at %s", dataset, cache_graph_path)
  else:
    cache_graph_path = None

  if isinstance(repo, str):
    repo = Repo(repo)

  # Select the right repo.
  if repo == Repo.AUTO:
    repo = Repo.WEB

  if repo == Repo.CNS:
    cns_name = traffic_cns_name(dataset, forecast_horizon_seconds)
    loader = functools.partial(dataset_loader.load_from_cns, name=cns_name)
  elif repo == Repo.WEB:

    def loader():
      if verbose:
        log.info(
            "Loading the %s data from %s",
            dataset,
            spec.speeds_url,
        )
      speeds = download_traffic_speeds(dataset)
      distances, locations = download_traffic_sensor_graph(dataset)
      return build_traffic_graph(
          speeds=speeds,
          distances=distances,
          locations=locations,
          forecast_horizon_seconds=forecast_horizon_seconds,
          query_step=query_step,
          adjacency_threshold=adjacency_threshold,
      )

  else:
    raise ValueError(f"Unsupported repo for the traffic datasets: {repo}")

  if cache_graph_path is None:
    return loader()
  else:
    return cache_lib.cache(cache_graph_path, loader)

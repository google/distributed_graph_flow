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

"""Utility to generate synthetic temporal alert regression graphs.

This module simulates a heterogeneous, bipartite temporal graph between hardware
monitoring devices and alert events, useful for benchmarking and testing
temporal
Graph Neural Network (GNN) regression tasks.

Graph Topology & Schema:
  * Node Set `hardware`:
    * `#id`: Primary ID (`hw_0`, `hw_1`, ...).
    * `time`: 1D array of irregularly sampled timestamps.
    * `signal`: 1D array of noisy sinusoidal telemetry values sampled at `time`.
  * Node Set `alerts`:
    * `#id`: Primary ID (`alert_0`, `alert_1`, ...).
    * `creation_time`: Timestamp when the alert was triggered.
    * `signal_regression`: Target continuous regression label.
  * Edge Set `hardware_to_alert`:
    * Directed edges linking hardware nodes to alert nodes. The number of
      connected hardware nodes per alert is sampled from a geometric (power
      decay) distribution controlled by `neighbor_decay`.

Target Formulation (`signal_regression`):
  For each alert triggered at t_create with lookback window W, the target label
  is computed by linearly interpolating discrete hardware signals over the
  window [t_create - W, t_create], summing the integrals across connected
  neighbors, and normalizing by W.
"""

import dataclasses
from typing import Optional
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import graph_in_memory as gf_graph_in_memory_lib
from dgf.src.validate import in_memory_graph as in_memory_graph_validate_lib
import numpy as np


@dataclasses.dataclass
class TemporalAlertGraphConfig:
  """Configuration for generating synthetic temporal alert graphs.

  Attributes:
    num_hardware: Number of hardware nodes to generate.
    num_alerts: Number of alert nodes to generate.
    start_time: Unix epoch start time in seconds.
    duration: Total simulation time interval in seconds.
    sample_interval_mean: Mean interval between hardware timestamps in seconds.
    sample_interval_jitter: Max jitter added to timestamp intervals in seconds.
    window_duration: Lookback window in seconds for the signal integral.
    neighbor_decay: Probability of adding an additional neighbor in each step
      (controls the geometric / power decay of alert in-degrees).
    seed: Random seed for reproducible graph generation.
  """

  num_hardware: int = 10
  num_alerts: int = 10
  start_time: int = 1700000000
  duration: int = 86400
  sample_interval_mean: int = 1200
  sample_interval_jitter: int = 300
  window_duration: int = 3600
  neighbor_decay: float = 0.8
  seed: int = 42


def generate_signal_regression_schema() -> schema_lib.GraphSchema:
  """Generates schema for a signal regression temporal heterogeneous graph.

  Returns:
    A GraphSchema for the signal regression task.
  """
  return schema_lib.GraphSchema(
      node_sets={
          "hardware": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "time": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                      shape=(None,),
                      is_timeseries=True,
                      is_creation_time=True,
                      group="time",
                  ),
                  "signal": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                      shape=(None,),
                      is_timeseries=True,
                      group="time",
                  ),
              }
          ),
          "alerts": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "creation_time": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.TIMESTAMP,
                      is_creation_time=True,
                  ),
                  "signal_regression": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
              }
          ),
      },
      edge_sets={
          "hardware_to_alert": schema_lib.EdgeSchema(
              source="hardware", target="alerts"
          ),
      },
  )


def generate_signal_regression_in_memory_graph(
    config: TemporalAlertGraphConfig = TemporalAlertGraphConfig(),
    *,
    schema: Optional[schema_lib.GraphSchema] = None,
) -> in_memory_graph_lib.InMemoryGraph:
  """Generates an in-memory heterogeneous temporal signal regression graph.

  Args:
    config: Configuration parameters for graph generation.
    schema: Optional schema to validate against; if None, generates one.

  Returns:
    An InMemoryGraph object.
  """
  if config.window_duration > config.duration:
    raise ValueError(
        f"'window_duration' ({config.window_duration}) must be less than or"
        f" equal to 'duration' ({config.duration})."
    )
  if not 0.0 <= config.neighbor_decay < 1.0:
    raise ValueError(
        f"'neighbor_decay' ({config.neighbor_decay}) must be in [0.0, 1.0)."
    )
  rng = np.random.RandomState(config.seed)

  hw_ids = [f"hw_{i}".encode("utf-8") for i in range(config.num_hardware)]
  hw_times_list = []
  hw_signals_list = []

  for _ in range(config.num_hardware):
    t = config.start_time
    times = []
    signals = []
    while t <= config.start_time + config.duration:
      times.append(int(t))
      phase = 2.0 * np.pi * (t - config.start_time) / float(config.duration)
      s = float(np.sin(phase) + rng.normal(0, 0.1))
      signals.append(s)
      step = config.sample_interval_mean + rng.randint(
          -config.sample_interval_jitter, config.sample_interval_jitter + 1
      )
      t += max(1, step)
    hw_times_list.append(np.array(times, dtype=np.int64))
    hw_signals_list.append(np.array(signals, dtype=np.float32))

  hw_id_array = np.array(hw_ids, dtype=np.bytes_)
  hw_time_array = np.array(hw_times_list, dtype=np.object_)
  hw_signal_array = np.array(hw_signals_list, dtype=np.object_)

  alert_ids = [f"alert_{i}".encode("utf-8") for i in range(config.num_alerts)]
  alert_creation_times = config.start_time + rng.randint(
      config.window_duration, config.duration + 1, size=config.num_alerts
  )

  edge_sources = []
  edge_targets = []
  alert_regression = []

  # Sample the number of connected hardware neighbors `k` per alert from a
  # geometric (power decay) distribution controlled by `neighbor_decay`, where
  # each step adds another neighbor with probability `neighbor_decay` and stops
  # with probability `1 - neighbor_decay`.
  neighbor_counts = np.minimum(
      config.num_hardware,
      rng.geometric(p=1.0 - config.neighbor_decay, size=config.num_alerts),
  )

  for alert_idx in range(config.num_alerts):
    t_create = int(alert_creation_times[alert_idx])
    t_start = t_create - config.window_duration
    k = int(neighbor_counts[alert_idx])
    chosen_hw_indices = rng.choice(
        config.num_hardware, size=k, replace=False
    )
    total_integral = 0.0
    for hw_idx in chosen_hw_indices:
      edge_sources.append(hw_idx)
      edge_targets.append(alert_idx)

      hw_t = hw_times_list[hw_idx]
      hw_s = hw_signals_list[hw_idx]

      if len(hw_t) == 0:
        continue

      def eval_signal(t_val: float) -> float:
        if t_val <= hw_t[0]:
          return float(hw_s[0])
        if t_val >= hw_t[-1]:
          return float(hw_s[-1])
        idx = int(np.searchsorted(hw_t, t_val))
        if hw_t[idx] == t_val:
          return float(hw_s[idx])
        t0, t1 = hw_t[idx - 1], hw_t[idx]
        s0, s1 = hw_s[idx - 1], hw_s[idx]
        frac = (t_val - t0) / (t1 - t0)
        return float(s0 + frac * (s1 - s0))

      mask = (hw_t >= t_start) & (hw_t <= t_create)
      window_t = [float(t_start)] + list(hw_t[mask]) + [float(t_create)]
      window_t = sorted(list(set(window_t)))

      integral_h = 0.0
      for j in range(len(window_t) - 1):
        ta = window_t[j]
        tb = window_t[j + 1]
        sa = eval_signal(ta)
        sb = eval_signal(tb)
        integral_h += 0.5 * (sa + sb) * (tb - ta)

      total_integral += integral_h

    # Compute the target regression label (`signal_regression`).
    # This label represents the sum of the time-averaged telemetry
    # signals across all connected hardware neighbors during the lookback
    # window [t_create - W, t_create]. To predict this target accurately, a
    # temporal GNN must learn both spatial aggregation over neighbor edges and
    # temporal interpolation over irregularly sampled historical signals.
    val = total_integral / float(config.window_duration)
    alert_regression.append(val)

  alert_id_array = np.array(alert_ids, dtype=np.bytes_)
  alert_time_array = np.array(alert_creation_times, dtype=np.int64)
  alert_reg_array = np.array(alert_regression, dtype=np.float32)

  adj = np.array([edge_sources, edge_targets], dtype=np.int64)

  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "hardware": in_memory_graph_lib.InMemoryNodeSet(
              features={
                  "#id": hw_id_array,
                  "time": hw_time_array,
                  "signal": hw_signal_array,
              },
              num_nodes=config.num_hardware,
          ),
          "alerts": in_memory_graph_lib.InMemoryNodeSet(
              features={
                  "#id": alert_id_array,
                  "creation_time": alert_time_array,
                  "signal_regression": alert_reg_array,
              },
              num_nodes=config.num_alerts,
          ),
      },
      edge_sets={
          "hardware_to_alert": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=adj
          ),
      },
  )
  if schema is None:
    schema = generate_signal_regression_schema()
  in_memory_graph_validate_lib.validate_graph(graph, schema)
  return graph


def generate_signal_regression_graph(
    path: str,
    config: TemporalAlertGraphConfig = TemporalAlertGraphConfig(),
    *,
    schema: Optional[schema_lib.GraphSchema] = None,
):
  """Generates a signal regression temporal graph on disk in GF format.

  Args:
    path: The directory path to write the generated GF graph.
    config: Configuration parameters for graph generation.
    schema: Optional schema to validate against and write to disk; if None,
      generates one.
  """
  if schema is None:
    schema = generate_signal_regression_schema()
  graph = generate_signal_regression_in_memory_graph(
      config=config,
      schema=schema,
  )
  gf_graph_in_memory_lib.write_graph(graph, schema, path)

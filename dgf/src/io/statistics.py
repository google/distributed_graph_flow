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

"""IO of feature statistics."""

import dataclasses
import apache_beam as beam
import dacite
from dgf.src.data import statistics as statistics_lib
from dgf.src.util import json as json_lib


def read_feature_statistics(path: str) -> statistics_lib.GraphFeatureStatistics:
  """Reads feature statistics from disk in a JSON format.

  Usage example:

  ```python
    # Compute and save statistics in Beam
    with beam.Pipeline() as p:
      graph = gf.io.beam.read_graphai_hgraph(p, <graph path>)
      stats = gf.analyse.feature_statistics(graph)
      gf.io.write_feature_statistics(stats, "path/to/output.json")

    # Load the feature stats
    stats = gf.io.read_feature_statistics("path/to/output.json")
  ```

  Args:
    path: Input path.

  Returns:
    The loaded statistics.
  """
  serialized_stats = json_lib.read_json(path)
  return dacite.from_dict(
      data_class=statistics_lib.GraphFeatureStatistics, data=serialized_stats
  )


def write_feature_statistics(
    stats: statistics_lib.GraphFeatureStatistics, path: str
):
  """Saves feature statistics to disk in a json format.

  Usage example:

  ```python
    stats = gf.io.read_feature_statistics("path/to/output.json")
    gf.io.write_feature_statistics(stats, "path/to/output_again.json", stats)
  ```

  Args:
    stats: The statistics to save.
    path: Output path.
  """
  json_lib.write_json(path, dataclasses.asdict(stats))


def write_feature_statistics_beam(
    stats: beam.PCollection[statistics_lib.GraphFeatureStatistics], path: str
):
  """Writes a beam pcollection of feature statistics to disk in json format.

  To save stats in a python object, use "write_feature_statistics" instead.

  Usage example:

  ```python
    with beam.Pipeline() as p:
      graph = gf.io.beam.read_graphai_hgraph(p, <graph path>)
      stats = gf.analyse.feature_statistics(graph)
      gf.io.beam.write_feature_statistics(stats, "path/to/output.json")
  ```

  Args:
    stats: A PCollection with a single statistics object.
    path: Output path.
  """

  def _write_to_json(
      stats: statistics_lib.GraphFeatureStatistics, path: str
  ) -> None:
    write_feature_statistics(stats, path)

  _ = stats | "Write to json" >> beam.Map(_write_to_json, path=path)

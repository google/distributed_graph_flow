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

"""Tests for statistics IO."""

import json
import math
import os
from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.data import histogram
from dgf.src.data import statistics as stats_lib
from dgf.src.io import statistics as io_stats
from dgf.src.util import test_util


class StatisticsIoTest(parameterized.TestCase):

  def test_topology_statistics_io(self):
    # 1. Create dummy topology statistics
    hist1 = histogram.Histogram(values=[1.0, 2.0], bins=[0.0, 1.0, 2.0])
    hist2 = histogram.Histogram(values=[3.0, 4.0], bins=[0.0, 1.5, 3.0])

    node_sets = {
        "nodes1": stats_lib.NodeSetTopologyStatistics(num_nodes=hist1),
        "nodes2": stats_lib.NodeSetTopologyStatistics(num_nodes=hist2),
    }
    edge_sets = {
        "edges1": stats_lib.EdgeSetTopologyStatistics(
            num_edges=hist1,
            in_degree_distribution=hist1,
            out_degree_distribution=hist2,
        )
    }
    stats = stats_lib.GraphTopologyStatistics(
        node_sets=node_sets, edge_sets=edge_sets
    )

    # 2. Write to temp file
    temp_dir = self.create_tempdir().full_path
    path = os.path.join(temp_dir, "topo_stats.json")
    io_stats.write_topology_statistics(stats, path)

    # 3. Read back
    loaded_stats = io_stats.read_topology_statistics(path)

    # 4. Assert equality
    test_util.assert_are_equal(self, stats, loaded_stats)

  def test_feature_statistics_io_roundtrip_with_nonfinite(self):
    stats = stats_lib.GraphFeatureStatistics(
        node_sets={
            "n1": stats_lib.FeatureSetStatistics(
                features={
                    "f_mixed": stats_lib.FeatureStatistics(
                        count=10,
                        minimum=1.5,
                        maximum=9.5,
                        quantiles=[1.5, 5.5, 9.5],
                        mean=5.5,
                        num_nan=3,
                        num_posinf=1,
                        num_neginf=2,
                    ),
                    "f_all_nan": stats_lib.FeatureStatistics(
                        count=5,
                        minimum=math.inf,
                        maximum=-math.inf,
                        quantiles=[],
                        mean=math.nan,
                        num_nan=5,
                    ),
                }
            )
        }
    )
    temp_dir = self.create_tempdir().full_path
    path = os.path.join(temp_dir, "feature_stats.json")
    io_stats.write_feature_statistics(stats, path)
    loaded_stats = io_stats.read_feature_statistics(path)
    test_util.assert_are_equal(self, stats, loaded_stats)

  def test_read_legacy_feature_statistics_without_new_fields(self):
    legacy_json = {
        "node_sets": {
            "n1": {
                "features": {
                    "f1": {
                        "count": 4,
                        "minimum": 1.0,
                        "maximum": 7.0,
                        "dictionary": {},
                        "quantiles": [1.0, 4.0, 7.0],
                    },
                    "f2": {
                        "count": 2,
                        "minimum": float("nan"),
                        "maximum": float("nan"),
                        "dictionary": {
                            "a": {"index": 0, "count": 2},
                        },
                        "quantiles": [],
                    },
                }
            }
        }
    }
    temp_dir = self.create_tempdir().full_path
    path = os.path.join(temp_dir, "legacy_stats.json")
    with open(path, "w") as f:
      json.dump(legacy_json, f)

    loaded_stats = io_stats.read_feature_statistics(path)
    expected_stats = stats_lib.GraphFeatureStatistics(
        node_sets={
            "n1": stats_lib.FeatureSetStatistics(
                features={
                    "f1": stats_lib.FeatureStatistics(
                        count=4,
                        minimum=1.0,
                        maximum=7.0,
                        dictionary={},
                        quantiles=[1.0, 4.0, 7.0],
                        mean=math.nan,
                        num_nan=0,
                        num_posinf=0,
                        num_neginf=0,
                    ),
                    "f2": stats_lib.FeatureStatistics(
                        count=2,
                        minimum=math.nan,
                        maximum=math.nan,
                        dictionary={
                            "a": stats_lib.DictionaryItem(index=0, count=2),
                        },
                        quantiles=[],
                        mean=math.nan,
                        num_nan=0,
                        num_posinf=0,
                        num_neginf=0,
                    ),
                }
            )
        }
    )
    test_util.assert_are_equal(self, loaded_stats, expected_stats)


if __name__ == "__main__":
  absltest.main()

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

import functools
import math
import os
import tempfile

from absl.testing import absltest
from apache_beam.testing import test_pipeline
from apache_beam.testing import util
from dgf.src.analyse import in_memory_feature_statistics
from dgf.src.data import statistics as statistics_lib
from dgf.src.io import tf_graph_sample
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util

test_util.disable_diff_truncation()


class InMemoryStatisticsTest(absltest.TestCase):

  def test_feature_statistics(self):
    with tempfile.TemporaryDirectory() as tmpdir:

      # Generate some toy data
      path = os.path.join(tmpdir, "samples@1.tfr.gz")
      gen_test_graph.generate_tf_graph_sample_in_tf_record(
          os.path.join(tmpdir, "samples-00000-of-00001.tfr.gz"),
          node_ids=True,
          edge_ids=False,
          variable_length=True,
      )
      schema = gen_test_graph.generate_schema(
          node_ids=True, semantic=True, variable_length=True
      )

      expected_stats = statistics_lib.GraphFeatureStatistics(
          node_sets={
              "n2": statistics_lib.FeatureSetStatistics(
                  features={
                      "#id": statistics_lib.FeatureStatistics(count=4),
                      "f3": statistics_lib.FeatureStatistics(
                          count=4,
                          minimum=4,
                          maximum=5,
                          quantiles=[4.0, 4.0, 5.0, 5.0],
                      ),
                      "f4": statistics_lib.FeatureStatistics(
                          count=4,
                          minimum=10,
                          maximum=11,
                          quantiles=[10.0000, 10.0000, 11.0000, 11.0000],
                      ),
                      "f5": statistics_lib.FeatureStatistics(
                          count=4,
                          minimum=11,
                          maximum=14,
                          quantiles=[11.0, 12.0, 13.0, 14.0],
                      ),
                  }
              ),
              "n1": statistics_lib.FeatureSetStatistics(
                  features={
                      "#id": statistics_lib.FeatureStatistics(count=4),
                      "f2": statistics_lib.FeatureStatistics(
                          count=4,
                      ),
                      "f1": statistics_lib.FeatureStatistics(
                          count=4,
                          dictionary={
                              "blue": statistics_lib.DictionaryItem(
                                  index=0, count=2
                              ),
                              "red": statistics_lib.DictionaryItem(
                                  index=1, count=2
                              ),
                          },
                      ),
                  }
              ),
          }
      )
      with test_pipeline.TestPipeline() as p:
        hgraph = tf_graph_sample.read_tfgnn_graphs_beam(p, path, schema=schema)
        stats = in_memory_feature_statistics.feature_statistics_from_graphs(
            hgraph, schema, num_quantiles=4
        )
        util.assert_that(
            stats,
            util.equal_to(
                [expected_stats],
                equals_fn=functools.partial(test_util.are_equal, abs_tol=0.001),
            ),
        )


if __name__ == "__main__":
  absltest.main()

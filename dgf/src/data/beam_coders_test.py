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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from apache_beam.coders import typecoders
from apache_beam.typehints import trivial_inference
from dgf.src.analyse import reservoir_sampling
from dgf.src.data import statistics
from dgf.src.util import gen_test_graph
from dgf.src.util import test_util


class BeamCodersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "in_memory_graph",
          gen_test_graph.generate_in_memory_graph(),
      ),
      (
          "schema",
          gen_test_graph.generate_schema(),
      ),
      (
          "feature_stats_accumulator",
          statistics.FeatureStatisticsAccumulator(
              count=1,
              minimum=2,
              maximum=3,
              dictionary=None,
              quantiles=reservoir_sampling.BatchReservoirSampling(),
          ),
      ),
  )
  def test_coder(self, value):
    value_type = trivial_inference.instance_to_type(value)
    value_coder = typecoders.registry.get_coder(value_type)

    serialized = value_coder.encode(value)
    decoded = value_coder.decode(serialized)
    test_util.assert_are_equal(self, value, decoded)

    logging.info(
        "value:%s coder:%s serialized_len:%d",
        type(value),
        value_coder,
        len(serialized),
    )


if __name__ == "__main__":
  absltest.main()

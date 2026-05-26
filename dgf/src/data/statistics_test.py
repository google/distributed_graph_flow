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

import math
from absl.testing import absltest
from dgf.src.data import statistics as statistics_lib
from dgf.src.util import test_util

test_util.disable_diff_truncation()


class StatisticsTest(absltest.TestCase):

  def test_str(self):
    stats = statistics_lib.GraphFeatureStatistics(
        node_sets={
            'n1': statistics_lib.FeatureSetStatistics(
                features={
                    'f1': statistics_lib.FeatureStatistics(
                        count=2,
                        minimum=math.inf,
                        maximum=-math.inf,
                        dictionary={
                            'red': statistics_lib.DictionaryItem(
                                index=0, count=1
                            ),
                            'blue': statistics_lib.DictionaryItem(
                                index=1, count=1
                            ),
                        },
                        quantiles=[],
                    ),
                    'f2': statistics_lib.FeatureStatistics(
                        count=2,
                        minimum=0.0,
                        maximum=3.0,
                        dictionary={},
                        quantiles=[0.0, 1.0, 2.0, 3.0],
                    ),
                }
            ),
            'n2': statistics_lib.FeatureSetStatistics(
                features={
                    'f3': statistics_lib.FeatureStatistics(
                        count=2,
                        minimum=4,
                        maximum=5,
                        dictionary={},
                        quantiles=[4.0, 4.3333, 4.6666, 5.0],
                    )
                }
            ),
        }
    )
    self.assertEqual(
        str(stats),
        """\
GraphFeatureStatistics:
  Node Sets (2):
    'n1':
      'f1': count=2, dictionary=(2)['red': 1, 'blue': 1]
      'f2': count=2, min=0.0000, max=3.0000, quantiles=(4)[0.0000, 1.0000, 2.0000, 3.0000]
    'n2':
      'f3': count=2, min=4.0000, max=5.0000, quantiles=(4)[4.0000, 4.3333, 4.6666, 5.0000]
""",
    )


if __name__ == '__main__':
  absltest.main()

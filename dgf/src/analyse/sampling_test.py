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

from absl.testing import absltest
from dgf.src.analyse import sampling as sampling_lib
from dgf.src.sampling import config as config_lib
from dgf.src.util import gen_test_graph


class SamplingTest(absltest.TestCase):

  def test_print_sampling_plan_basic(self):
    schema = gen_test_graph.generate_schema()
    simple_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1", num_hops=1, hop_width=3, reverse=False
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        simple_config, schema
    )

    output = sampling_lib.print_sampling_plan(plan, return_output=True)
    expected_output = """Sampling Plan:

Root: n1
├── e1 [width=3] ➔ n1
└── e2 [width=3] ➔ n2"""
    self.assertEqual(output, expected_output)

  def test_print_sampling_plan_no_header(self):
    schema = gen_test_graph.generate_schema()
    simple_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1", num_hops=1, hop_width=3, reverse=False
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        simple_config, schema
    )

    output = sampling_lib.print_sampling_plan(
        plan, return_output=True, header=False
    )
    expected_output = """Root: n1
├── e1 [width=3] ➔ n1
└── e2 [width=3] ➔ n2"""
    self.assertEqual(output, expected_output)

  def test_print_sampling_plan_complex(self):
    schema = gen_test_graph.generate_schema()
    simple_config = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1",
        num_hops=2,
        hop_width=3,
        reverse=False,
        with_replacement=True,
    )
    plan = config_lib.simple_sampling_config_to_sampling_plan(
        simple_config, schema
    )

    output = sampling_lib.print_sampling_plan(plan, return_output=True)
    expected_output = """Sampling Plan:

Root: n1 (with replacement)
├── e1 [width=3] ➔ n1
│   ├── e1 [width=3] ➔ n1
│   └── e2 [width=3] ➔ n2
└── e2 [width=3] ➔ n2"""
    self.assertEqual(output, expected_output)


if __name__ == "__main__":
  absltest.main()

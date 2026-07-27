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

"""Test for the basic buildable config class."""

from absl.testing import absltest
from absl.testing import parameterized
from dgf.src.learning.ten_lines import common
import numpy as np


class TenLines(parameterized.TestCase):

  def test_num_model_weights(self):
    params = {
        "layer1": {
            "w": np.ones((10, 5), dtype=np.float32),
            "b": np.zeros((5,), dtype=np.float32),
        },
        "layer2": {
            "w": np.ones((5, 2), dtype=np.int32),
            "b": np.zeros((2,), dtype=np.int32),
        },
    }
    weights = common.num_model_weights(params)
    self.assertEqual(weights, {"float32": 55, "int32": 12})

  def test_num_model_weights_none(self):
    self.assertEqual(common.num_model_weights(None), {})

  @parameterized.named_parameters(
      (
          "hmp",
          "heterogeneous_message_passing",
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
      ),
      (
          "hmp_alias_hmpnn",
          "hmpnn",
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
      ),
      (
          "hmp_upper",
          "HETEROGENEOUS_MESSAGE_PASSING",
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
      ),
      (
          "hmp_enum",
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
          common.Architecture.HETEROGENEOUS_MESSAGE_PASSING,
      ),
      (
          "hgat",
          "heterogeneous_graph_attention_network",
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
      ),
      (
          "hgat_alias_hgat",
          "hgat",
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
      ),
      (
          "hgat_alias_han",
          "han",
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
      ),
      (
          "hgat_enum",
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
          common.Architecture.HETEROGENEOUS_GRAPH_ATTENTION_NETWORK,
      ),
  )
  def test_parse_architecture_success(self, input_val, expected):
    self.assertEqual(common.parse_architecture(input_val), expected)

  def test_parse_architecture_invalid_fails(self):
    with self.assertRaisesRegex(ValueError, "Unknown architecture: invalid"):
      common.parse_architecture("invalid")

    with self.assertRaisesRegex(TypeError, "Expected Architecture or str"):
      common.parse_architecture(123)  # pytype: disable=wrong-arg-types

  def test_check_number_of_seeds_success(self):
    common.check_number_of_seeds(
        batch_size=10, num_training=20, num_validation=15, key="node"
    )
    common.check_number_of_seeds(
        batch_size=10, num_training=20, num_validation=None, key="node"
    )
    common.check_number_of_seeds(
        batch_size=10, num_training=None, num_validation=15, key="node"
    )

  def test_check_number_of_seeds_insufficient_training_fails(self):
    with self.assertRaisesRegex(
        ValueError,
        r"The number of training seed nodes \(5\) is smaller than the batch"
        r" size \(10\)\. Increase the number of training seed nodes or decrease"
        r" the batch size\.",
    ):
      common.check_number_of_seeds(
          batch_size=10, num_training=5, num_validation=15, key="node"
      )

  def test_check_number_of_seeds_insufficient_validation_fails(self):
    with self.assertRaisesRegex(
        ValueError,
        r"The number of validation seed nodes \(5\) is smaller than the batch"
        r" size \(10\)\. Increase the number of validation seed edges or"
        r" decrease"
        r" the batch size\.",
    ):
      common.check_number_of_seeds(
          batch_size=10, num_training=15, num_validation=5, key="edge"
      )


if __name__ == "__main__":
  absltest.main()

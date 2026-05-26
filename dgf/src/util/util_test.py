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
from absl.testing import parameterized
from dgf.src.util import util
import numpy as np


class TestUtilTest(absltest.TestCase):

  def test_format_duration_hours_minutes_seconds(self):
    self.assertEqual(util.format_duration(3665), "1h 1m 5s")
    self.assertEqual(util.format_duration(7200), "2h 0m 0s")
    self.assertEqual(util.format_duration(3600 + 60 * 30 + 15), "1h 30m 15s")

  def test_format_duration_minutes_seconds(self):
    self.assertEqual(util.format_duration(65), "1m 5s")
    self.assertEqual(util.format_duration(180), "3m 0s")
    self.assertEqual(util.format_duration(59), "59.00s")

  def test_format_duration_seconds(self):
    self.assertEqual(util.format_duration(56.32), "56.32s")
    self.assertEqual(util.format_duration(0.5), "0.50s")
    self.assertEqual(util.format_duration(0), "0.00s")
    self.assertEqual(util.format_duration(10.0), "10.00s")


class BatchTest(parameterized.TestCase):

  def test_batch_indices_generator_drop_remainder_false_no_shuffle(self):
    num_items = 10
    batch_size = 3
    generator = util.batch_indices_generator(
        items=num_items,
        batch_size=batch_size,
        drop_remainder=False,
        shuffle=False,
    )
    batches = list(generator)
    self.assertLen(batches, 4)
    self.assertEqual(batches[0].tolist(), [0, 1, 2])
    self.assertEqual(batches[1].tolist(), [3, 4, 5])
    self.assertEqual(batches[2].tolist(), [6, 7, 8])
    self.assertEqual(batches[3].tolist(), [9])

  def test_batch_indices_generator_drop_remainder_true_no_shuffle(self):
    num_items = 10
    batch_size = 3
    generator = util.batch_indices_generator(
        items=num_items,
        batch_size=batch_size,
        drop_remainder=True,
        shuffle=False,
    )
    batches = list(generator)
    self.assertLen(batches, 3)
    self.assertEqual(batches[0].tolist(), [0, 1, 2])
    self.assertEqual(batches[1].tolist(), [3, 4, 5])
    self.assertEqual(batches[2].tolist(), [6, 7, 8])

  def test_batch_indices_generator_shuffle(self):
    num_items = 10
    batch_size = 3
    generator = util.batch_indices_generator(
        items=num_items,
        batch_size=batch_size,
        drop_remainder=False,
        shuffle=True,
    )
    batches = list(generator)
    self.assertLen(batches, 4)
    all_elements = np.concatenate(batches)
    self.assertCountEqual(all_elements.tolist(), list(range(num_items)))

  def test_batch_indices_generator_num_items_less_than_batch_size(self):
    # drop_remainder=True
    generator = util.batch_indices_generator(
        items=2,
        batch_size=3,
        drop_remainder=True,
        shuffle=False,
    )
    batches = list(generator)
    self.assertEmpty(batches)

    # drop_remainder=False
    generator = util.batch_indices_generator(
        items=2,
        batch_size=3,
        drop_remainder=False,
        shuffle=False,
    )
    batches = list(generator)
    self.assertLen(batches, 1)
    self.assertEqual(batches[0].tolist(), [0, 1])

  def test_batch_indices_generator_empty(self):
    generator = util.batch_indices_generator(
        items=0,
        batch_size=3,
        drop_remainder=False,
        shuffle=False,
    )
    batches = list(generator)
    self.assertEmpty(batches)

  def test_batch_indices_generator_provided_items(self):
    generator = util.batch_indices_generator(
        items=np.array([10, 11, 12, 13, 14]),
        batch_size=2,
        drop_remainder=False,
        shuffle=False,
    )
    batches = list(generator)
    self.assertLen(batches, 3)
    self.assertEqual(batches[0].tolist(), [10, 11])
    self.assertEqual(batches[1].tolist(), [12, 13])
    self.assertEqual(batches[2].tolist(), [14])

  @parameterized.named_parameters(
      ("no_drop_remainder", 10, 3, False, 4),
      ("drop_remainder", 10, 3, True, 3),
      ("no_drop_remainder_exact", 9, 3, False, 3),
      ("drop_remainder_exact", 9, 3, True, 3),
      ("items_less_than_batch_size", 2, 3, False, 1),
      ("items_less_than_batch_size_drop_remainder", 2, 3, True, 0),
      ("no_items", 0, 3, False, 0),
      ("items_array", np.array([1, 2, 3, 4, 5]), 2, False, 3),
      ("items_array_drop_remainder", np.array([1, 2, 3, 4, 5]), 2, True, 2),
  )
  def test_num_batches(self, items, batch_size, drop_remainder, expected):
    self.assertEqual(
        util.num_batches(
            items=items, batch_size=batch_size, drop_remainder=drop_remainder
        ),
        expected,
    )


class SplitTrainValidTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("basic_split", 10, 0.2, 1, 2, 8),
      ("zero_ratio", 10, 0.0, 1, 0, 10),
      ("one_ratio", 10, 1.0, 1, 10, 0),
      ("ensure_at_least_one", 3, 0.1, 1, 1, 2),
      ("ensure_at_least_batch_size", 10, 0.1, 2, 2, 8),
  )
  def test_split_ratios(
      self,
      num_values,
      validation_ratio,
      batch_size,
      expected_valid,
      expected_train,
  ):
    train_idxs, valid_idxs = util.split_train_valid(
        num_values, validation_ratio, random_seed=42, batch_size=batch_size
    )
    self.assertLen(valid_idxs, expected_valid)
    self.assertLen(train_idxs, expected_train)
    all_idxs = np.concatenate([train_idxs, valid_idxs])
    self.assertCountEqual(all_idxs.tolist(), list(range(num_values)))

  @parameterized.named_parameters(
      ("cap_by_max", 10, 0.5, 1, 2, 2, 8),
      ("not_capped_by_max", 10, 0.2, 1, 5, 2, 8),
      ("cap_below_batch_size", 10, 0.5, 2, 1, 1, 9),
  )
  def test_split_max_num_valid_examples(
      self,
      num_values,
      validation_ratio,
      batch_size,
      max_num_valid_examples,
      expected_valid,
      expected_train,
  ):
    train_idxs, valid_idxs = util.split_train_valid(
        num_values,
        validation_ratio,
        random_seed=42,
        batch_size=batch_size,
        max_num_valid_examples=max_num_valid_examples,
    )
    self.assertLen(valid_idxs, expected_valid)
    self.assertLen(train_idxs, expected_train)
    all_idxs = np.concatenate([train_idxs, valid_idxs])
    self.assertCountEqual(all_idxs.tolist(), list(range(num_values)))

  def test_randomness_and_consistency(self):
    num_values = 100
    validation_ratio = 0.2

    # Consistency
    train_idxs1, valid_idxs1 = util.split_train_valid(
        num_values, validation_ratio, random_seed=42
    )
    train_idxs2, valid_idxs2 = util.split_train_valid(
        num_values, validation_ratio, random_seed=42
    )
    np.testing.assert_array_equal(train_idxs1, train_idxs2)
    np.testing.assert_array_equal(valid_idxs1, valid_idxs2)

    # Different seeds
    train_idxs3, _ = util.split_train_valid(
        num_values, validation_ratio, random_seed=43
    )
    self.assertFalse(np.array_equal(train_idxs1, train_idxs3))

  @parameterized.named_parameters(
      ("too_few_values", 1, 0.5, 1),
      ("insufficient_for_batch", 3, 0.5, 2),
  )
  def test_invalid_inputs(self, num_values, validation_ratio, batch_size):
    with self.assertRaises(ValueError):
      util.split_train_valid(
          num_values, validation_ratio, random_seed=42, batch_size=batch_size
      )


if __name__ == "__main__":
  absltest.main()

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
from dgf.src.analyse import text_detection
import numpy as np


class TextDetectionTest(absltest.TestCase):

  def test_text_bytes_detected(self):
    sentences = [
        f"This paper proposes a novel graph neural network for task {i}."
        .encode("utf-8")
        for i in range(10)
    ]
    self.assertTrue(text_detection.looks_like_text(np.array(sentences)))

  def test_text_str_detected(self):
    sentences = [
        f"We are looking for a senior engineer with {i} years experience."
        for i in range(10)
    ]
    self.assertTrue(text_detection.looks_like_text(sentences))

  def test_2d_single_column_detected(self):
    sentences = np.array([
        [
            f"Customer reported an issue with payment method number {i}."
            .encode("utf-8")
        ]
        for i in range(10)
    ])
    self.assertTrue(text_detection.looks_like_text(sentences))

  def test_2d_multi_column_rejected(self):
    sentences = np.array([
        [
            f"First sentence for row {i} with enough words.".encode("utf-8"),
            f"Second sentence for row {i} with enough words.".encode("utf-8"),
        ]
        for i in range(10)
    ])
    self.assertFalse(text_detection.looks_like_text(sentences))

  def test_categorical_values_rejected(self):
    categories = [b"Private", b"Self-emp-not-inc", b"Local-gov", b"State-gov"]
    self.assertFalse(text_detection.looks_like_text(categories * 10))

  def test_too_few_values_rejected(self):
    values = [f"Unique sentence number {i}.".encode() for i in range(9)]
    self.assertFalse(text_detection.looks_like_text(values))

  def test_collision_threshold(self):
    unique_values = [f"Unique sentence number {i}.".encode() for i in range(18)]
    # 1 collision out of 20 values (5%) -> True.
    self.assertTrue(
        text_detection.looks_like_text(unique_values + [b"dup", b"dup"])
    )
    # 2 collisions out of 20 values (10%) -> False (requires < 10%).
    self.assertFalse(
        text_detection.looks_like_text(unique_values[:17] + [b"dup"] * 3)
    )

  def test_unique_ids_rejected(self):
    self.assertFalse(
        text_detection.looks_like_text([f"user_{i}" for i in range(20)])
    )
    self.assertFalse(
        text_detection.looks_like_text(
            [f"{i:032x}".encode() for i in range(20)]
        )
    )

  def test_multi_word_threshold(self):
    sentences = [f"Unique sentence number {i}." for i in range(10)]
    ids = [f"id_{i}" for i in range(10)]
    # 50% multi-word values -> False (requires > 50%).
    self.assertFalse(text_detection.looks_like_text(sentences + ids))
    # 55% multi-word values -> True.
    self.assertTrue(
        text_detection.looks_like_text(sentences + ["One more."] + ids[:9])
    )

  def test_empty_values_ignored(self):
    values = [f"Unique sentence number {i}.".encode() for i in range(10)]
    self.assertTrue(text_detection.looks_like_text(values + [b"", b"  "] * 10))

  def test_all_empty_values_rejected(self):
    self.assertFalse(text_detection.looks_like_text([b"", b"   "] * 10))

  def test_numerical_array_rejected(self):
    self.assertFalse(
        text_detection.looks_like_text(np.arange(20, dtype=np.int64))
    )


if __name__ == "__main__":
  absltest.main()

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

import absl.testing.absltest as absltest
from dgf.src.data import schema as schema_lib
from dgf.src.io import tf_graph_common
import numpy as np
import tensorflow as tf


class TfGraphCommonTest(absltest.TestCase):

  def test_populate_features(self):
    example = tf.train.Example()
    feature_schema = {
        "f1": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.INTEGER_64, shape=()
        )
    }
    features = {"f1": np.array([42])}
    tf_graph_common.populate_features(
        example, 0, feature_schema, features, ignore_keys=()
    )
    self.assertEqual(example.features.feature["f1"].int64_list.value[0], 42)


if __name__ == "__main__":
  absltest.main()

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

"""Conversion between feature formats and other similar objects."""

from typing import Any, Dict

from dgf.src.data import schema as schema_lib
import numpy as np
import pyarrow as pa
import tensorflow as tf

# Mapping from FeatureFormat to TensorFlow dtypes.
FEATURE_FORMAT_TO_TF_DTYPE: Dict[schema_lib.FeatureFormat, tf.DType] = {
    schema_lib.FeatureFormat.INTEGER_64: tf.int64,
    schema_lib.FeatureFormat.INTEGER_32: tf.int32,
    schema_lib.FeatureFormat.FLOAT_32: tf.float32,
    schema_lib.FeatureFormat.FLOAT_64: tf.float64,
    schema_lib.FeatureFormat.BYTES: tf.string,
    schema_lib.FeatureFormat.BOOL: tf.bool,
}

# Mapping from TensorFlow dtypes to FeatureFormat.
TF_DTYPE_TO_FEATURE_FORMAT: Dict[tf.DType, schema_lib.FeatureFormat] = {
    tf.int64: schema_lib.FeatureFormat.INTEGER_64,
    tf.int32: schema_lib.FeatureFormat.INTEGER_32,
    tf.float32: schema_lib.FeatureFormat.FLOAT_32,
    tf.float64: schema_lib.FeatureFormat.FLOAT_64,
    tf.string: schema_lib.FeatureFormat.BYTES,
    tf.bool: schema_lib.FeatureFormat.BOOL,
}

# Mapping from NumPy dtypes to FeatureFormat.
NP_DTYPE_TO_FEATURE_FORMAT: Dict[Any, schema_lib.FeatureFormat] = {
    np.int64: schema_lib.FeatureFormat.INTEGER_64,
    np.int32: schema_lib.FeatureFormat.INTEGER_32,
    np.float32: schema_lib.FeatureFormat.FLOAT_32,
    np.float64: schema_lib.FeatureFormat.FLOAT_64,
    np.bytes_: schema_lib.FeatureFormat.BYTES,
    np.str_: schema_lib.FeatureFormat.BYTES,
    np.bool_: schema_lib.FeatureFormat.BOOL,
}

# Mapping from FeatureFormat to NumPy dtypes.
FEATURE_FORMAT_TO_NP_DTYPE: Dict[schema_lib.FeatureFormat, Any] = {
    schema_lib.FeatureFormat.INTEGER_64: np.int64,
    schema_lib.FeatureFormat.INTEGER_32: np.int32,
    schema_lib.FeatureFormat.FLOAT_32: np.float32,
    schema_lib.FeatureFormat.FLOAT_64: np.float64,
    schema_lib.FeatureFormat.BYTES: np.bytes_,
    schema_lib.FeatureFormat.BOOL: np.bool_,
}

# Mapping from FeatureFormat to Avro types.
FEATURE_FORMAT_TO_AVRO_DTYPE: Dict[schema_lib.FeatureFormat, str] = {
    schema_lib.FeatureFormat.INTEGER_32: "int",
    schema_lib.FeatureFormat.INTEGER_64: "long",
    schema_lib.FeatureFormat.FLOAT_32: "float",
    schema_lib.FeatureFormat.FLOAT_64: "double",
    schema_lib.FeatureFormat.BYTES: "bytes",
    schema_lib.FeatureFormat.BOOL: "boolean",
}

FEATURE_FORMAT_TO_PYARROW_DATA_TYPE: Dict[
    schema_lib.FeatureFormat, pa.DataType
] = {
    schema_lib.FeatureFormat.INTEGER_32: pa.int32(),
    schema_lib.FeatureFormat.INTEGER_64: pa.int64(),
    schema_lib.FeatureFormat.FLOAT_32: pa.float32(),
    schema_lib.FeatureFormat.FLOAT_64: pa.float64(),
    schema_lib.FeatureFormat.BYTES: pa.binary(),
    schema_lib.FeatureFormat.BOOL: pa.bool_(),
}

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

"""Machine Learning and Graph Neural Networks using JAX."""

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top,g-bad-import-order,reimported,disable=attribute-error

from dgf.src.learning.jax.common import get_activation
from dgf.src.learning.jax.common import jnp_dtype_from_string
from dgf.src.learning.jax.common import jnp_name_from_dtype
from dgf.src.learning.jax.common import JaxBaseConfig

# TODO(bmayer,gbm): Modify API documentation generation to support constants.
# from dgf.src.learning.jax.common import DEFAULT_DROPOUT_RATE
# from dgf.src.learning.jax.common import DEFAULT_MATRIX_PRECISION
# from dgf.src.learning.jax.common import DEFAULT_POINTWISE_NORM_PRECISION
# from dgf.src.learning.jax.common import DEFAULT_SOFTMAX_PRECISION

# from dgf.src.learning.jax.common import DEFAULT_NODESET_NAME
# from dgf.src.learning.jax.common import DEFAULT_NODE_FEATURE_NAME
# from dgf.src.learning.jax.common import DEFAULT_EDGESET_NAME
# from dgf.src.learning.jax.common import DEFAULT_EDGE_FEATURE_NAME
# from dgf.src.learning.jax.common import DEFAULT_HIDDEN_STATE_NAME


from dgf.src.learning.jax.flax_train import train

# Sub-namespaces
from dgf.src.api.jax import layers

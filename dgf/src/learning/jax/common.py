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

"""Common utilities and definitions for JAX in GraphFlow."""

import abc
import dataclasses
from typing import Any, Callable, Dict, Optional, Protocol

from absl import logging
from dgf.src.learning import config
from dgf.src.util import log
from flax.core import pretty_repr
import flax.linen as nn
import jax
import jax.nn as jax_nn
import jax.numpy as jnp

# Linear, Einsum, (matrices) etc.
# This will work well with CPU, TPU and modern GPUs. Older GPUs (e.g., v100)
# don't support bf16.
DEFAULT_MATRIX_PRECISION: jnp.dtype = jnp.bfloat16

# Full precision due to mean/var stats.
DEFAULT_POINTWISE_NORM_PRECISION: jnp.dtype = jnp.float32

# Softmax
DEFAULT_SOFTMAX_PRECISION: jnp.dtype = jnp.float32

# Final loss reduction should be accumulated in full precision
DEFAULT_LOSS_PRECISION: jnp.dtype = jnp.float32

# TODO(bmayer): Might be a good default value that could work for large or small
# datasets?
DEFAULT_DROPOUT_RATE: float = 0.3


class BuildableModule(Protocol):
  """Protocol for objects that define a `make` function and returns nn.Module."""

  def make(self) -> nn.Module:
    ...


class GenericLayer(Protocol):
  """Protocol for objects that define a `make` function and returns nn.Module."""

  def make(self, name: Optional[str] = None) -> nn.Module:
    ...

  def architecture(self) -> str:
    ...


# By default we assume a single node set named `nodes` with node features named
# # `initial_state`. Similarly we assume a single edge set named `edges` with
# an optional `initial_state`.
DEFAULT_NODESET_NAME = "nodes"
DEFAULT_NODE_FEATURE_NAME = "initial_state"
DEFAULT_EDGESET_NAME = "edges"
DEFAULT_EDGE_FEATURE_NAME = "initial_state"

DEFAULT_HIDDEN_STATE_NAME = "hidden_state"


def jnp_name_from_dtype(dtype: jnp.dtype) -> str:
  """Return a string name for a jnp.dtype object."""
  return dtype.__name__  # pyrefly: ignore[missing-attribute]


@dataclasses.dataclass(frozen=True, kw_only=True)
class PrecisionConfig:
  """Precision and regularization settings for JAX GNN layers.

  Separated from JaxBaseConfig to cleanly isolate precision/regularization
  concerns from dataset-specific fields (nodeset_name, edgeset_name, etc.)
  which should be derived from the graph schema at make() time.
  """

  matrix_precision: str = jnp_name_from_dtype(DEFAULT_MATRIX_PRECISION)
  pointwise_norm_precision: str = jnp_name_from_dtype(
      DEFAULT_POINTWISE_NORM_PRECISION
  )
  softmax_precision: str = jnp_name_from_dtype(DEFAULT_SOFTMAX_PRECISION)
  dropout_rate: float = DEFAULT_DROPOUT_RATE

  def __post_init__(self):
    _ = jnp_dtype_from_string(self.matrix_precision)
    _ = jnp_dtype_from_string(self.pointwise_norm_precision)
    _ = jnp_dtype_from_string(self.softmax_precision)
    if not 0.0 <= self.dropout_rate <= 1.0:
      raise ValueError(
          f"dropout_rate must be in [0, 1.0], got {self.dropout_rate}"
      )


@dataclasses.dataclass(frozen=True, kw_only=True)
class JaxBaseConfig(config.Config):
  """Base class for a GNN implemented in JAX."""

  matrix_precision: str = jnp_name_from_dtype(DEFAULT_MATRIX_PRECISION)
  pointwise_norm_precision: str = jnp_name_from_dtype(
      DEFAULT_POINTWISE_NORM_PRECISION
  )
  softmax_precision: str = jnp_name_from_dtype(DEFAULT_SOFTMAX_PRECISION)
  dropout_rate: float = DEFAULT_DROPOUT_RATE
  nodeset_name: str = DEFAULT_NODESET_NAME
  edgeset_name: str = DEFAULT_EDGESET_NAME
  input_node_feature: str = DEFAULT_NODE_FEATURE_NAME
  output_node_feature: str = DEFAULT_HIDDEN_STATE_NAME

  def __post_init__(self):
    # Validate that matrix_precision is a valid jnp.dtype.
    # Will throw an exception if we can't convert to dtype.
    _ = jnp_dtype_from_string(self.matrix_precision)
    _ = jnp_dtype_from_string(self.pointwise_norm_precision)
    _ = jnp_dtype_from_string(self.softmax_precision)
    if not 0.0 <= self.dropout_rate <= 1.0:
      raise ValueError(
          f"dropout_rate must be in [0, 1.0], got {self.dropout_rate}"
      )


def get_activation(name: str) -> Callable[..., Any]:
  """Get an activation function by (string) name."""
  try:
    return getattr(jax_nn, name)
  except AttributeError:
    logging.info("Did not find activation %s in jax.nn trying flax.", name)
    try:
      return getattr(nn, name)
    except AttributeError as exc:
      raise AttributeError(
          f"Activation {name} not found in jax.nn or flax.nn"
      ) from exc


def jnp_dtype_from_string(name: str) -> jnp.dtype:
  """Return a JAX numpy type from a string name."""
  try:
    return getattr(jnp, name)
  except AttributeError as exc:
    logging.error("Did not find dtype %s", name)
    raise AttributeError(f"dtype {name} is not found in jax.numpy") from exc


def log_info_shape(name: str, value: Any):
  """Log the shape of a JAX array or a pytree of JAX arrays.

  Args:
    name: The name of the value being logged.
    value: The value to log the shape of.
  """
  model_params_shapes = jax.tree_util.tree_map(lambda x: x.shape, value)
  log.info(f"{name}:\n{pretty_repr(model_params_shapes)}")


def log_info_value(name: str, value: Any):
  """Log the values of a pytree of JAX arrays during execution.

  Args:
    name: The name of the value being logged.
    value: The value to log the shape of.
  """
  jax.debug.print(name + ":\n{value}", value=value)


@jax.jit
def jit_gather_features(
    features: Dict[str, jax.Array], node_idxs: jax.Array
) -> Dict[str, jax.Array]:
  """Gathers features for specified node indices.

  Args:
    features: A dictionary of feature arrays.
    node_idxs: An array of node indices to gather.

  Returns:
    A dictionary of gathered feature arrays.
  """
  return {k: v[node_idxs] for k, v in features.items()}


class ArchitectureProvider(abc.ABC):
  """Abstract class that provides neural net architecture information."""

  @abc.abstractmethod
  def architecture(self) -> str:
    """Returns a string describing the architecture from top to bottom.

    Example:
      ```
      Dense(256)
      Norm(batch norm)
      Actiation(gelu)
      ```
    """
    raise NotImplementedError

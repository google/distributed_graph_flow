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

"""Text feature normalization using frozen pre-trained text embedding models."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import enum
import functools
import os
from typing import Any

import dataclasses_json
from dgf.src.data import schema as schema_lib
from dgf.src.transform import normalize_base
from dgf.src.util.weak_dep.weak_dep_keras_hub import keras_hub
from dgf.src.util.weak_dep.weak_dep_tensorflow import Module
from dgf.src.util.weak_dep.weak_dep_tensorflow import tf
from dgf.src.util.weak_dep.weak_dep_tensorflow_hub import hub
from dgf.src.util.weak_dep.weak_dep_tensorflow_text import tf_text
import numpy as np

_BATCH_SIZE = 1024
_KERAS_HUB_HANDLE_PREFIXES = ("tf://", "kaggle://", "hf://")


class TextEmbedderBackend(str, enum.Enum):
  """Backend library used to load and run a text embedding model."""

  TF_HUB = "TF_HUB"
  KERAS_HUB = "KERAS_HUB"


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class TextEmbedderConfig:
  """Configuration for a pre-trained frozen text embedding model.

  Attributes:
    name: Short identifier for display in reports and logs.
    handle: Model handle, preset name, URL, or filesystem path.
    dim: Output embedding dimension per text entry.
    backend: Whether to load via TensorFlow Hub or KerasHub.
    requires_tf_text: Whether `tensorflow_text` ops must be imported before
      loading the model.
    output_key: Optional dictionary key to extract when the model returns a dict
      of tensors.
    description: Human-readable summary of the model.
    sequence_length: Number of tokens of the KerasHub preprocessor. If None,
      uses the default of the preset.
  """

  name: str
  handle: str
  dim: int
  backend: TextEmbedderBackend = TextEmbedderBackend.TF_HUB
  requires_tf_text: bool = False
  output_key: str | None = None
  description: str = ""
  sequence_length: int | None = 128


def _select_handle(handles: list[str]) -> str:
  """Returns the first handle (the internal handles are stripped in OSS)."""
  # TODO(gbm): Find a better way to handle internal / external models.
  return handles[0]


BUILTIN_TEXT_EMBEDDERS: dict[str, TextEmbedderConfig] = {
    "nnlm-en-dim50": TextEmbedderConfig(
        name="nnlm-en-dim50",
        handle=_select_handle([
            "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2",
        ]),
        dim=50,
        backend=TextEmbedderBackend.TF_HUB,
        requires_tf_text=False,
        description=(
            "Token-based English text embedding trained on Google News 7B"
            " (50-d, TF-Hub)."
        ),
    ),
    "nnlm-en-dim128": TextEmbedderConfig(
        name="nnlm-en-dim128",
        handle=_select_handle([
            "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2",
        ]),
        dim=128,
        backend=TextEmbedderBackend.TF_HUB,
        requires_tf_text=False,
        description=(
            "Token-based English text embedding trained on Google News 7B"
            " (128-d, TF-Hub)."
        ),
    ),
    "use-en-dim512": TextEmbedderConfig(
        name="use-en-dim512",
        handle=_select_handle([
            "https://tfhub.dev/google/universal-sentence-encoder/4",
        ]),
        dim=512,
        backend=TextEmbedderBackend.TF_HUB,
        requires_tf_text=False,
        description=(
            "Universal Sentence Encoder DAN model for English text"
            " (512-d, TF-Hub)."
        ),
    ),
    "use-multi-dim512": TextEmbedderConfig(
        name="use-multi-dim512",
        handle=_select_handle([
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
        ]),
        dim=512,
        backend=TextEmbedderBackend.TF_HUB,
        requires_tf_text=True,
        description=(
            "Multilingual Universal Sentence Encoder covering 16 languages"
            " (512-d, TF-Hub)."
        ),
    ),
    "bert-tiny-en-dim128": TextEmbedderConfig(
        name="bert-tiny-en-dim128",
        handle=_select_handle([
            "bert_tiny_en_uncased",
        ]),
        dim=128,
        backend=TextEmbedderBackend.KERAS_HUB,
        requires_tf_text=True,
        output_key="pooled_output",
        description="2-layer uncased English BERT model (128-d, KerasHub).",
    ),
    "bert-small-en-dim512": TextEmbedderConfig(
        name="bert-small-en-dim512",
        handle=_select_handle([
            "bert_small_en_uncased",
        ]),
        dim=512,
        backend=TextEmbedderBackend.KERAS_HUB,
        requires_tf_text=True,
        output_key="pooled_output",
        description="4-layer uncased English BERT model (512-d, KerasHub).",
    ),
    "bge-base-en-dim768": TextEmbedderConfig(
        name="bge-base-en-dim768",
        handle=_select_handle([
            "bge_base_v1.5_en",
        ]),
        dim=768,
        backend=TextEmbedderBackend.KERAS_HUB,
        requires_tf_text=True,
        output_key="pooled_output",
        description=(
            "12-layer BGE v1.5 English sentence embedding model"
            " (768-d, KerasHub)."
        ),
    ),
    "embedding-gemma3-dim768": TextEmbedderConfig(
        name="embedding-gemma3-dim768",
        handle=_select_handle([
            "embedding_gemma3_300m",
        ]),
        dim=768,
        backend=TextEmbedderBackend.KERAS_HUB,
        requires_tf_text=True,
        output_key="pooled_output",
        description="300M Gemma 3 text embedding model (768-d, KerasHub).",
    ),
}


def _load_keras_hub_preprocessor(preset: str, sequence_length: int | None):
  """Loads the KerasHub text preprocessor of `preset`."""
  kwargs = {}
  if sequence_length is not None:
    kwargs["sequence_length"] = sequence_length
  try:
    return keras_hub.models.TextClassifierPreprocessor.from_preset(
        preset, **kwargs
    )
  except ValueError:
    return keras_hub.models.TextEmbedderPreprocessor.from_preset(
        preset, **kwargs
    )


def _to_bytes(value: Any) -> bytes:
  """Converts a text value to UTF-8 bytes."""
  if isinstance(value, bytes):
    return value
  if isinstance(value, str):
    return value.encode("utf-8", "replace")
  raise ValueError(f"Expected a text value, got {type(value)}: {value!r}")


class LoadedTextEmbedder(Module):
  """Wraps a frozen pre-trained text embedding model as a `tf.Module`.

  Instances are shared by all the users of a model (see `_load_text_embedder`),
  preventing weight duplication in RAM and in exported TensorFlow SavedModels.
  """

  def __init__(self, config: TextEmbedderConfig):
    super().__init__(name=f"text_embedder_{config.name.replace('-', '_')}")
    self._config = config
    # Assigning the models as attributes makes `tf.Module` track their
    # variables and resources (e.g., for SavedModel export).
    self._hub_module: Any = None
    self._preprocessor: Any = None
    self._backbone: Any = None

    # Fails early if a library is not installed.
    if config.backend == TextEmbedderBackend.TF_HUB:
      hub.force_import()
    else:
      keras_hub.force_import()
    if (
        config.requires_tf_text
        or config.backend == TextEmbedderBackend.KERAS_HUB
    ):
      tf_text.force_import()  # Registers the TF Text ops used by the model.

    with tf.device("/CPU:0"):
      if config.backend == TextEmbedderBackend.TF_HUB:
        self._hub_module = hub.load(config.handle)
      else:
        self._backbone = keras_hub.models.Backbone.from_preset(
            config.handle, trainable=False
        )
        self._preprocessor = _load_keras_hub_preprocessor(
            config.handle, config.sequence_length
        )
      dim = int(self._embed_flat(tf.constant([b""])).shape[-1])

    if config.dim <= 0:
      self._config = dataclasses.replace(config, dim=dim)
    elif dim != config.dim:
      raise ValueError(
          f"Text embedder {config.name!r} ({config.handle!r}) produced"
          f" embeddings of dimension {dim}, expected {config.dim}."
      )

  @property
  def config(self) -> TextEmbedderConfig:
    """Returns the resolved `TextEmbedderConfig`."""
    return self._config

  def __deepcopy__(self, memo: dict[int, Any]) -> LoadedTextEmbedder:
    # The model is frozen: copies of a normalizer share it.
    return self

  @property
  def dim(self) -> int:
    """Returns the output embedding dimension."""
    return self._config.dim

  def _embed_flat(self, texts: tf.Tensor) -> tf.Tensor:
    """Embeds a 1D `tf.string` tensor into a `[n, dim]` float32 tensor."""
    if self._hub_module is not None:
      outputs = self._hub_module(texts)
    else:
      outputs = self._backbone(self._preprocessor(texts), training=False)
    if isinstance(outputs, Mapping):
      if self._config.output_key not in outputs:
        raise ValueError(
            f"Text embedder {self._config.name!r} returned the outputs"
            f" {sorted(outputs)}. Set `output_key` to one of them."
        )
      outputs = outputs[self._config.output_key]
    if len(outputs.shape) != 2:
      raise ValueError(
          f"Text embedder {self._config.name!r} returned a tensor of shape"
          f" {outputs.shape}, expected [batch, dim]."
      )
    return tf.cast(outputs, tf.float32)

  def embed_tensorflow(self, texts: tf.Tensor) -> tf.Tensor:
    """Embeds a `tf.string` tensor of shape `[...]` into `[..., dim]`."""
    embedded = self._embed_flat(tf.reshape(texts, [-1]))
    return tf.reshape(embedded, tf.concat([tf.shape(texts), [self.dim]], 0))

  def embed_numpy(self, texts: np.ndarray) -> np.ndarray:
    """Embeds an array of strings of shape `[...]` into `[..., dim]`."""
    texts = np.asarray(texts)
    flat_texts = [_to_bytes(t) for t in texts.reshape(-1)]
    with tf.device("/CPU:0"):
      embeddings = [
          self._embed_flat(tf.constant(flat_texts[i : i + _BATCH_SIZE])).numpy()
          for i in range(0, len(flat_texts), _BATCH_SIZE)
      ]
    if not embeddings:
      return np.zeros(texts.shape + (self.dim,), dtype=np.float32)
    return np.concatenate(embeddings).reshape(texts.shape + (self.dim,))


@functools.cache
def _load_text_embedder(handle: str) -> LoadedTextEmbedder:
  """Loads the model of `handle`, shared by all its users."""
  return LoadedTextEmbedder(resolve_text_embedder_config(handle, dim=0))


def resolve_text_embedder_config(
    spec: str | TextEmbedderConfig,
    *,
    dim: int | None = None,
) -> TextEmbedderConfig:
  """Resolves a built-in name, handle string, or `TextEmbedderConfig`.

  Args:
    spec: Built-in embedder name, model handle / path, or config.
    dim: Embedding dimension of a custom handle. If None, the model is loaded to
      infer it. If 0, the dimension is left unknown.

  Returns:
    The resolved config.
  """
  if isinstance(spec, TextEmbedderConfig):
    return spec
  if not spec:
    raise ValueError(
        f"Expected a non-empty text embedder name or handle, got {spec!r}."
    )
  if spec in BUILTIN_TEXT_EMBEDDERS:
    return BUILTIN_TEXT_EMBEDDERS[spec]
  for builtin_config in BUILTIN_TEXT_EMBEDDERS.values():
    if builtin_config.handle == spec:
      return builtin_config
  if dim is None:
    return _load_text_embedder(spec).config

  is_keras_hub = spec.startswith(_KERAS_HUB_HANDLE_PREFIXES) or (
      os.path.exists(os.path.join(spec, "config.json"))
  )
  return TextEmbedderConfig(
      name=os.path.basename(spec.rstrip("/")) or "custom",
      handle=spec,
      dim=dim,
      backend=(
          TextEmbedderBackend.KERAS_HUB
          if is_keras_hub
          else TextEmbedderBackend.TF_HUB
      ),
      output_key="pooled_output" if is_keras_hub else None,
      description=f"Custom text embedder ({spec})",
      sequence_length=None,
  )


@normalize_base.normalizer_registry.register
@dataclasses_json.dataclass_json
@dataclasses.dataclass(kw_only=True)
class TextEmbeddingNormalizer(normalize_base.AbstractFeatureNormalizer):
  """Normalizes a TEXT feature into a dense vector using a pre-trained model.

  The frozen model is loaded on first use, and shared with the other
  normalizers using the same embedder. Only the embedder handle is serialized,
  not the model weights.

  Usage example:

  ```python
  normalizer = dgf.transform.TextEmbeddingNormalizer.create(
      "title", schema.node_sets["paper"].features["title"]
  )
  embeddings = normalizer.normalize_numpy(np.array([b"Graph neural networks"]))
  ```
  """

  output_feature_name: str
  output_shape: tuple[int | None, ...]
  embedder_handle: str = "nnlm-en-dim128"
  is_timeseries: bool = False
  group: str | None = None
  type: str = dataclasses.field(default="TextEmbeddingNormalizer", init=False)

  @classmethod
  def create(
      cls,
      feature_name: str,
      input_schema: schema_lib.FeatureSchema,
      embedder: str | TextEmbedderConfig = "nnlm-en-dim128",
  ) -> "TextEmbeddingNormalizer":
    """Creates a `TextEmbeddingNormalizer` for a `BYTES` feature."""
    if input_schema.format != schema_lib.FeatureFormat.BYTES:
      raise ValueError(
          f"Feature '{feature_name}' has format '{input_schema.format}', but "
          "TextEmbeddingNormalizer only supports BYTES features."
      )
    if not input_schema.is_static_shape():
      raise ValueError(
          "TextEmbeddingNormalizer requires fixed-length feature tensors, but"
          f" feature '{feature_name}' has a dynamic shape"
          f" ({input_schema.shape}). Please run PadAndCapTimeseries first."
      )
    cfg = resolve_text_embedder_config(embedder)
    handle = cfg.name if cfg.name in BUILTIN_TEXT_EMBEDDERS else cfg.handle
    input_shape = input_schema.shape or ()
    return TextEmbeddingNormalizer(
        input_feature=feature_name,
        output_feature_name=f"{feature_name}_TEXT_EMBEDDING",
        output_shape=tuple(input_shape) + (cfg.dim,),
        embedder_handle=handle,
        is_timeseries=input_schema.is_timeseries,
        group=normalize_base.output_timeseries_group(
            feature_name, input_schema
        ),
    )

  @property
  def embedding_dim(self) -> int:
    """Returns the output embedding dimension."""
    dim = self.output_shape[-1]
    assert dim is not None
    return int(dim)

  def embedder_config(self) -> TextEmbedderConfig:
    """Returns the embedder config, without loading the model if possible."""
    return resolve_text_embedder_config(
        self.embedder_handle, dim=self.embedding_dim
    )

  @functools.cached_property
  def embedder(self) -> LoadedTextEmbedder:
    """The text embedding model. Loaded on first access."""
    embedder = _load_text_embedder(self.embedder_handle)
    if embedder.dim != self.embedding_dim:
      raise ValueError(
          f"Text embedder {self.embedder_handle!r} produces embeddings of"
          f" dimension {embedder.dim}, expected {self.embedding_dim}."
      )
    return embedder

  def output_schema(self) -> schema_lib.FeatureSetSchema:
    return {
        self.output_feature_name: schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.FLOAT_32,
            semantic=schema_lib.FeatureSemantic.EMBEDDING,
            shape=self.output_shape,
            is_timeseries=self.is_timeseries,
            group=self.group,
        )
    }

  def normalize_numpy(self, value: np.ndarray) -> dict[str, np.ndarray]:
    return {self.output_feature_name: self.embedder.embed_numpy(value)}

  def tensorflow_resources(self) -> list[tf.Tensor]:
    return [self.embedder]  # pyrefly: ignore[bad-return]

  def normalize_tensorflow(self, value: tf.Tensor) -> dict[str, tf.Tensor]:
    return {self.output_feature_name: self.embedder.embed_tensorflow(value)}

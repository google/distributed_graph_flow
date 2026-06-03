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

"""Analaysis / data extraction from a schema."""

from typing import Tuple, Union
from dgf.src.data import schema as schema_lib
from dgf.src.util import log


def infer_most_likely_primary_key_or_none(
    name: str,
    schema: schema_lib.NodeSchema | schema_lib.EdgeSchema,
) -> str | None:
  """Determines the most likely primary key in a schema or returns None.

  Args:
    name: The name of the NodeSet or EdgeSet.
    schema: The NodeSet or EdgeSet schema to search for a primary key.

  Returns:
    The name of the feature inferred to be the primary key, or None if no
    such feature is found.
  """
  candidate_names = ["#id", "#ID", "id", "ID"]
  matching_candidates = []
  for candidate in candidate_names:
    if candidate in schema.features:
      matching_candidates.append(candidate)

  if len(matching_candidates) > 1:
    raise ValueError(
        f"No primary feature found in the schema of {name!r} (i.e."
        " semantic=PRIMARY_ID), but multiple feature that look like primary"
        f" key found ({matching_candidates!r}). The available features are:"
        f" {schema.features.keys()!r}. Solution: Specify the primary key in the"
        " schema."
    )

  if len(matching_candidates) == 0:
    return None

  return matching_candidates[0]


def infer_most_likely_primary_key(
    name: str,
    schema: schema_lib.NodeSchema | schema_lib.EdgeSchema,
) -> str:
  """Determines the most likely primary key in a schema without one explicited.

  Args:
    name: The name of the NodeSet or EdgeSet.
    schema: The NodeSet or EdgeSet schema to search for a primary key.

  Returns:
    The name of the feature inferred to be the primary key.
  """
  pk = infer_most_likely_primary_key_or_none(name, schema)
  if pk is None:
    candidate_names = ["#id", "#ID", "id", "ID"]
    raise ValueError(
        f"No primary feature found in the schema of {name!r} (i.e."
        " semantic=PRIMARY_ID) and no feature look like a primary key"
        f" ({candidate_names!r}). The available features are:"
        f" {schema.features.keys()!r}. Solution: Specify the primary key in the"
        " schema."
    )
  return pk


def primary_feature(
    name: str,
    schema: schema_lib.NodeSchema | schema_lib.EdgeSchema,
) -> str:
  """Gets the primary feature of a NodeSet or EdgeSet schema.

  The primary feature is the one with semantic type PRIMARY_ID.

  Args:
    name: The name of the NodeSet or EdgeSet.
    schema: The NodeSet or EdgeSet schema to search for the primary feature.

  Returns:
    The name of the primary feature.
  """
  feature = primary_feature_or_none(name, schema)
  if feature is None:
    raise ValueError(
        f"No primary feature found in the schema of {name!r}. This operation"
        " requires for the schema to define a primary key i.e. a feature with"
        " semantic=PRIMARY_ID. The available features are:"
        f" {schema.features.keys()!r}."
    )
  return feature


def primary_feature_or_none(
    name: str,
    schema: schema_lib.NodeSchema | schema_lib.EdgeSchema,
) -> str | None:
  """Gets the primary feature of a NodeSet or EdgeSet schema, or None.

  The primary feature is the one with semantic type PRIMARY_ID.

  Args:
    name: The name of the NodeSet or EdgeSet.
    schema: The NodeSet or EdgeSet schema to search for the primary feature.

  Returns:
    The name of the primary feature, or None if no primary feature is found.
    Raises ValueError if multiple primary features are found.
  """

  primary_features = []
  for feature_name, feature_schema in schema.features.items():
    if feature_schema.semantic == schema_lib.FeatureSemantic.PRIMARY_ID:
      primary_features.append(feature_name)

  if not primary_features:
    return None
  if len(primary_features) > 1:
    raise ValueError(
        f"Multiple primary features found in the schema of {name!r}:"
        f" {[f for f in primary_features]}. Currently, DGF does not support"
        " multi-primary keys."
    )

  return primary_features[0]


def _fix_tf_example_formats(features: schema_lib.FeatureSetSchema):
  """Fix invalid format / dtype in tf-example based graphs."""
  # Note: tf-example only support int64, float32 and bytes.

  for feature_name, feature_schema in features.items():
    fmt = feature_schema.format
    if (
        fmt == schema_lib.FeatureFormat.INTEGER_32
        or fmt == schema_lib.FeatureFormat.BOOL
    ):
      feature_schema.format = schema_lib.FeatureFormat.INTEGER_64
      log.info(
          "Converted feature '%s' format from %s to INTEGER_64 for TF Example"
          " compatibility",
          feature_name,
          fmt,
      )
    elif fmt == schema_lib.FeatureFormat.FLOAT_64:
      feature_schema.format = schema_lib.FeatureFormat.FLOAT_32
      log.info(
          "Converted feature '%s' format from FLOAT_64 to FLOAT_32 for TF"
          " Example compatibility",
          feature_name,
      )


def fix_schema(
    schema: schema_lib.GraphSchema,
    create_pound_id_as_fall_back: bool = False,
    fix_shape: bool = True,
    tf_example: bool = False,
):
  """Tries to fix broken/invalid schemas by inferring and setting primary keys.

  Args:
    schema: The GraphSchema to be fixed. This object is modified in place.
    create_pound_id_as_fall_back: If true and no primary id can be found, create
      one called "#id". This should only be used to consume old GraphAI
      datasets. Note that create_pound_id_as_fall_back=True does not check that
      the feaure "#id" is actually present in the data.
    fix_shapes: If true, fixes the extra None dimension added to all the shapes.
      This is a common issue with TF-GNN schemas.
    tf_example: If true assumes the data comes from a TensorFlow Example
      container. This format can only contain float32, int64, and bytes
      (tf.string) values.
  """

  def shape_is_suspicious(shape: schema_lib.Shape):
    return (
        shape is not None
        and len(shape) == 2
        and shape[0] is None
        and shape[1] is not None
    )

  def fix_suspicious_shape(
      feature_name,
      shape: schema_lib.Shape,
  ) -> schema_lib.Shape:
    log.info("Fix suspicious shape of feature '%s'", feature_name)
    assert shape_is_suspicious(shape)
    if shape[1] == 1:
      return tuple()
    else:
      return shape[1:]

  for nodeset_name, nodeset_def in schema.node_sets.items():
    if tf_example:
      _fix_tf_example_formats(nodeset_def.features)
    all_shapes_are_suspicious = True
    for _, feature_schema in nodeset_def.features.items():
      all_shapes_are_suspicious = (
          all_shapes_are_suspicious
          and shape_is_suspicious(feature_schema.shape)
      )
    if fix_shape and all_shapes_are_suspicious:
      for feature_name, feature_schema in nodeset_def.features.items():
        feature_schema.shape = fix_suspicious_shape(
            feature_name, feature_schema.shape
        )

    if primary_feature_or_none(nodeset_name, nodeset_def) is not None:
      continue
    # This nodeset has not primary key.
    if create_pound_id_as_fall_back:
      primary_key = infer_most_likely_primary_key_or_none(
          nodeset_name, nodeset_def
      )
      if primary_key is None:
        primary_key = "#id"
        nodeset_def.features[primary_key] = schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BYTES
        )
    else:
      primary_key = infer_most_likely_primary_key(nodeset_name, nodeset_def)
    nodeset_def.features[primary_key].semantic = (
        schema_lib.FeatureSemantic.PRIMARY_ID
    )
    log.info(
        "Automatically set primary key for nodeset '%s' to '%s'",
        nodeset_name,
        primary_key,
    )

  for edgeset_name, edgeset_def in schema.edge_sets.items():
    if tf_example:
      _fix_tf_example_formats(edgeset_def.features)
    all_shapes_are_suspicious = True
    for _, feature_schema in edgeset_def.features.items():
      all_shapes_are_suspicious = (
          all_shapes_are_suspicious
          and shape_is_suspicious(feature_schema.shape)
      )
    if fix_shape and all_shapes_are_suspicious:
      for feature_name, feature_schema in edgeset_def.features.items():
        feature_schema.shape = fix_suspicious_shape(
            feature_name, feature_schema.shape
        )

    if primary_feature_or_none(edgeset_name, edgeset_def) is not None:
      continue
    # This nodeset has not primary key.
    primary_key = infer_most_likely_primary_key_or_none(
        edgeset_name, edgeset_def
    )

    if primary_key is None:
      # If it okay for an edgeset not to have a primary key.
      continue

    edgeset_def.features[primary_key].semantic = (
        schema_lib.FeatureSemantic.PRIMARY_ID
    )
    log.info(
        "Fixed schema. Set primary key for edgeset '%s' to '%s'",
        edgeset_name,
        primary_key,
    )


def infer_schema_semantic(
    schema: schema_lib.GraphSchema,
    raise_on_error: bool = True,
    verbose: bool = True,
) -> schema_lib.GraphSchema:
  """Automatically detects the semantic of features with UNKNOWN semantic.

  Usage example:

  ```python
  schema = dgf.analyse.infer_schema_semantic(schema)
  ```

  The logic to infer the semantic is as follows:
  - bytes and booleans are considered categorical.
  - numerical values (float, integers) are considered numerical.
  - integer values where the name starts with "is_" are considered categorical.
  - features where the name starts with # are ignored.

  Args:
    schema: The GraphSchema to infer semantics for.
    raise_on_error: If True, raises a ValueError if a feature's semantic cannot
      be inferred. If False, logs a warning instead.
    verbose: Print message about non-trivial decisions.

  Returns:
    The modified GraphSchema.
  """
  for nodeset_name, nodeset_def in schema.node_sets.items():
    _infer_features_semantic(
        nodeset_def.features,
        container_name=f"nodeset '{nodeset_name}'",
        raise_on_error=raise_on_error,
        verbose=verbose,
    )

  for edgeset_name, edgeset_def in schema.edge_sets.items():
    _infer_features_semantic(
        edgeset_def.features,
        container_name=f"edgeset '{edgeset_name}'",
        raise_on_error=raise_on_error,
        verbose=verbose,
    )

  return schema


def _infer_features_semantic(
    features: schema_lib.FeatureSetSchema,
    container_name: str,
    raise_on_error: bool,
    verbose: bool,
):
  for feature_name, feature_schema in features.items():
    if feature_name.startswith("#"):
      if verbose:
        log.info(
            "Ignoring feature %r in %s because it starts with '#'.",
            feature_name,
            container_name,
        )
      continue
    if feature_schema.semantic != schema_lib.FeatureSemantic.UNKNOWN:
      continue

    fmt = feature_schema.format
    inferred = False
    if fmt in (schema_lib.FeatureFormat.BYTES, schema_lib.FeatureFormat.BOOL):
      feature_schema.semantic = schema_lib.FeatureSemantic.CATEGORICAL
      inferred = True
    elif fmt.is_numerical():
      if fmt.is_integer() and feature_name.startswith("is_"):
        feature_schema.semantic = schema_lib.FeatureSemantic.CATEGORICAL
        if verbose:
          log.info(
              "Inferred feature %r in %s as CATEGORICAL because it starts with"
              " 'is_'.",
              feature_name,
              container_name,
          )
      else:
        feature_schema.semantic = schema_lib.FeatureSemantic.NUMERICAL
      inferred = True

    if not inferred:
      msg = (
          f"Could not infer semantic for feature {feature_name!r} in"
          f" {container_name} with format {fmt!r}. Please specify the semantic"
          " manually in the schema."
      )
      if raise_on_error:
        raise ValueError(
            f"{msg} To disable this error and print a warning instead, set"
            " `raise_on_error=False` in `infer_schema_semantic`."
        )
      else:
        log.warning(msg)

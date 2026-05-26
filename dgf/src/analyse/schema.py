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


def fix_schema(
    schema: schema_lib.GraphSchema,
    create_pound_id_as_fall_back: bool = False,
):
  """Tries to fix broken/invalid schemas by inferring and setting primary keys.

  Args:
    schema: The GraphSchema to be fixed. This object is modified in place.
    create_pound_id_as_fall_back: If true and no primary id can be found, create
      one called "#id". This should only be used to consume old GraphAI
      datasets. Note that create_pound_id_as_fall_back=True does not check that
      the feaure "#id" is actually present in the data.
  """

  for nodeset_name, nodeset_def in schema.node_sets.items():
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

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

"""Vertex AI deployment utilities for GraphFlow models."""

import tempfile
from typing import Any, Optional

from dgf.src.learning.ten_lines import common
from dgf.src.util import filesystem
from dgf.src.util import log
from dgf.src.util.weak_dep.weak_dep_tensorflow import tf  # pylint: disable=g-importing-member
from google.cloud.aiplatform import aiplatform
import yaml


DEFAULT_SERVING_CONTAINER_URI = (
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
)


def _create_batched_serving_model(tf_model: Any) -> Any:
  """Wraps a GraphFlow TF model to handle Vertex AI's batched JSON instances.

  Vertex AI sends prediction requests as a JSON list of instances, which the
  container parses into a batched tensor (e.g. adding a [None] outermost
  dimension). GraphFlow models natively expect unbatched 1D arrays. This wrapper
  strips the batch dimension on the input, passes it to the model, and adds a
  batch dimension to the output to satisfy Vertex AI's response contract.

  Args:
    tf_model: The raw TensorFlow callable exported from a GraphFlow model.

  Returns:
    A wrapped `tf.Module` with a concrete `__call__` signature accepting batched
    inputs.
  """

  class BatchedServingWrapper(tf.Module):
    """Module wrapper stripping outer batch dimension."""

    def __init__(self, raw_model):
      super().__init__()
      self.raw_model = raw_model

    @tf.function
    def __call__(self, **kwargs):
      # Vertex AI wraps instances in a batch dimension. We strip it here (v[0])
      unbatched_kwargs = {k: v[0] for k, v in kwargs.items()}
      out = self.raw_model(**unbatched_kwargs)
      # Re-add the batch dimension to the predictions
      return {"predictions": tf.expand_dims(out, axis=0)}

  batched_model = BatchedServingWrapper(tf_model)

  # Dynamically trace tf.function using TensorSpecs to avoid needing dummy data
  unbatched_spec = tf_model.__call__.structured_input_signature[1]
  batched_spec = {}
  for k, spec in unbatched_spec.items():
    batched_shape = [None] + spec.shape.as_list()
    batched_spec[k] = tf.TensorSpec(
        shape=batched_shape, dtype=spec.dtype, name=k
    )

  batched_model.__call__ = batched_model.__call__.get_concrete_function(
      **batched_spec
  )
  return batched_model


def _save_model_to_gcs(
    model: common.Model,
    target_gcs_dir: str,
    instance_schema: dict[str, Any],
    prediction_schema: dict[str, Any],
    project: Optional[str] = None,
) -> None:
  """Saves the model and serving schemas to GCS if not already present."""
  done_path = f"{target_gcs_dir}/DONE"
  model_uuid = model.metadata.uuid

  gcs_save_needed = True
  if filesystem.exists(done_path, project=project):
    log.info(
        "Step 1 (Save to GCS): SKIP - Model already exists in GCS (%s)",
        model_uuid,
    )
    gcs_save_needed = False
  elif filesystem.exists(target_gcs_dir, project=project):
    raise RuntimeError(
        f"Corrupted GCS directory state: Directory {target_gcs_dir} already "
        "contains files, but 'DONE' file is missing. This indicates a "
        "previous upload crashed or was interrupted mid-way. "
        "Please delete the directory in GCS and try again."
    )

  if gcs_save_needed:
    log.info("Step 1 (Save to GCS): EXECUTE - Saving model to GCS...")

    with tempfile.TemporaryDirectory() as tmp_dir:
      model.save(tmp_dir)

      # Export to TF SavedModel for Vertex AI
      tf_model = model.to_tensorflow_function(consume_tf_graph_dict=True)

      batched_model = _create_batched_serving_model(tf_model)
      tf.saved_model.save(batched_model, tmp_dir)

      # Upload all files from tmp_dir to GCS via filesystem utility
      filesystem.copy_dir(tmp_dir, target_gcs_dir, project=project)

    # 1. Instance Schema (Request)
    instance_schema_yaml = yaml.dump(instance_schema, sort_keys=False)
    instance_schema_path = f"{target_gcs_dir}/instance_schema.yaml"
    filesystem.write_text(
        instance_schema_path, instance_schema_yaml, project=project
    )
    log.info("Instance Schema YAML saved to %s", instance_schema_path)

    # 2. Prediction Schema (Response)
    pred_yaml_str = yaml.dump(prediction_schema, sort_keys=False)
    pred_schema_path = f"{target_gcs_dir}/prediction_schema.yaml"
    filesystem.write_text(pred_schema_path, pred_yaml_str, project=project)
    log.info("Prediction Schema YAML saved to %s", pred_schema_path)

    filesystem.write_text(done_path, "", project=project)
    log.info("Model saved to %s", target_gcs_dir)


def _upload_to_vertex_ai(
    model: common.Model,
    target_gcs_dir: str,
    display_name: str,
    serving_container_image_uri: str,
) -> Any:
  """Uploads the model to Vertex AI Model Registry or reuses an existing version."""
  model_uuid = model.metadata.uuid
  log.info("Step 2 (Upload to Vertex): Checking existing models...")
  existing_models = aiplatform.Model.list(
      filter=f'display_name="{display_name}"', order_by="create_time desc"
  )

  vertex_model = None
  parent_model_name = None
  if existing_models:
    parent_model_name = existing_models[0].resource_name
    for existing_model in existing_models:
      # Verify both the custom uuid label and GCS artifact URI match
      existing_uri = (existing_model.uri or "").rstrip("/")
      if (
          existing_model.labels
          and existing_model.labels.get("uuid") == model_uuid
          and existing_uri == target_gcs_dir.rstrip("/")
      ):
        log.info(
            "Step 2 (Upload to Vertex): SKIP - Model UUID (%s) and GCS URI"
            " match.",
            model_uuid,
        )
        vertex_model = existing_model
        break

  if not vertex_model:
    log.info(
        "Step 2 (Upload to Vertex): EXECUTE - Uploading new model version..."
    )
    upload_kwargs = {
        "display_name": display_name,
        "artifact_uri": target_gcs_dir,
        "serving_container_image_uri": serving_container_image_uri,
        "labels": {"uuid": model_uuid},
        "sync": True,
    }
    if parent_model_name:
      upload_kwargs["parent_model"] = parent_model_name
      upload_kwargs["is_default_version"] = True

    # Attach predict schemata
    upload_kwargs["instance_schema_uri"] = (
        f"{target_gcs_dir}/instance_schema.yaml"
    )
    upload_kwargs["prediction_schema_uri"] = (
        f"{target_gcs_dir}/prediction_schema.yaml"
    )

    vertex_model = aiplatform.Model.upload(**upload_kwargs)

  return vertex_model


def _get_or_create_endpoint(
    endpoint_id: Optional[str],
    display_name: str,
    use_dedicated_endpoint: bool,
) -> Any:
  """Retrieves an existing Vertex AI Endpoint or creates a new one."""
  if endpoint_id:
    log.info(
        "Step 3 (Create Endpoint): SKIP - Bypassing creation to use"
        " provided endpoint_id %s",
        endpoint_id,
    )
    try:
      endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)
    except Exception as e:
      raise RuntimeError(
          f"Failed to load Vertex AI Endpoint with ID '{endpoint_id}'. Please"
          " verify that the endpoint exists in your GCP project and location."
          f" Original error: {e}"
      ) from e
  else:
    log.info("Step 3 (Create Endpoint): Checking existing endpoints...")
    existing_endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{display_name}"', order_by="create_time desc"
    )
    if existing_endpoints:
      log.info(
          "Step 3 (Create Endpoint): SKIP - Endpoint '%s' already exists.",
          display_name,
      )
      # Note: Reusing an existing endpoint does not automatically update the
      # serving model version on Vertex AI's side. Instead, Step 4
      # (_deploy_model_to_endpoint) compares the target model's version_id
      # against currently deployed models on this endpoint; if a new version
      # was uploaded in Step 2, Step 4 will deploy it with 100% traffic and
      # undeploy older model versions.
      endpoint = existing_endpoints[0]
    else:
      log.info(
          "Step 3 (Create Endpoint): EXECUTE - Creating new endpoint..."
      )

      # Vertex AI 'Dedicated Endpoints' increase max payload limits to 10MB
      # and provide VPC isolation.
      endpoint_kwargs = {
          "display_name": display_name,
          "sync": True,
      }
      if use_dedicated_endpoint:
        endpoint_kwargs["dedicated_endpoint_enabled"] = True

      endpoint = aiplatform.Endpoint.create(**endpoint_kwargs)

  return endpoint


def _deploy_model_to_endpoint(
    endpoint: Any,
    vertex_model: Any,
    machine_type: str,
    blocking: bool,
) -> None:
  """Deploys the target model version to the endpoint and cleans up old versions."""
  log.info("Step 4 (Deploy Model): Checking current deployments...")
  deployed_models = endpoint.list_models()
  already_deployed = False
  matched_deployed_id = None
  existing_deployed_ids = {d.id for d in deployed_models}

  target_base = vertex_model.resource_name
  target_ver = vertex_model.version_id

  for deployed in deployed_models:
    dep_model = deployed.model or ""
    if "@" in dep_model:
      dep_base, dep_ver = dep_model.split("@", 1)
    else:
      dep_base = dep_model
      dep_ver = deployed.model_version_id

    same_version = (dep_base == target_base) and (
        target_ver is None or dep_ver is None or str(target_ver) == str(dep_ver)
    )
    if same_version:
      deployed_machine = None
      if deployed.dedicated_resources:
        deployed_machine = (
            deployed.dedicated_resources.machine_spec.machine_type
        )

      if deployed_machine == machine_type:
        already_deployed = True
        matched_deployed_id = deployed.id
        log.info(
            "Step 4 (Deploy Model): SKIP - Model version is already deployed"
            " with matching hardware."
        )
        break

  if not already_deployed:
    log.info(
        "Step 4 (Deploy Model): EXECUTE - Deploying model to endpoint (this"
        " usually takes 10-15 minutes)..."
    )
    endpoint.deploy(
        model=vertex_model,
        machine_type=machine_type,
        traffic_percentage=100,
        sync=blocking,
    )
    log.info(
        "Model deployed successfully to endpoint: %s", endpoint.resource_name
    )

  # Cleanup: Undeploy ALL old models to free up compute resources
  if already_deployed:
    for deployed in deployed_models:
      if deployed.id != matched_deployed_id:
        log.info(
            "Undeploying old model (ID: %s) to prevent resource drain...",
            deployed.id,
        )
        endpoint.undeploy(deployed_model_id=deployed.id, sync=False)
  elif blocking:
    deployed_models = endpoint.list_models()
    for deployed in deployed_models:
      if deployed.id in existing_deployed_ids:
        log.info(
            "Undeploying old model (ID: %s) to prevent resource drain...",
            deployed.id,
        )
        endpoint.undeploy(deployed_model_id=deployed.id, sync=False)
  else:
    log.info(
        "Skipping immediate undeployment of old models because blocking=False"
        " (deployment is in progress)."
    )


def to_vertex_ai(
    model: common.Model,
    *,
    model_dir_on_gcs: str,
    display_name: str,
    location: str,
    project: Optional[str] = None,
    machine_type: str = "n1-standard-4",
    use_dedicated_endpoint: bool = True,
    serving_container_image_uri: str = DEFAULT_SERVING_CONTAINER_URI,
    endpoint_id: Optional[str] = None,
    blocking: bool = True,
) -> Any:
  """Deploy a GraphFlow GNN model as an inference endpoint on Vertex AI.

  Usage example:
    ```
    graph, schema = dgf.io.read_graph("/tmp/my_hgraph")
    model = dgf.learning.train_node_model(graph=graph, schema=schema, ...)
    endpoint = dgf.deploy.to_vertex_ai(
        model,
        model_dir_on_gcs="gs://my-bucket/models",
        display_name="my_gnn_endpoint",
        location="us-central1",
        project="my-gcp-project",
    )
    # Run online predictions or inspect the deployed endpoint on Vertex AI:
    response = endpoint.predict(instances=[{...}])
    ```

  Args:
    model: A trained DGF model (e.g. NodePredictionModel). Note: calling this
      function populates `model.serving_function_signature` so that it is
      persisted with the SavedModel artifact.
    model_dir_on_gcs: Base GCS directory where models should be saved.
    display_name: Display name for the Vertex AI Model and Endpoint. Used
      together with `model.metadata.uuid` to identify existing model versions in
      Vertex Model Registry, and to look up or create the target Endpoint (if
      `endpoint_id` is not provided).
    location: GCP Region.
    project: GCP Project ID.
    machine_type: Machine type for the Vertex AI Endpoint.
    use_dedicated_endpoint: Whether to use dedicated resources.
    serving_container_image_uri: Docker image URI for Vertex AI prediction.
    endpoint_id: If provided, deploy to this existing endpoint ID.
    blocking: If True, waits for endpoint creation and model deployment to
      finish.

  Returns:
    The Vertex AI Endpoint object.

  Raises:
    ValueError: If the model does not have a UUID in its metadata.
    NotImplementedError: If the model does not support Vertex AI serving schema
      extraction.
    RuntimeError: If GCS state is corrupted or explicit endpoint lookup fails.
  """

  aiplatform.init(project=project, location=location)

  # -----------------------------------------------------------------
  # Pre-flight Checks
  # -----------------------------------------------------------------
  if not getattr(model, "metadata", None) or not model.metadata.uuid:
    raise ValueError(
        "Model does not have a UUID in its metadata. A UUID must be assigned"
        " during training to ensure proper lineage tracking."
    )

  instance_schema, prediction_schema = model._extract_serving_schemata()  # pylint: disable=protected-access
  # Populate serving_function_signature on the model instance so model.save()
  # writes serving_function_signature.yaml into the SavedModel directory.
  model.serving_function_signature = yaml.dump(
      instance_schema, default_flow_style=False, sort_keys=False
  )

  # -----------------------------------------------------------------
  # Step 1: Save Model to GCS
  # -----------------------------------------------------------------
  # Construct target GCS directory using UUID to prevent overwrites
  model_dir_on_gcs = model_dir_on_gcs.rstrip("/")
  target_gcs_dir = f"{model_dir_on_gcs}/{model.metadata.uuid}"

  _save_model_to_gcs(
      model,
      target_gcs_dir,
      instance_schema=instance_schema,
      prediction_schema=prediction_schema,
      project=project,
  )

  # -----------------------------------------------------------------
  # Step 2: Upload to Vertex AI
  # -----------------------------------------------------------------
  vertex_model = _upload_to_vertex_ai(
      model,
      target_gcs_dir,
      display_name,
      serving_container_image_uri,
  )

  # -----------------------------------------------------------------
  # Step 3: Create Endpoint
  # -----------------------------------------------------------------
  endpoint = _get_or_create_endpoint(
      endpoint_id, display_name, use_dedicated_endpoint
  )

  # -----------------------------------------------------------------
  # Step 4: Deploy Model
  # -----------------------------------------------------------------
  _deploy_model_to_endpoint(endpoint, vertex_model, machine_type, blocking)

  return endpoint


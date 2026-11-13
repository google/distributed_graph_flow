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

import concurrent.futures
import tempfile
import time
from typing import Any, Literal, Optional

from dgf.src.learning.ten_lines import common
from dgf.src.util import filesystem
from dgf.src.util import log
from dgf.src.util.weak_dep.weak_dep_aiplatform import aiplatform  # pylint: disable=g-importing-member
from dgf.src.util.weak_dep.weak_dep_tensorflow import tf  # pylint: disable=g-importing-member
import jax
import yaml


# Optimized TF runtimes, required by the jax2tf XlaCallModule ops.
_SERVING_CONTAINER_URIS = {
    "cpu": (
        "us-docker.pkg.dev/vertex-ai-restricted/prediction/tf_opt-cpu.nightly:latest"
    ),
    "gpu": (
        "us-docker.pkg.dev/vertex-ai-restricted/prediction/tf_opt-gpu.nightly:latest"
    ),
}
_GPU_ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
_MAX_UPLOAD_WAIT_SECONDS = 2 * 60 * 60
_UPLOAD_POLL_INTERVAL_SECONDS = 30


def _create_batched_serving_model(tf_model: Any, input_names: set[str]) -> Any:
  """Wraps a GraphFlow TF model to run on each batched Vertex AI instance.

  Args:
    tf_model: TF function consuming a single TF graph dict.
    input_names: Inputs of the instance schema. The other inputs of tf_model
      (i.e., the nodesets and edgesets not visited by the sampling plan) are fed
      with empty values.

  Returns:
    A TF module consuming a batch of instances.
  """
  unbatched_spec = tf_model.__call__.structured_input_signature[1]

  class BatchedServingWrapper(tf.Module):
    """Module wrapper running the model on each instance of the batch."""

    def __init__(self, raw_model):
      super().__init__()
      self.raw_model = raw_model

    @tf.function(autograph=False)
    def __call__(self, **kwargs):
      def predict_instance(idx):
        instance = {k: v[idx] for k, v in kwargs.items()}
        for k, spec in unbatched_spec.items():
          if k not in instance:
            instance[k] = tf.zeros([d or 0 for d in spec.shape], spec.dtype)
        return self.raw_model(**instance)

      predictions = tf.map_fn(
          predict_instance,
          tf.range(tf.shape(next(iter(kwargs.values())))[0]),
          fn_output_signature=tf.TensorSpec(shape=None, dtype=tf.float32),
      )
      return {"predictions": predictions}

  batched_model = BatchedServingWrapper(tf_model)
  batched_model.__call__ = batched_model.__call__.get_concrete_function(**{
      k: tf.TensorSpec(
          shape=[None] + unbatched_spec[k].shape.as_list(),
          dtype=unbatched_spec[k].dtype,
          name=unbatched_spec[k].name,
      )
      for k in sorted(input_names)
  })
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
      # TODO(gbm): Use `model.to_tensorflow_saved_model` (requires support for
      # Vertex's batched instances).
      previous_version = jax.config.jax_export_calling_convention_version
      jax.config.update("jax_export_calling_convention_version", 9)
      try:
        tf_model = model.to_tensorflow_function(input_format="TF_GRAPH_DICT")
        batched_model = _create_batched_serving_model(
            tf_model, set(instance_schema["properties"])
        )
        tf.saved_model.save(batched_model, tmp_dir)
      finally:
        jax.config.update(
            "jax_export_calling_convention_version", previous_version
        )

      # Upload all files from tmp_dir to GCS via filesystem utility
      filesystem.copy_local_dir(tmp_dir, target_gcs_dir, project=project)

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


def _find_uploaded_model(
    display_name: str,
    model_uuid: str | None,
    target_gcs_dir: str,
    serving_container_image_uri: str,
) -> tuple[Any, str | None]:
  """Finds an uploaded model matching the UUID, GCS URI, and serving container."""
  existing_models = aiplatform.Model.list(
      filter=f'display_name="{display_name}"', order_by="create_time desc"
  )
  if not existing_models:
    return None, None
  for existing_model in existing_models:
    existing_uri = (existing_model.uri or "").rstrip("/")
    container_spec = existing_model.container_spec
    if (
        existing_model.labels
        and existing_model.labels.get("uuid") == model_uuid
        and existing_uri == target_gcs_dir.rstrip("/")
        and container_spec is not None
        and container_spec.image_uri == serving_container_image_uri
    ):
      return existing_model, existing_models[0].resource_name
  return None, existing_models[0].resource_name


def _upload_to_vertex_ai(
    model: common.Model,
    target_gcs_dir: str,
    display_name: str,
    serving_container_image_uri: str,
) -> Any:
  """Uploads the model to Vertex AI Model Registry or reuses an existing version."""
  model_uuid = model.metadata.uuid
  log.info("Step 2 (Upload to Vertex): Checking existing models...")
  vertex_model, parent_model_name = _find_uploaded_model(
      display_name, model_uuid, target_gcs_dir, serving_container_image_uri
  )
  if vertex_model:
    log.info(
        "Step 2 (Upload to Vertex): SKIP - Model UUID (%s), GCS URI and"
        " serving container match.",
        model_uuid,
    )
    return vertex_model

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

  try:
    return aiplatform.Model.upload(**upload_kwargs)
  except concurrent.futures.TimeoutError:
    # Model.upload times out after 15 minutes; poll until the upload finishes.
    log.info(
        "Step 2 (Upload to Vertex): The upload is taking longer than"
        " expected. Waiting for the model to appear in the Model Registry..."
    )
  deadline = time.time() + _MAX_UPLOAD_WAIT_SECONDS
  while time.time() < deadline:
    time.sleep(_UPLOAD_POLL_INTERVAL_SECONDS)
    vertex_model, _ = _find_uploaded_model(
        display_name, model_uuid, target_gcs_dir, serving_container_image_uri
    )
    if vertex_model:
      return vertex_model
  raise TimeoutError(
      f"The model {model_uuid} was not uploaded to the Vertex AI Model"
      f" Registry within {_MAX_UPLOAD_WAIT_SECONDS} seconds."
  )


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
    target: Literal["cpu", "gpu"],
    blocking: bool,
) -> None:
  """Deploys the target model version to the endpoint and cleans up old versions."""
  log.info("Step 4 (Deploy Model): Checking current deployments...")
  deployed_models = endpoint.list_models()
  already_deployed = False
  matched_deployed_id = None
  existing_deployed_ids = {d.id for d in deployed_models}

  target_version = vertex_model.version_id

  for deployed in deployed_models:
    deployed_model_name = deployed.model or ""
    if "@" in deployed_model_name:
      deployed_base, deployed_version = deployed_model_name.split("@", 1)
    else:
      deployed_base = deployed_model_name
      deployed_version = deployed.model_version_id

    same_version = (deployed_base == vertex_model.resource_name) and (
        target_version is None
        or deployed_version is None
        or str(target_version) == str(deployed_version)
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
        accelerator_type=_GPU_ACCELERATOR_TYPE if target == "gpu" else None,
        accelerator_count=1 if target == "gpu" else None,
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
    target: Literal["cpu", "gpu"] = "cpu",
    use_dedicated_endpoint: bool = True,
    serving_container_image_uri: Optional[str] = None,
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
    model: A trained DGF model (e.g. NodePredictionModel or
      LinkPredictionModel). Note: calling this function populates
      `model.serving_function_signature` so that it is persisted with the
      SavedModel artifact.
    model_dir_on_gcs: Base GCS directory where models should be saved.
    display_name: Display name for the Vertex AI Model and Endpoint. Used
      together with `model.metadata.uuid` to identify existing model versions in
      Vertex Model Registry, and to look up or create the target Endpoint (if
      `endpoint_id` is not provided).
    location: GCP Region.
    project: GCP Project ID.
    machine_type: Machine type for the Vertex AI Endpoint.
    target: Hardware running the model. With "gpu", a NVIDIA T4 GPU is attached
      to the machine (`machine_type` should be a N1 machine).
    use_dedicated_endpoint: Whether to use dedicated resources.
    serving_container_image_uri: Docker image URI for Vertex AI prediction.
      Defaults to the optimized TensorFlow runtime for `target`.
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
  if serving_container_image_uri is None:
    serving_container_image_uri = _SERVING_CONTAINER_URIS[target]

  # Pre-flight Checks
  if not model.metadata or not model.metadata.uuid:
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

  # Step 1: Save Model to GCS
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

  # Step 2: Upload to Vertex AI
  vertex_model = _upload_to_vertex_ai(
      model,
      target_gcs_dir,
      display_name,
      serving_container_image_uri,
  )

  # Step 3: Create Endpoint
  endpoint = _get_or_create_endpoint(
      endpoint_id, display_name, use_dedicated_endpoint
  )

  # Step 4: Deploy Model
  _deploy_model_to_endpoint(
      endpoint, vertex_model, machine_type, target, blocking
  )

  return endpoint

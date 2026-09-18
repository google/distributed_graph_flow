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

"""Hermetic unit tests for dgf.src.deploy.vertex covering all 6 Design Doc cases."""

import types
from typing import Any, Optional
from unittest import mock
from absl.testing import absltest
from dgf.src.deploy import vertex
from dgf.src.learning.ten_lines import common
from dgf.src.util.weak_dep.weak_dep_tensorflow import tf  # pylint: disable=g-importing-member
import yaml


class StatefulGCPMock:
  """Stateful mock of GCS and Vertex AI (Model Registry + Endpoints) per region."""

  def __init__(self):
    self.current_location = "us-central1"
    self.current_project = "test-project"
    # (bucket_name, blob_path) -> str content
    self.gcs_blobs = {}
    self.gcs_upload_calls = 0

    # (location, display_name) -> list of mock Model objects (newest first)
    self.models_by_name = {}
    self.model_upload_calls = []

    # (location, endpoint_id) -> mock Endpoint object
    self.endpoints_by_id = {}
    self.endpoint_create_calls = []
    self.endpoint_list_calls = 0
    self.deploy_calls = []
    self.undeploy_calls = []
    self._id_counter = 1000

  def _next_id(self) -> str:
    self._id_counter += 1
    return str(self._id_counter)

  def init_aiplatform(self, project=None, location=None):
    if project:
      self.current_project = project
    if location:
      self.current_location = location

  def create_storage_client(self, project=None):
    del project
    mock_client = mock.MagicMock()

    def get_bucket(bucket_name):
      mock_bucket = mock.MagicMock()

      def get_blob(blob_path):
        mock_blob = mock.MagicMock()
        key = (bucket_name, blob_path)
        mock_blob.exists.side_effect = lambda: key in self.gcs_blobs
        mock_blob.download_as_text.side_effect = lambda: self.gcs_blobs[key]

        def upload_str(content):
          self.gcs_blobs[key] = content
          self.gcs_upload_calls += 1

        def upload_file(local_path):
          self.gcs_blobs[key] = f"file:{local_path}"
          self.gcs_upload_calls += 1

        mock_blob.upload_from_string.side_effect = upload_str
        mock_blob.upload_from_filename.side_effect = upload_file
        return mock_blob

      def list_blobs(prefix="", max_results=None):
        del max_results
        matches = []
        for b_name, b_path in self.gcs_blobs:
          if b_name == bucket_name and b_path.startswith(prefix):
            matches.append(mock.MagicMock(name=b_path))
        return matches

      mock_bucket.blob.side_effect = get_blob
      mock_bucket.list_blobs.side_effect = list_blobs
      return mock_bucket

    mock_client.bucket.side_effect = get_bucket
    return mock_client

  def model_list(self, filter=None, order_by=None):  # pylint: disable=redefined-builtin
    del order_by
    # filter is of form: display_name="foo"
    display_name = (filter or "").split('"')[1]
    key = (self.current_location, display_name)
    return list(self.models_by_name.get(key, []))

  def model_upload(
      self,
      display_name,
      artifact_uri,
      serving_container_image_uri,
      labels=None,
      sync=True,
      parent_model=None,
      is_default_version=False,
      instance_schema_uri=None,
      prediction_schema_uri=None,
  ):
    self.model_upload_calls.append({
        "location": self.current_location,
        "display_name": display_name,
        "artifact_uri": artifact_uri,
        "labels": labels,
        "parent_model": parent_model,
        "is_default_version": is_default_version,
        "instance_schema_uri": instance_schema_uri,
        "prediction_schema_uri": prediction_schema_uri,
        "serving_container_image_uri": serving_container_image_uri,
        "sync": sync,
    })

    key = (self.current_location, display_name)
    existing = self.models_by_name.setdefault(key, [])
    if parent_model:
      base_resource_name = parent_model
      version_id = str(len(existing) + 1)
    else:
      model_id = self._next_id()
      base_resource_name = (
          f"projects/{self.current_project}/locations/{self.current_location}"
          f"/models/{model_id}"
      )
      version_id = "1"

    mock_model = types.SimpleNamespace(
        resource_name=base_resource_name,
        version_id=version_id,
        uri=artifact_uri,
        labels=dict(labels or {}),
        display_name=display_name,
    )
    # Insert at front so existing[0] is latest version
    existing.insert(0, mock_model)
    return mock_model

  def _make_endpoint_obj(self, endpoint_id, display_name, location):
    resource_name = (
        f"projects/{self.current_project}/locations/{location}"
        f"/endpoints/{endpoint_id}"
    )
    ep = types.SimpleNamespace(
        name=endpoint_id,
        resource_name=resource_name,
        display_name=display_name,
        location=location,
        _deployed_models=[],
    )

    def list_models():
      return list(ep._deployed_models)

    def deploy(
        model, machine_type="n1-standard-4", traffic_percentage=100, sync=True
    ):
      dep_id = f"dep-{self._next_id()}"
      self.deploy_calls.append({
          "endpoint": resource_name,
          "model": model.resource_name,
          "version_id": model.version_id,
          "machine_type": machine_type,
          "traffic_percentage": traffic_percentage,
          "sync": sync,
      })
      dep_obj = types.SimpleNamespace(
          id=dep_id,
          model=model.resource_name,
          model_version_id=str(model.version_id),
          dedicated_resources=types.SimpleNamespace(
              machine_spec=types.SimpleNamespace(machine_type=machine_type)
          ),
      )
      ep._deployed_models.append(dep_obj)

    def undeploy(deployed_model_id, sync=False):
      self.undeploy_calls.append({
          "endpoint": resource_name,
          "deployed_model_id": deployed_model_id,
          "sync": sync,
      })
      ep._deployed_models = [
          d for d in ep._deployed_models if d.id != deployed_model_id
      ]

    ep.list_models = list_models
    ep.deploy = deploy
    ep.undeploy = undeploy
    return ep

  def endpoint_create(
      self, display_name, sync=True, dedicated_endpoint_enabled=False
  ):
    endpoint_id = self._next_id()
    self.endpoint_create_calls.append({
        "location": self.current_location,
        "display_name": display_name,
        "endpoint_id": endpoint_id,
        "sync": sync,
        "dedicated_endpoint_enabled": dedicated_endpoint_enabled,
    })
    ep = self._make_endpoint_obj(
        endpoint_id, display_name, self.current_location
    )
    self.endpoints_by_id[(self.current_location, endpoint_id)] = ep
    return ep

  def endpoint_list(self, filter=None, order_by=None):  # pylint: disable=redefined-builtin
    del order_by
    self.endpoint_list_calls += 1
    display_name = (filter or "").split('"')[1]
    matches = [
        ep
        for (loc, _), ep in self.endpoints_by_id.items()
        if loc == self.current_location and ep.display_name == display_name
    ]
    return matches

  def endpoint_ctor(self, endpoint_name):
    key = (self.current_location, endpoint_name)
    if key not in self.endpoints_by_id:
      raise ValueError(
          f"Endpoint {endpoint_name} not found in {self.current_location}"
      )
    return self.endpoints_by_id[key]


class FakeGNNModel(common.Model):
  """Lightweight fake DGF model for hermetic deployment testing."""

  def __init__(self, model_uuid: str):
    super().__init__(data=None)
    self.metadata = common.Metadata(name="FakeGNNModel", uuid=model_uuid)

  @classmethod
  def name(cls) -> str:
    return "FakeGNNModel"

  def describe(self):
    return None

  def data(self):
    return None

  def _internal_save(self, path: str) -> None:
    pass

  def _internal_load(self, path: str) -> None:
    pass

  def save(self, tmp_dir: str):
    # Write a dummy file into tmp_dir so os.walk finds an artifact to upload
    with open(f"{tmp_dir}/saved_model.pb", "w") as f:
      f.write("dummy_model")

  def to_tensorflow_function(
      self,
      *,
      input_format: Any = None,
      consume_tf_graph_dict: Optional[bool] = None,
  ) -> Any:
    del input_format, consume_tf_graph_dict
    return mock.MagicMock()

  def _extract_serving_schemata(
      self,
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    sig = {
        "title": f"NodePrediction_TestGraph_user_{self.metadata.uuid}",
        "type": "object",
        "required": ["gnn_user_seed_node_idxs"],
        "x-google-graph": "TestGraph",
        "x-google-gnn-input-graphs": [{
            "input_node": "user",
            "sampling_plan": [{
                "edge": "follows",
                "width": 5,
            }],
        }],
        "properties": {"age": {"type": "integer"}},
    }
    pred_schema = {
        "type": "object",
        "properties": {"predictions": {"type": "array"}},
    }
    return sig, pred_schema


class VertexDeployTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.gcp = StatefulGCPMock()

    # Patch GCP & TF API calls with MagicMocks backed by StatefulGCPMock
    self.mock_aiplatform_init = self.enter_context(
        mock.patch.object(
            vertex.aiplatform, "init", side_effect=self.gcp.init_aiplatform
        )
    )
    self.mock_storage_client = self.enter_context(
        mock.patch.object(
            vertex.filesystem.storage,
            "Client",
            side_effect=self.gcp.create_storage_client,
        )
    )
    self.mock_model_list = self.enter_context(
        mock.patch.object(
            vertex.aiplatform.Model, "list", side_effect=self.gcp.model_list
        )
    )
    self.mock_model_upload = self.enter_context(
        mock.patch.object(
            vertex.aiplatform.Model, "upload", side_effect=self.gcp.model_upload
        )
    )
    self.mock_endpoint_cls = self.enter_context(
        mock.patch.object(
            vertex.aiplatform, "Endpoint", side_effect=self.gcp.endpoint_ctor
        )
    )
    self.mock_endpoint_create = mock.MagicMock(
        side_effect=self.gcp.endpoint_create
    )
    self.mock_endpoint_list = mock.MagicMock(side_effect=self.gcp.endpoint_list)
    self.mock_endpoint_cls.create = self.mock_endpoint_create
    self.mock_endpoint_cls.list = self.mock_endpoint_list

    self._real_create_batched_serving_model = (
        vertex._create_batched_serving_model
    )  # pylint: disable=protected-access
    self.mock_batched_wrapper = self.enter_context(
        mock.patch.object(
            vertex,
            "_create_batched_serving_model",
            return_value=mock.MagicMock(),
        )
    )
    self.mock_tf_saved_model_save = self.enter_context(
        mock.patch("tensorflow.saved_model.save")
    )

  def _reset_api_mocks(self):
    """Resets call histories on all mocked GCP/TF API functions between steps."""
    self.mock_aiplatform_init.reset_mock()
    self.mock_storage_client.reset_mock()
    self.mock_model_list.reset_mock()
    self.mock_model_upload.reset_mock()
    self.mock_endpoint_cls.reset_mock()
    self.mock_endpoint_create.reset_mock()
    self.mock_endpoint_list.reset_mock()
    self.mock_tf_saved_model_save.reset_mock()

  def _deploy_baseline(self):
    """Helper to establish the Case 0 baseline state in the mock GCP backend."""
    model = FakeGNNModel("uuid-1")
    ep0 = vertex.to_vertex_ai(
        model=model,
        model_dir_on_gcs="gs://test-bucket/dir1",
        display_name="gnn-model-v1",
        location="us-central1",
        machine_type="n2-standard-4",
        blocking=True,
    )
    self._reset_api_mocks()
    return model, ep0

  def test_preflight_validation_checks(self):
    model_no_uuid = FakeGNNModel("uuid-1")
    model_no_uuid.metadata.uuid = None
    with self.assertRaisesRegex(ValueError, "Model does not have a UUID"):
      vertex.to_vertex_ai(
          model=model_no_uuid,
          model_dir_on_gcs="gs://bucket/dir",
          display_name="test-model",
          location="us-central1",
      )

    class UnsupportedModel(FakeGNNModel):

      def _extract_serving_schemata(self):
        return super(FakeGNNModel, self)._extract_serving_schemata()

    model_no_sig = UnsupportedModel("uuid-1")
    with self.assertRaisesRegex(
        NotImplementedError, "does not support Vertex AI serving schema"
    ):
      vertex.to_vertex_ai(
          model=model_no_sig,
          model_dir_on_gcs="gs://bucket/dir",
          display_name="test-model",
          location="us-central1",
      )

  def test_corrupted_gcs_state_raises_error(self):
    model = FakeGNNModel("uuid-1")
    self.gcp.gcs_blobs[("my-bucket", "dir/uuid-1/partial_file.pb")] = (
        "partial-data"
    )
    with self.assertRaisesRegex(
        RuntimeError,
        "Corrupted GCS directory state: Directory gs://my-bucket/dir/uuid-1"
        " already contains files, but 'DONE' file is missing.",
    ):
      vertex.to_vertex_ai(
          model=model,
          model_dir_on_gcs="gs://my-bucket/dir",
          display_name="test-model",
          location="us-central1",
      )

  def test_case_0_initial_deploy_and_idempotency_check(self):
    model = FakeGNNModel("uuid-1")

    # 1. Initial Deploy (All 4 steps EXECUTE via mocked APIs)
    ep0 = vertex.to_vertex_ai(
        model=model,
        model_dir_on_gcs="gs://test-bucket/dir1",
        display_name="gnn-model-v1",
        location="us-central1",
        machine_type="n2-standard-4",
        blocking=True,
    )
    self.mock_tf_saved_model_save.assert_called_once()
    self.mock_model_upload.assert_called_once()
    self.assertIsNone(
        self.mock_model_upload.call_args.kwargs.get("parent_model")
    )
    self.mock_endpoint_create.assert_called_once_with(
        display_name="gnn-model-v1",
        sync=True,
        dedicated_endpoint_enabled=True,
    )
    self.assertLen(self.gcp.deploy_calls, 1)

    # 2. Idempotency Check (Re-run with NO changes -> All 4 steps SKIP)
    self._reset_api_mocks()
    ep0b = vertex.to_vertex_ai(
        model=model,
        model_dir_on_gcs="gs://test-bucket/dir1",
        display_name="gnn-model-v1",
        location="us-central1",
        machine_type="n2-standard-4",
        blocking=True,
    )
    self.assertEqual(ep0b.resource_name, ep0.resource_name)
    # Step 1 SKIP: TF SavedModel save not called
    self.mock_tf_saved_model_save.assert_not_called()
    # Step 2 SKIP: Model.upload not called
    self.mock_model_upload.assert_not_called()
    # Step 3 SKIP: Endpoint.create not called
    self.mock_endpoint_create.assert_not_called()
    # Step 4 SKIP: No new deploy or undeploy calls
    self.assertLen(self.gcp.deploy_calls, 1)
    self.assertEmpty(self.gcp.undeploy_calls)

  def test_case_1_model_retrained_new_uuid(self):
    """Case 1: `model` changes (new UUID) -> Step 1 EXEC, Step 2 EXEC (@2), Step 3 SKIP, Step 4 EXEC."""
    model, ep0 = self._deploy_baseline()
    model.metadata.uuid = "uuid-2"

    ep1 = vertex.to_vertex_ai(
        model=model,
        model_dir_on_gcs="gs://test-bucket/dir1",
        display_name="gnn-model-v1",
        location="us-central1",
        machine_type="n2-standard-4",
        blocking=True,
    )
    # Step 1 EXECUTED: Saved new model.UUID to GCS
    self.mock_tf_saved_model_save.assert_called_once()
    # Step 2 EXECUTED: Uploaded new model version under existing parent_model
    self.mock_model_upload.assert_called_once()
    self.assertIsNotNone(
        self.mock_model_upload.call_args.kwargs.get("parent_model")
    )
    self.assertTrue(
        self.mock_model_upload.call_args.kwargs.get("is_default_version")
    )
    # Step 3 SKIPPED: Reused existing endpoint
    self.mock_endpoint_create.assert_not_called()
    self.assertEqual(ep1.resource_name, ep0.resource_name)
    # Step 4 EXECUTED: Deployed version 2 and undeployed version 1
    self.assertLen(self.gcp.deploy_calls, 2)
    self.assertEqual(self.gcp.deploy_calls[-1]["version_id"], "2")
    self.assertLen(self.gcp.undeploy_calls, 1)

  def test_case_2_model_dir_on_gcs_changes(self):
    """Case 2: `model_dir_on_gcs` changes -> Step 1 EXEC, Step 2 EXEC (new URI), Step 3 SKIP, Step 4 EXEC."""
    model, ep0 = self._deploy_baseline()

    ep2 = vertex.to_vertex_ai(
        model=model,
        model_dir_on_gcs="gs://test-bucket/dir2",  # Changed GCS path
        display_name="gnn-model-v1",
        location="us-central1",
        machine_type="n2-standard-4",
        blocking=True,
    )
    # Step 1 EXECUTED: Saved to new GCS directory
    self.mock_tf_saved_model_save.assert_called_once()
    # Step 2 EXECUTED: Detected URI mismatch and uploaded new version
    self.mock_model_upload.assert_called_once()
    self.assertEqual(
        self.mock_model_upload.call_args.kwargs["artifact_uri"],
        "gs://test-bucket/dir2/uuid-1",
    )
    # Step 3 SKIPPED: Reused existing endpoint
    self.mock_endpoint_create.assert_not_called()
    self.assertEqual(ep2.resource_name, ep0.resource_name)
    # Step 4 EXECUTED: Deployed new version and cleaned up old version
    self.assertLen(self.gcp.deploy_calls, 2)
    self.assertLen(self.gcp.undeploy_calls, 1)

  def test_case_3_display_name_changes(self):
    """Case 3: `display_name` changes -> Step 1 SKIP, Step 2 EXEC, Step 3 EXEC, Step 4 EXEC."""
    model, ep0 = self._deploy_baseline()

    ep3 = vertex.to_vertex_ai(
        model=model,
        model_dir_on_gcs="gs://test-bucket/dir1",
        display_name="gnn-model-v2",  # Changed display_name
        location="us-central1",
        machine_type="n2-standard-4",
        blocking=True,
    )
    # Step 1 SKIPPED: GCS artifact already exists for uuid-1
    self.mock_tf_saved_model_save.assert_not_called()
    # Step 2 EXECUTED: Uploaded brand new Model resource (parent_model=None)
    self.mock_model_upload.assert_called_once()
    self.assertIsNone(
        self.mock_model_upload.call_args.kwargs.get("parent_model")
    )
    self.assertEqual(
        self.mock_model_upload.call_args.kwargs["display_name"], "gnn-model-v2"
    )
    # Step 3 EXECUTED: Created brand new Endpoint resource
    self.mock_endpoint_create.assert_called_once()
    self.assertNotEqual(ep3.resource_name, ep0.resource_name)
    # Step 4 EXECUTED: Deployed to new endpoint
    self.assertLen(self.gcp.deploy_calls, 2)
    self.assertEqual(self.gcp.deploy_calls[-1]["endpoint"], ep3.resource_name)

  def test_case_4_machine_type_changes(self):
    """Case 4: `machine_type` changes -> Step 1 SKIP, Step 2 SKIP, Step 3 SKIP, Step 4 EXEC."""
    model, ep0 = self._deploy_baseline()

    ep4 = vertex.to_vertex_ai(
        model=model,
        model_dir_on_gcs="gs://test-bucket/dir1",
        display_name="gnn-model-v1",
        location="us-central1",
        machine_type="n1-standard-8",  # Scaled hardware
        blocking=True,
    )
    # Steps 1, 2, 3 all SKIPPED
    self.mock_tf_saved_model_save.assert_not_called()
    self.mock_model_upload.assert_not_called()
    self.mock_endpoint_create.assert_not_called()
    self.assertEqual(ep4.resource_name, ep0.resource_name)
    # Step 4 EXECUTED: Redeployed on n1-standard-8 after hardware mismatch
    self.assertLen(self.gcp.deploy_calls, 2)
    self.assertEqual(self.gcp.deploy_calls[-1]["machine_type"], "n1-standard-8")
    self.assertLen(self.gcp.undeploy_calls, 1)

  def test_case_5_explicit_endpoint_id_provided(self):
    """Case 5: `endpoint_id` provided -> Step 1 SKIP, Step 2 SKIP, Step 3 SKIP (bypass search), Step 4 EXEC."""
    model, _ = self._deploy_baseline()
    # Create a second standalone target endpoint to deploy onto via explicit ID
    target_ep = self.gcp.endpoint_create(display_name="explicit-target-ep")
    self._reset_api_mocks()

    ep5 = vertex.to_vertex_ai(
        model=model,
        model_dir_on_gcs="gs://test-bucket/dir1",
        display_name="gnn-model-v1",
        location="us-central1",
        machine_type="n2-standard-4",
        endpoint_id=target_ep.name,  # Explicit endpoint_id
        blocking=True,
    )
    # Steps 1 & 2 SKIPPED
    self.mock_tf_saved_model_save.assert_not_called()
    self.mock_model_upload.assert_not_called()
    # Step 3 SKIPPED creation AND bypassed Endpoint.list search
    self.mock_endpoint_create.assert_not_called()
    self.mock_endpoint_list.assert_not_called()
    self.mock_endpoint_cls.assert_called_once_with(endpoint_name=target_ep.name)
    self.assertEqual(ep5.resource_name, target_ep.resource_name)
    # Step 4 EXECUTED: Deployed onto target_ep
    self.assertLen(self.gcp.deploy_calls, 2)
    self.assertEqual(
        self.gcp.deploy_calls[-1]["endpoint"], target_ep.resource_name
    )

  def test_case_6_location_changes(self):
    """Case 6: `location` changes -> Step 1 SKIP (global GCS), Step 2 EXEC, Step 3 EXEC, Step 4 EXEC."""
    model, _ = self._deploy_baseline()

    ep6 = vertex.to_vertex_ai(
        model=model,
        model_dir_on_gcs="gs://test-bucket/dir1",
        display_name="gnn-model-v1",
        location="us-east1",  # Changed GCP region
        machine_type="n2-standard-4",
        blocking=True,
    )
    # Step 1 SKIPPED: GCS is global and uuid-1 already exists in bucket
    self.mock_tf_saved_model_save.assert_not_called()
    # Step 2 EXECUTED: Uploaded new model to us-east1 registry
    self.mock_model_upload.assert_called_once()
    self.assertEqual(self.gcp.model_upload_calls[-1]["location"], "us-east1")
    # Step 3 EXECUTED: Created new endpoint in us-east1
    self.mock_endpoint_create.assert_called_once()
    self.assertIn("/locations/us-east1/", ep6.resource_name)
    # Step 4 EXECUTED: Deployed to us-east1 endpoint
    self.assertLen(self.gcp.deploy_calls, 2)
    self.assertEqual(self.gcp.deploy_calls[-1]["endpoint"], ep6.resource_name)

  def test_instance_schema_yaml_contains_graph_and_sampling_config(self):
    """Verifies that instance_schema.yaml uploaded to GCS contains x-google-graph and x-google-gnn-input-graphs."""
    model = FakeGNNModel(model_uuid="uuid-graph-schema")
    vertex.to_vertex_ai(
        model=model,
        model_dir_on_gcs="gs://test-bucket/dir1",
        display_name="gnn-model-graph-schema",
        location="us-central1",
        blocking=True,
    )
    yaml_str = self.gcp.gcs_blobs.get(
        ("test-bucket", "dir1/uuid-graph-schema/instance_schema.yaml")
    )
    self.assertIsNotNone(yaml_str)
    parsed = yaml.safe_load(yaml_str)
    expected_yaml = {
        "title": f"NodePrediction_TestGraph_user_{model.metadata.uuid}",
        "type": "object",
        "required": ["gnn_user_seed_node_idxs"],
        "x-google-graph": "TestGraph",
        "x-google-gnn-input-graphs": [{
            "input_node": "user",
            "sampling_plan": [{
                "edge": "follows",
                "width": 5,
            }],
        }],
        "properties": {"age": {"type": "integer"}},
    }
    self.assertEqual(parsed, expected_yaml)

  def test_model_uuid_match_not_first_in_list(self):
    """Verifies that _upload_to_vertex_ai searches all matching models by UUID."""
    model_v1 = FakeGNNModel("uuid-v1")
    model_v2 = FakeGNNModel("uuid-v2")

    # Deploy v1 first, then v2 under the same display_name (so v2 is
    # existing_models[0] and v1 is existing_models[1]).
    vertex.to_vertex_ai(
        model=model_v1,
        model_dir_on_gcs="gs://test-bucket/dir1",
        display_name="shared-display-name",
        location="us-central1",
    )
    vertex.to_vertex_ai(
        model=model_v2,
        model_dir_on_gcs="gs://test-bucket/dir1",
        display_name="shared-display-name",
        location="us-central1",
    )

    # Now re-deploy v1; it should find v1 at index 1 in existing_models and
    # skip Model.upload.
    self._reset_api_mocks()
    vertex.to_vertex_ai(
        model=model_v1,
        model_dir_on_gcs="gs://test-bucket/dir1",
        display_name="shared-display-name",
        location="us-central1",
    )
    self.mock_model_upload.assert_not_called()

  def test_create_batched_serving_model(self):
    """Verifies _create_batched_serving_model strips input batch dim and adds output batch dim."""

    class DummyRawModel(tf.Module):

      @tf.function
      def __call__(self, **kwargs):
        return kwargs["x"] * 2.0

    raw_model = DummyRawModel()
    raw_model.__call__ = raw_model.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=[3], dtype=tf.float32, name="x")
    )

    batched_model = self._real_create_batched_serving_model(raw_model)
    out = batched_model.__call__(
        x=tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
    )
    self.assertIn("predictions", out)
    self.assertEqual(out["predictions"].shape.as_list(), [1, 3])
    self.assertSequenceAlmostEqual(
        out["predictions"].numpy().tolist()[0], [2.0, 4.0, 6.0]
    )


if __name__ == "__main__":
  absltest.main()

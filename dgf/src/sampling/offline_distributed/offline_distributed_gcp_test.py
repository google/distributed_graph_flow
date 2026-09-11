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

"""Unit tests for offline_distributed_gcp."""

import json
from unittest import mock
from absl.testing import absltest
from google.cloud.aiplatform import aiplatform
from dgf.src.sampling import config as config_lib
from dgf.src.sampling.offline_distributed import offline_distributed_gcp
from dgf.src.util import gen_test_graph
from dgf.src.util import log


class OfflineDistributedGcpTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_schema = gen_test_graph.generate_schema()
    self.simple_plan = config_lib.SimpleSamplingConfig(
        seed_nodeset="n1",
        num_hops=2,
        hop_width=10,
    )

  @mock.patch.object(offline_distributed_gcp.filesystem, "open_write")
  @mock.patch.object(aiplatform, "CustomJob")
  def test_offline_distributed_sampler_gcp_blocking_success(
      self, mock_custom_job_cls, mock_open_write
  ):
    mock_file = mock.MagicMock()
    mock_open_write.return_value.__enter__.return_value = mock_file
    mock_job = mock.MagicMock()
    mock_job.resource_name = (
        "projects/test-proj/locations/us-central1/customJobs/12345"
    )
    mock_job.state = aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED
    mock_custom_job_cls.return_value = mock_job

    with log.capture_logs(log_info=True) as logs:
      job = offline_distributed_gcp.offline_distributed_sampler_gcp(
          input_path="gs://my_bucket/graph",
          output_path="gs://my_bucket/samples",
          plan=self.simple_plan,
          schema=self.mock_schema,
          project="test-proj",
          region="us-central1",
          num_workers=3,
          num_seeds=500,
          blocking=True,
          poll_interval=0.01,
      )

    self.assertEqual(job, mock_job)
    mock_custom_job_cls.assert_called_once()
    _, kwargs = mock_custom_job_cls.call_args
    self.assertEqual(kwargs["project"], "test-proj")
    self.assertEqual(kwargs["location"], "us-central1")
    self.assertEqual(kwargs["staging_bucket"], "gs://my_bucket/samples_staging")
    self.assertEqual(kwargs["base_output_dir"], "gs://my_bucket/samples")

    specs = kwargs["worker_pool_specs"]
    self.assertLen(specs, 1)
    self.assertEqual(specs[0]["machine_spec"]["machine_type"], "n1-highmem-4")
    container_args = specs[0]["container_spec"]["args"]
    self.assertIn("--input_graph=gs://my_bucket/graph", container_args)
    self.assertIn("--output_samples=gs://my_bucket/samples", container_args)
    self.assertIn(
        "--sampling_config=gs://my_bucket/samples/sampling_config.json",
        container_args,
    )
    self.assertIn("--num_seeds=500", container_args)
    self.assertIn("--num_workers=3", container_args)
    self.assertIn("--max_num_workers=3", container_args)
    self.assertIn("--runner=dataflow", container_args)

    mock_job.submit.assert_called_once()
    mock_open_write.assert_called_once_with(
        "gs://my_bucket/samples/sampling_config.json"
    )
    written_data = "".join(call.args[0] for call in mock_file.write.call_args_list)
    parsed_plan = json.loads(written_data)
    self.assertEqual(parsed_plan["root"]["nodeset"], "n1")

    # Verify that console and logging URLs and duration are logged
    log_messages = " ".join(l.text for l in logs)
    self.assertIn(
        "https://console.cloud.google.com/vertex-ai/locations/us-central1/training/12345?project=test-proj",
        log_messages,
    )
    self.assertIn(
        "https://console.cloud.google.com/logs/viewer?project=test-proj&resource=ml_job%2Fjob_id%2F12345",
        log_messages,
    )
    self.assertIn("Total duration:", log_messages)

  @mock.patch.object(offline_distributed_gcp.filesystem, "open_write")
  @mock.patch.object(aiplatform, "CustomJob")
  def test_offline_distributed_sampler_gcp_state_transitions(
      self, mock_custom_job_cls, mock_open_write
  ):
    mock_open_write.return_value.__enter__.return_value = mock.MagicMock()
    mock_job = mock.MagicMock()
    mock_job.resource_name = (
        "projects/test-proj/locations/us-central1/customJobs/12345"
    )
    type(mock_job).state = mock.PropertyMock(
        side_effect=[
            aiplatform.gapic.JobState.JOB_STATE_PENDING,
            aiplatform.gapic.JobState.JOB_STATE_RUNNING,
            aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED,
        ]
    )
    mock_custom_job_cls.return_value = mock_job

    with log.capture_logs(log_info=True) as logs:
      job = offline_distributed_gcp.offline_distributed_sampler_gcp(
          input_path="gs://my_bucket/graph",
          output_path="gs://my_bucket/samples",
          plan=self.simple_plan,
          schema=self.mock_schema,
          project="test-proj",
          blocking=True,
          poll_interval=0.001,
      )

    self.assertEqual(job, mock_job)
    mock_job.submit.assert_called_once()
    log_messages = " ".join(l.text for l in logs)
    self.assertIn("Pending", log_messages)
    self.assertIn("Running", log_messages)
    self.assertIn("CustomJob completed successfully", log_messages)

  @mock.patch.object(offline_distributed_gcp.filesystem, "open_write")
  @mock.patch.object(aiplatform, "CustomJob")
  def test_offline_distributed_sampler_gcp_non_blocking(
      self, mock_custom_job_cls, mock_open_write
  ):
    mock_open_write.return_value.__enter__.return_value = mock.MagicMock()
    mock_job = mock.MagicMock()
    mock_job.resource_name = (
        "projects/test-proj/locations/us-central1/customJobs/12345"
    )
    mock_custom_job_cls.return_value = mock_job

    job = offline_distributed_gcp.offline_distributed_sampler_gcp(
        input_path="gs://my_bucket/graph",
        output_path="gs://my_bucket/samples",
        plan=self.simple_plan,
        schema=self.mock_schema,
        project="test-proj",
        blocking=False,
    )

    self.assertEqual(job, mock_job)
    mock_job.submit.assert_called_once()

  @mock.patch.object(offline_distributed_gcp.filesystem, "open_write")
  @mock.patch.object(aiplatform, "CustomJob")
  def test_offline_distributed_sampler_gcp_submit_failure(
      self, mock_custom_job_cls, mock_open_write
  ):
    mock_open_write.return_value.__enter__.return_value = mock.MagicMock()
    mock_job = mock.MagicMock()
    mock_job.submit.side_effect = RuntimeError("CustomJob submission error")
    mock_custom_job_cls.return_value = mock_job

    with log.capture_logs() as logs:
      with self.assertRaises(RuntimeError):
        offline_distributed_gcp.offline_distributed_sampler_gcp(
            input_path="gs://my_bucket/graph",
            output_path="gs://my_bucket/samples",
            plan=self.simple_plan,
            schema=self.mock_schema,
            project="test-proj",
            blocking=True,
        )

    self.assertTrue(any(l.severity == log.Severity.ERROR for l in logs))

  @mock.patch.object(offline_distributed_gcp.filesystem, "open_write")
  @mock.patch.object(aiplatform, "CustomJob")
  def test_offline_distributed_sampler_gcp_job_failure(
      self, mock_custom_job_cls, mock_open_write
  ):
    mock_open_write.return_value.__enter__.return_value = mock.MagicMock()
    mock_job = mock.MagicMock()
    mock_job.resource_name = (
        "projects/test-proj/locations/us-central1/customJobs/12345"
    )
    type(mock_job).state = mock.PropertyMock(
        return_value=aiplatform.gapic.JobState.JOB_STATE_FAILED
    )
    mock_job.error = "Out of memory error in worker"
    mock_custom_job_cls.return_value = mock_job

    with log.capture_logs() as logs:
      with self.assertRaises(RuntimeError):
        offline_distributed_gcp.offline_distributed_sampler_gcp(
            input_path="gs://my_bucket/graph",
            output_path="gs://my_bucket/samples",
            plan=self.simple_plan,
            schema=self.mock_schema,
            project="test-proj",
            blocking=True,
            poll_interval=0.001,
        )

    self.assertTrue(any(l.severity == log.Severity.ERROR for l in logs))

  @mock.patch.object(offline_distributed_gcp.io_schema, "read_schema")
  @mock.patch.object(offline_distributed_gcp.filesystem, "open_write")
  @mock.patch.object(aiplatform, "CustomJob")
  def test_plan_conversion_and_schema_loading(
      self, mock_custom_job_cls, mock_open_write, mock_read_schema
  ):
    mock_open_write.return_value.__enter__.return_value = mock.MagicMock()
    mock_job = mock.MagicMock()
    mock_job.state = aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED
    mock_custom_job_cls.return_value = mock_job
    mock_read_schema.return_value = self.mock_schema

    offline_distributed_gcp.offline_distributed_sampler_gcp(
        input_path="gs://my_bucket/graph",
        output_path="gs://my_bucket/samples",
        plan=self.simple_plan,
        schema=None,
        project="test-proj",
        blocking=True,
        poll_interval=0.001,
    )

    mock_read_schema.assert_called_once_with("gs://my_bucket/graph/schema.json")

  @mock.patch.object(offline_distributed_gcp, "_get_default_gcp_project")
  def test_missing_project_raises_error(self, mock_get_project):
    mock_get_project.return_value = None

    with self.assertRaises(ValueError):
      offline_distributed_gcp.offline_distributed_sampler_gcp(
          input_path="gs://my_bucket/graph",
          output_path="gs://my_bucket/samples",
          plan=self.simple_plan,
          schema=self.mock_schema,
          project=None,
      )

  @mock.patch.object(offline_distributed_gcp.filesystem, "open_write")
  @mock.patch.object(aiplatform, "CustomJob")
  def test_custom_parameters(self, mock_custom_job_cls, mock_open_write):
    mock_open_write.return_value.__enter__.return_value = mock.MagicMock()
    mock_job = mock.MagicMock()
    mock_custom_job_cls.return_value = mock_job

    offline_distributed_gcp.offline_distributed_sampler_gcp(
        input_path="gs://my_bucket/graph",
        output_path="gs://my_bucket/samples",
        plan=self.simple_plan,
        schema=self.mock_schema,
        project="custom-proj",
        region="europe-west1",
        num_workers=10,
        num_seeds=None,
        temp_location="gs://custom/temp",
        staging_location="gs://custom/staging",
        display_name="my-custom-display-name",
        blocking=False,
    )

    mock_custom_job_cls.assert_called_once()
    _, kwargs = mock_custom_job_cls.call_args
    self.assertEqual(kwargs["display_name"], "my-custom-display-name")
    self.assertEqual(kwargs["project"], "custom-proj")
    self.assertEqual(kwargs["location"], "europe-west1")
    self.assertEqual(kwargs["staging_bucket"], "gs://custom/staging")

    container_args = kwargs["worker_pool_specs"][0]["container_spec"]["args"]
    self.assertIn("--num_seeds=0", container_args)
    self.assertIn("--num_workers=10", container_args)
    self.assertIn("--max_num_workers=10", container_args)
    self.assertIn("--region=europe-west1", container_args)
    self.assertIn("--project=custom-proj", container_args)
    self.assertIn("--staging_location=gs://custom/staging", container_args)
    self.assertIn("--temp_location=gs://custom/temp", container_args)

  def test_invalid_input_path_raises_error(self):
    with self.assertRaisesRegex(
        ValueError, "input_path must be a Google Cloud Storage path"
    ):
      offline_distributed_gcp.offline_distributed_sampler_gcp(
          input_path="/local/path/to/graph",
          output_path="gs://my_bucket/samples",
          plan=self.simple_plan,
          schema=self.mock_schema,
          project="test-proj",
      )

  def test_invalid_output_path_raises_error(self):
    with self.assertRaisesRegex(
        ValueError, "output_path must be a Google Cloud Storage path"
    ):
      offline_distributed_gcp.offline_distributed_sampler_gcp(
          input_path="gs://my_bucket/graph",
          output_path="/local/path/to/samples",
          plan=self.simple_plan,
          schema=self.mock_schema,
          project="test-proj",
      )

  def test_format_duration(self):
    self.assertEqual(offline_distributed_gcp._format_duration(45), "45s")
    self.assertEqual(offline_distributed_gcp._format_duration(125), "2m 05s")
    self.assertEqual(offline_distributed_gcp._format_duration(3600), "60m 00s")

  def test_get_job_urls(self):
    console_url, logs_url = offline_distributed_gcp._get_job_urls(
        project="my-proj",
        region="us-central1",
        resource_name="projects/123/locations/us-central1/customJobs/456",
    )
    self.assertEqual(
        console_url,
        "https://console.cloud.google.com/vertex-ai/locations/us-central1/training/456?project=my-proj",
    )
    self.assertEqual(
        logs_url,
        "https://console.cloud.google.com/logs/viewer?project=my-proj&resource=ml_job%2Fjob_id%2F456",
    )

    empty_console, empty_logs = offline_distributed_gcp._get_job_urls(
        project="my-proj", region="us-central1", resource_name=""
    )
    self.assertEqual(
        empty_console,
        "https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=my-proj",
    )
    self.assertEqual(
        empty_logs,
        "https://console.cloud.google.com/logs/viewer?project=my-proj",
    )

  def test_get_job_stage_and_name(self):
    stage, name = offline_distributed_gcp._get_job_stage_and_name(
        aiplatform.gapic.JobState.JOB_STATE_PENDING
    )
    self.assertEqual(stage, 1)
    self.assertEqual(name, "Pending")

    stage, name = offline_distributed_gcp._get_job_stage_and_name(
        aiplatform.gapic.JobState.JOB_STATE_RUNNING
    )
    self.assertEqual(stage, 2)
    self.assertEqual(name, "Running")

    stage, name = offline_distributed_gcp._get_job_stage_and_name(
        aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED
    )
    self.assertEqual(stage, 3)
    self.assertEqual(name, "Succeeded")


if __name__ == "__main__":
  absltest.main()

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

"""Runs the distributed graph sampler on GCP using Vertex AI and Dataflow."""

import datetime
import os
import subprocess
import time
from typing import Any, Optional, Union

from dgf.src.data import schema as schema_lib
from dgf.src.io import schema as io_schema
from dgf.src.sampling import config as config_lib
from dgf.src.util import filesystem
from dgf.src.util import log
from google.cloud.aiplatform import aiplatform
import tqdm

_DEFAULT_IMAGE_URI = (
    "us-central1-docker.pkg.dev/graph-flow/glassbox-repo/sampler:latest"
)
_WORKER_BINARY = "/google3/third_party/py/dgf/src/bin/google/offline_distributed_sampling_gcp"
_WORKER_MACHINE_TYPE = "n1-highmem-4"
_DEFAULT_POLL_INTERVAL_SEC = 5.0

_CONSOLE_JOB_URL_TEMPLATE = "https://console.cloud.google.com/vertex-ai/locations/{region}/training/{job_id}?project={project}"
_CONSOLE_JOBS_LIST_URL_TEMPLATE = "https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project}"
_LOGGING_JOB_URL_TEMPLATE = "https://console.cloud.google.com/logs/viewer?project={project}&resource=ml_job%2Fjob_id%2F{job_id}"
_LOGGING_ROOT_URL_TEMPLATE = (
    "https://console.cloud.google.com/logs/viewer?project={project}"
)

_TERMINAL_FAILURE_STATES = {
    aiplatform.gapic.JobState.JOB_STATE_FAILED,
    aiplatform.gapic.JobState.JOB_STATE_CANCELLED,
    aiplatform.gapic.JobState.JOB_STATE_EXPIRED,
    aiplatform.gapic.JobState.JOB_STATE_PAUSED,
}

_STAGES = [
    "1/4: Job submitted",
    "2/4: Provisioning Vertex AI worker",
    "3/4: Running Dataflow distributed sampler",
    "4/4: Completed",
]


def _format_duration(seconds: float) -> str:
  """Formats a duration in seconds into human-readable format."""
  sec = int(seconds)
  if sec >= 60:
    return f"{sec // 60}m {sec % 60:02d}s"
  return f"{sec}s"


def _get_default_gcp_project() -> Optional[str]:
  """Gets the active GCP project from environment or gcloud config."""
  for env_var in (
      "GOOGLE_CLOUD_PROJECT",
      "CLOUDSDK_CORE_PROJECT",
      "GCP_PROJECT",
  ):
    if env_val := os.environ.get(env_var):
      return env_val
  try:
    res = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0 and res.stdout.strip():
      return res.stdout.strip()
  except Exception:  # pylint: disable=broad-except
    pass
  return None


def _validate_gcs_path(path: str, param_name: str) -> str:
  """Validates that a path is a Google Cloud Storage path starting with 'gs://'."""
  if not path.startswith("gs://"):
    msg = (
        f"{param_name} must be a Google Cloud Storage path starting with"
        f" 'gs://', got: '{path}'. You can copy data to a GCS bucket using:\n "
        " gcloud storage cp -r <local_path> gs://<bucket_name>/<path>"
    )
    log.error("%s", msg)
    raise ValueError(msg)
  return path.rstrip("/")


def _validate_paths(input_path: str, output_path: str) -> tuple[str, str]:
  """Validates and normalizes input and output GCS paths."""
  return (
      _validate_gcs_path(input_path, "input_path"),
      _validate_gcs_path(output_path, "output_path"),
  )


def _get_job_urls(
    project: str, region: str, resource_name: str
) -> tuple[str, str]:
  """Returns the Cloud Console and Cloud Logging URLs for a CustomJob."""
  job_id = resource_name.split("/")[-1] if resource_name else ""
  if job_id:
    console_url = _CONSOLE_JOB_URL_TEMPLATE.format(
        region=region, job_id=job_id, project=project
    )
    logs_url = _LOGGING_JOB_URL_TEMPLATE.format(project=project, job_id=job_id)
  else:
    console_url = _CONSOLE_JOBS_LIST_URL_TEMPLATE.format(project=project)
    logs_url = _LOGGING_ROOT_URL_TEMPLATE.format(project=project)
  return console_url, logs_url


def _get_job_stage_and_name(state: Any) -> tuple[int, str]:
  """Maps a JobState enum or string to a (stage_index, human_readable_name)."""
  state_enum = getattr(state, "name", str(state))
  state_name = state_enum.removeprefix("JOB_STATE_").capitalize()

  if (
      state == aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED
      or state_enum == "JOB_STATE_SUCCEEDED"
  ):
    return 3, state_name
  if (
      state == aiplatform.gapic.JobState.JOB_STATE_RUNNING
      or "RUNNING" in state_enum
  ):
    return 2, state_name
  return 1, state_name


def _write_sampling_plan(
    input_path: str,
    output_path: str,
    plan: Union[config_lib.SimpleSamplingConfig, config_lib.SamplingPlan],
    schema: Optional[schema_lib.GraphSchema],
) -> str:
  """Converts sampling config to plan if needed and writes it to GCS."""
  sampling_config_path = f"{output_path}/sampling_config.json"

  if isinstance(plan, config_lib.SimpleSamplingConfig):
    if schema is None:
      schema_path = f"{input_path}/schema.json"
      log.info("Reading graph schema from %s", schema_path)
      schema = io_schema.read_schema(schema_path)
    plan = config_lib.simple_sampling_config_to_sampling_plan(plan, schema)

  log.info("Writing sampling plan to %s", sampling_config_path)
  with filesystem.open_write(sampling_config_path) as f:
    f.write(plan.to_json(indent=2))  # pyrefly: ignore[missing-attribute]

  return sampling_config_path


def _create_custom_job(
    input_path: str,
    output_path: str,
    sampling_config_path: str,
    project: str,
    region: str,
    num_workers: int,
    num_seeds: Optional[int],
    staging_location: str,
    temp_location: str,
    display_name: str,
) -> aiplatform.CustomJob:
  """Builds and instantiates the Vertex AI CustomJob."""
  args = [
      f"--input_graph={input_path}",
      f"--output_samples={output_path}",
      f"--sampling_config={sampling_config_path}",
      f"--num_seeds={num_seeds if num_seeds is not None else 0}",
      "--runner=dataflow",
      f"--project={project}",
      f"--region={region}",
      f"--worker_machine_type={_WORKER_MACHINE_TYPE}",
      f"--num_workers={num_workers}",
      f"--max_num_workers={num_workers}",
      "--autoscaling_algorithm=NONE",
      f"--staging_location={staging_location}",
      f"--temp_location={temp_location}",
      "--environment_type=DOCKER",
      f"--sdk_container_image={_DEFAULT_IMAGE_URI}",
      f"--worker_binary={_WORKER_BINARY}",
  ]

  worker_pool_specs = [{
      "machine_spec": {
          "machine_type": _WORKER_MACHINE_TYPE,
      },
      "replica_count": 1,
      "container_spec": {
          "image_uri": _DEFAULT_IMAGE_URI,
          "args": args,
      },
  }]

  return aiplatform.CustomJob(
      display_name=display_name,
      worker_pool_specs=worker_pool_specs,
      project=project,
      location=region,
      staging_bucket=staging_location,
      base_output_dir=output_path,
  )


def _monitor_job(
    job: aiplatform.CustomJob,
    project: str,
    region: str,
    poll_interval: float,
) -> None:
  """Monitors a submitted CustomJob until completion with progress reporting."""
  resource_name = getattr(job, "resource_name", "") or ""
  _, logs_url = _get_job_urls(project, region, resource_name)

  pbar = tqdm.tqdm(
      total=len(_STAGES),
      desc=f"Distributed Sampler [{_STAGES[0]}]",
      unit="stage",
  )
  pbar.update(1)

  start_time = time.time()
  last_state_name = None
  current_stage = 1

  try:
    while True:
      state = job.state
      stage_idx, state_name = _get_job_stage_and_name(state)
      elapsed_str = _format_duration(time.time() - start_time)

      if state_name != last_state_name:
        log.info(
            "Job status changed to: %s (elapsed: %s)", state_name, elapsed_str
        )
        last_state_name = state_name

      if (
          state in _TERMINAL_FAILURE_STATES
          or "FAILED" in state_name.upper()
          or "CANCEL" in state_name.upper()
      ):
        pbar.close()
        job_error = getattr(job, "error", None)
        error_msg = (
            f"Vertex AI CustomJob '{resource_name}' failed with status"
            f" {state_name}. Error: {job_error}\nView logs: {logs_url}"
        )
        log.error("%s", error_msg)
        raise RuntimeError(error_msg)

      if (
          state == aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED
          or stage_idx == 3
      ):
        if current_stage < 3:
          pbar.update(3 - current_stage)
        pbar.set_description(f"Distributed Sampler [{_STAGES[3]}]")
        pbar.set_postfix_str(f"Success in {elapsed_str}")
        pbar.close()
        log.info(
            "CustomJob completed successfully: %s in %s",
            resource_name,
            elapsed_str,
        )
        break

      if stage_idx > current_stage:
        pbar.update(stage_idx - current_stage)
        current_stage = stage_idx

      pbar.set_description(f"Distributed Sampler [{_STAGES[stage_idx]}]")
      pbar.set_postfix_str(f"Status: {state_name} | Elapsed: {elapsed_str}")
      pbar.refresh()
      time.sleep(poll_interval)
  except Exception as e:
    pbar.close()
    if not isinstance(e, RuntimeError):
      log.error("CustomJob monitoring error: %s", e)
    raise


def offline_distributed_sampler_gcp(
    input_path: str,
    output_path: str,
    plan: Union[config_lib.SimpleSamplingConfig, config_lib.SamplingPlan],
    schema: Optional[schema_lib.GraphSchema] = None,
    *,
    blocking: bool = True,
    project: Optional[str] = None,
    region: str = "us-central1",
    num_workers: int = 5,
    num_seeds: Optional[int] = None,
    temp_location: Optional[str] = None,
    staging_location: Optional[str] = None,
    display_name: Optional[str] = None,
    poll_interval: float = _DEFAULT_POLL_INTERVAL_SEC,
) -> aiplatform.CustomJob:
  """Runs the offline distributed graph sampler on GCP.

  Submits a Vertex AI CustomJob running the Glassbox container image to
  execute the distributed sampling pipeline on Apache Beam / Dataflow.

  Usage example:

  ```python
  job = dgf.sampling.offline_distributed_sampler_gcp(
      input_path="gs://my_bucket/my_graph",
      output_path="gs://my_bucket/my_samples",
      plan=dgf.sampling.SimpleSamplingConfig(
          seed_nodeset="paper",
          num_hops=2,
          hop_width=10,
      ),
      blocking=True,
  )
  ```

  Args:
    input_path: GCS path to the input GraphFlow graph directory (must start with
      'gs://').
    output_path: GCS path to write sampled graphs and schema to (must start with
      'gs://').
    plan: Sampling configuration (`SimpleSamplingConfig` or `SamplingPlan`).
    schema: Optional graph schema. If None and `plan` is `SimpleSamplingConfig`,
      it is read from `{input_path}/schema.json`.
    blocking: If True, waits for the job to complete while logging progress.
    project: GCP project ID. If None, it is resolved from the environment.
    region: GCP region to run the Vertex AI job and Dataflow workers in.
    num_workers: Number of Dataflow workers.
    num_seeds: Optional number of seeds to sample. If None, samples all nodes.
    temp_location: Optional GCS temporary directory for Dataflow.
    staging_location: Optional GCS staging directory for Dataflow and Vertex AI.
    display_name: Optional display name for the Vertex AI CustomJob.
    poll_interval: Polling interval in seconds when `blocking=True`.

  Returns:
    The `google.cloud.aiplatform.CustomJob` instance.
  """
  input_path, output_path = _validate_paths(input_path, output_path)

  if project is None:
    project = _get_default_gcp_project()
    if not project:
      raise ValueError(
          "GCP project must be specified or configured in the environment."
      )
    log.info("Using GCP project: %s", project)

  if temp_location is None:
    temp_location = f"{output_path}_temp"
  if staging_location is None:
    staging_location = f"{output_path}_staging"
  if display_name is None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    display_name = f"dgf-distributed-sampler-glassbox-{timestamp}"

  start_time = time.time()
  sampling_config_path = _write_sampling_plan(
      input_path=input_path,
      output_path=output_path,
      plan=plan,
      schema=schema,
  )

  job = _create_custom_job(
      input_path=input_path,
      output_path=output_path,
      sampling_config_path=sampling_config_path,
      project=project,
      region=region,
      num_workers=num_workers,
      num_seeds=num_seeds,
      staging_location=staging_location,
      temp_location=temp_location,
      display_name=display_name,
  )

  log.info("Submitting Vertex AI CustomJob '%s'...", display_name)
  try:
    job.submit()
  except Exception as e:
    log.error("Failed to submit CustomJob: %s", e)
    raise

  resource_name = getattr(job, "resource_name", "") or ""
  console_url, logs_url = _get_job_urls(project, region, resource_name)
  log.info("Vertex AI CustomJob created: %s", resource_name or display_name)
  log.info("  Cloud Console: %s", console_url)
  log.info("  Cloud Logging: %s", logs_url)

  if blocking:
    _monitor_job(
        job=job,
        project=project,
        region=region,
        poll_interval=poll_interval,
    )
    total_duration = _format_duration(time.time() - start_time)
    log.info("Total duration: %s", total_duration)

  return job

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

"""Common utilities for supporting differnt types of Apache Beam Runners.
"""

import abc
import dataclasses
from typing import Any, Dict, Optional
import apache_beam as beam


def runner_from_options(options: Dict[str, Any]) -> beam.runners.PipelineRunner:
  """Returns a Beam runner based on the provided options."""
  return runner_from_name(options["runner"])


def runner_from_name(name: str) -> beam.runners.PipelineRunner:
  """Returns a Beam runner based on the provided name."""
  if name in ["DirectRunner", "DataflowRunner"]:
    return beam.runners.create_runner(name)
  else:
    raise ValueError(f"Unsupported runner: {name}")


def program_started(name: str):
  """Call this function at the beginning of all GraphFlow Beam jobs."""
  if name in ["DirectRunner", "DataflowRunner"]:
    print("Program Started.")
    return
  else:
    raise ValueError(f"Unsupported runner: {name}")


class RunnerBuildableConfig(abc.ABC):
  """Abstract base class for an Apache Beam runner configuration.
  """

  @abc.abstractmethod
  def to_options_dict(self) -> Dict[str, Any]:
    pass

  def make(self) -> beam.runners.PipelineRunner:
    pipeline_options = self.to_options_dict()
    return beam.runner.PipelineRunner(pipeline_options)


@dataclasses.dataclass(frozen=True)
class LocalRunnerConfig(RunnerBuildableConfig):
  """Configuration for running a pipeline locally (DirectRunner)."""

  def to_options_dict(self) -> Dict[str, Any]:
    """Returns options for the DirectRunner."""
    return {"runner": "DirectRunner"}


@dataclasses.dataclass(frozen=True)
class DataflowRunnerConfig(RunnerBuildableConfig):
  """Configuration for running a pipeline on Cloud Dataflow."""
  project: str
  region: str
  temp_location: str
  job_name: str

  # --- Optional but common parameters ---
  service_account_email: Optional[str] = None
  subnetwork: Optional[str] = None
  machine_type: Optional[str] = None
  num_workers: Optional[int] = None
  max_num_workers: Optional[int] = None

  # The modern option for Beam SDKs (2.30.0+)
  sdk_container_image: Optional[str] = None

  # The legacy option for older Beam SDKs
  worker_harness_container_image: Optional[str] = None

  # Path to a setup.py file for installing dependencies on workers
  setup_file: Optional[str] = None

  def to_options_dict(self) -> Dict[str, Any]:
    """Returns options for the DataflowRunner."""
    opts = {
        "runner": "DataflowRunner",
        "project": self.project,
        "region": self.region,
        "temp_location": self.temp_location,
        "job_name": self.job_name
    }

    # Fill in optional fields w/o overriding defaults.
    if self.service_account_email is not None:
      opts["service_account_email"] = self.service_account_email
    if self.subnetwork is not None:
      opts["subnetwork"] = self.subnetwork
    if self.machine_type is not None:
      opts["machine_type"] = self.machine_type
    if self.num_workers is not None:
      opts["num_workers"] = self.num_workers
    if self.max_num_workers is not None:
      opts["max_num_workers"] = self.max_num_workers
    if self.setup_file is not None:
      opts["setup_file"] = self.setup_file
    if self.sdk_container_image is not None:
      opts["sdk_container_image"] = self.sdk_container_image
    if self.worker_harness_container_image is not None:
      opts["worker_harness_container_image"] = (
          self.worker_harness_container_image
      )

    # Filter out any None values so they don't override Beam defaults
    return opts

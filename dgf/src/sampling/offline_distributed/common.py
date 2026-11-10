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

"""Utilities shared by the offline distributed samplers (GCP and Borg)."""

from dgf.src.data import schema as schema_lib
from dgf.src.io import schema as io_schema
from dgf.src.sampling import config as config_lib
from dgf.src.util import filesystem
from dgf.src.util import log

# Containers the input seeds can be read from. SSTable is excluded: it is a DAX
# sink that the sampler cannot read seeds from.
SUPPORTED_INPUT_SEED_CONTAINERS = frozenset({"TFRECORD", "RECORDIO"})

SAMPLING_CONFIG_FILENAME = "sampling_config.json"


def validate_seed_args(
    num_seeds: int | None,
    num_samples_per_seed: int,
    input_seed_container: str,
    filter_seed_node: str | None = None,
    input_seeds: str | None = None,
) -> None:
  """Validates the arguments controlling the selection of the seeds.

  Args:
    num_seeds: Number of seeds to select among the available ones, or None.
    num_samples_per_seed: Number of samples to generate for each seed.
    input_seed_container: Container of the input seeds.
    filter_seed_node: Optional filter on the seed nodes.
    input_seeds: Optional path to input seeds.

  Raises:
    ValueError: If the arguments are inconsistent.
  """
  if num_seeds is not None and num_seeds < 0:
    raise ValueError(f"num_seeds cannot be negative, got {num_seeds}.")
  if num_samples_per_seed < 1:
    raise ValueError(
        "num_samples_per_seed cannot be less than one, got"
        f" {num_samples_per_seed}."
    )
  if num_seeds and num_samples_per_seed > 1:
    raise ValueError(
        f"num_seeds ({num_seeds}) and num_samples_per_seed"
        f" ({num_samples_per_seed}) are exclusive: use num_seeds to generate"
        " fewer samples than there are seed nodes, and num_samples_per_seed to"
        " generate more."
    )
  if input_seed_container not in SUPPORTED_INPUT_SEED_CONTAINERS:
    raise ValueError(
        f"Unsupported input_seed_container {input_seed_container!r}. Supported"
        f" containers are {sorted(SUPPORTED_INPUT_SEED_CONTAINERS)}."
    )
  if filter_seed_node is not None and input_seeds is not None:
    raise ValueError(
        "filter_seed_node and input_seeds are exclusive: filter_seed_node only"
        " applies to the seed nodes taken from the root nodeset of the sampling"
        " plan."
    )


def write_sampling_plan(
    input_path: str,
    output_path: str,
    plan: config_lib.SimpleSamplingConfig | config_lib.SamplingPlan,
    schema: schema_lib.GraphSchema | None,
) -> str:
  """Converts the sampling config to a plan if needed and writes it."""
  sampling_config_path = f"{output_path}/{SAMPLING_CONFIG_FILENAME}"

  if isinstance(plan, config_lib.SimpleSamplingConfig):
    if schema is None:
      schema_path = f"{input_path}/schema.json"
      log.info("Reading graph schema from %s", schema_path)
      schema = io_schema.read_schema(schema_path)
    plan = config_lib.simple_sampling_config_to_sampling_plan(plan, schema)
  # Early check. The Go sampler validates the plan again.
  config_lib.check_supported_options(
      plan,
      schema,
      "offline distributed sampler",
      support_non_with_replacement=True,
      support_multi_visit=True,
      support_temporal_sampling=True,
      support_last_temporal_strategy=True,
  )

  if not filesystem.is_gcs_path(output_path):
    filesystem.makedirs(output_path)
  log.info("Writing sampling plan to %s", sampling_config_path)
  with filesystem.open_write(sampling_config_path) as f:
    f.write(plan.to_json(indent=2))  # pyrefly: ignore[missing-attribute]

  return sampling_config_path

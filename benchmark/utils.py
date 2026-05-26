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

"""Utilities for benchmarking."""

import abc
import dataclasses
import math
import time
from typing import List, Optional, Union

_NAME_WIDTH = 50
_TIME_WIDTH = 15


class Benchmark(abc.ABC):
  """A basic benchmark.

  The user is expected to implement either "run" + "num_units" (to control
  precicely the benchmark) or only "run_unit" (which is called multiple times
  for ~5 seconds).
  """

  automatic_num_units: int = -1
  max_runtime_seconds: float = 5
  unit_multiplicator: int = 1

  def setup(self):
    """Run once before 'run'."""
    pass

  def run(self):
    """Function to benchmark. Can be run multiple times."""

    # Estimate how often the timer should be checked to reduce timer checking
    # overhead. We want for the benchmark to run for "max_runtime_seconds" (e.g.
    #  5 seconds) and for the timer check to be done "optimal_split" times (to
    # avoid the overhead).
    # TODO(gbm): Do a binary search.
    num_estimation_runs = 5
    optimal_split = 10
    begin_estimation_time = time.perf_counter()
    for _ in range(num_estimation_runs):
      self.run_unit()
    estimation_duration = time.perf_counter() - begin_estimation_time
    timer_check_interval = math.ceil(
        min(
            100,
            self.max_runtime_seconds
            * num_estimation_runs
            / estimation_duration
            / optimal_split,
        )
    )
    print(f"Will check timer every {timer_check_interval} run units")

    num_units = 0
    begin = time.perf_counter()
    while True:
      for _ in range(timer_check_interval):
        self.run_unit()
      num_units += timer_check_interval * self.unit_multiplicator
      if time.perf_counter() - begin >= self.max_runtime_seconds:
        break
    print("num_units:", num_units)
    self.automatic_num_units = num_units

  def run_unit(self):
    """Function to benchmark. Can be run multiple times.

    If the user overrides "run", this method is not called.
    """
    pass

  def clean(self):
    """Run once after all the 'run' calls."""
    pass

  @abc.abstractmethod
  def name(self) -> str:
    """Name of the benchmark."""
    pass

  def num_units(self) -> int:
    """Num of units.

    Only need to be overriden if the user defines "run" instead of "run_unit".
    Used to show the time per units. Called after "clean".
    """
    return self.automatic_num_units

  def set_unit_multiplicator(self, unit_multiplicator: int):
    """Tells how many units are actually run when calling "run_unit".

    This allows to benchmark batch processing fairly.
    """
    self.unit_multiplicator = unit_multiplicator

  def details(self) -> str:
    """Extra details about the benchmark. Called after "clean"."""
    return ""


@dataclasses.dataclass
class BenchmarkResult:
  """Results of a benchark."""

  name: str
  num_units: Optional[int]
  details: str
  wall_time_seconds: float
  cpu_time_seconds: float

  def __str__(self):
    name_and_result = self.name + "; " + self.details + f" ({self.num_units})"
    return (
        f"{self.wall_time_seconds:{_TIME_WIDTH}.5f}    "
        f"{self.cpu_time_seconds:{_TIME_WIDTH}.5f}    "
        f"{self.wall_time_seconds / self.num_units:{_TIME_WIDTH}.5f}    "
        f"{self.num_units / self.wall_time_seconds:{_TIME_WIDTH}.5f}    "
        f"{name_and_result:<{_NAME_WIDTH}}"
    )


class Benchmarker:
  """Measure the execution time of a functions record the results.

  Usage example:

  ```python
  b = Benchmarker()

  # benchmark f1
  def f1():
    # something
    pass
  b.run("f1", f1)

  # benchmark f2
  def f2():
    # something
    pass
  b.run("f2", f2)

  # Print the results
  b.print_results()
  ```
  """

  def __init__(self):
    self._results: List[Union[BenchmarkResult, None]] = []

  def run(
      self,
      benchmark: Benchmark,
      repetitions: int,
      warmup_repetitions: int,
  ):
    """Measure the execution time of fun and print+record the results.

    Args:
      benchmark: The Benchmark instance to run.
      repetitions: The number of times to run the benchmark for measurement.
      warmup_repetitions: The number of times to run the benchmark for warmup
        before measurement.
    """

    print(f"Run {benchmark.name()}")
    print("\tsetup")
    begin_setup_time = time.perf_counter()
    benchmark.setup()
    end_setup_time = time.perf_counter()
    print(f"\t\tsetup time: {end_setup_time - begin_setup_time:.5f} seconds")

    if warmup_repetitions > 0:
      print("\twarmup")
    for _ in range(warmup_repetitions):
      benchmark.run()

    print("\trun")
    begin_wall_time = time.perf_counter()
    begin_cpu_time = time.process_time()
    for _ in range(repetitions):
      benchmark.run()
    end_wall_time = time.perf_counter()
    end_cpu_time = time.process_time()

    print("\tclean")
    benchmark.clean()

    result = BenchmarkResult(
        name=benchmark.name(),
        num_units=benchmark.num_units(),
        details=benchmark.details(),
        wall_time_seconds=(end_wall_time - begin_wall_time) / repetitions,
        cpu_time_seconds=(end_cpu_time - begin_cpu_time) / repetitions,
    )
    print(f"Completed: {result}")
    self._results.append(result)

  def add_separator(self):
    """Adds a separator to the results."""
    self._results.append(None)

  def print_results(self):
    """Prints all measures."""
    header = (
        f"{'Wall time (s)':<{_TIME_WIDTH}}    "
        f"{'CPU time (s)':<{_TIME_WIDTH}}    "
        f"{'Wall time (s)/unit':<{_TIME_WIDTH}}    "
        f"{'units/s':<{_TIME_WIDTH}}    "
        f"{'Name':<{_NAME_WIDTH}}"
    )
    sep_length = len(header)
    print("=" * sep_length)
    print(header)
    print("=" * sep_length)
    for idx, result in enumerate(self._results):
      if result is None:
        if idx != 0:
          print("-" * sep_length, flush=True)
      else:
        print(result)
    print("=" * sep_length, flush=True)

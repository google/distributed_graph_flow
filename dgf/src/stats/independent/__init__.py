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

"""Finite-sample confidence intervals for evaluation metrics on independent samples."""

from dgf.src.stats.independent import bound
from dgf.src.stats.independent import interval
from dgf.src.stats.independent import jax_accumulator
from dgf.src.stats.independent import metrics
from dgf.src.stats.independent import sketch
from dgf.src.stats.independent.interval import confidence_interval
from dgf.src.stats.independent.interval import Result
from dgf.src.stats.independent.interval import Status
from dgf.src.stats.independent.sketch import IndependentSketch
from dgf.src.stats.independent.sketch import R2Sketch

__all__ = [
    "Result",
    "Status",
    "confidence_interval",
    "IndependentSketch",
    "R2Sketch",
    "bound",
    "interval",
    "metrics",
    "sketch",
    "jax_accumulator",
]

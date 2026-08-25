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

"""Graph algorithms operating on GbbsGraphHandle.

Example:

gbbs_handle = dgf.gbbs.loader.load_graph_from_path(
    "/path/to/graph",
    symmetric=True,
)
result = dgf.gbbs.connected_components.connected_components(
    gbbs_handle,
    dgf.gbbs.connected_components.SimpleUnionAsyncCCParams(),
)
print(f"Number of nodes: {len(result.labels)}")
print(f"Number of components: {result.num_components}")
print(f"Largest component size: {result.largest_component_size}")
"""

import contextlib
import threading
import time
from typing import Optional, Union

from dgf.src.gbbs import _gbbs_ext
import tqdm

# Re-export param types for convenience.
SimpleUnionAsyncCCParams = _gbbs_ext.SimpleUnionAsyncCCParams
BfsCCParams = _gbbs_ext.BfsCCParams
LabelPropagationCCParams = _gbbs_ext.LabelPropagationCCParams
WorkEfficientCCParams = _gbbs_ext.WorkEfficientCCParams
ShiloachVishkinCCParams = _gbbs_ext.ShiloachVishkinCCParams
StronglyConnectedComponentsParams = _gbbs_ext.StronglyConnectedComponentsParams

ConnectedComponentsResult = _gbbs_ext.ConnectedComponentsResult
CCSession = _gbbs_ext.CCSession

# Type alias matching the C++ CCParams variant.
CCParams = Union[
    SimpleUnionAsyncCCParams,
    BfsCCParams,
    LabelPropagationCCParams,
    WorkEfficientCCParams,
    ShiloachVishkinCCParams,
    StronglyConnectedComponentsParams,
]


def validate_graph_params(
    graph: _gbbs_ext.GbbsGraphHandle,
    params: CCParams,
) -> None:
  """Validate that *params* are compatible with *graph*.

  Union-find algorithms (``SimpleUnionAsyncCCParams``,
  ``ShiloachVishkinCCParams``) work on both symmetric and asymmetric
  (directed) graphs because ``unite(u, v)`` is inherently symmetric.

  BFS / label-propagation algorithms (``BfsCCParams``,
  ``LabelPropagationCCParams``, ``WorkEfficientCCParams``) only follow
  outgoing edges, so they produce incorrect components on directed graphs.

  Args:
    graph: A loaded ``GbbsGraphHandle``.
    params: The algorithm parameters to validate.

  Raises:
    ValueError: If the algorithm requires a symmetric graph but the graph
      was constructed as asymmetric (directed).
  """
  if _gbbs_ext.RequiresSymmetricGraph(params) and not graph.is_symmetric():
    raise ValueError(
        f"{type(params).__name__} requires a symmetric (undirected) graph, "
        "but the graph was built as asymmetric (directed). Use "
        "SimpleUnionAsyncCCParams or ShiloachVishkinCCParams for directed "
        "graphs."
    )


# Phase enum mirrors C++ CCSession::Phase.
_PHASE_LABELS = {
    0: "Not started",
    1: "Running CC algorithm",
    2: "Copying labels",
    3: "Computing histogram",
    4: "Computing stats",
    5: "Done",
}


@contextlib.contextmanager
def _track_cc_progress(session: CCSession, poll_interval: float = 0.5):
  """Polls CCSession from a daemon thread to report phase transitions."""
  stop = threading.Event()
  last_phase = -1
  pbar = None

  def _poller():
    nonlocal last_phase, pbar
    while not stop.wait(poll_interval):
      phase = session.phase()
      if phase != last_phase:
        if pbar is not None:
          pbar.close()
          pbar = None

        if phase == 1:
          # CC algorithm — opaque, show elapsed time spinner.
          pbar = tqdm.tqdm(
              desc="CC algorithm",
              bar_format="{desc}: {elapsed}",
              disable=False,
          )
        elif phase in (2, 3, 4):
          pbar = tqdm.tqdm(
              desc=_PHASE_LABELS[phase],
              bar_format="{desc}...",
              disable=False,
          )
        elif phase == 5:
          # Done — print summary.
          duration = session.algorithm_duration_ms()
          n_nodes = session.num_nodes()
          n_comp = session.num_components()
          largest = session.largest_component_size()
          tqdm.tqdm.write(
              f"CC complete: {n_comp} components, "
              f"largest={largest}, "
              f"algorithm={duration}ms, "
              f"nodes={n_nodes}"
          )
        last_phase = phase

  t = threading.Thread(target=_poller, daemon=True)
  t.start()
  try:
    yield
  finally:
    stop.set()
    t.join(timeout=2.0)
    if pbar is not None:
      pbar.close()


def connected_components(
    graph: _gbbs_ext.GbbsGraphHandle,
    params: Optional[CCParams] = None,
    progress: bool = True,
) -> ConnectedComponentsResult:
  """Compute connected components of a GBBS graph.

  Args:
    graph: A loaded GbbsGraphHandle.
    params: Algorithm parameters. Defaults to SimpleUnionAsyncCCParams(). Use
      SimpleUnionAsyncCCParams or ShiloachVishkinCCParams for directed
      (asymmetric) graphs. BfsCCParams, LabelPropagationCCParams, and
      WorkEfficientCCParams require a symmetric (undirected) graph.
    progress: If True, display phase-level progress to stderr.

  Returns:
    ConnectedComponentsResult with .labels (numpy uint32 array),
    .num_components, .largest_component_size, .largest_component_id.

  Raises:
    ValueError: If the algorithm requires a symmetric graph but the
      graph is asymmetric.
  """
  if params is None:
    params = SimpleUnionAsyncCCParams()
  validate_graph_params(graph, params)
  session = CCSession()

  if progress:
    with _track_cc_progress(session):
      return _gbbs_ext.RunConnectedComponents(graph, params, session)
  else:
    return _gbbs_ext.RunConnectedComponents(graph, params, session)

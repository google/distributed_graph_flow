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

"""Generic predicate filtering for sequences of InMemoryGraph objects.

Useful for post-processing graph samples or subgraphs.

TODO(bmayer): Consider interface to iterator / generator and to replace the
InMemoryGraphPredicate by a callable. This way, this filtering could be
composed with other generator functions without having to keep the full data in
memory.

e.g.

# Simple version
def filter_graphs(
    graphs: Iterator[in_memory_graph.InMemoryGraph],
    predicate:Callable[[InMemoryGraph],bool])
    -> Iterator[in_memory_graph.InMemoryGraph]:
    for graph in graphs:
        if predicate(graph):
            yield graph

# Groupped version
filter_graphs(
    graphs: Iterator[in_memory_graph.InMemoryGraph],
    predicates:Sequence[Callable[[InMemoryGraph],bool]])
    -> Iterator[Tuple[in_memory_graph.InMemoryGraph, ...]
    for graph in graphs:
        yield tuple(g for p in predicates if p(g))
"""

import dataclasses
from typing import Any, List, Optional, Protocol, Sequence, TypeVar

from dgf.src.data import in_memory_graph as dgf_in_memory_graph
import numpy as np
import tqdm

InMemoryGraph = dgf_in_memory_graph.InMemoryGraph
T = TypeVar("T")


class InMemoryGraphPredicate(Protocol[T]):
  """Protocol for implementing a predicate on dgf.data.InMemoryGraph."""

  def __call__(self, graph: InMemoryGraph) -> bool:
    ...


@dataclasses.dataclass
class NumNodesPredicate:
  """Predicate for filtering by number of nodes.

  Note the `num_nodes` field on an InMemoryNodeSet is optional. It is up to the
  caller to check that it is set.

  Returns `True` if the number of nodes for the `nodset_name` is in the
  range [lower, upper): lower <= number of nodes < upper.

  Attributes:
    lower: The lower bound for the number of nodes. Defaults to np.inf
    upper: The upper bound for the number of nodes. Defaults to -np.inf
    nodeset_name: Optional name of node set to consider. If None (default), will
      count (sum) all the nodes in all nodesets.
  """

  # TODO(bmayer): Consider supporting List[str].
  nodeset_name: Optional[str] = None
  lower: int = -np.inf  # pyrefly: ignore[bad-assignment]
  upper: int = np.inf  # pyrefly: ignore[bad-assignment]

  def __post_init__(self):
    if self.lower > self.upper:
      raise ValueError(f"{self.lower=} must be less than {self.upper=}")

  def __call__(self, graph: InMemoryGraph) -> bool:
    if self.nodeset_name is None:
      nn = 0
      for _, ns in graph.node_sets.items():
        nn += ns.num_nodes  # pyrefly: ignore[unsupported-operation]
    else:
      nn = graph.node_sets[self.nodeset_name].num_nodes

    if self.lower <= nn < self.upper:  # pyrefly: ignore[unsupported-operation]
      return True

    return False


@dataclasses.dataclass
class ContainsLabelPredicate:
  """Predicate for filtering subgraphs if they have a positive label.

  Attributes:
    nodeset_name: Target nodeset name.
    feature_name: The feature name we want to filter the `label` on.
    label: The label value we want to filter.
    aggregator: "ANY" or "ALL" aggregator function.
  """

  nodeset_name: str
  feature_name: str
  label: Any = 1

  # TODO(bmayer): Better typing.
  aggregator: str = "ANY"

  def __call__(self, graph: InMemoryGraph) -> bool:
    labels = graph.node_sets[self.nodeset_name].features[self.feature_name]

    if self.aggregator == "ANY":
      return (labels == self.label).any()
    elif self.aggregator == "ALL":
      return (labels == self.label).all()
    else:
      raise ValueError(
          f"Unknown aggregator: {self.aggregator}. Must be one of ['ANY',"
          " 'ALL']"
      )


def filter_graphs(
    graphs: Sequence[InMemoryGraph],
    predicates: Sequence[InMemoryGraphPredicate],
    verbose: bool = True,
    tqdm_desc: str = "Filtering Graphs",
) -> List[List[InMemoryGraph]]:
  """Filters a sequence of graphs based on user defined predicates.

  Simple single-thread alternative to built in filter for chaining multiple
  predicates over a single pass of len(graphs).

  Args:
    graphs: A sequence of dgf.data.InMemoryGraph objects
    predicates: A sequence of callables that returns a boolean for each input
      graph.
    verbose: Print filtering progress bar.
    tqdm_desc: String passed as description to tqdm progress bar.

  Returns:
    A list of len(prediates) lists which contain all graphs that satisfy each
    predicate.
  """
  ret = [[] for _ in range(len(predicates))]

  for graph in tqdm.tqdm(graphs, disable=not verbose, desc=tqdm_desc):
    for i, predicate in enumerate(predicates):
      if predicate(graph):
        ret[i].append(graph)

  return ret

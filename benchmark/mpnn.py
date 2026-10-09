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

"""Benchmarking of message passing (MPNN) layers on synthetic graph samples.

Each `Scenario` targets one structural regime (node/edge ratio, nodeset size
vs. 65536, padding) that selects a different code path in the layers.
"""

from collections.abc import Callable, Sequence
import dataclasses
import enum
from typing import Any
import dgf
from dgf.benchmark import utils as benchmark_utils
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.util import log
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# Thresholds used by `hetero_gnn.should_sort_edges` and `sort_edges_by_dst`.
_SORT_MIN_EDGES = 50000
_NARROW_INDEX_MAX_NODES = 65536


class LayerType(enum.Enum):
  """The MPNN layer implementation to benchmark."""

  HETERO_GNN = "HeterogeneousGraphConvolution"
  HETERO_GAT = "HeterogeneousGraphAttentionNetwork"
  GRAPHSAGE = "GraphSAGE"
  GCN = "GCN"


class Mode(enum.Enum):
  """Execution mode to benchmark."""

  FORWARD = "fwd"
  FORWARD_BACKWARD = "fwd+bwd"


@dataclasses.dataclass(frozen=True)
class Scenario:
  """A synthetic graph sample configuration.

  Attributes:
    name: Short identifier of the scenario.
    num_nodesets: Number of node sets.
    num_edgesets: Number of edge sets.
    treeness: Probability that a sampled neighbor is a new node (1.0 = pure
      tree; lower values create loops).
    batch_size: Number of samples merged in a batch.
    num_hops: Number of sampling hops.
    hop_width: Default number of neighbors sampled per plan edge.
    edgeset_hop_widths: Per-edgeset override of `hop_width`.
    padding_relative_margin: Relative padding margin (padding edges all point to
      the sentinel node).
    description: Regime exercised by this scenario.
  """

  name: str
  num_nodesets: int
  num_edgesets: int
  treeness: float
  batch_size: int
  num_hops: int = 2
  hop_width: int = 5
  edgeset_hop_widths: dict[str, int] = dataclasses.field(default_factory=dict)
  padding_relative_margin: float = 0.1
  description: str = ""


# A "regime" describes the shape of a relation in the merged batch: regime 1
# relations have more edges than nodes (e.g. paper->topic in MAG), regime 2
# relations have many more nodes than edges (e.g. author->institution). The
# scenarios also vary the number of nodes per nodeset (below / above 65536),
# the number of edges per edgeset (below / above 50000) and the padding ratio.
DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="small_tree",
        num_nodesets=4,
        num_edgesets=4,
        treeness=1.0,
        batch_size=32,
        description="Pure trees, mix of regime 1 and 2. N < 65536.",
    ),
    Scenario(
        name="small_loops",
        num_nodesets=4,
        num_edgesets=4,
        treeness=0.3,
        batch_size=32,
        description="Many loops, almost all relations in regime 1. N < 65536.",
    ),
    Scenario(
        name="mixed_relations",
        num_nodesets=4,
        num_edgesets=4,
        treeness=0.7,
        batch_size=128,
        hop_width=10,
        edgeset_hop_widths={"e2": 1},
        description=(
            "MAG-like mix: 'e2' has far more source nodes than edges (regime"
            " 2), the others more edges than nodes (regime 1)."
        ),
    ),
    Scenario(
        name="many_edges_few_nodes",
        num_nodesets=1,
        num_edgesets=1,
        treeness=0.2,
        batch_size=128,
        hop_width=15,
        description="Homogeneous, E >= 50000 but N < 65536.",
    ),
    Scenario(
        name="many_edges_many_nodes",  # Previously called "large_sorted".
        num_nodesets=1,
        num_edgesets=1,
        treeness=0.9,
        batch_size=256,
        hop_width=15,
        description="Homogeneous, E >= 50000 and N >= 65536.",
    ),
    Scenario(
        name="heavy_padding",
        num_nodesets=4,
        num_edgesets=4,
        treeness=0.3,
        batch_size=32,
        padding_relative_margin=1.0,
        description="'small_loops' with 100% padding (high degree sentinel).",
    ),
)


def generate_synthetic_schema(
    num_nodesets: int,
    num_edgesets: int,
    dims: int,
    num_hops: int = 2,
) -> dgf.data.GraphSchema:
  """Generates a synthetic schema with an "embedding" feature per nodeset.

  The first `min(num_nodesets - 1, num_edgesets)` edgesets form a tree rooted
  at "n0" of depth at most `num_hops`; remaining edgesets cycle across nodeset
  pairs.
  """
  if num_nodesets < 1:
    raise ValueError(f"num_nodesets must be >= 1, got {num_nodesets}")
  if num_edgesets < 1:
    raise ValueError(f"num_edgesets must be >= 1, got {num_edgesets}")
  if dims < 1:
    raise ValueError(f"dims must be >= 1, got {dims}")

  node_sets = {
      f"n{i}": dgf.data.NodeSchema(
          features={
              "embedding": dgf.data.FeatureSchema(
                  format=dgf.data.FeatureFormat.FLOAT_32,
                  semantic=dgf.data.FeatureSemantic.EMBEDDING,
                  shape=(dims,),
              )
          }
      )
      for i in range(num_nodesets)
  }

  max_tree_depth = max(1, num_hops)
  depths = [0]
  edge_sets = {}
  for i in range(num_edgesets):
    if num_nodesets == 1:
      src_idx = 0
      dst_idx = 0
    elif i < num_nodesets - 1:
      eligible_srcs = [u for u, d in enumerate(depths) if d < max_tree_depth]
      src_idx = eligible_srcs[i % len(eligible_srcs)]
      dst_idx = i + 1
      depths.append(depths[src_idx] + 1)
    else:
      k = i - (num_nodesets - 1)
      src_idx = k % num_nodesets
      dst_idx = (src_idx + (k // num_nodesets) + 1) % num_nodesets
    edge_sets[f"e{i}"] = dgf.data.EdgeSchema(
        source=f"n{src_idx}",
        target=f"n{dst_idx}",
    )

  return dgf.data.GraphSchema(node_sets=node_sets, edge_sets=edge_sets)


def generate_synthetic_sample(
    schema: dgf.data.GraphSchema,
    sampling_plan: sampling_config_lib.SamplingPlan,
    dims: int,
    treeness: float,
    rng: np.random.Generator,
) -> dgf.data.InMemoryGraph:
  """Generates one synthetic GraphSAGE-like sample.

  Mirrors `SampleBuilder::RecursiveGrow` of the C++ sampler: depth-first
  expansion of the plan from a single seed node, sampling between
  `hop_width // 2` and `hop_width` neighbors per plan edge. Each neighbor is a
  new node with probability `treeness`, otherwise an already visited one
  (closing a loop). Edges are sorted and deduplicated per edgeset.
  """
  if not 0.0 <= treeness <= 1.0:
    raise ValueError(f"treeness must be in [0.0, 1.0], got {treeness}")

  num_nodes: dict[str, int] = {ns_name: 0 for ns_name in schema.node_sets}
  edges: dict[str, list[tuple[int, int]]] = {
      es_name: [] for es_name in schema.edge_sets
  }

  seed_nodeset = sampling_plan.root.nodeset
  num_nodes[seed_nodeset] = 1

  def rec_grow(
      plan_node: sampling_config_lib.PlanNode, source_sampled_node: int
  ):
    for plan_edge in plan_node.children:
      target_nodeset = plan_edge.node.nodeset
      edgeset_name = plan_edge.edgeset
      min_width = max(1, plan_edge.hop_width // 2)
      num_neighbors = int(rng.integers(min_width, plan_edge.hop_width + 1))

      for _ in range(num_neighbors):
        curr_target_nodes = num_nodes[target_nodeset]
        if curr_target_nodes == 0 or rng.random() < treeness:
          target_sampled_node = curr_target_nodes
          num_nodes[target_nodeset] = curr_target_nodes + 1
        else:
          if target_nodeset == plan_node.nodeset and curr_target_nodes > 1:
            # Pick a distinct existing node to form a non-trivial loop.
            r = int(rng.integers(0, curr_target_nodes - 1))
            target_sampled_node = r + 1 if r >= source_sampled_node else r
          else:
            target_sampled_node = int(rng.integers(0, curr_target_nodes))

        if not plan_edge.reversed:
          edges[edgeset_name].append((source_sampled_node, target_sampled_node))
        else:
          edges[edgeset_name].append((target_sampled_node, source_sampled_node))

        if plan_edge.node.children:
          rec_grow(plan_edge.node, target_sampled_node)

  rec_grow(sampling_plan.root, 0)

  node_sets = {}
  for ns_name in schema.node_sets:
    n = num_nodes[ns_name]
    emb = rng.standard_normal(size=(n, dims), dtype=np.float32)
    node_sets[ns_name] = dgf.data.InMemoryNodeSet(
        num_nodes=n,
        features={"embedding": emb},
    )

  edge_sets = {}
  for es_name in schema.edge_sets:
    deduped_edges = sorted(set(edges[es_name]))
    if deduped_edges:
      adj = np.array(deduped_edges, dtype=np.int64).T
    else:
      adj = np.zeros((2, 0), dtype=np.int64)
    edge_sets[es_name] = dgf.data.InMemoryEdgeSet(adjacency=adj)

  return dgf.data.InMemoryGraph(node_sets=node_sets, edge_sets=edge_sets)


def _override_hop_widths(
    plan: sampling_config_lib.SamplingPlan, edgeset_hop_widths: dict[str, int]
) -> None:
  """Overrides, in place, the hop width of the given edgesets in a plan."""
  unknown = set(edgeset_hop_widths) - sampling_config_lib.edgesets_in_plan(plan)
  if unknown:
    raise ValueError(f"Edgesets {sorted(unknown)} are not in the sampling plan")

  def rec(node: sampling_config_lib.PlanNode):
    for child in node.children:
      if child.edgeset in edgeset_hop_widths:
        child.hop_width = edgeset_hop_widths[child.edgeset]
      rec(child.node)

  rec(plan.root)


def generate_padded_jax_batch(
    schema: dgf.data.GraphSchema,
    scenario: Scenario,
    dims: int,
    seed: int = 0,
) -> dgf.data.JaxInMemoryGraph:
  """Generates a batch of synthetic samples, merges and pads them for JAX."""
  sampling_config = sampling_config_lib.SimpleSamplingConfig(
      seed_nodeset="n0",
      num_hops=scenario.num_hops,
      hop_width=scenario.hop_width,
      reverse=True,
      with_replacement=False,
      multi_visit=True,
  )
  sampling_plan = sampling_config_lib.simple_sampling_config_to_sampling_plan(
      sampling_config, schema
  )
  _override_hop_widths(sampling_plan, scenario.edgeset_hop_widths)

  rng = np.random.default_rng(seed)
  samples = [
      generate_synthetic_sample(
          schema=schema,
          sampling_plan=sampling_plan,
          dims=dims,
          treeness=scenario.treeness,
          rng=rng,
      )
      for _ in range(scenario.batch_size)
  ]

  unpadded_merged, _ = dgf.transform.GraphMerger(
      schema=schema, padding=None, sentinel_offset=False
  )(samples)
  padding = dgf.analyse.padding_from_graph_generator(
      schema=schema,
      graphs=iter([unpadded_merged]),
      relative_margin=scenario.padding_relative_margin,
  )
  padded_merged, _ = dgf.transform.GraphMerger(
      schema=schema, padding=padding, sentinel_offset=False
  )(samples)
  return dgf.convert.graph_to_jax_graph(padded_merged)


def describe_relations(
    jax_graph: dgf.data.JaxInMemoryGraph, schema: dgf.data.GraphSchema
) -> str:
  """Returns a one-line summary `name[N_src->N_dst:E r?/r? [sort] [u16]]`.

  `r1`/`r2` is the regime (node- vs. edge-level projection) in the forward and
  reverse directions; `sort` means edges would be sorted on GPU; `u16` means
  both nodesets have fewer than 65536 nodes.
  """

  def regime(n_src: int, n_dst: int, e: int) -> str:
    return "r1" if n_src + 2 * n_dst <= 3 * e else "r2"

  parts = []
  for es_name, es_schema in schema.edge_sets.items():
    n_src = jax_graph.node_sets[es_schema.source].num_nodes
    n_dst = jax_graph.node_sets[es_schema.target].num_nodes
    assert n_src is not None and n_dst is not None
    e = int(jax_graph.edge_sets[es_name].adjacency.shape[1])
    sort = e >= _SORT_MIN_EDGES and max(n_src, n_dst) >= _NARROW_INDEX_MAX_NODES
    narrow = max(n_src, n_dst) < _NARROW_INDEX_MAX_NODES
    parts.append(
        f"{es_name}[{n_src}->{n_dst}:{e}"
        f" {regime(n_src, n_dst, e)}/{regime(n_dst, n_src, e)}"
        f"{' sort' if sort else ''}{' u16' if narrow else ''}]"
    )
  return " ".join(parts)


class _StackedMPNN(nn.Module):
  """Stacks multiple MPNN layers sequentially."""

  make_layer: Callable[[str], nn.Module]
  num_layers: int

  @nn.compact
  def __call__(
      self,
      graph: dgf.data.JaxInMemoryGraph,
      training: bool,
  ) -> dgf.data.JaxInMemoryGraph:
    for i in range(self.num_layers):
      graph = self.make_layer(f"layer_{i}")(graph, training=training)
    return graph


def _make_model(
    schema: dgf.data.GraphSchema,
    layer_type: LayerType,
    num_layers: int,
    dims: int,
    num_heads: int,
) -> nn.Module:
  """Creates a stack of `num_layers` MPNN layers of the given type."""
  if layer_type == LayerType.HETERO_GNN:
    layer_cfg = dgf.jax.layers.HeterogeneousGraphConvolutionConfig(dims=dims)
  elif layer_type == LayerType.HETERO_GAT:
    layer_cfg = dgf.jax.layers.HeterogeneousGraphAttentionNetworkConfig(
        dims=dims, num_heads=num_heads
    )
  elif layer_type == LayerType.GRAPHSAGE:
    layer_cfg = dgf.jax.layers.HeterogeneousGraphConvolutionConfig.graphsage(
        dims=dims
    )
  elif layer_type == LayerType.GCN:
    layer_cfg = dgf.jax.layers.HeterogeneousGraphConvolutionConfig.gcn(
        dims=dims
    )
  else:
    raise ValueError(f"Unsupported layer_type: {layer_type}")
  return _StackedMPNN(
      make_layer=lambda name: layer_cfg.make(schema, name=name),
      num_layers=num_layers,
  )


class RunMPNN(benchmark_utils.Benchmark):
  """Benchmarks one (model, mode) step on a padded JAX batch."""

  def __init__(
      self,
      *,
      jax_graph: dgf.data.JaxInMemoryGraph,
      schema: dgf.data.GraphSchema,
      scenario: Scenario,
      model: nn.Module,
      variables: Any,
      layer_type: LayerType,
      mode: Mode,
      num_layers: int,
      dims: int,
      max_runtime_seconds: float,
  ):
    self.jax_graph = jax_graph
    self.schema = schema
    self.scenario = scenario
    self.model = model
    self.variables = variables
    self.layer_type = layer_type
    self.mode = mode
    self.num_layers = num_layers
    self.dims = dims
    self.max_runtime_seconds = max_runtime_seconds
    self.set_unit_multiplicator(scenario.batch_size)

    self._step_fn: Callable[[Any, dgf.data.JaxInMemoryGraph], Any] | None = None
    self._total_nodes = sum(
        ns.num_nodes or 0 for ns in self.jax_graph.node_sets.values()
    )
    self._total_edges = sum(
        int(es.adjacency.shape[1]) for es in self.jax_graph.edge_sets.values()
    )

  def impl_name(self) -> str:
    return f"{self.layer_type.value} {self.scenario.name}"

  def setup(self):
    model = self.model
    if self.mode == Mode.FORWARD:
      self._step_fn = jax.jit(lambda p, g: model.apply(p, g, training=False))
    elif self.mode == Mode.FORWARD_BACKWARD:

      def loss_fn(p, g):
        out_g: Any = model.apply(p, g, training=False)
        return sum(
            jnp.sum(ns.features["embedding"] ** 2)
            for ns in out_g.node_sets.values()
        )

      self._step_fn = jax.jit(
          jax.value_and_grad(loss_fn, argnums=(0, 1), allow_int=True)
      )
    else:
      raise ValueError(f"Unsupported mode: {self.mode}")

    # Compile outside of the timed region.
    out = self._step_fn(self.variables, self.jax_graph)
    jax.block_until_ready(out)

  def run_unit(self):
    assert self._step_fn is not None
    out = self._step_fn(self.variables, self.jax_graph)
    jax.block_until_ready(out)

  def details(self) -> str:
    return (
        f"mode={self.mode.value}"
        f" layers={self.num_layers}"
        f" dims={self.dims}"
        f" treeness={self.scenario.treeness}"
        f" batch={self.scenario.batch_size}"
        f" hops={self.scenario.num_hops}"
        f" width={self.scenario.hop_width}"
        f" padding={self.scenario.padding_relative_margin}"
        f" nodes={self._total_nodes}"
        f" edges={self._total_edges}"
        f" backend={jax.default_backend()}"
        f" relations={describe_relations(self.jax_graph, self.schema)}"
    )


def mpnn(
    *,
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
    layer_types: Sequence[LayerType] = tuple(LayerType),
    list_dims: Sequence[int] = (128,),
    list_num_layers: Sequence[int] = (1, 3),
    num_heads: int = 4,
    max_runtime_seconds: float = 2.0,
):
  """Benchmarks MPNN layers on synthetic batches.

  Samples are generated, merged and padded once per (scenario, dims) before
  the benchmark loop. Compilation happens in `setup`, so no warmup is needed.
  """
  benchmarker = benchmark_utils.Benchmarker()

  for scenario in scenarios:
    for dims in list_dims:
      log.info(
          "Scenario '%s' (dims=%d): %s",
          scenario.name,
          dims,
          scenario.description,
      )
      schema = generate_synthetic_schema(
          num_nodesets=scenario.num_nodesets,
          num_edgesets=scenario.num_edgesets,
          dims=dims,
          num_hops=scenario.num_hops,
      )
      jax_graph = generate_padded_jax_batch(schema, scenario, dims=dims)
      log.info("Relations: %s", describe_relations(jax_graph, schema))

      for num_layers in list_num_layers:
        for layer_type in layer_types:
          model = _make_model(schema, layer_type, num_layers, dims, num_heads)
          variables = model.init(
              jax.random.PRNGKey(0), jax_graph, training=False
          )
          for mode in [Mode.FORWARD, Mode.FORWARD_BACKWARD]:
            benchmarker.run(
                RunMPNN(
                    jax_graph=jax_graph,
                    schema=schema,
                    scenario=scenario,
                    model=model,
                    variables=variables,
                    layer_type=layer_type,
                    mode=mode,
                    num_layers=num_layers,
                    dims=dims,
                    max_runtime_seconds=max_runtime_seconds,
                ),
                repetitions=1,
                warmup_repetitions=0,
            )
        benchmarker.add_separator()

  benchmarker.print_results()

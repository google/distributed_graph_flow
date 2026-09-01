#include "dgf/src/gbbs/connected_components.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/gbbs/benchmarks/Connectivity/BFSCC/Connectivity.h"
#include "third_party/gbbs/benchmarks/Connectivity/LabelPropagation/Connectivity.h"
#include "third_party/gbbs/benchmarks/Connectivity/ShiloachVishkin/Connectivity.h"
#include "third_party/gbbs/benchmarks/Connectivity/SimpleUnionAsync/Connectivity.h"
#include "third_party/gbbs/benchmarks/Connectivity/WorkEfficientSDB14/Connectivity.h"
#include "third_party/gbbs/benchmarks/Connectivity/common.h"
#include "third_party/gbbs/benchmarks/StronglyConnectedComponents/RandomGreedyBGSS16/StronglyConnectedComponents.h"
#include "third_party/gbbs/gbbs/bridge.h"
#include "third_party/parlay/include/parlay/monoid.h"
#include "third_party/parlay/include/parlay/parallel.h"
#include "third_party/parlay/include/parlay/primitives.h"

namespace dgf::gbbs {

namespace {

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

using ::gbbs::parent;
using ::gbbs::sequence;

// Helper: set session phase if session is non-null.
inline void SetPhase(CCSession* session, CCSession::Phase phase) {
  if (session) session->phase.store(phase, std::memory_order_relaxed);
}

// Fused accumulator for single-pass stats computation over counts.
struct CCStats {
  size_t num_components = 0;
  size_t largest_size = 0;
  uint32_t largest_id = 0;
};

ConnectedComponentsResult BuildResult(sequence<parent> labels,
                                      CCSession* session) {
  const size_t n = labels.size();

  // Phase: copy labels -> std::vector<uint32_t>.
  SetPhase(session, CCSession::kCopyingLabels);
  std::vector<uint32_t> vec(n);
  parlay::parallel_for(0, n, [&](size_t i) { vec[i] = labels[i]; });

  // Phase: parallel histogram.
  SetPhase(session, CCSession::kComputingHistogram);
  auto counts = parlay::histogram_by_index(labels, n);

  // Phase: fused reduce — compute all stats in a single pass over counts.
  SetPhase(session, CCSession::kComputingStats);

  auto per_bucket = parlay::delayed_tabulate(n, [&](size_t i) -> CCStats {
    if (counts[i] > 0) {
      return {1, counts[i], static_cast<uint32_t>(i)};
    }
    return {};
  });

  auto stats = parlay::reduce(
      per_bucket,
      parlay::binary_op(
          [](const CCStats& a, const CCStats& b) -> CCStats {
            return {
                a.num_components + b.num_components,
                a.largest_size >= b.largest_size ? a.largest_size
                                                 : b.largest_size,
                a.largest_size >= b.largest_size ? a.largest_id : b.largest_id,
            };
          },
          CCStats{}));

  // Publish final stats to session.
  if (session) {
    session->num_components.store(static_cast<int64_t>(stats.num_components),
                                  std::memory_order_relaxed);
    session->largest_component_size.store(
        static_cast<int64_t>(stats.largest_size), std::memory_order_relaxed);
  }

  SetPhase(session, CCSession::kDone);
  return {std::move(vec), stats.num_components, stats.largest_size,
          stats.largest_id};
}

}  // namespace

// Returns true if the algorithm requires a symmetric (undirected) graph.
//
// Union-find based algorithms (SimpleUnionAsync, ShiloachVishkin) work on
// both symmetric and asymmetric graphs because unite(u,v) is inherently
// symmetric — seeing any edge between u and v is enough to merge their
// components.
//
// BFS and label-propagation based algorithms (BFS-CC, LabelPropagation,
// WorkEfficient) only follow outgoing edges via edgeMap, so they produce
// incorrect weakly connected components on directed graphs: vertices
// reachable only via incoming edges are placed in separate components.
bool RequiresSymmetricGraph(const CCParams& params) {
  return std::visit(
      overloaded{
          [](const SimpleUnionAsyncCCParams&) { return false; },
          [](const ShiloachVishkinCCParams&) { return false; },
          [](const StronglyConnectedComponentsParams&) { return false; },
          [](const BfsCCParams&) { return true; },
          [](const LabelPropagationCCParams&) { return true; },
          [](const WorkEfficientCCParams&) { return true; },
      },
      params);
}

absl::StatusOr<ConnectedComponentsResult> RunConnectedComponents(
    GbbsGraphHandle& graph, const CCParams& params, CCSession* session) {
  // Validate graph type x algorithm compatibility.
  if (RequiresSymmetricGraph(params) && !graph.is_symmetric()) {
    return absl::InvalidArgumentError(
        "This CC algorithm requires a symmetric (undirected) graph. "
        "Use SimpleUnionAsyncCCParams or ShiloachVishkinCCParams for "
        "directed graphs.");
  }

  // --- CC Algorithm (opaque GBBS internals) ---
  SetPhase(session, CCSession::kRunningAlgorithm);
  auto t0 = absl::Now();

  auto labels = graph.Visit([&](auto& g) -> sequence<parent> {
    return std::visit(
        overloaded{
            [&](const SimpleUnionAsyncCCParams&) {
              return ::gbbs::simple_union_find::SimpleUnionAsync(g);
            },

            [&](const BfsCCParams&) { return ::gbbs::bfs_cc::CC(g); },

            [&](const LabelPropagationCCParams& p) {
              if (p.use_permutation) {
                return ::gbbs::labelprop_cc::CC<true>(g);
              }
              return ::gbbs::labelprop_cc::CC<false>(g);
            },

            [&](const WorkEfficientCCParams& p) {
              return ::gbbs::workefficient_cc::CC(g, p.beta, p.pack, p.permute);
            },

            [&](const ShiloachVishkinCCParams&) {
              using Graph = std::decay_t<decltype(g)>;
              auto sv = ::gbbs::shiloachvishkin_cc::SVAlgorithm<Graph>(g);
              size_t n = g.n;
              auto parents = sequence<parent>::from_function(
                  n, [](size_t i) { return static_cast<parent>(i); });
              sv.initialize(parents);
              sv.template compute_components<::gbbs::no_sampling>(parents);
              return parents;
            },

            [&](const StronglyConnectedComponentsParams& p) {
              auto scc_labels = ::gbbs::StronglyConnectedComponents(g, p.beta);
              auto parents = sequence<parent>::from_function(
                  scc_labels.size(),
                  [&](size_t i) { return static_cast<parent>(scc_labels[i]); });
              return parents;
            },
        },
        params);
  });

  // Record algorithm duration and node count.
  if (session) {
    session->algorithm_duration_ms.store(
        absl::ToInt64Milliseconds(absl::Now() - t0), std::memory_order_relaxed);
    session->num_nodes.store(static_cast<int64_t>(labels.size()),
                             std::memory_order_relaxed);
  }

  // --- Build result (tracked by phase) ---
  return BuildResult(std::move(labels), session);
}

}  // namespace dgf::gbbs

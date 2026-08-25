#ifndef THIRD_PARTY_PY_DGF_SRC_GBBS_CONNECTED_COMPONENTS_H_
#define THIRD_PARTY_PY_DGF_SRC_GBBS_CONNECTED_COMPONENTS_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "dgf/src/gbbs/gbbs_graph_handle.h"

namespace dgf::gbbs {

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

// Result of a connected components computation.
struct ConnectedComponentsResult {
  std::vector<uint32_t> labels;  // labels[i] = component ID for vertex i.
  size_t num_components;
  size_t largest_component_size;
  uint32_t largest_component_id;
};

// ---------------------------------------------------------------------------
// Session (progress tracking)
// ---------------------------------------------------------------------------

// Lock-free progress tracking for connected components computation.
// Python poller reads atomics every ~500ms.
struct CCSession {
  enum Phase : int32_t {
    kNotStarted = 0,
    kRunningAlgorithm = 1,    // Opaque GBBS computation.
    kCopyingLabels = 2,       // Copying labels to output vector.
    kComputingHistogram = 3,  // Parallel histogram over component IDs.
    kComputingStats = 4,      // Fused reduce: num_components + largest.
    kDone = 5,
  };

  // Current execution phase.
  std::atomic<int32_t> phase{kNotStarted};

  // Wall-clock duration of the CC algorithm phase (milliseconds).
  // Set at kRunningAlgorithm -> kCopyingLabels transition.
  std::atomic<int64_t> algorithm_duration_ms{0};

  // Summary stats — populated as they become known.
  std::atomic<int64_t> num_nodes{0};
  std::atomic<int64_t> num_components{0};
  std::atomic<int64_t> largest_component_size{0};

  // Convenience accessors for nanobind.
  int32_t get_phase() const { return phase.load(std::memory_order_relaxed); }
  int64_t get_algorithm_duration_ms() const {
    return algorithm_duration_ms.load(std::memory_order_relaxed);
  }
  int64_t get_num_nodes() const {
    return num_nodes.load(std::memory_order_relaxed);
  }
  int64_t get_num_components() const {
    return num_components.load(std::memory_order_relaxed);
  }
  int64_t get_largest_component_size() const {
    return largest_component_size.load(std::memory_order_relaxed);
  }
};

// ---------------------------------------------------------------------------
// CC Algorithm parameter structs.
// ---------------------------------------------------------------------------

struct SimpleUnionAsyncCCParams {};

struct BfsCCParams {};

struct LabelPropagationCCParams {
  bool use_permutation = false;
};

struct WorkEfficientCCParams {
  double beta = 0.2;
  bool pack = false;
  bool permute = false;
};

struct ShiloachVishkinCCParams {};

// Set of all supported CC algorithms.
using CCParams = std::variant<SimpleUnionAsyncCCParams, BfsCCParams,
                              LabelPropagationCCParams, WorkEfficientCCParams,
                              ShiloachVishkinCCParams>;

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

// Returns true if the algorithm requires a symmetric (undirected) graph.
bool RequiresSymmetricGraph(const CCParams& params);

// ---------------------------------------------------------------------------
// Dispatch function
// ---------------------------------------------------------------------------

// Run a connected components algorithm on a GBBS graph.
// Returns InvalidArgumentError if the algorithm requires a symmetric graph
// but the graph is asymmetric (directed).
// If `session` is non-null, phase transitions and stats are written to it
// via relaxed atomics for lock-free progress tracking from Python.
absl::StatusOr<ConnectedComponentsResult> RunConnectedComponents(
    GbbsGraphHandle& graph, const CCParams& params,
    CCSession* session = nullptr);

}  // namespace dgf::gbbs

#endif  // THIRD_PARTY_PY_DGF_SRC_GBBS_CONNECTED_COMPONENTS_H_

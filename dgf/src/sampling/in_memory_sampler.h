// Reusable Non-nanobind code (i.e. c++ code that can be used without nanobind)
// for the in memory sampler. This code is tested in in_memory_sampler_test.cc
//
// This code can be called from any c++ lib/bin.

#ifndef DGF_SRC_SAMPLING_IN_MEMORY_SAMPLER_H_
#define DGF_SRC_SAMPLING_IN_MEMORY_SAMPLER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"

namespace dgf::sampling::in_memory_sampler {

// Index to encode the index of a node in a sampled graph.
// TODO(gbm): Add control flag.
typedef std::uint32_t SampleIdx;

// Index to encode the index of a node or edge in the in-memory input graph.
typedef std::size_t InputIdx;

// Timestamp type for temporal graphs.
typedef std::int64_t Timestamp;

// Index for fast retrieval of target nodes from source nodes in a directed
// graph. Represents outgoing edges where the `i`-th source node's targets are
// in `target_node_idxs` from index `source_blocks[i]` to
// `source_blocks[i+1]-1`.
//
// TODO(gbm): Should those be arrays? (e.g., numpy or jax array).
// TODO(gbm): Augment structure to support weighted sampling.
struct Empty {};

template <bool HasTimestamp, bool HasEdgeIdx>
struct Edge {
  InputIdx source;
  InputIdx target;
  [[no_unique_address]] std::conditional_t<HasTimestamp, Timestamp, Empty>
      timestamp;
  [[no_unique_address]] std::conditional_t<HasEdgeIdx, InputIdx, Empty>
      edge_idx;
};

struct AdjacencyIndex {
  // `source_blocks[i]` is the start index in `target_node_idxs` for source `i`.
  // Has size `n + 1`, where `n` is the number of source nodes.
  // `source_blocks[n]` equals `target_node_idxs.size()`.
  std::vector<InputIdx> source_blocks;
  // Contains target node indices. Segments `[source_blocks[i],
  // source_blocks[i+1])` are sorted.
  std::vector<InputIdx> target_node_idxs;
  // Parallel array containing edge timestamps, if the edgeset has timestamps.
  // Otherwise, is empty.
  std::vector<Timestamp> timestamps;
  // Parallel array containing edge indices, if edge masking is enabled.
  // Otherwise, is empty.
  std::vector<InputIdx> edge_idxs;

  template <bool HasTimestamp, bool HasEdgeIdx>
  static absl::StatusOr<AdjacencyIndex> CreateFromEdgeList(
      std::vector<Edge<HasTimestamp, HasEdgeIdx>>&& edges,
      size_t num_source_nodes, size_t num_target_nodes);

  static AdjacencyIndex CreateEmpty(size_t num_source_nodes,
                                    size_t num_target_nodes);

  // Example
  // =======
  // Suppose the following edges: 0->0, 0->1, 2->3. AdjacencyIndex will be as
  // follow:
  //   source_blocks = {0, 2, 2, 3}
  //   target_node_idxs = {0, 1, 3}
  //
  // Access example
  // ==============
  // The edges+destinations starting from node 'i' can be accessed with:
  //
  // size_t start_idx = source_blocks[i]
  // size_t end_idx = source_blocks[i+1]
  // size_t len = end_idx - start_idx
  // targets = absl::MakeConstSpan(target_node_idxs).subspan(start_idx, len)

  std::span<const InputIdx> Targets(InputIdx source_node) const {
    DCHECK_LT(source_node + 1, source_blocks.size());
    InputIdx start_idx = source_blocks[source_node];
    InputIdx end_idx = source_blocks[source_node + 1];
    DCHECK_LE(start_idx, end_idx);
    DCHECK_LE(end_idx, target_node_idxs.size());
    InputIdx num_neighbors = end_idx - start_idx;
    return absl::MakeConstSpan(target_node_idxs)
        .subspan(start_idx, num_neighbors);
  }

  // Checks if a directed edge exists from `source_node` to `target_node`.
  bool HasEdge(InputIdx source_node, InputIdx target_node) const {
    auto targets = Targets(source_node);
    return std::binary_search(targets.begin(), targets.end(), target_node);
  }

  // Returns a string representation of the AdjacencyIndex for debugging.
  std::string to_string() const {
    auto format_cropped = [](const auto& vec) {
      constexpr size_t kMaxSize = 10;
      std::string result = absl::StrJoin(
          absl::MakeConstSpan(vec).first(std::min(vec.size(), kMaxSize)), ", ");
      if (vec.size() > kMaxSize) {
        absl::StrAppend(&result, ", ...");
      }
      return result;
    };

    return absl::StrCat("AdjacencyIndex(source_blocks=(", source_blocks.size(),
                        ")[", format_cropped(source_blocks),
                        "], target_node_idxs=(", target_node_idxs.size(), ")[",
                        format_cropped(target_node_idxs), "])");
  }

  // Samples up to `num_samples` unique target nodes connected to the given
  // `source_node` uniformly at random without replacement. The sampled target
  // node indices are appended to the `result` vector.
  absl::Status SampleRandomUniform(InputIdx source_node,
                                   std::size_t num_samples,
                                   std::vector<InputIdx>* result,
                                   std::mt19937_64* rng,
                                   InputIdx masked_edge_idx = -1) const;

  // Samples the first `num_samples` target nodes connected to the given
  // `source_node` (sorted by source node id). The sampled target node indices
  // are appended to the `result` vector.
  absl::Status SampleFirst(InputIdx source_node, std::size_t num_samples,
                           std::vector<InputIdx>* result,
                           InputIdx masked_edge_idx = -1) const;

  // Samples up to `num_samples` unique target nodes anterior to
  // `seed_timestamp` uniformly at random without replacement.
  absl::Status SampleRandomUniformWithTimestamp(InputIdx source_node,
                                                Timestamp seed_timestamp,
                                                std::size_t num_samples,
                                                std::vector<InputIdx>* result,
                                                std::mt19937_64* rng) const;

  // Samples the first `num_samples` target nodes anterior to `seed_timestamp`.
  absl::Status SampleFirstWithTimestamp(InputIdx source_node,
                                        Timestamp seed_timestamp,
                                        std::size_t num_samples,
                                        std::vector<InputIdx>* result) const;
};

// Creates map "m" such that "m[string_list[i]] = i".
absl::flat_hash_map<std::string, int> IndexStringValues(
    absl::Span<const std::string> string_list);

// Defines the order of operations for sampling. This C++ struct mirrors the
// `SamplingPlan` class in `config.py` and is used for faster, multi-threaded
// execution.
struct SamplingPlan {
  struct Edge;
  struct Node {
    // Unique & dense identifier of the plan's node.
    int nodeset_idx;
    std::vector<Edge> children;

    // Unique and dense index of the this Node in the sampling plan. Computed by
    // calling "ComputeStepIdx".
    int step_idx = -1;

    // Debug string
    std::string to_string(int indent) const {
      std::string prefix(indent * 2, ' ');
      auto child_formatter = [&](std::string* out, const Edge& child) {
        absl::StrAppend(out, child.to_string(indent + 1));
      };
      std::string children_part =
          children.empty()
              ? "[]"
              : absl::StrCat("[\n",
                             absl::StrJoin(children, ",\n", child_formatter),
                             "\n", prefix, "]");
      return absl::StrCat(prefix, "Node(nodeset_idx=", nodeset_idx,
                          ", children=", children_part, ")");
    }
  };

  struct Edge {
    int edgeset_idx;
    bool reversed;
    std::unique_ptr<Node> node;
    int hop_width;

    // Debug string
    std::string to_string(int indent) const {
      std::string prefix(indent * 2, ' ');
      return absl::StrCat(prefix, "Edge(edgeset_idx=", edgeset_idx,
                          ", reversed=", reversed, ", hop_width=", hop_width,
                          ", node=\n", node->to_string(indent + 1), ")");
    }
  };

  std::unique_ptr<Node> root;
  bool with_replacement;

  // Total number of steps. "step_idx" in "Node" are in [0, num_steps).
  size_t num_steps;

  // Mapping between step index and the corresponding node.
  // Does not own the Node*.
  // Guarantee that "step_to_nodes[i]->step_idx == i".
  std::vector<Node*> step_to_nodes;

  // Assigns unique `step_idx` to each `Node` for efficient lookups. Call only
  // after the plan is fully built.
  void ComputeStepIdx();

  // Returns the Node for a given `step_idx`. Requires `ComputeStepIdx` to have
  // been called before.
  absl::StatusOr<const Node&> StepIdxToNode(int step_idx);

  // Debug string
  std::string to_string() const {
    return absl::StrCat("SamplingPlan(root=\n", root->to_string(0),
                        ",\n  with_replacement=", with_replacement, "\n)");
  }
};

}  // namespace dgf::sampling::in_memory_sampler

#endif  // DGF_SRC_SAMPLING_IN_MEMORY_SAMPLER_H_

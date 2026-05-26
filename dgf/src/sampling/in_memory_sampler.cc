#include "dgf/src/sampling/in_memory_sampler.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "dgf/src/util/util.h"

namespace dgf::sampling::in_memory_sampler {

absl::Status AdjacencyIndex::SampleRandomUniform(
    const InputIdx source_node, const std::size_t num_samples,
    std::vector<InputIdx>* result, std::mt19937_64* rng,
    InputIdx masked_edge_idx) const {
  if (!timestamps.empty()) {
    return absl::FailedPreconditionError(
        "Cannot use SampleRandomUniform when timestamps are available. Use "
        "SampleRandomUniformWithTimestamp instead.");
  }

  if (source_node + 1 >= source_blocks.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The source node idx ", source_node,
        " is out of bounds. Should be lower than ", source_blocks.size() - 1));
  }
  result->clear();

  // Get the list of candidate target nodes.
  const InputIdx start_idx = source_blocks[source_node];
  const InputIdx end_idx = source_blocks[source_node + 1];
  const InputIdx num_neighbors = end_idx - start_idx;
  std::uniform_int_distribution<size_t> dist(0, num_neighbors - 1);

  if (num_neighbors == 0) {
    return absl::OkStatus();
  }

  // Check if there is any masked edge for this node.
  bool has_masked_edge = false;
  if (masked_edge_idx != -1) {
    DGF_STATUS_CHECK(!edge_idxs.empty());
    for (size_t i = 0; i < num_neighbors; i++) {
      if (edge_idxs[start_idx + i] == masked_edge_idx) {
        has_masked_edge = true;
        break;
      }
    }
  }

  if (has_masked_edge) {
    // Count valid neighbors.
    size_t num_valid_neighbors = 0;
    for (size_t i = 0; i < num_neighbors; i++) {
      if (edge_idxs[start_idx + i] != masked_edge_idx) {
        num_valid_neighbors++;
      }
    }

    if (num_valid_neighbors == 0) {
      return absl::OkStatus();  // All neighbors are masked
    }

    if (num_samples >= num_valid_neighbors) {
      // Sample all available valid neighbors.
      result->reserve(result->size() + num_valid_neighbors);
      for (size_t i = 0; i < num_neighbors; i++) {
        if (edge_idxs[start_idx + i] != masked_edge_idx) {
          result->push_back(target_node_idxs[start_idx + i]);
        }
      }
    } else {
      if (num_samples < 16 && num_samples * 2 < num_valid_neighbors) {
        // Fast path for small sample sizes with masking.
        size_t selected_indices[16];
        for (size_t i = 0; i < num_samples; i++) {
          while (true) {
            size_t idx = dist(*rng);
            if (edge_idxs[start_idx + idx] == masked_edge_idx) {
              continue;  // Skip masked
            }
            bool duplicate = false;
            for (size_t j = 0; j < i; ++j) {
              if (selected_indices[j] == idx) {
                duplicate = true;
                break;
              }
            }
            if (!duplicate) {
              selected_indices[i] = idx;
              result->push_back(target_node_idxs[start_idx + idx]);
              break;
            }
          }
        }
      } else {
        // Fallback to reservoir sampling.
        size_t initial_size = result->size();
        size_t valid_count = 0;
        for (size_t i = 0; i < num_neighbors; i++) {
          if (edge_idxs[start_idx + i] != masked_edge_idx) {
            valid_count++;
            if (result->size() - initial_size < num_samples) {
              result->push_back(target_node_idxs[start_idx + i]);
            } else {
              std::uniform_int_distribution<size_t> dist(0, valid_count - 1);
              size_t r = dist(*rng);
              if (r < num_samples) {
                (*result)[initial_size + r] = target_node_idxs[start_idx + i];
              }
            }
          }
        }
      }
    }
  } else {
    // No masking (or no masked edge found for this node).
    auto first = target_node_idxs.begin() + start_idx;
    auto last = target_node_idxs.begin() + end_idx;
    if (num_samples >= num_neighbors) {
      // Sample all available neighbors.
      result->insert(result->end(), first, last);
    } else {
      // Sample num_samples unique neighbors.
      if (num_samples < 16 && num_samples * 2 < num_neighbors) {
        // Fast path for small sample sizes.
        size_t selected_indices[16];
        for (size_t i = 0; i < num_samples; i++) {
          while (true) {
            size_t idx = dist(*rng);
            bool duplicate = false;
            for (size_t j = 0; j < i; ++j) {
              if (selected_indices[j] == idx) {
                duplicate = true;
                break;
              }
            }
            if (!duplicate) {
              selected_indices[i] = idx;
              result->push_back(*(first + idx));
              break;
            }
          }
        }
      } else {
        std::sample(first, last, std::back_inserter(*result), num_samples,
                    *rng);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status AdjacencyIndex::SampleFirst(InputIdx source_node,
                                         std::size_t num_samples,
                                         std::vector<InputIdx>* result,
                                         InputIdx masked_edge_idx) const {
  if (!timestamps.empty()) {
    return absl::FailedPreconditionError(
        "Cannot use SampleRandomUniform when timestamps are available. Use "
        "SampleRandomUniformWithTimestamp instead.");
  }

  if (source_node + 1 >= source_blocks.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The source node idx ", source_node,
        " is out of bounds. Should be lower than ", source_blocks.size() - 1));
  }
  result->clear();

  // Get the list of candidate target nodes.
  InputIdx start_idx = source_blocks[source_node];
  InputIdx end_idx = source_blocks[source_node + 1];
  InputIdx num_neighbors = end_idx - start_idx;

  if (masked_edge_idx != -1 && !edge_idxs.empty()) {
    DGF_STATUS_CHECK(!edge_idxs.empty());
    for (size_t i = 0; i < num_neighbors && result->size() < num_samples; i++) {
      if (edge_idxs[start_idx + i] != masked_edge_idx) {
        result->push_back(target_node_idxs[start_idx + i]);
      }
    }
  } else {
    std::size_t count = std::min(num_samples, num_neighbors);
    if (count > 0) {
      auto first = target_node_idxs.begin() + start_idx;
      auto last = first + count;
      result->insert(result->end(), first, last);
    }
  }
  return absl::OkStatus();
}

absl::flat_hash_map<std::string, int> IndexStringValues(
    absl::Span<const std::string> string_list) {
  absl::flat_hash_map<std::string, int> map;
  for (size_t i = 0; i < string_list.size(); i++) {
    map[string_list[i]] = i;
  }
  return map;
}

AdjacencyIndex AdjacencyIndex::CreateEmpty(const size_t num_source_nodes,
                                           const size_t num_target_nodes) {
  AdjacencyIndex index;
  index.source_blocks.assign(num_source_nodes + 1, 0);
  return index;
}

template <bool HasTimestamp, bool HasEdgeIdx>
absl::StatusOr<AdjacencyIndex> AdjacencyIndex::CreateFromEdgeList(
    std::vector<Edge<HasTimestamp, HasEdgeIdx>>&& edges,
    const size_t num_source_nodes, const size_t num_target_nodes) {
  auto custom_less = [&](const Edge<HasTimestamp, HasEdgeIdx>& a,
                         const Edge<HasTimestamp, HasEdgeIdx>& b) {
    if (a.source != b.source) return a.source < b.source;
    if constexpr (HasTimestamp) {
      if (a.timestamp != b.timestamp) return a.timestamp < b.timestamp;
    }
    return a.target < b.target;
  };

  std::sort(edges.begin(), edges.end(), custom_less);

  AdjacencyIndex index;
  index.source_blocks.resize(num_source_nodes + 1);
  index.target_node_idxs.reserve(edges.size());
  if constexpr (HasTimestamp) index.timestamps.reserve(edges.size());
  if constexpr (HasEdgeIdx) index.edge_idxs.reserve(edges.size());

  std::size_t current_source = 0;
  index.source_blocks[0] = 0;
  for (size_t i = 0; i < edges.size(); i++) {
    InputIdx source_node = edges[i].source;
    InputIdx target_node = edges[i].target;
    while (source_node > current_source) {
      index.source_blocks[++current_source] = index.target_node_idxs.size();
    }
    index.target_node_idxs.push_back(target_node);
    if constexpr (HasTimestamp) index.timestamps.push_back(edges[i].timestamp);
    if constexpr (HasEdgeIdx) index.edge_idxs.push_back(edges[i].edge_idx);
  }
  while (current_source < num_source_nodes) {
    index.source_blocks[++current_source] = index.target_node_idxs.size();
  }
  return index;
}

// Explicit instantiations
template absl::StatusOr<AdjacencyIndex> AdjacencyIndex::CreateFromEdgeList<
    false, false>(std::vector<Edge<false, false>>&& edges, size_t, size_t);
template absl::StatusOr<AdjacencyIndex> AdjacencyIndex::CreateFromEdgeList<
    true, false>(std::vector<Edge<true, false>>&& edges, size_t, size_t);
template absl::StatusOr<AdjacencyIndex> AdjacencyIndex::CreateFromEdgeList<
    false, true>(std::vector<Edge<false, true>>&& edges, size_t, size_t);

void SamplingPlan::ComputeStepIdx() {
  step_to_nodes.clear();
  if (!root) {
    return;
  }
  std::queue<Node*> q;
  q.push(root.get());
  while (!q.empty()) {
    Node* current_node = q.front();
    q.pop();
    current_node->step_idx = step_to_nodes.size();
    step_to_nodes.push_back(current_node);
    for (const auto& edge : current_node->children) {
      if (edge.node) {
        q.push(edge.node.get());
      }
    }
  }
  num_steps = step_to_nodes.size();
}

absl::StatusOr<const SamplingPlan::Node&> SamplingPlan::StepIdxToNode(
    int step_idx) {
  if (step_to_nodes.empty()) {
    return absl::InvalidArgumentError(
        "The sampling plan is empty. Call ComputeStepIdx.");
  }
  if (step_idx < 0 || step_idx >= step_to_nodes.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "step_idx ", step_idx, " is out of bounds. Should be between 0 and ",
        step_to_nodes.size() - 1));
  }
  return *step_to_nodes[step_idx];
}

absl::Status AdjacencyIndex::SampleRandomUniformWithTimestamp(
    const InputIdx source_node, Timestamp seed_timestamp,
    const std::size_t num_samples, std::vector<InputIdx>* result,
    std::mt19937_64* rng) const {
  if (source_node + 1 >= source_blocks.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The source node idx ", source_node,
        " is out of bounds. Should be lower than ", source_blocks.size() - 1));
  }
  result->clear();

  InputIdx start_idx = source_blocks[source_node];
  InputIdx end_idx = source_blocks[source_node + 1];

  if (timestamps.empty()) {
    return SampleRandomUniform(source_node, num_samples, result, rng);
  }

  auto first_time = timestamps.begin() + start_idx;
  auto last_time = timestamps.begin() + end_idx;
  auto end_valid_it = std::upper_bound(first_time, last_time, seed_timestamp);
  InputIdx num_neighbors = std::distance(first_time, end_valid_it);

  auto first = target_node_idxs.begin() + start_idx;
  auto last = first + num_neighbors;
  if (num_samples >= num_neighbors) {
    result->insert(result->end(), first, last);
  } else {
    std::sample(first, last, std::back_inserter(*result), num_samples, *rng);
  }
  return absl::OkStatus();
}

absl::Status AdjacencyIndex::SampleFirstWithTimestamp(
    InputIdx source_node, Timestamp seed_timestamp, std::size_t num_samples,
    std::vector<InputIdx>* result) const {
  if (source_node + 1 >= source_blocks.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The source node idx ", source_node,
        " is out of bounds. Should be lower than ", source_blocks.size() - 1));
  }
  result->clear();

  InputIdx start_idx = source_blocks[source_node];
  InputIdx end_idx = source_blocks[source_node + 1];

  if (timestamps.empty()) {
    return SampleFirst(source_node, num_samples, result);
  }

  auto first_time = timestamps.begin() + start_idx;
  auto last_time = timestamps.begin() + end_idx;
  auto end_valid_it = std::upper_bound(first_time, last_time, seed_timestamp);
  InputIdx num_neighbors = std::distance(first_time, end_valid_it);

  std::size_t count = std::min(num_samples, (std::size_t)num_neighbors);
  if (count > 0) {
    auto first = target_node_idxs.begin() + start_idx;
    auto last = first + count;
    result->insert(result->end(), first, last);
  }
  return absl::OkStatus();
}

}  // namespace dgf::sampling::in_memory_sampler

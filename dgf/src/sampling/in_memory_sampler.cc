#include "dgf/src/sampling/in_memory_sampler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "dgf/src/util/util.h"

namespace dgf::sampling::in_memory_sampler {
namespace {

template <typename T>
bool GatherTypedRows(const GatherRowsTask& task) {
  const T* src = reinterpret_cast<const T*>(task.src);
  T* dst = reinterpret_cast<T*>(task.dst);
  for (std::size_t i = task.begin; i < task.end; i++) {
    const InputIdx j = task.idxs[i];
    if (j >= task.src_num_rows) return false;
    dst[i] = src[j];
  }
  return true;
}

// Appends `num_samples` distinct values sampled uniformly at random without
// replacement from the `num_candidates` values starting at `first`. Requires
// `num_samples < num_candidates`.
//
// The cost is O(num_samples) when `num_samples` is small compared to
// `num_candidates` (e.g., sampling the neighbors of a high-degree node), and
// O(num_candidates) otherwise.
void SampleDistinct(std::vector<InputIdx>::const_iterator first,
                    const std::size_t num_candidates,
                    const std::size_t num_samples,
                    std::vector<InputIdx>* result, Rng* rng) {
  DCHECK_LT(num_samples, num_candidates);
  if (num_samples * 2 >= num_candidates) {
    // Selection sampling. O(num_candidates) = O(num_samples).
    std::sample(first, first + num_candidates, std::back_inserter(*result),
                num_samples, *rng);
  } else if (num_samples < 16) {
    // Rejection sampling with a linear scan for duplicates.
    size_t selected_indices[16];
    for (size_t i = 0; i < num_samples; i++) {
      while (true) {
        size_t idx = absl::Uniform<size_t>(*rng, 0, num_candidates);
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
    // Rejection sampling with a hash set for duplicates. Since less than half
    // of the candidates are selected, each draw succeeds with probability
    // > 1/2.
    absl::flat_hash_set<size_t> selected_indices;
    selected_indices.reserve(num_samples);
    while (selected_indices.size() < num_samples) {
      const size_t idx = absl::Uniform<size_t>(*rng, 0, num_candidates);
      if (selected_indices.insert(idx).second) {
        result->push_back(*(first + idx));
      }
    }
  }
}

}  // namespace

absl::Status AdjacencyIndex::SampleRandomUniform(
    const InputIdx source_node, const std::size_t num_samples,
    std::vector<InputIdx>* result, Rng* rng, InputIdx masked_edge_idx) const {
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
            size_t idx = absl::Uniform<size_t>(*rng, 0, num_neighbors);
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
              size_t r = absl::Uniform<size_t>(*rng, 0, valid_count);
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
      SampleDistinct(first, num_neighbors, num_samples, result, rng);
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

void SamplerIndex::ComputeStepSlots() {
  step_slots_.assign(plan_.num_steps, -1);
  num_step_slots_.assign(nodesets_.size(), 0);
  for (const SamplingPlan::Node* node : plan_.step_to_nodes) {
    if (!node->children.empty()) {
      step_slots_[node->step_idx] = num_step_slots_[node->nodeset_idx]++;
    }
  }
}

SampleIdx SampleBuilder::AddNode(NodeSet& nodeset, InputIdx node_idx) {
  const SampleIdx sampled_node_idx =
      nodeset.sampled_node_idx_to_node_idx.size();
  nodeset.sampled_node_idx_to_node_idx.push_back(node_idx);
  nodeset.expanded_steps.resize(
      nodeset.expanded_steps.size() + nodeset.words_per_node, 0);
  return sampled_node_idx;
}

bool SampleBuilder::MarkExpanded(NodeSet& nodeset, SampleIdx sampled_node_idx,
                                 int slot) {
  uint64_t& word =
      nodeset.expanded_steps[sampled_node_idx * nodeset.words_per_node +
                             slot / 64];
  const uint64_t mask = uint64_t{1} << (slot % 64);
  if (word & mask) {
    return false;
  }
  word |= mask;
  return true;
}

absl::Status SampleBuilder::RecursiveGrow(
    const SamplerIndex& index, const SamplingPlan::Node& plan_node,
    InputIdx source_node, SampleIdx source_sampled_node,
    std::optional<Timestamp> seed_timestamp, InputIdx masked_edge_idx,
    int depth) {
  DGF_STATUS_CHECK(depth < recursion_cache.size());
  auto& cache_node_idxs = recursion_cache[depth];
  cache_node_idxs.clear();

  for (const auto& plan_edge : plan_node.children) {
    // Get the edge data to sample from.
    const SamplerIndex::EdgeSet& edgeset =
        index.edgesets_[plan_edge.edgeset_idx];
    const std::optional<AdjacencyIndex>& edges =
        plan_edge.reversed ? edgeset.backward_index : edgeset.forward_index;
    DGF_STATUS_CHECK(edges.has_value());

    // Sample target nodes / edges.
    if (seed_timestamp.has_value()) {
      DGF_STATUS_CHECK(index.edgeset_to_mask_idx_ == -1);
      if (index.debug_sampling_) {
        DGF_RETURN_IF_ERROR(edges->SampleFirstWithTimestamp(
            source_node, *seed_timestamp, plan_edge.hop_width,
            &cache_node_idxs));
      } else {
        DGF_RETURN_IF_ERROR(edges->SampleRandomUniformWithTimestamp(
            source_node, *seed_timestamp, plan_edge.hop_width, &cache_node_idxs,
            &rng));
      }
    } else {
      InputIdx current_masked_edge_idx = -1;
      if (plan_edge.edgeset_idx == index.edgeset_to_mask_idx_) {
        current_masked_edge_idx = masked_edge_idx;
      }

      if (index.debug_sampling_) {
        DGF_RETURN_IF_ERROR(edges->SampleFirst(source_node, plan_edge.hop_width,
                                               &cache_node_idxs,
                                               current_masked_edge_idx));
      } else {
        DGF_RETURN_IF_ERROR(edges->SampleRandomUniform(
            source_node, plan_edge.hop_width, &cache_node_idxs, &rng,
            current_masked_edge_idx));
      }
    }

    // Recursively sample the sub-nodes.
    const SamplingPlan::Node& target_plan_node = *plan_edge.node;
    auto& sample_target_nodeset = nodesets[target_plan_node.nodeset_idx];
    auto& sample_edgeset = edgesets[plan_edge.edgeset_idx];
    // -1 if the target step is a leaf i.e. there is nothing to expand.
    const int step_slot = index.step_slots_[target_plan_node.step_idx];

    const auto add_edge = [&](SampleIdx src, SampleIdx trg) {
      if (!plan_edge.reversed) {
        sample_edgeset.edges.push_back({src, trg});
      } else {
        sample_edgeset.edges.push_back({trg, src});
      }
    };

    for (const auto target_node : cache_node_idxs) {
      // Record the target node and create/reuse a sampled node.
      bool recurse = step_slot >= 0;

      SampleIdx target_sampled_node;
      if (!with_replacement) {
        // Sampling without replacement.
        // Note: The new sampled node idx is the number of sampled nodes (and
        // not the size of the map) since a seed node can be repeated.
        auto [target_sampled_node_it, inserted] =
            sample_target_nodeset.node_idx_to_sampled_node_idx.try_emplace(
                target_node,
                sample_target_nodeset.sampled_node_idx_to_node_idx.size());
        target_sampled_node = target_sampled_node_it->second;
        add_edge(source_sampled_node, target_sampled_node);

        if (inserted) {
          AddNode(sample_target_nodeset, target_node);
        }

        if (!multi_visit && recurse) {
          // Further expand the node iff it was not already expanded at this
          // plan step. Note that the same node can be expanded multiple
          // times through different plan steps.
          recurse = MarkExpanded(sample_target_nodeset, target_sampled_node,
                                 step_slot);
        }

      } else {
        // Sampling with replacement.
        target_sampled_node = AddNode(sample_target_nodeset, target_node);
        add_edge(source_sampled_node, target_sampled_node);
      }

      // Build the sub-sample.
      if (recurse) {
        DGF_RETURN_IF_ERROR(RecursiveGrow(index, target_plan_node, target_node,
                                          target_sampled_node, seed_timestamp,
                                          masked_edge_idx, depth + 1));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status SampleBuilder::Grow(const SamplerIndex& index,
                                 std::span<const InputIdx> seed_node_idxs,
                                 std::optional<Timestamp> seed_timestamp,
                                 InputIdx masked_edge_idx, GrowMode mode) {
  const bool subgraph = mode == GrowMode::kSubGraph;
  with_replacement = !subgraph && index.plan_.with_replacement;
  multi_visit = !subgraph && index.plan_.multi_visit;

  // Pre-allocate recursion cache to avoid reallocations during recursion.
  if (recursion_cache.size() < index.plan_.num_steps) {
    recursion_cache.resize(index.plan_.num_steps);
  }

  // Reuse capacity of edgesets and nodesets.
  if (edgesets.size() < index.edgesets_.size()) {
    edgesets.resize(index.edgesets_.size());
  }
  for (auto& es : edgesets) {
    es.edges.clear();
  }

  if (nodesets.size() < index.nodesets_.size()) {
    nodesets.resize(index.nodesets_.size());
  }
  const bool track_expanded_steps = !with_replacement && !multi_visit;
  for (std::size_t i = 0; i < nodesets.size(); i++) {
    auto& ns = nodesets[i];
    ns.sampled_node_idx_to_node_idx.clear();
    ns.node_idx_to_sampled_node_idx.clear();
    ns.expanded_steps.clear();
    ns.words_per_node =
        track_expanded_steps ? (index.num_step_slots_[i] + 63) / 64 : 0;
  }

  // Record the seed nodes as the first sampled nodes.
  const SamplingPlan::Node& root = *index.plan_.root;
  auto& seed_nodeset = nodesets[root.nodeset_idx];
  const std::size_t num_seed_nodeset_nodes =
      index.nodesets_[root.nodeset_idx].num_nodes;
  for (const InputIdx seed_node_idx : seed_node_idxs) {
    if (seed_node_idx >= num_seed_nodeset_nodes) {
      return absl::InvalidArgumentError(
          absl::StrCat("Seed node idx ", seed_node_idx,
                       " is out of bounds. Should be lower than ",
                       num_seed_nodeset_nodes, "."));
    }
    const SampleIdx sample_seed_node_idx = AddNode(seed_nodeset, seed_node_idx);
    if (!with_replacement) {
      // Note: If a seed node is repeated, the last occurrence is used.
      seed_nodeset.node_idx_to_sampled_node_idx[seed_node_idx] =
          sample_seed_node_idx;
    }
  }

  // Recursive transversal.
  for (std::size_t i = 0; i < seed_node_idxs.size(); i++) {
    const InputIdx seed_node_idx = seed_node_idxs[i];
    const SampleIdx sample_seed_node_idx =
        with_replacement
            ? static_cast<SampleIdx>(i)
            : seed_nodeset.node_idx_to_sampled_node_idx[seed_node_idx];
    DGF_RETURN_IF_ERROR(RecursiveGrow(index, root, seed_node_idx,
                                      sample_seed_node_idx, seed_timestamp,
                                      masked_edge_idx,
                                      /*depth=*/0));
  }

  // Deduplicate the edges. Without replacement, the same edge can be reached
  // several times (e.g., from both of its nodes when `reverse=True`).
  // Note: We also sort edges in debug mode.
  const bool dedup_edges = !with_replacement;
  if (dedup_edges || index.debug_sampling_) {
    for (auto& sampled_edgeset : edgesets) {
      std::sort(sampled_edgeset.edges.begin(), sampled_edgeset.edges.end());
      if (dedup_edges) {
        sampled_edgeset.edges.erase(std::unique(sampled_edgeset.edges.begin(),
                                                sampled_edgeset.edges.end()),
                                    sampled_edgeset.edges.end());
      }
    }
  }
  return absl::OkStatus();
}

absl::Status AdjacencyIndex::SampleRandomUniformWithTimestamp(
    const InputIdx source_node, Timestamp seed_timestamp,
    const std::size_t num_samples, std::vector<InputIdx>* result,
    Rng* rng) const {
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
    SampleDistinct(first, num_neighbors, num_samples, result, rng);
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

MergeLayout ComputeMergeLayout(
    const std::vector<std::vector<std::size_t>>& sample_num_nodes,
    const std::vector<std::vector<std::size_t>>& sample_num_edges,
    const MergePadding& padding) {
  const auto compute_offsets =
      [](const std::vector<std::vector<std::size_t>>& counts,
         const std::vector<std::optional<std::size_t>>& padded,
         std::vector<std::vector<std::size_t>>* offsets,
         std::vector<std::size_t>* totals) {
        offsets->resize(counts.size());
        totals->resize(counts.size());
        for (std::size_t set_idx = 0; set_idx < counts.size(); set_idx++) {
          auto& set_offsets = (*offsets)[set_idx];
          set_offsets.assign(counts[set_idx].size() + 1, 0);
          for (std::size_t i = 0; i < counts[set_idx].size(); i++) {
            set_offsets[i + 1] = set_offsets[i] + counts[set_idx][i];
          }
          (*totals)[set_idx] = padded[set_idx].value_or(set_offsets.back());
        }
      };
  MergeLayout layout;
  compute_offsets(sample_num_nodes, padding.num_nodes, &layout.node_offsets,
                  &layout.num_nodes);
  compute_offsets(sample_num_edges, padding.num_edges, &layout.edge_offsets,
                  &layout.num_edges);
  return layout;
}

std::optional<PaddingOverflow> FindPaddingOverflow(
    const MergeLayout& layout, const MergePadding& padding) {
  for (std::size_t i = 0; i < layout.node_offsets.size(); i++) {
    const std::size_t required = layout.node_offsets[i].back() + 1;
    if (padding.num_nodes[i].has_value() && *padding.num_nodes[i] < required) {
      return PaddingOverflow{true, static_cast<int>(i), required,
                             *padding.num_nodes[i]};
    }
  }
  for (std::size_t i = 0; i < layout.edge_offsets.size(); i++) {
    const std::size_t required = layout.edge_offsets[i].back();
    if (padding.num_edges[i].has_value() && *padding.num_edges[i] < required) {
      return PaddingOverflow{false, static_cast<int>(i), required,
                             *padding.num_edges[i]};
    }
  }
  return std::nullopt;
}

void FillPaddingEdgeEndpoints(const MergeLayout& layout,
                              const std::size_t nodeset_idx, int64_t* begin,
                              int64_t* const end) {
  const std::size_t num_nodes = layout.num_nodes[nodeset_idx];
  const std::size_t num_real_nodes = layout.node_offsets[nodeset_idx].back();
  DCHECK_GT(num_nodes, num_real_nodes);
  const std::size_t num_sentinels = num_nodes - num_real_nodes;
  const auto last_node = static_cast<int64_t>(num_nodes - 1);
  std::size_t slot = 0;
  for (; begin != end; ++begin) {
    *begin = last_node - static_cast<int64_t>(slot);
    if (++slot == num_sentinels) slot = 0;
  }
}

bool GatherRows(const GatherRowsTask& task) {
  if (task.idxs == nullptr) {
    std::memset(task.dst + task.begin * task.row_bytes, 0,
                (task.end - task.begin) * task.row_bytes);
    return true;
  }
  switch (task.row_bytes) {
    case 8:
      return GatherTypedRows<int64_t>(task);
    case 4:
      return GatherTypedRows<int32_t>(task);
    case 2:
      return GatherTypedRows<int16_t>(task);
    case 1:
      return GatherTypedRows<int8_t>(task);
    default:
      for (std::size_t i = task.begin; i < task.end; i++) {
        const InputIdx j = task.idxs[i];
        if (j >= task.src_num_rows) return false;
        std::memcpy(task.dst + i * task.row_bytes,
                    task.src + j * task.row_bytes, task.row_bytes);
      }
      return true;
  }
}

void AppendGatherRowsTasks(const char* src, const std::size_t src_num_rows,
                           const InputIdx* idxs, const std::size_t num_idxs,
                           char* dst, const std::size_t num_dst_rows,
                           const std::size_t row_bytes,
                           const std::size_t rows_per_task,
                           std::vector<GatherRowsTask>* tasks) {
  DCHECK_GT(rows_per_task, 0);
  for (std::size_t begin = 0; begin < num_idxs; begin += rows_per_task) {
    tasks->push_back({src, src_num_rows, dst, idxs, begin,
                      std::min(num_idxs, begin + rows_per_task), row_bytes});
  }
  for (std::size_t begin = num_idxs; begin < num_dst_rows;
       begin += rows_per_task) {
    tasks->push_back({src, src_num_rows, dst, /*idxs=*/nullptr, begin,
                      std::min(num_dst_rows, begin + rows_per_task),
                      row_bytes});
  }
}

}  // namespace dgf::sampling::in_memory_sampler

// Nanobind extension code for the in memory sampler.
// This code is tested in in_memory_sampler_test.py
//
// This code can only be called from python.
//
// The main object of the in memory sampler are as follow:
//   - Sampler: An in-memory representation of the graph optimized for sampling.
//   - SamplingPlan: The sampling plan i.e., the meta-graph of nodeset/edgesets
//     to visit.
//   - SampleBuilder: A temporary object used to accumulate node/edge indices
//     during the sampling process, and then convert them into a graph sample.
//   - AdjacencyIndex: A set of edges indexed for efficient sampling.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <latch>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"  // IWYU pragma: keep
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "dgf/src/data/schema.h"
#include "dgf/src/data/schema_nb.h"
#include "dgf/src/sampling/in_memory_sampler.h"
#include "dgf/src/sampling/in_memory_sampler_nb.h"
#include "dgf/src/util/concurrency.h"
#include "dgf/src/util/nanobind_util.h"
#include "dgf/src/util/status_caster.h"
#include "dgf/src/util/util.h"

namespace nb = nanobind;

namespace dgf::sampling::in_memory_sampler {

struct SampleBuilder;  // Forward declaration

// In-memory index used for graph sampling. This struct stores node and edge
// indices, as well as edge pairs, but does not include feature values.
struct Sampler {
  struct EdgeSet {
    std::string name;
    int source_nodeset = -1;
    int target_nodeset = -1;
    bool need_forward = false;
    bool need_backward = false;
    std::optional<AdjacencyIndex> forward_index;
    std::optional<AdjacencyIndex> backward_index;
  };

  struct NodeSet {
    std::string name;
    std::size_t num_nodes;

    // List of edgesets where this nodeset is a source.
    std::vector<int> as_source_edgeset;

    // List of edgestes where this nodeset is a target.
    std::vector<int> as_target_edgeset;
  };

  // Sampling plan i.e. order of the sampling operations.
  SamplingPlan plan_;

  // Per edge-set information indexed by edgeset index.
  std::vector<EdgeSet> edgesets_;

  // Per nodeset-set information indexed by nodeset index.
  std::vector<NodeSet> nodesets_;

  // Index of python modules
  ModuleIndex module_index_;

  // Random number generator.
  // TODO(gbm): Create a pool for when using multi-threaded sampling.
  std::mt19937_64 rng_;

  // If true, the sampler runs a deterministic sampling useful for debugging:
  //   - When sampling k unique edges, the k edges comming from the k nodes with
  //   the lowest ids are sampled (and sorted).
  bool debug_sampling_ = false;

  util::concurrency::ThreadPool thread_pool;

  std::unique_ptr<data::GraphSchema> schema_;
  bool has_temporal_edgesets_ = false;
  int edgeset_to_mask_idx_ = -1;

  // Pool of available SampleBuilders to reuse them across seeds and calls.
  std::vector<std::unique_ptr<SampleBuilder>> sample_builder_pool_;
  std::mutex sample_builder_pool_mutex_;

  std::unique_ptr<SampleBuilder> AcquireSampleBuilder();
  void ReleaseSampleBuilder(std::unique_ptr<SampleBuilder> builder);

  Sampler(int num_threads) : thread_pool(num_threads) {}

  std::string __str__() const {
    auto map_formatter = [](std::string* out,
                            const std::pair<const std::string, int>& pair) {
      absl::StrAppend(out, pair.first, ": ", pair.second);
    };

    // Sort nodeset_index_ by key for consistent string representation.
    std::vector<std::pair<std::string, int>> nodeset_items(
        schema_->nodeset_name_to_idx.begin(),
        schema_->nodeset_name_to_idx.end());
    std::sort(nodeset_items.begin(), nodeset_items.end());
    const std::string nodeset_str =
        absl::StrJoin(nodeset_items, ", ", map_formatter);

    // Sort edgeset_index_ by key for consistent string representation.
    std::vector<std::pair<std::string, int>> edgeset_items(
        schema_->edgeset_name_to_idx.begin(),
        schema_->edgeset_name_to_idx.end());
    std::sort(edgeset_items.begin(), edgeset_items.end());
    const std::string edgeset_str =
        absl::StrJoin(edgeset_items, ", ", map_formatter);

    // Format nodesets_.
    std::vector<std::string> nodeset_parts;
    nodeset_parts.reserve(nodesets_.size());
    for (size_t i = 0; i < nodesets_.size(); ++i) {
      nodeset_parts.push_back(absl::StrCat(
          "{idx=", i, ", num_nodes=", nodesets_[i].num_nodes, "}"));
    }
    const std::string nodesets_str = absl::StrJoin(nodeset_parts, ", ");

    return absl::StrCat("Sampler(\n  plan=", plan_.to_string(),
                        ",\n  nodeset_index={", nodeset_str, "}",
                        ",\n  edgeset_index={", edgeset_str, "}",
                        ",\n  nodesets=[", nodesets_str, "]\n)");
  }

  Sampler() = default;

  // Creates a new sample.
  absl::StatusOr<nb::list> Sample(
      const nb::ndarray<int64_t, nb::numpy, nb::shape<-1>>& seed_node_idxs,
      std::optional<nb::ndarray<int64_t, nb::numpy, nb::shape<-1>>>
          seed_timestamps = std::nullopt,
      std::optional<nb::ndarray<int64_t, nb::numpy, nb::shape<-1>>>
          masked_edge_idxs = std::nullopt);

  // Extracts the graph subset around the provided seed nodes.
  absl::StatusOr<nb::object> SubGraph(
      const std::vector<InputIdx>& seed_node_idx);

  // Extracts a graph subset around each provided seed nodes.
  absl::StatusOr<nb::list> MultiSubGraphs(
      const std::vector<InputIdx>& seed_node_idx);

  // Negative sampling using a random walk.
  //
  // For each of the given seed nodes (`seed_node_idxs`), which belong to the
  // source nodeset of the specified `edgeset_idx`, this method samples
  // `num_samples_per_seeds` nodes from the target nodeset of `edgeset_idx`. The
  // result is an array of shape `[seed_node_idxs.size(),
  // num_samples_per_seeds]` containing the indices of the sampled nodes in the
  // target nodeset.
  //
  // Target nodes that are direct neighbors of a seed node are excluded from the
  // sampled negative nodes.
  absl::StatusOr<nb::ndarray<int64_t, nb::numpy, nb::shape<-1, -1>>>
  RandomWalkNegativeSampling(
      const nb::ndarray<int64_t, nb::numpy, nb::shape<-1>>& seed_node_idxs,
      int target_edgeset_idx, int num_walks, int num_negatives_per_seed);

  // Returns the edgeset index give an edgeset name. Fails if the edgeset does
  // not exist.
  absl::StatusOr<int> EdgesetNameToEdgesetIdx(std::string& name);

  // Index the edgeset data.
  absl::Status IndexEdgeSets(const nb::object& py_graph,
                             nb::dict edgeset_timestamp_features);
};

// A struct to hold data during the creation of a sample.
struct SampleBuilder {
  struct EdgeSet {
    // The pair of sampled edges (i.e., src/target node index).
    absl::btree_set<std::pair<SampleIdx, SampleIdx>> edges;
  };

  struct NodeSet {
    // Mapping from node idx in the sample to node idx in the original graph.
    std::vector<InputIdx> sampled_node_idx_to_node_idx;
    // Inverse mapping of "sampled_node_idx_to_node_idx".
    // Only used when sampling without replacement.
    absl::btree_map<InputIdx, SampleIdx> node_idx_to_sampled_node_idx;
  };

  // Per edge-set information indexed by edgeset index.
  std::vector<EdgeSet> edgesets;
  // Per nodeset-set information indexed by nodeset index.
  std::vector<NodeSet> nodesets;

  // Cache to avoid heap allocations during recursive sampling.
  // `recursion_cache[depth]` stores the temporary sampled node indices at that
  // depth.
  std::vector<std::vector<InputIdx>> recursion_cache;

  // Random number generator.
  std::mt19937_64 rng;

  explicit SampleBuilder(std::mt19937_64::result_type rng_seed)
      : rng(rng_seed) {}

  absl::Status RecursiveGrow(const Sampler& sampler,
                             const SamplingPlan::Node& plan_node,
                             InputIdx source_node,
                             SampleIdx source_sampled_node,
                             std::optional<Timestamp> seed_timestamp,
                             InputIdx masked_edge_idx, int depth) {
    DGF_STATUS_CHECK(depth < recursion_cache.size());
    auto& cache_node_idxs = recursion_cache[depth];
    cache_node_idxs.clear();

    for (const auto& plan_edge : plan_node.children) {
      // Get the edge data to sample from.
      const Sampler::EdgeSet& edgeset =
          sampler.edgesets_[plan_edge.edgeset_idx];
      const std::optional<AdjacencyIndex>& edges =
          plan_edge.reversed ? edgeset.backward_index : edgeset.forward_index;
      DGF_STATUS_CHECK(edges.has_value());

      // Sample target nodes / edges.
      if (seed_timestamp.has_value()) {
        DGF_STATUS_CHECK(sampler.edgeset_to_mask_idx_ == -1);
        if (sampler.debug_sampling_) {
          DGF_RETURN_IF_ERROR(edges->SampleFirstWithTimestamp(
              source_node, *seed_timestamp, plan_edge.hop_width,
              &cache_node_idxs));
        } else {
          DGF_RETURN_IF_ERROR(edges->SampleRandomUniformWithTimestamp(
              source_node, *seed_timestamp, plan_edge.hop_width,
              &cache_node_idxs, &rng));
        }
      } else {
        InputIdx current_masked_edge_idx = -1;
        if (plan_edge.edgeset_idx == sampler.edgeset_to_mask_idx_) {
          current_masked_edge_idx = masked_edge_idx;
        }

        if (sampler.debug_sampling_) {
          DGF_RETURN_IF_ERROR(
              edges->SampleFirst(source_node, plan_edge.hop_width,
                                 &cache_node_idxs, current_masked_edge_idx));
        } else {
          DGF_RETURN_IF_ERROR(edges->SampleRandomUniform(
              source_node, plan_edge.hop_width, &cache_node_idxs, &rng,
              current_masked_edge_idx));
        }
      }

      // Recursively sample the sub-nodes.
      auto& sample_target_nodeset = nodesets[plan_edge.node->nodeset_idx];
      auto& sample_edgeset = edgesets[plan_edge.edgeset_idx];
      for (const auto target_node : cache_node_idxs) {
        // Record the target node and create/reuse a sampled node.

        SampleIdx target_sampled_node;
        if (!sampler.plan_.with_replacement) {
          // Sampling without replacement.
          auto [target_sampled_node_it, inserted] =
              sample_target_nodeset.node_idx_to_sampled_node_idx.try_emplace(
                  target_node,
                  sample_target_nodeset.node_idx_to_sampled_node_idx.size());
          target_sampled_node = target_sampled_node_it->second;

          if (inserted) {
            sample_target_nodeset.sampled_node_idx_to_node_idx.push_back(
                target_node);
            DGF_STATUS_CHECK(
                sample_target_nodeset.sampled_node_idx_to_node_idx.size() ==
                sample_target_nodeset.node_idx_to_sampled_node_idx.size());
          }
          // Record the edge, if it is a new edge.
          if (!plan_edge.reversed) {
            sample_edgeset.edges.insert(
                {source_sampled_node, target_sampled_node});
          } else {
            sample_edgeset.edges.insert(
                {target_sampled_node, source_sampled_node});
          }
        } else {
          // Sampling with replacement.
          target_sampled_node =
              sample_target_nodeset.sampled_node_idx_to_node_idx.size();
          sample_target_nodeset.sampled_node_idx_to_node_idx.push_back(
              target_node);
          // Record the edge, if it is a new edge.
          if (!plan_edge.reversed) {
            sample_edgeset.edges.insert(
                {source_sampled_node, target_sampled_node});
          } else {
            sample_edgeset.edges.insert(
                {target_sampled_node, source_sampled_node});
          }
        }

        // Build the sub-sample.
        DGF_RETURN_IF_ERROR(RecursiveGrow(sampler, *plan_edge.node, target_node,
                                          target_sampled_node, seed_timestamp,
                                          masked_edge_idx, depth + 1));
      }
    }
    return absl::OkStatus();
  }

  absl::Status Grow(const Sampler& sampler, InputIdx seed_node_idx,
                    std::optional<Timestamp> seed_timestamp,
                    InputIdx masked_edge_idx) {
    // Pre-allocate recursion cache to avoid reallocations during recursion.
    if (recursion_cache.size() < sampler.plan_.num_steps) {
      recursion_cache.resize(sampler.plan_.num_steps);
    }

    // Reuse capacity of edgesets and nodesets.
    if (edgesets.size() < sampler.edgesets_.size()) {
      edgesets.resize(sampler.edgesets_.size());
    }
    for (auto& es : edgesets) {
      es.edges.clear();
    }

    if (nodesets.size() < sampler.nodesets_.size()) {
      nodesets.resize(sampler.nodesets_.size());
    }
    for (auto& ns : nodesets) {
      ns.sampled_node_idx_to_node_idx.clear();
      ns.node_idx_to_sampled_node_idx.clear();
    }

    // Record the seed node as the first sampled node.
    SampleIdx sample_seed_node_idx = 0;
    auto& seed_nodeset = nodesets[sampler.plan_.root->nodeset_idx];
    seed_nodeset.sampled_node_idx_to_node_idx.push_back(seed_node_idx);
    if (!sampler.plan_.with_replacement) {
      seed_nodeset.node_idx_to_sampled_node_idx[seed_node_idx] =
          sample_seed_node_idx;
    }

    // Recursive transversal.
    return RecursiveGrow(sampler, *sampler.plan_.root, seed_node_idx,
                         sample_seed_node_idx, seed_timestamp, masked_edge_idx,
                         /*depth=*/0);
  }

  absl::StatusOr<nb::object> ExportToInMemoryGraph(const Sampler& sampler) {
    nb::dict py_nodesets;
    nb::dict py_edgesets;
    for (std::size_t nodeset_idx = 0; nodeset_idx < nodesets.size();
         nodeset_idx++) {
      const auto& nodeset_name = sampler.nodesets_[nodeset_idx].name;
      const auto& sampled_nodeset = nodesets[nodeset_idx];
      const int num_nodes = sampled_nodeset.sampled_node_idx_to_node_idx.size();
      nb::dict py_features;
      py_features[sampler.module_index_.key_idx_feature] =
          NodeIdxsToNumpyArray(sampled_nodeset.sampled_node_idx_to_node_idx);
      py_nodesets[string_to_py_str(nodeset_name)] =
          sampler.module_index_.nodeset_cls(num_nodes, py_features);
    }

    for (std::size_t edgeset_idx = 0; edgeset_idx < edgesets.size();
         edgeset_idx++) {
      const auto& edgeset = sampler.edgesets_[edgeset_idx];
      const auto& sampled_edgeset = edgesets[edgeset_idx];

      auto py_adjacency = EdgesToNumpyArray(sampled_edgeset.edges);
      py_edgesets[string_to_py_str(edgeset.name)] =
          sampler.module_index_.edgeset_cls(std::move(py_adjacency));
    }
    return sampler.module_index_.graph_cls(py_nodesets, py_edgesets);
  }
};

std::unique_ptr<SampleBuilder> Sampler::AcquireSampleBuilder() {
  std::lock_guard<std::mutex> lock(sample_builder_pool_mutex_);
  if (sample_builder_pool_.empty()) {
    return std::make_unique<SampleBuilder>(rng_());
  }
  auto builder = std::move(sample_builder_pool_.back());
  sample_builder_pool_.pop_back();
  return builder;
}

void Sampler::ReleaseSampleBuilder(std::unique_ptr<SampleBuilder> builder) {
  std::lock_guard<std::mutex> lock(sample_builder_pool_mutex_);
  sample_builder_pool_.push_back(std::move(builder));
}

// Creates a graph sample starting from a given seed node.
// The returned `nb::object` is an instance of `InMemoryGraph`
// containing the sampled subgraph.
absl::StatusOr<nb::list> Sampler::Sample(
    const nb::ndarray<int64_t, nb::numpy, nb::shape<-1>>& seed_node_idxs,
    std::optional<nb::ndarray<int64_t, nb::numpy, nb::shape<-1>>>
        seed_timestamps,
    std::optional<nb::ndarray<int64_t, nb::numpy, nb::shape<-1>>>
        masked_edge_idxs) {
  auto seed_node_idxs_view = seed_node_idxs.view();
  std::size_t num_seeds = seed_node_idxs_view.shape(0);

  if (seed_timestamps.has_value()) {
    if (!has_temporal_edgesets_) {
      return absl::InvalidArgumentError(
          "seed_timestamps provided but no temporal edgesets configured. Set "
          "the 'edgeset_timestamp_features' field in the sampling plan.");
    }
    if (num_seeds != seed_timestamps->view().shape(0)) {
      return absl::InvalidArgumentError(
          "seed_node_idxs and seed_timestamps must have the same size");
    }
  }
  if (masked_edge_idxs.has_value()) {
    if (edgeset_to_mask_idx_ == -1) {
      return absl::InvalidArgumentError(
          "masked_edge_idxs provided but no edgeset to mask configured.");
    }
    if (num_seeds != masked_edge_idxs->view().shape(0)) {
      return absl::InvalidArgumentError(
          "seed_node_idxs and masked_edge_idxs must have the same size");
    }
  }

  // Pre-allocate pool if needed.
  {
    std::lock_guard<std::mutex> lock(sample_builder_pool_mutex_);
    while (sample_builder_pool_.size() < thread_pool.num_threads()) {
      sample_builder_pool_.push_back(std::make_unique<SampleBuilder>(rng_()));
    }
  }

  std::vector<std::unique_ptr<SampleBuilder>> active_builders(num_seeds);

  // Create samples
  {
    // Release the GIL during the non-python sampling process.
    nb::gil_scoped_release release;

    // Generate seeds sequentially in the main thread to ensure determinism.
    std::vector<uint64_t> seeds(num_seeds);
    for (size_t i = 0; i < num_seeds; ++i) {
      seeds[i] = rng_();
    }

    // Start the sampling.
    absl::Status global_status;
    std::mutex global_status_mutex;
    std::latch latch(num_seeds);

    for (size_t seed_idx = 0; seed_idx < num_seeds; seed_idx++) {
      const auto seed_node_idx =
          static_cast<InputIdx>(seed_node_idxs_view(seed_idx));
      std::optional<Timestamp> seed_timestamp = std::nullopt;
      if (seed_timestamps.has_value()) {
        seed_timestamp = seed_timestamps->view()(seed_idx);
      }
      InputIdx masked_edge_idx = -1;
      if (masked_edge_idxs.has_value()) {
        masked_edge_idx = masked_edge_idxs->view()(seed_idx);
      }
      const uint64_t seed = seeds[seed_idx];
      thread_pool.Schedule([this, seed_idx, seed, &active_builders,
                            seed_node_idx, seed_timestamp, masked_edge_idx,
                            &latch, &global_status_mutex, &global_status]() {
        std::unique_ptr<SampleBuilder> sample_builder = AcquireSampleBuilder();
        sample_builder->rng.seed(seed);

        const auto status = sample_builder->Grow(
            *this, seed_node_idx, seed_timestamp, masked_edge_idx);

        active_builders[seed_idx] = std::move(sample_builder);
        latch.count_down();

        // If the sampling failed, record the failure.
        if (!status.ok()) {
          util::concurrency::MutexLock l(global_status_mutex);
          global_status.Update(status);
        }
      });
    }

    // Wait for all the sampling to be done.
    latch.wait();

    // Return an error if any of the samplers failed.
    if (!global_status.ok()) {
      // Release builders back to pool even on failure.
      for (auto& builder : active_builders) {
        if (builder != nullptr) {
          ReleaseSampleBuilder(std::move(builder));
        }
      }
      return global_status;
    }
  }

  // Convert samples into python objects.
  nb::list graphs;

  for (auto& sample_builder : active_builders) {
    DGF_ASSIGN_OR_RETURN(auto graph,
                         sample_builder->ExportToInMemoryGraph(*this));
    graphs.append(graph);
    // Release builder back to pool.
    ReleaseSampleBuilder(std::move(sample_builder));
  }
  return graphs;
}

namespace {

// Thread-local visited node tracking cache to completely avoid heap allocation
// and O(N_graph) resets during subgraph extraction loops.
//
// Instead of allocating a fresh visited array of size N_graph per seed,
// each OS worker thread in the pool reuses its own thread-local cache
// instance. Resets between seeds are performed in O(1) by incrementing
// `generation`.
struct ThreadLocalVisited {
  struct NodesetVisited {
    // visited_gen[node_idx] stores the generation number when node_idx was
    // visited in the current seed's BFS tree.
    // If visited_gen[node_idx] == generation, the node has been visited in
    // the current seed.
    std::vector<size_t> visited_gen;

    // sample_idxs[node_idx] maps the global input node index to its local
    // sampled output subgraph index.
    std::vector<SampleIdx> sample_idxs;

    // visited_step_gen[node_idx * num_steps + step_idx] stores the generation
    // number when node_idx was visited during a specific plan step.
    // Flattened 2D vector for memory locality.
    std::vector<size_t> visited_step_gen;

    // Number of steps in the sampling plan.
    size_t num_steps = 0;
  };
  std::vector<NodesetVisited> nodesets;
  size_t generation = 0;

  void ResetOrResize(const Sampler* sampler) {
    nodesets.resize(sampler->nodesets_.size());
    for (size_t i = 0; i < sampler->nodesets_.size(); ++i) {
      size_t num_nodes = sampler->nodesets_[i].num_nodes;
      size_t num_steps = sampler->plan_.num_steps;
      nodesets[i].num_steps = num_steps;

      nodesets[i].visited_gen.resize(num_nodes, 0);
      nodesets[i].sample_idxs.resize(num_nodes, 0);
      nodesets[i].visited_step_gen.resize(num_nodes * num_steps, 0);
    }
    generation++;
    if (generation == 0) {
      for (auto& ns : nodesets) {
        std::fill(ns.visited_gen.begin(), ns.visited_gen.end(), 0);
        std::fill(ns.visited_step_gen.begin(), ns.visited_step_gen.end(), 0);
      }
      generation = 1;
    }
  }
};

thread_local ThreadLocalVisited tl_visited;

}  // namespace

// Helper struct and methods to extract a subgraph by performing a breadth-first
// search starting from a set of seed nodes.
//
// Usage example:
//  WorkingNodeset a;
//  a.ExtractSubGraph(...);
//  return a.SubGraphToPyhon()
struct SubGraphExtractor {
  static constexpr SampleIdx kNonVisited =
      std::numeric_limits<SampleIdx>::max();

  struct WorkingNodeset {
    // List of the input node idxs to return: "i \in node_idxs iff visited[i]
    // is true".
    //
    // This arrays also define the mapping from input to output node idxs.
    std::vector<InputIdx> node_idxs;
  };
  std::vector<WorkingNodeset> working_nodesets;

  struct WorkingEdgeset {
    // Source and target output node idxs.
    absl::btree_set<std::pair<SampleIdx, SampleIdx>> edges;
  };
  std::vector<WorkingEdgeset> working_edgesets;

  absl::Status RecursiveGrow(const Sampler& sampler,
                             const SamplingPlan::Node& plan_node,
                             InputIdx source_node,
                             SampleIdx source_sampled_node,
                             std::mt19937_64* rng) {
    for (const auto& plan_edge : plan_node.children) {
      // Get the edge data to sample from.
      const Sampler::EdgeSet& edgeset =
          sampler.edgesets_[plan_edge.edgeset_idx];
      const std::optional<AdjacencyIndex>& edges =
          plan_edge.reversed ? edgeset.backward_index : edgeset.forward_index;
      DGF_STATUS_CHECK(edges.has_value());

      // Sample target nodes / edges.
      std::vector<InputIdx> cache_node_idxs;
      if (sampler.debug_sampling_) {
        DGF_RETURN_IF_ERROR(edges->SampleFirst(source_node, plan_edge.hop_width,
                                               &cache_node_idxs));
      } else {
        DGF_RETURN_IF_ERROR(edges->SampleRandomUniform(
            source_node, plan_edge.hop_width, &cache_node_idxs, rng));
      }

      // Recursively sample the sub-nodes.
      int target_nodeset_idx = plan_edge.node->nodeset_idx;
      auto& sample_target_nodeset = working_nodesets[target_nodeset_idx];
      auto& tl_nodeset = tl_visited.nodesets[target_nodeset_idx];
      auto& sample_edgeset = working_edgesets[plan_edge.edgeset_idx];

      for (const auto target_node : cache_node_idxs) {
        bool is_visited =
            (tl_nodeset.visited_gen[target_node] == tl_visited.generation);
        size_t step_idx = plan_edge.node->step_idx;
        bool step_visited =
            is_visited &&
            (tl_nodeset.visited_step_gen[target_node * tl_nodeset.num_steps +
                                         step_idx] == tl_visited.generation);

        bool continue_recusion = true;
        size_t effective_target_sampled_node;
        if (is_visited) {
          // The node was already visited.
          effective_target_sampled_node = tl_nodeset.sample_idxs[target_node];
          if (step_visited) {
            // The node was already visited with this sampling plan step.
            continue_recusion = false;
          } else {
            // This node was not already visited with this sampling plan step.
            tl_nodeset.visited_step_gen[target_node * tl_nodeset.num_steps +
                                        step_idx] = tl_visited.generation;
          }
        } else {
          // This is a new node; index it + remember it.
          effective_target_sampled_node =
              sample_target_nodeset.node_idxs.size();
          tl_nodeset.visited_gen[target_node] = tl_visited.generation;
          tl_nodeset.sample_idxs[target_node] = effective_target_sampled_node;
          sample_target_nodeset.node_idxs.push_back(target_node);
          tl_nodeset
              .visited_step_gen[target_node * tl_nodeset.num_steps + step_idx] =
              tl_visited.generation;
        }

        // Try to add the edges (automatic dedup).
        if (!plan_edge.reversed) {
          sample_edgeset.edges.insert(
              {source_sampled_node, effective_target_sampled_node});
        } else {
          sample_edgeset.edges.insert(
              {effective_target_sampled_node, source_sampled_node});
        }

        if (!continue_recusion) {
          continue;
        }

        // Build the sub-sample.
        DGF_RETURN_IF_ERROR(RecursiveGrow(sampler, *plan_edge.node, target_node,
                                          effective_target_sampled_node, rng));
      }
    }
    return absl::OkStatus();
  }

  // Extracts the subgraph.
  absl::Status ExtractSubGraph(const std::vector<InputIdx>& seed_node_idxs,
                               const Sampler* sampler, std::mt19937_64* rng) {
    // Initialize the working memory.
    working_nodesets.resize(sampler->nodesets_.size());
    working_edgesets.resize(sampler->edgesets_.size());

    // Reset or resize thread-local structures in O(1)
    tl_visited.ResetOrResize(sampler);

    // Record the seed nodes.
    for (const auto seed_node_idx : seed_node_idxs) {
      int nodeset_idx = sampler->plan_.root->nodeset_idx;
      auto& nodeset = working_nodesets[nodeset_idx];
      auto& tl_nodeset = tl_visited.nodesets[nodeset_idx];

      const SampleIdx sample_seed_node_idx = nodeset.node_idxs.size();
      nodeset.node_idxs.push_back(seed_node_idx);

      // Mark visited in current generation
      tl_nodeset.visited_gen[seed_node_idx] = tl_visited.generation;
      tl_nodeset.sample_idxs[seed_node_idx] = sample_seed_node_idx;

      size_t step_idx = sampler->plan_.root->step_idx;
      tl_nodeset
          .visited_step_gen[seed_node_idx * tl_nodeset.num_steps + step_idx] =
          tl_visited.generation;
    }

    // Grow the graph.
    for (const auto seed_node_idx : seed_node_idxs) {
      int nodeset_idx = sampler->plan_.root->nodeset_idx;
      auto& tl_nodeset = tl_visited.nodesets[nodeset_idx];
      const SampleIdx sample_seed_node_idx =
          tl_nodeset.sample_idxs[seed_node_idx];
      DGF_RETURN_IF_ERROR(RecursiveGrow(*sampler, *sampler->plan_.root,
                                        seed_node_idx, sample_seed_node_idx,
                                        rng));
    }
    return absl::OkStatus();
  }

  // Convert the extracted sub-graph into a python graph.
  absl::StatusOr<nb::object> SubGraphToPyhon(const Sampler* sampler) const {
    nb::dict py_nodesets;
    nb::dict py_edgesets;
    for (std::size_t nodeset_idx = 0; nodeset_idx < working_nodesets.size();
         nodeset_idx++) {
      const auto& nodeset_name = sampler->nodesets_[nodeset_idx].name;
      const auto& working_nodeset = working_nodesets[nodeset_idx];
      const int num_nodes = working_nodeset.node_idxs.size();
      nb::dict py_features;
      py_features[sampler->module_index_.key_idx_feature] =
          NodeIdxsToNumpyArray(working_nodeset.node_idxs);
      py_nodesets[string_to_py_str(nodeset_name)] =
          sampler->module_index_.nodeset_cls(num_nodes, py_features);
    }

    for (std::size_t edgeset_idx = 0; edgeset_idx < working_edgesets.size();
         edgeset_idx++) {
      const auto& edgeset = sampler->edgesets_[edgeset_idx];
      const auto& working_edgeset = working_edgesets[edgeset_idx];

      auto py_adjacency = EdgesToNumpyArray(working_edgeset.edges);
      py_edgesets[string_to_py_str(edgeset.name)] =
          sampler->module_index_.edgeset_cls(std::move(py_adjacency));
    }
    return sampler->module_index_.graph_cls(py_nodesets, py_edgesets);
  }
};

struct RandomWalkNegativeSamplerHelper {
  const Sampler* sampler;
  const Sampler::EdgeSet* edgeset;
  size_t num_target_nodes;
  bool is_homogeneous;
  int num_walks;
  int num_negatives_per_seed;

  absl::Status SampleForSeed(InputIdx seed_node, int64_t* output_for_seed_node,
                             std::mt19937_64* local_rng) const {
    absl::flat_hash_map<InputIdx, int> visit_counts;
    const int target_nodeset_idx = edgeset->target_nodeset;

    // Phase 1: Walk Simulation
    for (int walk_idx = 0; walk_idx < num_walks; walk_idx++) {
      InputIdx cur_node = seed_node;
      const SamplingPlan::Node* cur_plan_node = sampler->plan_.root.get();

      while (!cur_plan_node->children.empty()) {
        std::uniform_int_distribution<size_t> e_dist(
            0, cur_plan_node->children.size() - 1);
        const auto& plan_edge = cur_plan_node->children[e_dist(*local_rng)];

        const auto& current_edgeset = sampler->edgesets_[plan_edge.edgeset_idx];
        const auto& index = plan_edge.reversed ? current_edgeset.backward_index
                                               : current_edgeset.forward_index;
        if (!index.has_value()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Edgeset '", current_edgeset.name,
                           "' does not have the required index. Ensure it is "
                           "properly configured in the SamplingPlan."));
        }

        auto neighbors = index->Targets(cur_node);
        if (neighbors.empty()) break;

        std::uniform_int_distribution<size_t> n_dist(0, neighbors.size() - 1);
        cur_node = neighbors[n_dist(*local_rng)];
        cur_plan_node = plan_edge.node.get();

        if (cur_plan_node->nodeset_idx == target_nodeset_idx) {
          if (is_homogeneous && cur_node == seed_node) continue;
          visit_counts[cur_node]++;
        }
      }
    }

    // Phase 2: Filtering Out Invalid Negatives
    struct Candidate {
      InputIdx node;
      int count;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(visit_counts.size());

    for (const auto& pair : visit_counts) {
      InputIdx target_node = pair.first;
      if (!edgeset->forward_index->HasEdge(seed_node, target_node)) {
        candidates.push_back({target_node, pair.second});
      }
    }

    // Phase 3: Selection (Top-K Ranking)
    const int num_extracted =
        std::min(static_cast<int>(candidates.size()), num_negatives_per_seed);
    if (num_extracted < candidates.size()) {
      std::nth_element(candidates.begin(), candidates.begin() + num_extracted,
                       candidates.end(),
                       [](const Candidate& a, const Candidate& b) {
                         return a.count > b.count;
                       });
    }
    std::sort(candidates.begin(), candidates.begin() + num_extracted,
              [](const Candidate& a, const Candidate& b) {
                return a.count > b.count;
              });

    for (int i = 0; i < num_extracted; i++) {
      output_for_seed_node[i] = static_cast<int64_t>(candidates[i].node);
    }
    int num_filled = num_extracted;

    // Phase 4: Robust Fallback
    if (num_filled < num_negatives_per_seed) {
      // Note: The remaining slots are filled by randomly sampling nodes with
      // replacement. We only ensure the sampled node is not the seed node,
      // without checking for existing edges.
      std::uniform_int_distribution<InputIdx> t_dist(0, num_target_nodes - 1);
      while (num_filled < num_negatives_per_seed) {
        InputIdx candidate = t_dist(*local_rng);
        if (is_homogeneous && candidate == seed_node) {
          continue;
        }
        output_for_seed_node[num_filled++] = static_cast<int64_t>(candidate);
      }
    }
    return absl::OkStatus();
  }
};

absl::StatusOr<nb::ndarray<int64_t, nb::numpy, nb::shape<-1, -1>>>
Sampler::RandomWalkNegativeSampling(
    const nb::ndarray<int64_t, nb::numpy, nb::shape<-1>>& seed_node_idxs,
    int target_edgeset_idx, int num_walks, int num_negatives_per_seed) {
  if (target_edgeset_idx < 0 || target_edgeset_idx >= edgesets_.size()) {
    return absl::InvalidArgumentError("Invalid target_edgeset_idx");
  }
  const auto& edgeset = edgesets_[target_edgeset_idx];
  if (!edgeset.forward_index.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Edgeset '", edgeset.name, "' does not have a forward index. ",
        "Ensure it is traversed forward in the SamplingPlan."));
  }

  const size_t num_target_nodes = nodesets_[edgeset.target_nodeset].num_nodes;
  const bool is_homogeneous =
      (edgeset.source_nodeset == edgeset.target_nodeset);

  RandomWalkNegativeSamplerHelper helper{
      this,           &edgeset,  num_target_nodes,
      is_homogeneous, num_walks, num_negatives_per_seed};

  auto seed_node_idxs_view = seed_node_idxs.view();
  const size_t num_seeds = seed_node_idxs_view.shape(0);

  int64_t* output_data = new int64_t[num_seeds * num_negatives_per_seed];

  std::vector<std::mt19937_64> rngs;
  rngs.reserve(num_seeds);

  absl::Status global_status;
  std::mutex global_status_mutex;

  {
    nb::gil_scoped_release release;

    for (size_t i = 0; i < num_seeds; i++) {
      rngs.emplace_back(rng_());
    }

    std::latch latch(num_seeds);

    for (size_t seed_idx = 0; seed_idx < num_seeds; seed_idx++) {
      const InputIdx seed_node =
          static_cast<InputIdx>(seed_node_idxs_view(seed_idx));
      int64_t* output_for_seed_node =
          output_data + (seed_idx * num_negatives_per_seed);

      thread_pool.Schedule([&helper, seed_node, output_for_seed_node, seed_idx,
                            &rngs, &latch, &global_status_mutex,
                            &global_status]() {
        absl::Status status = helper.SampleForSeed(
            seed_node, output_for_seed_node, &rngs[seed_idx]);
        latch.count_down();

        if (!status.ok()) {
          util::concurrency::MutexLock l(global_status_mutex);
          global_status.Update(status);
        }
      });
    }

    latch.wait();
    if (!global_status.ok()) {
      delete[] output_data;
      return global_status;
    }
  }

  nb::capsule owner(output_data,
                    [](void* p) noexcept { delete[] (int64_t*)p; });
  return nb::ndarray<int64_t, nb::numpy, nb::shape<-1, -1>>(
      output_data, {num_seeds, static_cast<size_t>(num_negatives_per_seed)},
      owner);
}

absl::StatusOr<int> Sampler::EdgesetNameToEdgesetIdx(std::string& name) {
  auto it = schema_->edgeset_name_to_idx.find(name);
  if (it == schema_->edgeset_name_to_idx.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Edgeset '", name, "' not found"));
  }
  return it->second;
}

absl::StatusOr<nb::object> Sampler::SubGraph(
    const std::vector<InputIdx>& seed_node_idxs) {
  SubGraphExtractor extractor;
  {
    // Release the GIL.
    nb::gil_scoped_release release;
    // Create a local RNG
    std::mt19937_64 rng(rng_());
    DGF_RETURN_IF_ERROR(extractor.ExtractSubGraph(seed_node_idxs, this, &rng));
  }

  // Convert to a python object.
  return extractor.SubGraphToPyhon(this);
}

absl::StatusOr<nb::list> Sampler::MultiSubGraphs(
    const std::vector<InputIdx>& seed_node_idxs) {
  std::size_t num_seeds = seed_node_idxs.size();
  std::vector<SubGraphExtractor> extractors(num_seeds);
  // TODO(gbm): Don't re-create mt19937_64s at each sampling.
  std::vector<std::mt19937_64> rngs;
  rngs.reserve(num_seeds);

  {
    // Release GIL for parallel execution
    nb::gil_scoped_release release;

    for (size_t i = 0; i < num_seeds; ++i) {
      rngs.emplace_back(rng_());
    }

    absl::Status global_status;
    std::mutex global_status_mutex;
    std::latch latch(num_seeds);

    for (size_t i = 0; i < num_seeds; i++) {
      thread_pool.Schedule([&extractors, this, &seed_node_idxs, i, &rngs,
                            &latch, &global_status_mutex, &global_status]() {
        const auto status =
            extractors[i].ExtractSubGraph({seed_node_idxs[i]}, this, &rngs[i]);
        latch.count_down();

        if (!status.ok()) {
          util::concurrency::MutexLock l(global_status_mutex);
          global_status.Update(status);
        }
      });
    }
    latch.wait();
    DGF_RETURN_IF_ERROR(global_status);
  }

  nb::list graphs;
  for (size_t i = 0; i < num_seeds; ++i) {
    DGF_ASSIGN_OR_RETURN(auto graph, extractors[i].SubGraphToPyhon(this));
    graphs.append(graph);
  }
  return graphs;
}

// Builds an AdjacencyIndex from a numpy adjacency matrix.
absl::StatusOr<AdjacencyIndex> BuildAdjacencyIndex(
    const Adjacency& py_adjacency, std::optional<TimestampsArray> py_timestamps,
    bool reversed, std::size_t num_source_nodes, std::size_t num_target_nodes,
    bool populate_edge_idxs = false) {
  if (reversed) {
    std::swap(num_source_nodes, num_target_nodes);
  }
  const auto py_adjacency_view = py_adjacency.view();
  std::size_t num_edges = py_adjacency_view.shape(1);

  auto get_and_validate_edge =
      [&](std::size_t i) -> absl::StatusOr<std::pair<InputIdx, InputIdx>> {
    int64_t source_node = py_adjacency_view(0, i);
    int64_t target_node = py_adjacency_view(1, i);
    if (reversed) {
      std::swap(source_node, target_node);
    }
    if (source_node < 0 || source_node >= num_source_nodes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid source node index ", source_node,
          ". The node index should be between 0 and ", num_source_nodes, "."));
    }
    if (target_node < 0 || target_node >= num_target_nodes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid target node index ", target_node,
          ". The node index should be between 0 and ", num_target_nodes, "."));
    }
    return std::make_pair(static_cast<InputIdx>(source_node),
                          static_cast<InputIdx>(target_node));
  };

  if (py_timestamps.has_value()) {
    DGF_STATUS_CHECK(!populate_edge_idxs);
    const auto timestamps_view = py_timestamps->view();
    std::vector<Edge<true, false>> edges;
    edges.reserve(num_edges);
    for (std::size_t i = 0; i < num_edges; ++i) {
      DGF_ASSIGN_OR_RETURN(const auto edge, get_and_validate_edge(i));
      Edge<true, false> e;
      e.source = edge.first;
      e.target = edge.second;
      e.timestamp = static_cast<Timestamp>(timestamps_view(i));
      edges.push_back(e);
    }
    return AdjacencyIndex::CreateFromEdgeList<true, false>(
        std::move(edges), num_source_nodes, num_target_nodes);
  } else if (populate_edge_idxs) {
    DGF_STATUS_CHECK(!py_timestamps.has_value());
    std::vector<Edge<false, true>> edges;
    edges.reserve(num_edges);
    for (std::size_t i = 0; i < num_edges; ++i) {
      DGF_ASSIGN_OR_RETURN(const auto edge, get_and_validate_edge(i));
      Edge<false, true> e;
      e.source = edge.first;
      e.target = edge.second;
      e.edge_idx = static_cast<InputIdx>(i);
      edges.push_back(e);
    }
    return AdjacencyIndex::CreateFromEdgeList<false, true>(
        std::move(edges), num_source_nodes, num_target_nodes);
  } else {
    std::vector<Edge<false, false>> edges;
    edges.reserve(num_edges);
    for (std::size_t i = 0; i < num_edges; ++i) {
      DGF_ASSIGN_OR_RETURN(const auto edge, get_and_validate_edge(i));
      Edge<false, false> e;
      e.source = edge.first;
      e.target = edge.second;
      edges.push_back(e);
    }
    return AdjacencyIndex::CreateFromEdgeList<false, false>(
        std::move(edges), num_source_nodes, num_target_nodes);
  }
}

absl::Status Sampler::IndexEdgeSets(const nb::object& py_graph,
                                    nb::dict edgeset_timestamp_features) {
  for (auto item : edgeset_timestamp_features) {
    std::string edgeset_name = nb::cast<std::string>(item.first);
    if (schema_->edgeset_name_to_idx.find(edgeset_name) ==
        schema_->edgeset_name_to_idx.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Edgeset '", edgeset_name, "' does not exist in schema"));
    }
  }
  if (!edgeset_timestamp_features.empty()) {
    has_temporal_edgesets_ = true;
  }

  // Index the nodesets.
  nodesets_.assign(schema_->nodeset_name_to_idx.size(), {});
  DGF_GET_ATTR_OR_RETURN(nb::dict, py_node_sets, py_graph, "node_sets");
  for (const auto nodeset : schema_->nodeset_name_to_idx) {
    DGF_ASSIGN_OR_RETURN(const nb::object py_nodeset,
                         GetItemFromPyDict<nb::object>(
                             py_node_sets, string_to_py_str(nodeset.first)));
    DGF_GET_ATTR_OR_RETURN(std::size_t, num_nodes, py_nodeset, "num_nodes");
    nodesets_[nodeset.second].num_nodes = num_nodes;
    nodesets_[nodeset.second].name = nodeset.first;
  }

  // List the edgesets. Infer the structure from the plan (instead of the graph
  // schema).
  edgesets_.assign(schema_->edgeset_name_to_idx.size(), {});
  std::function<absl::Status(const SamplingPlan::Node&)> scan_plan =
      [&](const SamplingPlan::Node& node) -> absl::Status {
    for (const auto& child : node.children) {
      auto& edgeset = edgesets_[child.edgeset_idx];
      if (!child.reversed) {
        DGF_STATUS_CHECK(edgeset.source_nodeset == -1 ||
                         edgeset.source_nodeset == node.nodeset_idx);
        DGF_STATUS_CHECK(edgeset.target_nodeset == -1 ||
                         edgeset.target_nodeset == child.node->nodeset_idx);

        edgeset.source_nodeset = node.nodeset_idx;
        edgeset.target_nodeset = child.node->nodeset_idx;

        nodesets_[edgeset.source_nodeset].as_source_edgeset.push_back(
            child.edgeset_idx);
        nodesets_[edgeset.target_nodeset].as_target_edgeset.push_back(
            child.edgeset_idx);

        edgeset.need_forward = true;
      } else {
        DGF_STATUS_CHECK(edgeset.source_nodeset == -1 ||
                         edgeset.source_nodeset == child.node->nodeset_idx);
        DGF_STATUS_CHECK(edgeset.target_nodeset == -1 ||
                         edgeset.target_nodeset == node.nodeset_idx);

        edgeset.source_nodeset = child.node->nodeset_idx;
        edgeset.target_nodeset = node.nodeset_idx;

        nodesets_[edgeset.source_nodeset].as_source_edgeset.push_back(
            child.edgeset_idx);
        nodesets_[edgeset.target_nodeset].as_target_edgeset.push_back(
            child.edgeset_idx);

        edgeset.need_backward = true;
      }
      DGF_RETURN_IF_ERROR(scan_plan(*child.node));
    }
    return absl::OkStatus();
  };
  DGF_RETURN_IF_ERROR(scan_plan(*plan_.root));

  // Populate the edgeset names.
  for (const auto& edgeset : schema_->edgeset_name_to_idx) {
    edgesets_[edgeset.second].name = edgeset.first;
  }

  // Load and index the edges in memory.
  DGF_GET_ATTR_OR_RETURN(nb::dict, py_edge_sets, py_graph, "edge_sets");
  for (int edgeset_idx = 0; edgeset_idx < edgesets_.size(); edgeset_idx++) {
    // Get the adjacency.
    auto& edgeset = edgesets_[edgeset_idx];
    DGF_ASSIGN_OR_RETURN(const nb::object py_edgeset,
                         GetItemFromPyDict<nb::object>(
                             py_edge_sets, string_to_py_str(edgeset.name)));
    DGF_GET_ATTR_OR_RETURN(Adjacency, py_adjacency, py_edgeset, "adjacency");

    std::optional<TimestampsArray> py_timestamps = std::nullopt;

    nb::str py_edgeset_name = string_to_py_str(edgeset.name);
    if (edgeset_timestamp_features.contains(py_edgeset_name)) {
      DGF_ASSIGN_OR_RETURN(nb::str timestamp_feature_name,
                           GetItemFromPyDict<nb::str>(
                               edgeset_timestamp_features, py_edgeset_name));
      DGF_GET_ATTR_OR_RETURN(nb::dict, py_features, py_edgeset, "features");
      DGF_ASSIGN_OR_RETURN(TimestampsArray extracted_timestamps,
                           GetItemFromPyDict<TimestampsArray>(
                               py_features, timestamp_feature_name));
      py_timestamps = extracted_timestamps;
    }

    // Index the adjacency.
    bool populate_edge_idxs = (edgeset_idx == edgeset_to_mask_idx_);
    if (edgeset.need_forward) {
      DGF_ASSIGN_OR_RETURN(
          edgeset.forward_index,
          BuildAdjacencyIndex(py_adjacency, py_timestamps, false,
                              nodesets_[edgeset.source_nodeset].num_nodes,
                              nodesets_[edgeset.target_nodeset].num_nodes,
                              populate_edge_idxs));
    }
    if (edgeset.need_backward) {
      DGF_ASSIGN_OR_RETURN(
          edgeset.backward_index,
          BuildAdjacencyIndex(py_adjacency, py_timestamps, true,
                              nodesets_[edgeset.source_nodeset].num_nodes,
                              nodesets_[edgeset.target_nodeset].num_nodes,
                              populate_edge_idxs));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Sampler>> CreateSampler(
    const nb::object& graph, const nb::object& plan, const bool debug_sampling,
    const size_t num_threads, const int64_t seed, const nb::object& py_schema,
    std::optional<std::string> edgeset_to_mask = std::nullopt,
    nb::dict edgeset_timestamp_features = nb::dict()) {
  auto sampler = std::make_unique<Sampler>(num_threads);

  // Import the schema
  DGF_ASSIGN_OR_RETURN(sampler->schema_, data::CreateGraphSchema(py_schema));
  DGF_ASSIGN_OR_RETURN(sampler->plan_,
                       CreateSamplingPlan(plan, *sampler->schema_));

  if (edgeset_to_mask.has_value()) {
    auto it = sampler->schema_->edgeset_name_to_idx.find(*edgeset_to_mask);
    if (it == sampler->schema_->edgeset_name_to_idx.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Edgeset '", *edgeset_to_mask, "' does not exist in schema"));
    }
    sampler->edgeset_to_mask_idx_ = it->second;
  }

  DGF_RETURN_IF_ERROR(
      sampler->IndexEdgeSets(graph, edgeset_timestamp_features));
  if (seed >= 0) {
    sampler->rng_.seed(seed);
  } else {
    sampler->rng_.seed(std::random_device()());
  }
  sampler->debug_sampling_ = debug_sampling;
  return std::move(sampler);
}

NB_MODULE(_in_memory_sampler_ext, m) {
  nb::class_<Sampler>(m, "Sampler")
      .def("Sample", ValueOrThrowWrapper(&Sampler::Sample),
           nb::arg("seed_node_idx"), nb::arg("seed_timestamps") = nb::none(),
           nb::arg("masked_edge_idxs") = nb::none())
      .def("SubGraph", ValueOrThrowWrapper(&Sampler::SubGraph))
      .def("MultiSubGraphs", ValueOrThrowWrapper(&Sampler::MultiSubGraphs))
      .def("RandomWalkNegativeSampling",
           ValueOrThrowWrapper(&Sampler::RandomWalkNegativeSampling),
           nb::arg("seed_node_idxs"), nb::arg("target_edgeset_idx"),
           nb::arg("num_walks"), nb::arg("num_negatives_per_seed"))
      .def("EdgesetNameToEdgesetIdx",
           ValueOrThrowWrapper(&Sampler::EdgesetNameToEdgesetIdx))
      .def("__str__", &Sampler::__str__);

  nb::class_<AdjacencyIndex>(m, "AdjacencyIndex")
      .def("__str__", &AdjacencyIndex::to_string);

  m.def("CreateSampler", ValueOrThrowWrapper(CreateSampler), nb::arg("graph"),
        nb::arg("plan"), nb::arg("debug_sampling"), nb::arg("num_threads"),
        nb::arg("seed"), nb::arg("py_schema"),
        nb::arg("edgeset_to_mask") = nb::none(),
        nb::arg("edgeset_timestamp_features") = nb::dict());

  m.def("BuildAdjacencyIndex", ValueOrThrowWrapper(BuildAdjacencyIndex),
        nb::arg("py_adjacency"), nb::arg("py_timestamps") = nb::none(),
        nb::arg("reversed"), nb::arg("num_source_nodes"),
        nb::arg("num_target_nodes"), nb::arg("populate_edge_idxs") = false);
}

}  // namespace dgf::sampling::in_memory_sampler

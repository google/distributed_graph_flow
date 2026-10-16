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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <latch>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/random/distributions.h"
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

namespace dgf::sampling::in_memory_sampler {

struct Sampler;
struct SampledBatch;
struct MergeConfig;
struct RawMergedBatch;

// Sample builders borrowed from the pool of a `Sampler`, and returned to the
// pool on destruction.
class PooledBuilders {
 public:
  PooledBuilders() = default;
  PooledBuilders(Sampler* sampler, std::size_t num_builders);
  ~PooledBuilders();
  PooledBuilders(PooledBuilders&& other) noexcept;
  PooledBuilders& operator=(PooledBuilders&& other) noexcept;
  PooledBuilders(const PooledBuilders&) = delete;
  PooledBuilders& operator=(const PooledBuilders&) = delete;

  std::span<const std::unique_ptr<SampleBuilder>> builders() const {
    return builders_;
  }

 private:
  Sampler* sampler_ = nullptr;
  std::vector<std::unique_ptr<SampleBuilder>> builders_;
};

// In-memory index used for graph sampling. This struct stores node and edge
// indices, as well as edge pairs, but does not include feature values.
struct Sampler : SamplerIndex {
  // Index of python modules
  ModuleIndex module_index_;

  // Random number generator.
  // Not guarded by a mutex: Only used by the calling thread (the thread pool
  // workers get their own generator). A sampler should not be called
  // concurrently from several threads.
  // TODO(gbm): Create a pool for when using multi-threaded sampling.
  Rng rng_;

  util::concurrency::ThreadPool thread_pool;

  std::unique_ptr<data::GraphSchema> schema_;
  bool has_temporal_edgesets_ = false;

  // Pool of available SampleBuilders to reuse them across seeds and calls.
  util::concurrency::Mutex sample_builder_pool_mutex_;
  std::vector<std::unique_ptr<SampleBuilder>> sample_builder_pool_
      ABSL_GUARDED_BY(sample_builder_pool_mutex_);

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
      const NodeIdxs& seed_node_idxs,
      std::optional<TimestampsArray> seed_timestamps = std::nullopt,
      std::optional<EdgeIdxs> masked_edge_idxs = std::nullopt);

  // Samples and merges a batch of subgraphs. Returns
  // ((InMemoryGraph, offsets), None, None) if the batch fits in the padding, or
  // (None, SampledBatch, (is_nodeset, set_name, required, padded)) on padding
  // overflow so the caller can split the batch without re-sampling.
  absl::StatusOr<nb::tuple> SampleMerged(
      const NodeIdxs& seed_node_idxs,
      std::optional<TimestampsArray> seed_timestamps,
      std::optional<EdgeIdxs> masked_edge_idxs, const MergeConfig& config);

  // Same as `SampleMerged`, but for the sub-graphs of `MultiSubGraphs`.
  absl::StatusOr<nb::tuple> MultiSubGraphsMerged(const NodeIdxs& seed_node_idxs,
                                                 const MergeConfig& config);

  // Implementation of `SampleMerged` and `MultiSubGraphsMerged`.
  absl::StatusOr<nb::tuple> SampleMergedImpl(
      const NodeIdxs& seed_node_idxs,
      const std::optional<TimestampsArray>& seed_timestamps,
      const std::optional<EdgeIdxs>& masked_edge_idxs,
      const MergeConfig& config, GrowMode mode);

  // Implementation of `Sample` and `MultiSubGraphs`.
  absl::StatusOr<nb::list> SampleImpl(
      const NodeIdxs& seed_node_idxs,
      const std::optional<TimestampsArray>& seed_timestamps,
      const std::optional<EdgeIdxs>& masked_edge_idxs, GrowMode mode);

  // Converts a sample into a python `InMemoryGraph`.
  absl::StatusOr<nb::object> ExportToInMemoryGraph(
      const SampleBuilder& builder) const;

  // Grows one sample per seed node in the thread pool. Called without the GIL.
  absl::StatusOr<PooledBuilders> GrowSamplesNoGil(
      const NodeIdxs& seed_node_idxs,
      const std::optional<TimestampsArray>& seed_timestamps,
      const std::optional<EdgeIdxs>& masked_edge_idxs, GrowMode mode);

  // Gets `num_builders` sample builders from the pool (or creates them). Only
  // used by `PooledBuilders`.
  std::vector<std::unique_ptr<SampleBuilder>> AcquireBuilders(
      std::size_t num_builders);

  // Returns sample builders to the pool for re-use. Only used by
  // `PooledBuilders`.
  void ReleaseBuilders(std::vector<std::unique_ptr<SampleBuilder>>* builders);

  // Merges the builders [begin, end) into raw buffers. Called without the GIL.
  // If the samples overflow the padding, populates `overflow` and returns
  // `std::nullopt`.
  absl::StatusOr<std::optional<RawMergedBatch>> MergeBuildersNoGil(
      std::span<const std::unique_ptr<SampleBuilder>> builders,
      std::size_t begin, std::size_t end, const MergeConfig& config,
      PaddingOverflow* overflow);

  // Wraps `raw` into a python tuple (InMemoryGraph, offsets).
  nb::tuple ExportMergedBatch(const MergeConfig& config,
                              RawMergedBatch* raw) const;

  // Converts `overflow` into a python tuple (is_nodeset, set_name, required,
  // padded).
  nb::tuple ExportPaddingOverflow(const PaddingOverflow& overflow) const;

  // Creates the configuration of `SampleMerged` and `SampledBatch::Merge`.
  //
  // Args:
  //   padding_num_nodes: Dict nodeset name -> padded number of nodes.
  //   padding_num_edges: Dict edgeset name -> padded number of edges.
  //   features: Dict nodeset name -> list of features to export, as tuples
  //     (name, source values or None, output numpy dtype, output reshape
  //     argument). The source values are C-contiguous arrays (one row per
  //     node) gathered in c++. If None, the feature is exported as None (to be
  //     gathered in python).
  //   return_node_idxs: Whether to export the "#idx" feature.
  absl::StatusOr<std::unique_ptr<MergeConfig>> CreateMergeConfig(
      const nb::dict& padding_num_nodes, const nb::dict& padding_num_edges,
      const nb::dict& features, bool return_node_idxs);

  // Extracts the graph subset around the provided seed nodes.
  absl::StatusOr<nb::object> SubGraph(
      const std::vector<InputIdx>& seed_node_idx);

  // Extracts a graph subset around each provided seed nodes.
  absl::StatusOr<nb::list> MultiSubGraphs(const NodeIdxs& seed_node_idxs);

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
  RandomWalkNegativeSampling(const NodeIdxs& seed_node_idxs,
                             int target_edgeset_idx, int num_walks,
                             int num_negatives_per_seed);

  // Returns the edgeset index give an edgeset name. Fails if the edgeset does
  // not exist.
  absl::StatusOr<int> EdgesetNameToEdgesetIdx(std::string& name);

  // Index the edgeset data.
  absl::Status IndexEdgeSets(const nb::object& py_graph,
                             nb::dict edgeset_timestamp_features);
};

// Pre-computed configuration of `Sampler::SampleMerged` and
// `SampledBatch::Merge`. Created once with `Sampler::CreateMergeConfig` and
// used for all the batches.
struct MergeConfig {
  struct Feature {
    nb::str name;
    // If false, the feature is exported as None and gathered in python (e.g.
    // object arrays).
    bool gather_in_cc = false;
    RawArray src;
    const char* src_data = nullptr;
    std::size_t src_num_rows = 0;
    std::size_t row_bytes = 0;
    // The output is a uint8 array of shape [num_nodes, row_bytes] converted
    // with `.view(dtype).reshape(reshape)`.
    nb::object dtype;
    nb::tuple reshape;
  };

  MergePadding padding;
  // Features to export, indexed by nodeset idx.
  std::vector<std::vector<Feature>> features;
  // Whether to export the "#idx" feature, indexed by nodeset idx.
  std::vector<bool> export_node_idxs;

  MergeConfig() = default;
  MergeConfig(const MergeConfig&) = delete;
  MergeConfig& operator=(const MergeConfig&) = delete;
};

// Raw buffers produced by `Sampler::MergeBuildersNoGil`, before wrapping into
// python objects.
struct RawMergedBatch {
  MergeLayout layout;
  std::vector<std::unique_ptr<InputIdx[]>> node_bufs;
  std::vector<std::unique_ptr<int64_t[]>> edge_bufs;
  std::vector<std::vector<std::unique_ptr<uint8_t[]>>> feature_bufs;
};

// Samples kept in c++ memory when a batch overflows the padding, to be split
// and exported with `Merge`. The python object holding a `SampledBatch` keeps
// the `Sampler` alive.
struct SampledBatch {
  nb::object py_sampler;
  Sampler* sampler = nullptr;
  PooledBuilders builders;

  SampledBatch(Sampler* sampler, PooledBuilders builders)
      : py_sampler(nb::find(sampler)),
        sampler(sampler),
        builders(std::move(builders)) {}
  SampledBatch(const SampledBatch&) = delete;
  SampledBatch& operator=(const SampledBatch&) = delete;

  // Merges the samples [begin, end). Returns ((InMemoryGraph, offsets), None)
  // if the samples fit in the padding, or (None, (is_nodeset, set_name,
  // required, padded)) on padding overflow.
  absl::StatusOr<nb::tuple> Merge(std::size_t begin, std::size_t end,
                                  const MergeConfig& config);
};

// Creates a graph sample starting from a given seed node.
// The returned `nb::object` is an instance of `InMemoryGraph`
// containing the sampled subgraph.
absl::StatusOr<nb::list> Sampler::Sample(
    const NodeIdxs& seed_node_idxs,
    std::optional<TimestampsArray> seed_timestamps,
    std::optional<EdgeIdxs> masked_edge_idxs) {
  return SampleImpl(seed_node_idxs, seed_timestamps, masked_edge_idxs,
                    GrowMode::kPlan);
}

absl::StatusOr<nb::list> Sampler::MultiSubGraphs(
    const NodeIdxs& seed_node_idxs) {
  return SampleImpl(seed_node_idxs, std::nullopt, std::nullopt,
                    GrowMode::kSubGraph);
}

absl::StatusOr<nb::list> Sampler::SampleImpl(
    const NodeIdxs& seed_node_idxs,
    const std::optional<TimestampsArray>& seed_timestamps,
    const std::optional<EdgeIdxs>& masked_edge_idxs, const GrowMode mode) {
  PooledBuilders builders;
  {
    nb::gil_scoped_release release;
    DGF_ASSIGN_OR_RETURN(builders,
                         GrowSamplesNoGil(seed_node_idxs, seed_timestamps,
                                          masked_edge_idxs, mode));
  }

  // Convert samples into python objects.
  nb::list graphs;
  for (const auto& sample_builder : builders.builders()) {
    DGF_ASSIGN_OR_RETURN(auto graph, ExportToInMemoryGraph(*sample_builder));
    graphs.append(graph);
  }
  return graphs;
}

absl::StatusOr<nb::object> Sampler::ExportToInMemoryGraph(
    const SampleBuilder& builder) const {
  nb::dict py_nodesets;
  nb::dict py_edgesets;
  for (std::size_t nodeset_idx = 0; nodeset_idx < builder.nodesets.size();
       nodeset_idx++) {
    const auto& nodeset_name = nodesets_[nodeset_idx].name;
    const auto& sampled_nodeset = builder.nodesets[nodeset_idx];
    const int num_nodes = sampled_nodeset.sampled_node_idx_to_node_idx.size();
    nb::dict py_features;
    py_features[module_index_.key_idx_feature] =
        NodeIdxsToNumpyArray(sampled_nodeset.sampled_node_idx_to_node_idx);
    py_nodesets[string_to_py_str(nodeset_name)] =
        module_index_.nodeset_cls(num_nodes, py_features);
  }

  for (std::size_t edgeset_idx = 0; edgeset_idx < builder.edgesets.size();
       edgeset_idx++) {
    const auto& edgeset = edgesets_[edgeset_idx];
    const auto& sampled_edgeset = builder.edgesets[edgeset_idx];

    auto py_adjacency = EdgesToNumpyArray(sampled_edgeset.edges);
    py_edgesets[string_to_py_str(edgeset.name)] =
        module_index_.edgeset_cls(std::move(py_adjacency));
  }
  return module_index_.graph_cls(py_nodesets, py_edgesets);
}

PooledBuilders::PooledBuilders(Sampler* sampler, const std::size_t num_builders)
    : sampler_(sampler), builders_(sampler->AcquireBuilders(num_builders)) {}

PooledBuilders::~PooledBuilders() {
  if (sampler_ != nullptr) {
    sampler_->ReleaseBuilders(&builders_);
  }
}

PooledBuilders::PooledBuilders(PooledBuilders&& other) noexcept
    : sampler_(std::exchange(other.sampler_, nullptr)),
      builders_(std::move(other.builders_)) {}

PooledBuilders& PooledBuilders::operator=(PooledBuilders&& other) noexcept {
  // `other` releases the previous builders of `this` when destroyed.
  std::swap(sampler_, other.sampler_);
  std::swap(builders_, other.builders_);
  return *this;
}

std::vector<std::unique_ptr<SampleBuilder>> Sampler::AcquireBuilders(
    const std::size_t num_builders) {
  std::vector<std::unique_ptr<SampleBuilder>> builders(num_builders);
  util::concurrency::MutexLock lock(sample_builder_pool_mutex_);
  for (auto& builder : builders) {
    if (!sample_builder_pool_.empty()) {
      builder = std::move(sample_builder_pool_.back());
      sample_builder_pool_.pop_back();
    } else {
      builder = std::make_unique<SampleBuilder>();
    }
  }
  return builders;
}

absl::StatusOr<PooledBuilders> Sampler::GrowSamplesNoGil(
    const NodeIdxs& seed_node_idxs,
    const std::optional<TimestampsArray>& seed_timestamps,
    const std::optional<EdgeIdxs>& masked_edge_idxs, const GrowMode mode) {
  auto seed_node_idxs_view = seed_node_idxs.view();
  std::size_t num_seeds = seed_node_idxs_view.shape(0);

  // `kSubGraph` ignores the temporal options. `kPlan` requires the seed
  // timestamps to avoid sampling future edges by mistake.
  if (mode == GrowMode::kPlan && has_temporal_edgesets_ &&
      !seed_timestamps.has_value()) {
    return absl::InvalidArgumentError(
        "seed_timestamps must be provided with temporal sampling.");
  }
  if (seed_timestamps.has_value()) {
    if (!has_temporal_edgesets_) {
      return absl::InvalidArgumentError(
          "seed_timestamps provided but no temporal edgesets configured. Mark "
          "a node or edge feature as 'is_creation_time' in the graph schema.");
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

  PooledBuilders builders(this, num_seeds);

  // Generate seeds sequentially in the main thread to ensure determinism.
  std::vector<uint64_t> seeds(num_seeds);
  for (size_t i = 0; i < num_seeds; i++) {
    seeds[i] = rng_();
  }

  // Start the sampling.
  absl::Status global_status;
  util::concurrency::Mutex global_status_mutex;
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
    SampleBuilder* sample_builder = builders.builders()[seed_idx].get();
    thread_pool.Schedule([this, sample_builder, seed, seed_node_idx,
                          seed_timestamp, masked_edge_idx, mode, &latch,
                          &global_status_mutex, &global_status]() {
      sample_builder->rng = MakeRng(seed);

      const auto status = sample_builder->Grow(
          *this, {&seed_node_idx, 1}, seed_timestamp, masked_edge_idx, mode);

      // Record the failure before counting down: Once the latch reaches zero,
      // the calling thread can return and destroy `global_status`.
      if (!status.ok()) {
        util::concurrency::MutexLock l(global_status_mutex);
        global_status.Update(status);
      }

      latch.count_down();
    });
  }

  // Wait for all the sampling to be done.
  latch.wait();

  // Return an error if any of the samplers failed.
  DGF_RETURN_IF_ERROR(global_status);
  return builders;
}

void Sampler::ReleaseBuilders(
    std::vector<std::unique_ptr<SampleBuilder>>* builders) {
  util::concurrency::MutexLock lock(sample_builder_pool_mutex_);
  for (auto& builder : *builders) {
    if (builder) {
      sample_builder_pool_.push_back(std::move(builder));
    }
  }
  builders->clear();
}

absl::StatusOr<std::optional<RawMergedBatch>> Sampler::MergeBuildersNoGil(
    std::span<const std::unique_ptr<SampleBuilder>> builders,
    const std::size_t begin, const std::size_t end, const MergeConfig& config,
    PaddingOverflow* overflow) {
  // Minimum number of feature rows gathered by a task.
  constexpr std::size_t kMinRowsPerGatherTask = 2048;

  if (begin > end || end > builders.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid sample range [", begin, ", ", end,
                     ") for a batch of ", builders.size(), " samples"));
  }
  const std::size_t num_nodesets = nodesets_.size();
  const std::size_t num_edgesets = edgesets_.size();
  const std::size_t num_samples = end - begin;

  std::vector<std::vector<std::size_t>> num_nodes(num_nodesets);
  std::vector<std::vector<std::size_t>> num_edges(num_edgesets);
  for (std::size_t ns = 0; ns < num_nodesets; ns++) {
    num_nodes[ns].reserve(num_samples);
  }
  for (std::size_t es = 0; es < num_edgesets; es++) {
    num_edges[es].reserve(num_samples);
  }
  for (std::size_t i = begin; i < end; i++) {
    for (std::size_t ns = 0; ns < num_nodesets; ns++) {
      num_nodes[ns].push_back(
          builders[i]->nodesets[ns].sampled_node_idx_to_node_idx.size());
    }
    for (std::size_t es = 0; es < num_edgesets; es++) {
      num_edges[es].push_back(builders[i]->edgesets[es].edges.size());
    }
  }

  RawMergedBatch raw;
  raw.layout = ComputeMergeLayout(num_nodes, num_edges, config.padding);
  if (const auto maybe_overflow =
          FindPaddingOverflow(raw.layout, config.padding);
      maybe_overflow.has_value()) {
    *overflow = *maybe_overflow;
    return std::nullopt;
  }

  // Allocate the outputs.
  // The node idxs are exported as int64 numpy arrays.
  static_assert(sizeof(InputIdx) == sizeof(int64_t));
  raw.node_bufs.resize(num_nodesets);
  for (std::size_t ns = 0; ns < num_nodesets; ns++) {
    raw.node_bufs[ns] = std::make_unique_for_overwrite<InputIdx[]>(
        std::max<std::size_t>(1, raw.layout.num_nodes[ns]));
  }
  raw.edge_bufs.resize(num_edgesets);
  for (std::size_t es = 0; es < num_edgesets; es++) {
    raw.edge_bufs[es] = std::make_unique_for_overwrite<int64_t[]>(
        std::max<std::size_t>(1, 2 * raw.layout.num_edges[es]));
  }
  raw.feature_bufs.resize(num_nodesets);
  std::size_t total_feature_rows = 0;
  for (std::size_t ns = 0; ns < num_nodesets; ns++) {
    raw.feature_bufs[ns].reserve(config.features[ns].size());
    for (const auto& feature : config.features[ns]) {
      if (!feature.gather_in_cc) {
        raw.feature_bufs[ns].push_back(nullptr);
        continue;
      }
      raw.feature_bufs[ns].push_back(
          std::make_unique_for_overwrite<uint8_t[]>(std::max<std::size_t>(
              1, raw.layout.num_nodes[ns] * feature.row_bytes)));
      total_feature_rows += raw.layout.num_nodes[ns];
    }
  }

  // Feature gathering tasks (~4 tasks per thread).
  const std::size_t rows_per_task = std::max<std::size_t>(
      kMinRowsPerGatherTask,
      total_feature_rows / (4 * std::max(1, thread_pool.num_threads())) + 1);
  std::vector<GatherRowsTask> gather_tasks;
  for (std::size_t ns = 0; ns < num_nodesets; ns++) {
    for (std::size_t f = 0; f < config.features[ns].size(); f++) {
      const auto& feature = config.features[ns][f];
      if (!feature.gather_in_cc) continue;
      AppendGatherRowsTasks(
          feature.src_data, feature.src_num_rows, raw.node_bufs[ns].get(),
          raw.layout.node_offsets[ns].back(),
          reinterpret_cast<char*>(raw.feature_bufs[ns][f].get()),
          raw.layout.num_nodes[ns], feature.row_bytes, rows_per_task,
          &gather_tasks);
    }
  }

  // Copy the graph structure. Each sample writes in its own slice.
  util::concurrency::ConcurrentForLoop(
      std::min<std::size_t>(num_samples,
                            std::max(1, thread_pool.num_threads())),
      &thread_pool, num_samples,
      [&](std::size_t, std::size_t begin_item, std::size_t end_item) {
        for (std::size_t i = begin_item; i < end_item; i++) {
          const SampleBuilder& b = *builders[begin + i];
          for (std::size_t ns = 0; ns < num_nodesets; ns++) {
            const auto& src = b.nodesets[ns].sampled_node_idx_to_node_idx;
            std::copy(src.begin(), src.end(),
                      raw.node_bufs[ns].get() + raw.layout.node_offsets[ns][i]);
          }
          for (std::size_t es = 0; es < num_edgesets; es++) {
            const auto& edgeset = schema_->edgesets[es];
            const std::size_t src_offset =
                raw.layout.node_offsets[edgeset.source_nodeset][i];
            const std::size_t trg_offset =
                raw.layout.node_offsets[edgeset.target_nodeset][i];
            int64_t* dst_src =
                raw.edge_bufs[es].get() + raw.layout.edge_offsets[es][i];
            int64_t* dst_trg = dst_src + raw.layout.num_edges[es];
            for (const auto& [src, trg] : b.edgesets[es].edges) {
              *(dst_src++) = static_cast<int64_t>(src + src_offset);
              *(dst_trg++) = static_cast<int64_t>(trg + trg_offset);
            }
          }
        }
      });

  // Padding of the graph structure.
  for (std::size_t ns = 0; ns < num_nodesets; ns++) {
    std::fill(raw.node_bufs[ns].get() + raw.layout.node_offsets[ns].back(),
              raw.node_bufs[ns].get() + raw.layout.num_nodes[ns], 0);
  }
  for (std::size_t es = 0; es < num_edgesets; es++) {
    if (!config.padding.num_edges[es].has_value()) continue;
    const auto& edgeset = schema_->edgesets[es];
    int64_t* dst_src = raw.edge_bufs[es].get();
    int64_t* dst_trg = dst_src + raw.layout.num_edges[es];
    const std::size_t num_real = raw.layout.edge_offsets[es].back();
    FillPaddingEdgeEndpoints(raw.layout, edgeset.source_nodeset,
                             dst_src + num_real,
                             dst_src + raw.layout.num_edges[es]);
    FillPaddingEdgeEndpoints(raw.layout, edgeset.target_nodeset,
                             dst_trg + num_real,
                             dst_trg + raw.layout.num_edges[es]);
  }

  // Gather the features (requires the node idxs).
  std::atomic<bool> any_out_of_bounds = false;
  util::concurrency::ConcurrentForLoop(
      gather_tasks.size(), &thread_pool, gather_tasks.size(),
      [&](std::size_t, std::size_t begin_item, std::size_t end_item) {
        for (std::size_t i = begin_item; i < end_item; i++) {
          if (!GatherRows(gather_tasks[i])) {
            any_out_of_bounds.store(true, std::memory_order_relaxed);
          }
        }
      });
  if (any_out_of_bounds.load()) {
    return absl::InvalidArgumentError(
        "Node indices out of the bounds of the feature values. Make sure the "
        "features have one value per node.");
  }
  return raw;
}

nb::tuple Sampler::ExportMergedBatch(const MergeConfig& config,
                                     RawMergedBatch* raw) const {
  const std::size_t num_nodesets = nodesets_.size();
  const std::size_t num_edgesets = edgesets_.size();
  nb::dict py_nodesets;
  nb::dict py_offsets;
  for (std::size_t ns = 0; ns < num_nodesets; ns++) {
    const std::size_t num_nodes = raw->layout.num_nodes[ns];
    nb::dict py_features;
    for (std::size_t f = 0; f < config.features[ns].size(); f++) {
      const auto& feature = config.features[ns][f];
      if (!feature.gather_in_cc) {
        py_features[feature.name] = nb::none();
        continue;
      }
      uint8_t* data = raw->feature_bufs[ns][f].release();
      nb::capsule owner(data, [](void* p) noexcept { delete[] (uint8_t*)p; });
      const nb::object arr = nb::cast(nb::ndarray<uint8_t, nb::numpy>(
          data, {num_nodes, feature.row_bytes}, owner));
      py_features[feature.name] =
          arr.attr("view")(feature.dtype).attr("reshape")(feature.reshape);
    }
    InputIdx* idx_data = raw->node_bufs[ns].release();
    nb::capsule idx_owner(idx_data,
                          [](void* p) noexcept { delete[] (InputIdx*)p; });
    if (config.export_node_idxs[ns]) {
      py_features[module_index_.key_idx_feature] = NodeIdxs(
          reinterpret_cast<int64_t*>(idx_data), {num_nodes}, idx_owner);
    }
    const nb::str name = string_to_py_str(nodesets_[ns].name);
    py_nodesets[name] = module_index_.nodeset_cls(num_nodes, py_features);
    py_offsets[name] = CCVectorToNumpyArray<int64_t, std::size_t>(
        raw->layout.node_offsets[ns]);
  }
  nb::dict py_edgesets;
  for (std::size_t es = 0; es < num_edgesets; es++) {
    int64_t* data = raw->edge_bufs[es].release();
    nb::capsule owner(data, [](void* p) noexcept { delete[] (int64_t*)p; });
    py_edgesets[string_to_py_str(edgesets_[es].name)] =
        module_index_.edgeset_cls(
            Adjacency(data, {2, raw->layout.num_edges[es]}, owner));
  }
  return nb::make_tuple(module_index_.graph_cls(py_nodesets, py_edgesets),
                        py_offsets);
}

nb::tuple Sampler::ExportPaddingOverflow(
    const PaddingOverflow& overflow) const {
  const std::string& name = overflow.is_nodeset
                                ? nodesets_[overflow.set_idx].name
                                : edgesets_[overflow.set_idx].name;
  return nb::make_tuple(overflow.is_nodeset, name, overflow.required,
                        overflow.padded);
}

absl::StatusOr<nb::tuple> Sampler::SampleMerged(
    const NodeIdxs& seed_node_idxs,
    std::optional<TimestampsArray> seed_timestamps,
    std::optional<EdgeIdxs> masked_edge_idxs, const MergeConfig& config) {
  return SampleMergedImpl(seed_node_idxs, seed_timestamps, masked_edge_idxs,
                          config, GrowMode::kPlan);
}

absl::StatusOr<nb::tuple> Sampler::MultiSubGraphsMerged(
    const NodeIdxs& seed_node_idxs, const MergeConfig& config) {
  return SampleMergedImpl(seed_node_idxs, std::nullopt, std::nullopt, config,
                          GrowMode::kSubGraph);
}

absl::StatusOr<nb::tuple> Sampler::SampleMergedImpl(
    const NodeIdxs& seed_node_idxs,
    const std::optional<TimestampsArray>& seed_timestamps,
    const std::optional<EdgeIdxs>& masked_edge_idxs, const MergeConfig& config,
    const GrowMode mode) {
  PooledBuilders builders;
  std::optional<RawMergedBatch> raw;
  PaddingOverflow overflow;
  {
    nb::gil_scoped_release release;
    DGF_ASSIGN_OR_RETURN(builders,
                         GrowSamplesNoGil(seed_node_idxs, seed_timestamps,
                                          masked_edge_idxs, mode));
    DGF_ASSIGN_OR_RETURN(
        raw, MergeBuildersNoGil(builders.builders(), 0,
                                builders.builders().size(), config, &overflow));
  }
  if (raw.has_value()) {
    return nb::make_tuple(ExportMergedBatch(config, &*raw), nb::none(),
                          nb::none());
  }
  return nb::make_tuple(
      nb::none(), std::make_unique<SampledBatch>(this, std::move(builders)),
      ExportPaddingOverflow(overflow));
}

absl::StatusOr<nb::tuple> SampledBatch::Merge(const std::size_t begin,
                                              const std::size_t end,
                                              const MergeConfig& config) {
  std::optional<RawMergedBatch> raw;
  PaddingOverflow overflow;
  {
    nb::gil_scoped_release release;
    DGF_ASSIGN_OR_RETURN(
        raw, sampler->MergeBuildersNoGil(builders.builders(), begin, end,
                                         config, &overflow));
  }
  if (raw.has_value()) {
    return nb::make_tuple(sampler->ExportMergedBatch(config, &*raw),
                          nb::none());
  }
  return nb::make_tuple(nb::none(), sampler->ExportPaddingOverflow(overflow));
}

absl::StatusOr<std::unique_ptr<MergeConfig>> Sampler::CreateMergeConfig(
    const nb::dict& padding_num_nodes, const nb::dict& padding_num_edges,
    const nb::dict& features, const bool return_node_idxs) {
  auto config = std::make_unique<MergeConfig>();
  config->padding.num_nodes.assign(nodesets_.size(), std::nullopt);
  config->padding.num_edges.assign(edgesets_.size(), std::nullopt);
  config->features.resize(nodesets_.size());
  config->export_node_idxs.assign(nodesets_.size(), return_node_idxs);

  for (const auto item : padding_num_nodes) {
    DGF_ASSIGN_OR_RETURN(const int idx,
                         GetItem(schema_->nodeset_name_to_idx,
                                 nb::cast<std::string>(item.first)));
    config->padding.num_nodes[idx] = nb::cast<std::size_t>(item.second);
  }
  for (const auto item : padding_num_edges) {
    const auto name = nb::cast<std::string>(item.first);
    DGF_ASSIGN_OR_RETURN(const int idx,
                         GetItem(schema_->edgeset_name_to_idx, name));
    config->padding.num_edges[idx] = nb::cast<std::size_t>(item.second);
    const auto& edgeset = schema_->edgesets[idx];
    if (!config->padding.num_nodes[edgeset.source_nodeset].has_value() ||
        !config->padding.num_nodes[edgeset.target_nodeset].has_value()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Padding for edge set '", name,
          "' requires sentinel nodes on both source and target node sets."));
    }
  }

  for (const auto item : features) {
    DGF_ASSIGN_OR_RETURN(const int ns,
                         GetItem(schema_->nodeset_name_to_idx,
                                 nb::cast<std::string>(item.first)));
    for (const auto py_feature : nb::cast<nb::list>(item.second)) {
      const auto py_tuple = nb::cast<nb::tuple>(py_feature);
      DGF_STATUS_CHECK(py_tuple.size() == 4);
      MergeConfig::Feature feature;
      feature.name = nb::cast<nb::str>(py_tuple[0]);
      if (py_tuple[1].is_none()) {
        // Gathered in python, which requires the node idxs.
        config->export_node_idxs[ns] = true;
      } else {
        feature.gather_in_cc = true;
        feature.src = nb::cast<RawArray>(py_tuple[1]);
        DGF_STATUS_CHECK(feature.src.ndim() >= 1);
        feature.src_data = static_cast<const char*>(feature.src.data());
        feature.src_num_rows = feature.src.shape(0);
        feature.row_bytes = feature.src.itemsize();
        for (std::size_t d = 1; d < feature.src.ndim(); d++) {
          feature.row_bytes *= feature.src.shape(d);
        }
      }
      feature.dtype = py_tuple[2];
      feature.reshape = nb::cast<nb::tuple>(py_tuple[3]);
      config->features[ns].push_back(std::move(feature));
    }
  }
  return config;
}

struct RandomWalkNegativeSamplerHelper {
  const Sampler* sampler;
  const Sampler::EdgeSet* edgeset;
  size_t num_target_nodes;
  bool is_homogeneous;
  int num_walks;
  int num_negatives_per_seed;

  absl::Status SampleForSeed(InputIdx seed_node, int64_t* output_for_seed_node,
                             Rng* local_rng) const {
    absl::flat_hash_map<InputIdx, int> visit_counts;
    const int target_nodeset_idx = edgeset->target_nodeset;

    // Phase 1: Walk Simulation
    for (int walk_idx = 0; walk_idx < num_walks; walk_idx++) {
      InputIdx cur_node = seed_node;
      const SamplingPlan::Node* cur_plan_node = sampler->plan_.root.get();

      while (!cur_plan_node->children.empty()) {
        const auto& plan_edge = cur_plan_node->children[absl::Uniform<size_t>(
            *local_rng, 0, cur_plan_node->children.size())];

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

        cur_node =
            neighbors[absl::Uniform<size_t>(*local_rng, 0, neighbors.size())];
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
    // Ties are broken by node index: `visit_counts` iterates in an order that
    // changes between processes.
    const auto more_visited = [](const Candidate& a, const Candidate& b) {
      if (a.count != b.count) return a.count > b.count;
      return a.node < b.node;
    };
    const int num_extracted =
        std::min(static_cast<int>(candidates.size()), num_negatives_per_seed);
    if (num_extracted < candidates.size()) {
      std::nth_element(candidates.begin(), candidates.begin() + num_extracted,
                       candidates.end(), more_visited);
    }
    std::sort(candidates.begin(), candidates.begin() + num_extracted,
              more_visited);

    for (int i = 0; i < num_extracted; i++) {
      output_for_seed_node[i] = static_cast<int64_t>(candidates[i].node);
    }
    int num_filled = num_extracted;

    // Phase 4: Robust Fallback
    if (num_filled < num_negatives_per_seed) {
      // Note: The remaining slots are filled by randomly sampling nodes with
      // replacement. We only ensure the sampled node is not the seed node,
      // without checking for existing edges.
      while (num_filled < num_negatives_per_seed) {
        const InputIdx candidate =
            absl::Uniform<InputIdx>(*local_rng, 0, num_target_nodes);
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
Sampler::RandomWalkNegativeSampling(const NodeIdxs& seed_node_idxs,
                                    int target_edgeset_idx, int num_walks,
                                    int num_negatives_per_seed) {
  if (target_edgeset_idx < 0 || target_edgeset_idx >= edgesets_.size()) {
    return absl::InvalidArgumentError("Invalid target_edgeset_idx");
  }
  const auto& edgeset = edgesets_[target_edgeset_idx];
  if (!edgeset.forward_index.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Edgeset '", edgeset.name, "' does not have a forward index. ",
        "Ensure it is traversed forward in the SamplingPlan."));
  }

  if (num_walks < 0 || num_negatives_per_seed < 0) {
    return absl::InvalidArgumentError(
        "num_walks and num_negatives_per_seed must be non-negative.");
  }
  // The walks start from the seed nodes at the root of the plan, and the
  // negatives are checked against the edges of the seed nodes.
  if (plan_.root->nodeset_idx != edgeset.source_nodeset) {
    return absl::InvalidArgumentError(
        absl::StrCat("The source node set of edgeset '", edgeset.name,
                     "' should be the seed node set of the sampling plan."));
  }

  const size_t num_target_nodes = nodesets_[edgeset.target_nodeset].num_nodes;
  const bool is_homogeneous =
      (edgeset.source_nodeset == edgeset.target_nodeset);
  // The fallback samples nodes different from the seed node.
  const size_t min_num_target_nodes = is_homogeneous ? 2 : 1;
  if (num_negatives_per_seed > 0 && num_target_nodes < min_num_target_nodes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Not enough target nodes to sample negatives for edgeset '",
        edgeset.name, "'."));
  }

  RandomWalkNegativeSamplerHelper helper{
      this,           &edgeset,  num_target_nodes,
      is_homogeneous, num_walks, num_negatives_per_seed};

  auto seed_node_idxs_view = seed_node_idxs.view();
  const size_t num_seeds = seed_node_idxs_view.shape(0);
  const size_t num_source_nodes = nodesets_[edgeset.source_nodeset].num_nodes;
  for (size_t seed_idx = 0; seed_idx < num_seeds; seed_idx++) {
    const int64_t seed_node = seed_node_idxs_view(seed_idx);
    if (seed_node < 0 || seed_node >= num_source_nodes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Seed node idx ", seed_node,
          " is out of bounds. Should be lower than ", num_source_nodes, "."));
    }
  }

  int64_t* output_data = new int64_t[num_seeds * num_negatives_per_seed];

  std::vector<Rng> rngs;
  rngs.reserve(num_seeds);

  absl::Status global_status;
  std::mutex global_status_mutex;

  {
    nb::gil_scoped_release release;

    for (size_t i = 0; i < num_seeds; i++) {
      rngs.push_back(MakeRng(rng_()));
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
        // Record the failure before counting down: Once the latch reaches zero,
        // the calling thread can return and destroy `global_status`.
        if (!status.ok()) {
          util::concurrency::MutexLock l(global_status_mutex);
          global_status.Update(status);
        }

        latch.count_down();
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
  PooledBuilders builders(this, 1);
  SampleBuilder& builder = *builders.builders().front();
  {
    // Release the GIL.
    nb::gil_scoped_release release;
    builder.rng = MakeRng(rng_());
    DGF_RETURN_IF_ERROR(builder.Grow(*this, seed_node_idxs, std::nullopt,
                                     /*masked_edge_idx=*/-1,
                                     GrowMode::kSubGraph));
  }

  // Convert to a python object.
  return ExportToInMemoryGraph(builder);
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
  for (const auto& nodeset : schema_->nodeset_name_to_idx) {
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

        edgeset.need_forward = true;
      } else {
        DGF_STATUS_CHECK(edgeset.source_nodeset == -1 ||
                         edgeset.source_nodeset == child.node->nodeset_idx);
        DGF_STATUS_CHECK(edgeset.target_nodeset == -1 ||
                         edgeset.target_nodeset == node.nodeset_idx);

        edgeset.source_nodeset = child.node->nodeset_idx;
        edgeset.target_nodeset = node.nodeset_idx;

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

  // Check that the plan follows the edgesets as defined in the schema.
  for (int edgeset_idx = 0; edgeset_idx < edgesets_.size(); edgeset_idx++) {
    const auto& edgeset = edgesets_[edgeset_idx];
    const auto& schema_edgeset = schema_->edgesets[edgeset_idx];
    if (edgeset.source_nodeset != -1 &&
        (edgeset.source_nodeset != schema_edgeset.source_nodeset ||
         edgeset.target_nodeset != schema_edgeset.target_nodeset)) {
      return absl::InvalidArgumentError(
          absl::StrCat("The sampling plan traverses edgeset '", edgeset.name,
                       "' between the wrong node sets."));
    }
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
  sampler->ComputeStepSlots();
  if (seed >= 0) {
    sampler->rng_ = MakeRng(seed);
  }
  // Note: If seed < 0, `rng_` keeps its default, non-deterministic seeding.
  sampler->debug_sampling_ = debug_sampling;
  return std::move(sampler);
}

NB_MODULE(_in_memory_sampler_ext, m) {
  nb::class_<Sampler>(m, "Sampler")
      .def("Sample", ValueOrThrowWrapper(&Sampler::Sample),
           nb::arg("seed_node_idx"), nb::arg("seed_timestamps") = nb::none(),
           nb::arg("masked_edge_idxs") = nb::none())
      .def("SampleMerged", ValueOrThrowWrapper(&Sampler::SampleMerged),
           nb::arg("seed_node_idx"), nb::arg("seed_timestamps") = nb::none(),
           nb::arg("masked_edge_idxs") = nb::none(), nb::arg("config"))
      .def("CreateMergeConfig",
           ValueOrThrowWrapper(&Sampler::CreateMergeConfig),
           nb::arg("padding_num_nodes"), nb::arg("padding_num_edges"),
           nb::arg("features"), nb::arg("return_node_idxs"))
      .def("SubGraph", ValueOrThrowWrapper(&Sampler::SubGraph))
      .def("MultiSubGraphs", ValueOrThrowWrapper(&Sampler::MultiSubGraphs),
           nb::arg("seed_node_idx"))
      .def("MultiSubGraphsMerged",
           ValueOrThrowWrapper(&Sampler::MultiSubGraphsMerged),
           nb::arg("seed_node_idx"), nb::arg("config"))
      .def("RandomWalkNegativeSampling",
           ValueOrThrowWrapper(&Sampler::RandomWalkNegativeSampling),
           nb::arg("seed_node_idxs"), nb::arg("target_edgeset_idx"),
           nb::arg("num_walks"), nb::arg("num_negatives_per_seed"))
      .def("EdgesetNameToEdgesetIdx",
           ValueOrThrowWrapper(&Sampler::EdgesetNameToEdgesetIdx))
      .def("__str__", &Sampler::__str__);

  nb::class_<MergeConfig>(m, "MergeConfig");

  nb::class_<SampledBatch>(m, "SampledBatch")
      .def("Merge", ValueOrThrowWrapper(&SampledBatch::Merge), nb::arg("begin"),
           nb::arg("end"), nb::arg("config"));

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

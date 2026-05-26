#ifndef DGF_SRC_DATA_IN_MEMORY_GRAPH_H_
#define DGF_SRC_DATA_IN_MEMORY_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"

namespace dgf::data {

// Arbitrary rank with known (non-ragged) shapes.
template <typename T>
struct TensorView {
  absl::Span<T> data;  // Generic pointer with length.
  std::vector<size_t> shape;
  std::vector<int64_t> strides;
};

template <>
struct TensorView<std::string_view> {
  // Flat views of strings with numpy padding discarded.
  // Note: The data is not owned.
  std::vector<std::string_view> data;
  std::vector<size_t> shape;
  std::vector<int64_t> strides;
  std::vector<size_t> itemsizes;
};

struct FeaturesView {
  // TODO(bmayer): Migrate to vector<AbstractTensorFlow> and index wrt schema
  // order.
  absl::flat_hash_map<std::string, TensorView<float>> float_features;
  absl::flat_hash_map<std::string, TensorView<int64_t>> int64_features;
  absl::flat_hash_map<std::string, TensorView<std::string_view>> bytes_features;
};

// Topological connections. Weights and IDs can be stored in features aligned
// with each edge index.
struct AdjacencyView {
  absl::Span<int64_t> source;
  absl::Span<int64_t> target;
};

// TODO(bmayer): Remove `name` if we adopt graph/schema coupling.
struct NodeSetView {
  std::string name;
  int64_t num_nodes;
  FeaturesView features;
};

// TODO(bmayer): Remove `name` if we adopt graph/schema coupling.
struct EdgeSetView {
  std::string name;
  AdjacencyView adjacency;
  FeaturesView features;
};

// An un-owned view of graph data held in memory on a single machine.
// `context` is an optional nodeset that conventionally holds information
// pertaining to the entire graph.
// `node_sets` are named graph pieces with features.
// `edge_sets` maintain connections between `node_sets`.
// `context`, `node_sets` and `edge_sets` keep a copy of the graph piece name or
// the GraphView builder can keep the objects aligned to the schema
// representation for fast indexing. See the in-memory sampler for an example.
// TODO(bmayer): Add string/rep for debugging.
struct GraphView {
  std::optional<NodeSetView> context;
  std::vector<NodeSetView> node_sets;
  std::vector<EdgeSetView> edge_sets;
};
}  // namespace dgf::data

#endif  // DGF_SRC_DATA_IN_MEMORY_GRAPH_H_

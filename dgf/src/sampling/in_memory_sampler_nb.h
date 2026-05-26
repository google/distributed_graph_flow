// Reusable Nanobind code for the in memory sampler.
//
// This code can only be called in c++ lib/bin with exception support.

#ifndef DGF_SRC_SAMPLING_IN_MEMORY_SAMPLER_NB_H_
#define DGF_SRC_SAMPLING_IN_MEMORY_SAMPLER_NB_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"  // IWYU pragma: keep
#include "dgf/src/data/schema.h"
#include "dgf/src/sampling/in_memory_sampler.h"
#include "dgf/src/util/nanobind_util.h"

namespace dgf::sampling::in_memory_sampler {
namespace nb = nanobind;

// Numpy array definitions for dense adjacency lists and node idxs.
typedef nb::ndarray<int64_t, nb::numpy, nb::shape<2, -1>> Adjacency;
typedef nb::ndarray<int64_t, nb::numpy, nb::shape<-1>> NodeIdxs;
typedef nb::ndarray<int64_t, nb::numpy, nb::shape<-1>> TimestampsArray;

// Create an adjacency numpy array (of shape [2, num_edges]) from a container of
// node idx pairs (or something equivalent). The type T should be integer
// compatible.
template <typename T>
Adjacency EdgesToNumpyArray(const T& src) {
  using EdgeType = typename T::value_type;
  static_assert(std::is_integral_v<decltype(std::declval<EdgeType>().first)>,
                "The first element of the edge must be an integer type.");
  static_assert(std::is_integral_v<decltype(std::declval<EdgeType>().second)>,
                "The second element of the edge must be an integer type.");

  // TODO(gbm): Add option to export np.int32.
  const std::size_t rows = 2;
  const std::size_t cols = src.size();
  int64_t* data = new int64_t[rows * cols];
  int64_t* cur_src = data;
  int64_t* cur_trg = data + cols;
  for (const auto edge : src) {
    // Note: The data is stored edge-minor / src-target-major.
    // TODO(gbm): Would be be more efficient for the GNN computation to use an
    // edge-major representation?
    *(cur_src++) = edge.first;
    *(cur_trg++) = edge.second;
  }
  nb::capsule owner(data, [](void* p) noexcept { delete[] (int64_t*)p; });
  return Adjacency(
      /* data = */ data,
      /* shape = */ {rows, cols},
      /* owner = */ owner);
}

// Create an adjacency numpy array (of shape [2, num_edges]) from a container of
// node idxs (or something equivalent) where sucessive node ids define edges.
// For example, the array [1,2,3,4] defines the edges 1->2 and 3->4.
template <typename T>
Adjacency EdgesMergedListToNumpyArray(const T& src) {
  // TODO(gbm): Add option to export np.int32.
  const size_t num_edges = src.size() / 2;
  const size_t rows = 2;
  const size_t cols = num_edges;

  int64_t* data = new int64_t[rows * cols];
  int64_t* cur_src = data;
  int64_t* cur_trg = data + cols;

  size_t src_idx = 0;
  for (size_t edge_idx = 0; edge_idx < num_edges; edge_idx++) {
    *(cur_src++) = src[src_idx++];
    *(cur_trg++) = src[src_idx++];
  }

  nb::capsule owner(data, [](void* p) noexcept { delete[] (int64_t*)p; });
  return Adjacency(
      /* data = */ data,
      /* shape = */ {rows, cols},
      /* owner = */ owner);
}

// Create a node idxs numpy array.
NodeIdxs NodeIdxsToNumpyArray(const std::vector<InputIdx>& src);

// Python modules / classes / object of the returned objects.
// Should be created inside the GIL.
struct ModuleIndex {
  nanobind::module_ in_memory_graph_mod;
  nanobind::object graph_cls;
  nanobind::object nodeset_cls;
  nanobind::object edgeset_cls;
  nanobind::str key_idx_feature;  // The string "#idx".
  nanobind::str key_id_feature;   // The string "#id".

  ModuleIndex();
};

// Creates a c++ sampling plan from a python sampling plan.
//
// Args:
//   py_plan: The python `SamplingPlan` object.
//   nodeset_index: A map from nodeset names to their integer indices.
//   edgeset_index: A map from edgeset names to their integer indices.
absl::StatusOr<SamplingPlan> CreateSamplingPlan(
    const nanobind::object& py_plan, const data::GraphSchema& schema);

}  // namespace dgf::sampling::in_memory_sampler

#endif  // DGF_SRC_SAMPLING_IN_MEMORY_SAMPLER_H_

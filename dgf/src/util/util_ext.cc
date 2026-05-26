// Collection of utilities implemented in c++ and available in python.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"  // IWYU pragma: keep
#include "dgf/src/util/nanobind_util.h"
#include "dgf/src/util/status_caster.h"
#include "dgf/src/util/util.h"

namespace nb = nanobind;

namespace dgf {
namespace {
using NodeIdx = int64_t;
using EdgeIdx = int64_t;
typedef nb::ndarray<NodeIdx, nb::numpy, nb::shape<2, -1>> Adjacency;
}  // namespace

// Indexes edges in an efficient mapping (source node, target node) -> edge idx.
class EdgeIndexer {
 public:
  // Create an index.
  static absl::StatusOr<EdgeIndexer*> Create(const Adjacency& adjacencies) {
    auto indexer = std::make_unique<EdgeIndexer>();
    const auto adjacency_view = adjacencies.view();
    std::size_t num_edges = adjacency_view.shape(1);
    for (std::size_t i = 0; i < num_edges; ++i) {
      NodeIdx src = adjacency_view(0, i);
      NodeIdx tgt = adjacency_view(1, i);
      indexer->index_[{src, tgt}] = i;
    }
    return indexer.release();
  }

  // Query the index. If the pair (source, target) does not exist, return -1.
  EdgeIdx query(const NodeIdx source_node_idx,
                const NodeIdx target_node_idx) const {
    auto it = index_.find({source_node_idx, target_node_idx});
    if (it == index_.end()) {
      return -1;
    }
    return it->second;
  }

  // Query multiple edges. Returns an array of edge indices.
  absl::StatusOr<nb::ndarray<EdgeIdx, nb::numpy, nb::shape<-1>>> QueryArray(
      const Adjacency& queries) const {
    const auto queries_view = queries.view();
    std::size_t num_queries = queries_view.shape(1);
    EdgeIdx* result_data = new EdgeIdx[num_queries];

    for (std::size_t i = 0; i < num_queries; ++i) {
      NodeIdx src = queries_view(0, i);
      NodeIdx tgt = queries_view(1, i);
      auto it = index_.find({src, tgt});
      if (it == index_.end()) {
        result_data[i] = -1;
      } else {
        result_data[i] = it->second;
      }
    }

    nanobind::capsule owner(result_data,
                            [](void* p) noexcept { delete[] (EdgeIdx*)p; });
    return nanobind::ndarray<EdgeIdx, nb::numpy, nb::shape<-1>>(
        /* data = */ result_data,
        /* shape = */ {num_queries},
        /* owner = */ owner);
  }

 private:
  absl::flat_hash_map<std::pair<NodeIdx, NodeIdx>, EdgeIdx> index_;
};

NB_MODULE(util_ext, m) {
  nb::class_<EdgeIndexer>(m, "EdgeIndexer")
      .def("query", &EdgeIndexer::query)
      .def("query_array", ValueOrThrowWrapper(&EdgeIndexer::QueryArray));

  m.def("CreateEdgeIndexer", ValueOrThrowWrapper(EdgeIndexer::Create));
}

}  // namespace dgf

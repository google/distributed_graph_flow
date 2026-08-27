#ifndef THIRD_PARTY_PY_DGF_SRC_GBBS_GBBS_GRAPH_HANDLE_H_
#define THIRD_PARTY_PY_DGF_SRC_GBBS_GBBS_GRAPH_HANDLE_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <variant>

#include "third_party/gbbs/gbbs/graph.h"
#include "third_party/gbbs/gbbs/vertex.h"

namespace dgf::gbbs {

using GbbsSymmetric = ::gbbs::symmetric_graph<::gbbs::symmetric_vertex, float>;
using GbbsAsymmetric =
    ::gbbs::asymmetric_graph<::gbbs::asymmetric_vertex, float>;

// Holder of a GBBS graph.
class GbbsGraphHandle {
 public:
  GbbsGraphHandle(GbbsSymmetric g) : graph_(std::move(g)) {}
  GbbsGraphHandle(GbbsAsymmetric g) : graph_(std::move(g)) {}

  size_t num_nodes() const {
    return std::visit([](const auto& g) { return g.n; }, graph_);
  }
  size_t num_edges() const {
    return std::visit([](const auto& g) { return g.m; }, graph_);
  }

  bool is_symmetric() const {
    return std::holds_alternative<GbbsSymmetric>(graph_);
  }

  // Allow visiting the underlying variant directly.
  template <typename Visitor>
  auto Visit(Visitor&& visitor) const {
    return std::visit(std::forward<Visitor>(visitor), graph_);
  }

  template <typename Visitor>
  auto Visit(Visitor&& visitor) {
    return std::visit(std::forward<Visitor>(visitor), graph_);
  }

 private:
  std::variant<GbbsSymmetric, GbbsAsymmetric> graph_;
};

// Create a GBBS graph handle from source, target (and optionally weight)
// arrays. The arrays must be contiguous and of size `num_edges`. `edge_weights`
// can be nullptr if the graph is unweighted.
std::unique_ptr<GbbsGraphHandle> CreateGbbsGraphHandle(
    size_t num_nodes, size_t num_edges, const int64_t* edge_sources,
    const int64_t* edge_targets, const float* edge_weights, bool symmetric);

}  // namespace dgf::gbbs

#endif  // THIRD_PARTY_PY_DGF_SRC_GBBS_GBBS_GRAPH_HANDLE_H_

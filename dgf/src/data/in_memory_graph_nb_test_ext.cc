#include <cstddef>

#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "dgf/src/data/in_memory_graph_nb.h"
#include "dgf/src/util/status_caster.h"
#include "dgf/src/util/util.h"

namespace dgf {
namespace {
namespace nb = nanobind;

absl::StatusOr<size_t> CountNumNodes(nb::object obj) {
  DGF_ASSIGN_OR_RETURN(const auto graph, ::dgf::data::Graph::Create(obj));
  size_t num_nodes = 0;
  for (const auto& node_set : graph.view.node_sets) {
    num_nodes += node_set.num_nodes;
  }
  return num_nodes;
}

absl::StatusOr<size_t> CountNumEdges(nb::object obj) {
  DGF_ASSIGN_OR_RETURN(const auto graph, ::dgf::data::Graph::Create(obj));
  size_t num_edges = 0;
  for (const auto& edge_set : graph.view.edge_sets) {
    num_edges += edge_set.adjacency.source.size();
  }
  return num_edges;
}

NB_MODULE(in_memory_graph_nb_test_ext, m) {
  m.def("CountNumNodes", ValueOrThrowWrapper(CountNumNodes));
  m.def("CountNumEdges", ValueOrThrowWrapper(CountNumEdges));
}
}  // namespace
}  // namespace dgf
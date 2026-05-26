#ifndef DGF_SRC_DATA_IN_MEMORY_GRAPH_NB_H_
#define DGF_SRC_DATA_IN_MEMORY_GRAPH_NB_H_
#include <vector>

#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "dgf/src/data/in_memory_graph.h"

namespace dgf::data {
// A wrapper around dgf::data::GraphView that ensures that all objects that the
// view references maintain a positive refcount for the lifetime of the view.
// After the `Graph` is populated (e.g., using Create) it should be safe to
// release the GIL and access the graph's components in a multithreaded
// environment.
struct Graph {
  static absl::StatusOr<Graph> Create(nanobind::object obj);
  GraphView view;
  std::vector<nanobind::object> refs_;
};

// Creates Graph view(s) using a single thread. It should be safe to release the
// GIL and use multi-threading on the return value.
absl::StatusOr<std::vector<Graph>> CreateGraphs(nanobind::sequence seq);

}  // namespace dgf::data

#endif  // DGF_SRC_DATA_IN_MEMORY_GRAPH_NB_H_

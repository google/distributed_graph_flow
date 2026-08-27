#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unique_ptr.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include "third_party/gbbs/gbbs/macros.h"
#include "third_party/parlay/include/parlay/parallel.h"
#include "third_party/parlay/include/parlay/scheduler.h"
#include "dgf/src/gbbs/gbbs_graph_handle.h"

namespace nb = nanobind;

namespace dgf::gbbs {

// Keep the global scheduler alive as long as the module is loaded.
static parlay::scheduler_pointer global_scheduler;

NB_MODULE(_gbbs_ext, m) {
  // Initialize Parlay workers to hardware concurrency (or environment
  // variable).
  if (const auto env_p = std::getenv("PARLAY_NUM_THREADS")) {
    global_scheduler = parlay::initialize_scheduler(std::stoi(env_p));
  } else {
    global_scheduler =
        parlay::initialize_scheduler(std::thread::hardware_concurrency());
  }

  m.def(
      "set_num_parlay_workers",
      [](unsigned int num_workers) {
        // Technically this resets the global scheduler with new threads.
        global_scheduler = parlay::initialize_scheduler(num_workers);
      },
      nb::arg("num_workers"),
      "Sets the number of worker threads for Parlay / GBBS operations.");

  m.def(
      "num_parlay_workers",
      []() { return parlay::GetScheduler()->num_workers(); },
      "Returns the number of active worker threads in Parlay / GBBS.");

  m.def(
      "shutdown_parlay",
      []() {
        // Drop the strong reference to gracefully join threads before atexit
        global_scheduler.shared.reset();
      },
      "Cleanly joins Parlay threads. Recommended for atexit hooks.");

  nb::class_<GbbsGraphHandle>(m, "GbbsGraphHandle")
      .def("num_nodes", &GbbsGraphHandle::num_nodes)
      .def("num_edges", &GbbsGraphHandle::num_edges);

  m.def(
      "create_gbbs_graph_handle",
      [](size_t num_nodes,
         nb::ndarray<int64_t, nb::numpy, nb::shape<2, -1>, nb::c_contig>
             adjacency,
         std::optional<
             nb::ndarray<float, nb::numpy, nb::shape<-1>, nb::c_contig>>
             weights,
         bool symmetric) -> std::unique_ptr<GbbsGraphHandle> {
        nb::gil_scoped_release release;
        const size_t num_edges = adjacency.shape(1);
        const int64_t* edge_sources = adjacency.data();
        const int64_t* edge_targets = adjacency.data() + num_edges;
        if (weights.has_value()) {
          if (weights->shape(0) != num_edges) {
            throw std::invalid_argument(
                "number of weight elements must match the number of edges");
          }
        }
        const float* edge_weights =
            weights.has_value() ? weights->data() : nullptr;

        return CreateGbbsGraphHandle(num_nodes, num_edges, edge_sources,
                                     edge_targets, edge_weights, symmetric);
      },
      nb::arg("num_nodes"), nb::arg("adjacency"),
      nb::arg("weights") = std::nullopt, nb::arg("symmetric") = true,
      "Creates a GBBS graph from COO adjacency arrays and optional weights.");
}

}  // namespace dgf::gbbs

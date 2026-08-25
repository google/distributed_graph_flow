#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include "third_party/gbbs/gbbs/macros.h"
#include "third_party/parlay/include/parlay/parallel.h"
#include "third_party/parlay/include/parlay/scheduler.h"
#include "dgf/src/gbbs/connected_components.h"
#include "dgf/src/gbbs/gbbs_graph_handle.h"
#include "dgf/src/util/status_caster.h"

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
      .def("num_edges", &GbbsGraphHandle::num_edges)
      .def("is_symmetric", &GbbsGraphHandle::is_symmetric,
           "True if the graph was built as symmetric (undirected).");

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

  // ---------------------------------------------------------------------------
  // Connected Components: Session
  // ---------------------------------------------------------------------------

  nb::class_<CCSession>(m, "CCSession",
                        "Lock-free progress tracking for CC computation.")
      .def(nb::init<>())
      .def("phase", &CCSession::get_phase,
           "Current execution phase (0=NotStarted .. 5=Done).")
      .def("algorithm_duration_ms", &CCSession::get_algorithm_duration_ms,
           "Wall-clock duration of the CC algorithm phase in milliseconds.")
      .def("num_nodes", &CCSession::get_num_nodes,
           "Number of nodes in the graph (set after algorithm completes).")
      .def("num_components", &CCSession::get_num_components,
           "Number of connected components found.")
      .def("largest_component_size", &CCSession::get_largest_component_size,
           "Size of the largest connected component.");

  // ---------------------------------------------------------------------------
  // Connected Components: Algorithm Parameters
  // ---------------------------------------------------------------------------

  nb::class_<SimpleUnionAsyncCCParams>(
      m, "SimpleUnionAsyncCCParams",
      "Lock-free parallel union-find CC.\n"
      "Works on both symmetric and asymmetric (directed) graphs.")
      .def(nb::init<>());

  nb::class_<BfsCCParams>(
      m, "BfsCCParams",
      "BFS-based CC. Requires a symmetric (undirected) graph.")
      .def(nb::init<>());

  nb::class_<LabelPropagationCCParams>(
      m, "LabelPropagationCCParams",
      "Label propagation CC. Requires a symmetric (undirected) graph.\n\n"
      "Attributes:\n"
      "  use_permutation: If True, randomly permute vertices before "
      "propagation.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](LabelPropagationCCParams* self, bool use_permutation) {
            new (self) LabelPropagationCCParams{use_permutation};
          },
          nb::arg("use_permutation") = false)
      .def_rw("use_permutation", &LabelPropagationCCParams::use_permutation,
              "Random vertex permutation before propagation.");

  nb::class_<WorkEfficientCCParams>(
      m, "WorkEfficientCCParams",
      "LDD-based work-efficient CC. Requires a symmetric (undirected) "
      "graph.\n\n"
      "Attributes:\n"
      "  beta: LDD decomposition parameter (default 0.2).\n"
      "  pack: Enable edge packing optimization.\n"
      "  permute: Random permutation of vertices.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](WorkEfficientCCParams* self, double beta, bool pack,
             bool permute) {
            new (self) WorkEfficientCCParams{beta, pack, permute};
          },
          nb::arg("beta") = 0.2, nb::arg("pack") = false,
          nb::arg("permute") = false)
      .def_rw("beta", &WorkEfficientCCParams::beta, "LDD decomposition ratio.")
      .def_rw("pack", &WorkEfficientCCParams::pack,
              "Edge packing optimization.")
      .def_rw("permute", &WorkEfficientCCParams::permute,
              "Random vertex permutation.");

  nb::class_<ShiloachVishkinCCParams>(
      m, "ShiloachVishkinCCParams",
      "Shiloach-Vishkin hook-compress CC.\n"
      "Works on both symmetric and asymmetric (directed) graphs.")
      .def(nb::init<>());

  // ---------------------------------------------------------------------------
  // Connected Components: Result
  // ---------------------------------------------------------------------------

  nb::class_<ConnectedComponentsResult>(
      m, "ConnectedComponentsResult",
      "Result of a connected components computation.")
      .def_prop_ro(
          "labels",
          [](const ConnectedComponentsResult& r) {
            return nb::ndarray<nb::numpy, const uint32_t, nb::shape<-1>>(
                r.labels.data(), {r.labels.size()});
          },
          "Component labels as a numpy uint32 array (zero-copy view).")
      .def_ro("num_components", &ConnectedComponentsResult::num_components,
              "Number of connected components.")
      .def_ro("largest_component_size",
              &ConnectedComponentsResult::largest_component_size,
              "Size of the largest component.")
      .def_ro("largest_component_id",
              &ConnectedComponentsResult::largest_component_id,
              "Component ID of the largest component.");

  // ---------------------------------------------------------------------------
  // Connected Components: Dispatch
  // ---------------------------------------------------------------------------

  m.def(
      "RunConnectedComponents",
      [](GbbsGraphHandle& graph, const CCParams& params, CCSession* session) {
        nb::gil_scoped_release release;
        return ValueOrThrow(RunConnectedComponents(graph, params, session));
      },
      nb::arg("graph"), nb::arg("params"),
      nb::arg("session").none() = nb::none(),
      "Run a connected components algorithm on a GBBS graph.\n\n"
      "Returns InvalidArgumentError if the algorithm requires a symmetric\n"
      "graph but the graph is asymmetric (directed).");
}

}  // namespace dgf::gbbs

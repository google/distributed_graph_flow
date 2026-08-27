#include "dgf/src/gbbs/gbbs_graph_handle.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <tuple>

#include "third_party/gbbs/gbbs/macros.h"
#include "third_party/parlay/include/parlay/parallel.h"
#include "third_party/parlay/include/parlay/sequence.h"

namespace dgf::gbbs {

std::unique_ptr<GbbsGraphHandle> CreateGbbsGraphHandle(
    size_t num_nodes, size_t num_edges, const int64_t* edge_sources,
    const int64_t* edge_targets, const float* weights, bool symmetric) {
  std::cout << "CreateGbbsGraphHandle called with num_nodes=" << num_nodes
            << ", num_edges=" << num_edges << ", symmetric=" << symmetric
            << std::endl;
  using EdgeTuple = std::tuple<::gbbs::uintE, ::gbbs::uintE, float>;
  parlay::sequence<EdgeTuple> edges(num_edges);

  if (weights != nullptr) {
    std::cout << "Handling weights" << std::endl;
    parlay::parallel_for(0, num_edges, [&](size_t i) {
      edges[i] = std::make_tuple(static_cast<::gbbs::uintE>(edge_sources[i]),
                                 static_cast<::gbbs::uintE>(edge_targets[i]),
                                 weights[i]);
    });
  } else {
    std::cout << "Handling no weights" << std::endl;
    parlay::parallel_for(0, num_edges, [&](size_t i) {
      edges[i] =
          std::make_tuple(static_cast<::gbbs::uintE>(edge_sources[i]),
                          static_cast<::gbbs::uintE>(edge_targets[i]), 1.0f);
    });
  }

  if (symmetric) {
    return std::make_unique<GbbsGraphHandle>(
        GbbsSymmetric::from_edges(edges, num_nodes));
  } else {
    return std::make_unique<GbbsGraphHandle>(
        GbbsAsymmetric::from_edges(edges, num_nodes));
  }
}

}  // namespace dgf::gbbs

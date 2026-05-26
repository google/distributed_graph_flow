#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "dgf/src/data/in_memory_graph.h"  // IWYU pragma: keep
#include "dgf/src/data/in_memory_graph_nb.h"
#include "dgf/src/io/tf_graph_sample.h"
#include "dgf/src/util/concurrency.h"
#include "dgf/src/util/status_caster.h"
#include "dgf/src/util/util.h"

namespace dgf::tf_graph_sample_ext {
namespace {
namespace nb = nanobind;
constexpr size_t kTargetBlockSize = 1000;

absl::StatusOr<nb::bytes> SerializeGraph(nb::object in_memory_graph) {
  DGF_ASSIGN_OR_RETURN(const auto graph,
                       dgf::data::Graph::Create(in_memory_graph));

  data::tensorflow::Example example;
  dgf::tf_graph_sample::GraphToTfgnnExample(graph.view, &example);

  const std::string serialized_example = example.SerializeAsString();
  return nb::bytes(serialized_example.data(), serialized_example.size());
}

absl::StatusOr<std::string> DebugStringFromGraph(nb::object in_memory_graph) {
  DGF_ASSIGN_OR_RETURN(const auto graph,
                       dgf::data::Graph::Create(in_memory_graph));
  data::tensorflow::Example example;
  dgf::tf_graph_sample::GraphToTfgnnExample(graph.view, &example);
  return example.DebugString();
}

absl::StatusOr<std::vector<nb::bytes>> SerializeGraphs(
    nb::sequence in_memory_graphs, int num_threads) {
  DGF_ASSIGN_OR_RETURN(const auto graphs,
                       dgf::data::CreateGraphs(in_memory_graphs));
  const size_t num_graphs = graphs.size();
  std::vector<std::string> serialized_strings(num_graphs);
  std::vector<nb::bytes> serialized_graphs(num_graphs);

  if (num_threads < 0) {
    nb::gil_scoped_release release;

    LOG(INFO) << "num_threads: " << num_threads << " using a single thread.";
    data::tensorflow::Example example;
    for (size_t i = 0; i < num_graphs; ++i) {
      example.Clear();
      dgf::tf_graph_sample::GraphToTfgnnExample(graphs[i].view, &example);
      serialized_strings[i] = example.SerializeAsString();
    }
  } else {
    // Release the GIL, go fast...
    nb::gil_scoped_release release;

    dgf::util::concurrency::ThreadPool thread_pool(num_threads);

    // Guessing something on [1, 100] blocks with 1000 items per block will work
    // well.
    const size_t num_blocks = std::min(
        (size_t)num_threads,
        std::clamp(num_graphs / kTargetBlockSize, size_t{1}, size_t{100}));

    util::concurrency::ConcurrentForLoop(
        num_blocks, &thread_pool, num_graphs,
        [&graphs, &serialized_strings](size_t block_idx, size_t begin_item_idx,
                                       size_t end_item_idx) {
          data::tensorflow::Example example;
          for (size_t i = begin_item_idx; i < end_item_idx; ++i) {
            example.Clear();
            dgf::tf_graph_sample::GraphToTfgnnExample(graphs[i].view, &example);
            serialized_strings[i] = example.SerializeAsString();
          }
        });
  }
  // Once we have the GIL again, we can move the serialized graphs to python.
  for (size_t i = 0; i < num_graphs; ++i) {
    serialized_graphs[i] =
        nb::bytes(serialized_strings[i].data(), serialized_strings[i].size());
  }

  return serialized_graphs;
}

NB_MODULE(tf_graph_sample_ext, m) {
  m.def("debug_string", ValueOrThrowWrapper(DebugStringFromGraph),
        R"doc(C++ extension that converts a an InMemoryGraph object to the TFGNN
      format TF-Example debug string (usefule for testing).)doc");
  m.def("serialize_graph", ValueOrThrowWrapper(SerializeGraph),
        R"doc(C++ extension that converts a list of InMemoryGraph objects to a 
      list of tf.Example protos serialized to bytes.)doc");
  m.def("serialize_graphs", ValueOrThrowWrapper(SerializeGraphs),
        nb::arg("in_memory_graphs"), nb::arg("num_threads") = -1,
        R"doc(C++ extension that converts a list of InMemoryGraph objects to a 
              list of tf.Example protos serialized to bytes using 
              multiple threads.)doc");
}

}  // namespace
}  // namespace dgf::tf_graph_sample_ext

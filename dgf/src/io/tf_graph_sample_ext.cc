#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "dgf/src/data/in_memory_graph.h"  // IWYU pragma: keep
#include "dgf/src/data/in_memory_graph_nb.h"
#include "dgf/src/data/schema.h"
#include "dgf/src/data/schema_nb.h"
#include "dgf/src/io/tf_graph_sample.h"
#include "dgf/src/io/tf_graph_sample_parser.h"
#include "dgf/src/util/concurrency.h"
#include "dgf/src/util/nanobind_util.h"
#include "dgf/src/util/status_caster.h"
#include "dgf/src/util/util.h"

// NumPy C API. Imported in the module initialization.
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include "numpy/arrayobject.h"

namespace dgf::tf_graph_sample_ext {
namespace {
namespace nb = nanobind;
namespace tgs = ::dgf::tf_graph_sample;

constexpr size_t kTargetBlockSize = 1000;
// Number of parsing jobs per thread. Small jobs balance the work between the
// threads (as graphs have different sizes).
constexpr int kNumParsingBlocksPerThread = 4;

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

// Converts a decoded array into a NumPy array without copy. Uses the NumPy C
// API as it is significantly faster than nanobind's ndarray.
nb::object ArrayToNumpy(tgs::Array& array) {
  PyArray_Descr* descr = nullptr;
  switch (array.format) {
    case tgs::Format::INTEGER_64:
      descr = PyArray_DescrFromType(NPY_INT64);
      break;
    case tgs::Format::INTEGER_32:
      descr = PyArray_DescrFromType(NPY_INT32);
      break;
    case tgs::Format::FLOAT_32:
      descr = PyArray_DescrFromType(NPY_FLOAT32);
      break;
    case tgs::Format::FLOAT_64:
      descr = PyArray_DescrFromType(NPY_FLOAT64);
      break;
    case tgs::Format::BOOL:
      descr = PyArray_DescrFromType(NPY_BOOL);
      break;
    case tgs::Format::BYTES:
      descr = PyArray_DescrNewFromType(NPY_STRING);
      if (descr) {
        PyDataType_SET_ELSIZE(descr, static_cast<npy_intp>(array.itemsize));
      }
      break;
  }
  if (!descr) {
    throw nb::python_error();
  }
  static_assert(sizeof(npy_intp) == sizeof(int64_t));
  // Note: `PyArray_NewFromDescr` steals the reference to `descr`.
  nb::object result = nb::steal(PyArray_NewFromDescr(
      &PyArray_Type, descr, static_cast<int>(array.shape.size()),
      reinterpret_cast<npy_intp*>(array.shape.data()),
      /*strides=*/nullptr, array.data.get(), NPY_ARRAY_CARRAY,
      /*obj=*/nullptr));
  if (!result.is_valid()) {
    throw nb::python_error();
  }
  // The capsule owns the data, and is owned by the NumPy array.
  PyObject* owner =
      PyCapsule_New(array.data.get(), /*name=*/nullptr, [](PyObject* capsule) {
        std::free(PyCapsule_GetPointer(capsule, /*name=*/nullptr));
      });
  if (!owner) {
    throw nb::python_error();
  }
  array.data.release();
  // Note: `PyArray_SetBaseObject` steals the reference to `owner`.
  if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(result.ptr()),
                            owner) < 0) {
    throw nb::python_error();
  }
  return result;
}

// A feature of the Python schema.
struct PyFeature {
  // Index of the feature in the decoded node/edge set.
  int idx;
  nb::str name;
  // Key of the feature in the graph samples.
  nb::str key;
  // The `FeatureSchema`.
  nb::object schema;
};

// A node/edge set of the Python schema.
struct PySet {
  // Index of the set in the decoded graph.
  int idx;
  nb::str name;
  std::vector<PyFeature> features;
};

// Lists the node/edge sets and features of a Python schema. `py_sets` is a
// dict of `NodeSchema` or `EdgeSchema`, and `cc_sets` is the corresponding
// `nodesets` or `edgesets` of the C++ schema.
template <typename CCSets>
absl::StatusOr<std::vector<PySet>> ListPySets(
    const nb::dict& py_sets, const CCSets& cc_sets,
    const absl::flat_hash_map<std::string, int>& set_name_to_idx,
    std::string (*feature_key)(std::string_view, std::string_view)) {
  std::vector<PySet> sets;
  for (const auto [py_set_name, py_set] : py_sets) {
    const std::string set_name = nb::cast<std::string>(py_set_name);
    PySet& set = sets.emplace_back();
    set.name = nb::borrow<nb::str>(py_set_name);
    DGF_ASSIGN_OR_RETURN(set.idx, GetItem(set_name_to_idx, set_name));
    const auto& featureset = cc_sets[set.idx].featureset;
    DGF_GET_ATTR_OR_RETURN(nb::dict, py_features, py_set, "features");
    for (const auto [py_feature_name, py_feature] : py_features) {
      const std::string feature_name = nb::cast<std::string>(py_feature_name);
      PyFeature& feature = set.features.emplace_back();
      DGF_ASSIGN_OR_RETURN(
          feature.idx, GetItem(featureset.feature_name_to_idx, feature_name));
      feature.name = nb::borrow<nb::str>(py_feature_name);
      feature.key = string_to_py_str(feature_key(set_name, feature_name));
      feature.schema = nb::borrow<nb::object>(py_feature);
    }
  }
  return sets;
}

// Parsing resources shared between the parser and the futures. Contains Python
// objects i.e. should only be destroyed with the GIL. The background jobs only
// reference the (pure C++) decoder.
struct ParserCore {
  std::shared_ptr<const tgs::GraphSampleDecoder> decoder;
  nb::object graph_cls;
  nb::object node_set_cls;
  nb::object edge_set_cls;
  // Builds a ragged feature value from the decoded flat values and row lengths.
  nb::object build_ragged_feature;
  // The node/edge sets and features in the order of the Python schema.
  std::vector<PySet> node_sets;
  std::vector<PySet> edge_sets;
  // Destroyed first. The thread pool is idle as the futures wait for their
  // jobs.
  std::unique_ptr<util::concurrency::ThreadPool> thread_pool;

  nb::dict FeaturesToPython(std::vector<tgs::DecodedFeature>& features,
                            const std::vector<PyFeature>& py_features,
                            int64_t num_items) const {
    nb::dict result;
    for (const PyFeature& py_feature : py_features) {
      tgs::DecodedFeature& feature = features[py_feature.idx];
      nb::object values = ArrayToNumpy(feature.values);
      if (feature.row_lengths.empty()) {
        result[py_feature.name] = values;
        continue;
      }
      nb::list row_lengths;
      for (auto& row_length : feature.row_lengths) {
        row_lengths.append(ArrayToNumpy(row_length));
      }
      result[py_feature.name] = build_ragged_feature(
          values, row_lengths, num_items, py_feature.key, py_feature.schema);
    }
    return result;
  }

  // Converts a decoded graph into an InMemoryGraph.
  nb::object GraphToPython(tgs::DecodedGraph& graph) const {
    nb::dict py_node_sets;
    for (const PySet& py_set : node_sets) {
      auto& node_set = graph.node_sets[py_set.idx];
      py_node_sets[py_set.name] =
          node_set_cls(nb::int_(node_set.num_nodes),
                       FeaturesToPython(node_set.features, py_set.features,
                                        node_set.num_nodes));
    }
    nb::dict py_edge_sets;
    for (const PySet& py_set : edge_sets) {
      auto& edge_set = graph.edge_sets[py_set.idx];
      py_edge_sets[py_set.name] =
          edge_set_cls(ArrayToNumpy(edge_set.adjacency),
                       FeaturesToPython(edge_set.features, py_set.features,
                                        edge_set.num_edges));
    }
    return graph_cls(py_node_sets, py_edge_sets);
  }
};

// Result of an asynchronous parsing.
class ParseFuture {
 public:
  ParseFuture(std::shared_ptr<const ParserCore> core, nb::object serialized)
      : core_(std::move(core)),
        serialized_(std::move(serialized)),
        state_(std::make_shared<State>()) {}

  ~ParseFuture() {
    if (!state_->done.HasBeenNotified()) {
      // The jobs are reading the bytes in "serialized_".
      nb::gil_scoped_release release;
      state_->done.WaitForNotification();
    }
  }

  absl::Status Start() {
    auto inputs = SequenceOfBytesToStringViews(serialized_);
    if (!inputs.ok()) {
      state_->done.Notify();
      return inputs.status();
    }
    state_->inputs = *std::move(inputs);
    state_->outputs.resize(state_->inputs.size());
    util::concurrency::ThreadPool* thread_pool = core_->thread_pool.get();
    nb::gil_scoped_release release;
    // Without threads, the parsing is done in the calling thread.
    util::concurrency::ScheduleConcurrentForLoop(
        std::max(1, thread_pool->num_threads() * kNumParsingBlocksPerThread),
        thread_pool, state_->inputs.size(),
        // The jobs should not own any Python object.
        [state = state_, decoder = core_->decoder](size_t block_idx,
                                                   size_t begin, size_t end) {
          const absl::Status status = decoder->DecodeBatch(
              absl::MakeConstSpan(state->inputs).subspan(begin, end - begin),
              absl::MakeSpan(state->outputs).subspan(begin, end - begin));
          if (!status.ok()) {
            absl::MutexLock lock(state->mutex);
            state->status.Update(status);
          }
        },
        [state = state_]() { state->done.Notify(); });
    return absl::OkStatus();
  }

  bool Done() const { return state_->done.HasBeenNotified(); }

  // Waits for the parsing to be done, and returns the InMemoryGraphs.
  absl::StatusOr<nb::list> Result() {
    if (consumed_) {
      return absl::FailedPreconditionError("The result was already consumed");
    }
    {
      nb::gil_scoped_release release;
      state_->done.WaitForNotification();
    }
    consumed_ = true;
    {
      absl::MutexLock lock(state_->mutex);
      if (!state_->status.ok()) {
        return state_->status;
      }
    }
    nb::list graphs;
    for (auto& graph : state_->outputs) {
      graphs.append(core_->GraphToPython(graph));
    }
    state_->outputs.clear();
    serialized_ = nb::none();
    return graphs;
  }

 private:
  struct State {
    std::vector<std::string_view> inputs;
    std::vector<tgs::DecodedGraph> outputs;
    absl::Notification done;
    absl::Mutex mutex;
    absl::Status status;
  };

  std::shared_ptr<const ParserCore> core_;
  nb::object serialized_;
  std::shared_ptr<State> state_;
  bool consumed_ = false;
};

// Parses serialized TF-GNN graph samples into InMemoryGraphs.
class GraphSampleParser {
 public:
  // `build_ragged_feature` is called as `build_ragged_feature(values,
  // row_lengths, num_items, feature_key, feature_schema)` to build the value of
  // the variable-length features. If `num_threads` is 0, decodes in the
  // calling thread.
  static absl::StatusOr<std::unique_ptr<GraphSampleParser>> Create(
      nb::object schema, std::optional<std::string> import_node_ids,
      std::optional<std::string> import_edge_ids,
      nb::object build_ragged_feature, nb::object graph_cls,
      nb::object node_set_cls, nb::object edge_set_cls, int num_threads) {
    DGF_ASSIGN_OR_RETURN(std::unique_ptr<data::GraphSchema> cc_schema,
                         data::CreateGraphSchema(schema));
    auto core = std::make_shared<ParserCore>();
    core->graph_cls = graph_cls;
    core->node_set_cls = node_set_cls;
    core->edge_set_cls = edge_set_cls;
    core->build_ragged_feature = build_ragged_feature;
    DGF_GET_ATTR_OR_RETURN(nb::dict, py_node_sets, schema, "node_sets");
    DGF_GET_ATTR_OR_RETURN(nb::dict, py_edge_sets, schema, "edge_sets");
    DGF_ASSIGN_OR_RETURN(
        core->node_sets,
        ListPySets(py_node_sets, cc_schema->nodesets,
                   cc_schema->nodeset_name_to_idx, &tgs::NodeKey));
    DGF_ASSIGN_OR_RETURN(
        core->edge_sets,
        ListPySets(py_edge_sets, cc_schema->edgesets,
                   cc_schema->edgeset_name_to_idx, &tgs::EdgeKey));
    DGF_ASSIGN_OR_RETURN(
        core->decoder, tgs::GraphSampleDecoder::Create(
                           std::move(*cc_schema), import_node_ids.value_or(""),
                           import_edge_ids.value_or("")));
    core->thread_pool = std::make_unique<util::concurrency::ThreadPool>(
        std::max(0, num_threads));
    return std::unique_ptr<GraphSampleParser>(
        new GraphSampleParser(std::move(core)));
  }

  // Starts parsing a sequence of serialized graph samples in the background.
  absl::StatusOr<std::unique_ptr<ParseFuture>> Submit(nb::object serialized) {
    auto future = std::make_unique<ParseFuture>(core_, std::move(serialized));
    DGF_RETURN_IF_ERROR(future->Start());
    return future;
  }

  // Parses a sequence of serialized graph samples.
  absl::StatusOr<nb::list> Parse(nb::object serialized) {
    DGF_ASSIGN_OR_RETURN(auto future, Submit(std::move(serialized)));
    return future->Result();
  }

 private:
  explicit GraphSampleParser(std::shared_ptr<ParserCore> core)
      : core_(std::move(core)) {}

  std::shared_ptr<const ParserCore> core_;
};

NB_MODULE(tf_graph_sample_ext, m) {
  if (_import_array() < 0) {
    throw nb::python_error();
  }
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

  nb::class_<ParseFuture>(m, "ParseFuture")
      .def("done", &ParseFuture::Done,
           "Checks if the parsing is done (without blocking).")
      .def("result", ValueOrThrowWrapper(&ParseFuture::Result),
           "Waits for the parsing to be done, and returns the list of "
           "InMemoryGraphs. Can only be called once.");

  nb::class_<GraphSampleParser>(m, "GraphSampleParser")
      .def_static("create", ValueOrThrowWrapper(GraphSampleParser::Create),
                  nb::arg("schema"), nb::arg("import_node_ids").none(),
                  nb::arg("import_edge_ids").none(),
                  nb::arg("build_ragged_feature"), nb::arg("graph_cls"),
                  nb::arg("node_set_cls"), nb::arg("edge_set_cls"),
                  nb::arg("num_threads"))
      .def("submit", ValueOrThrowWrapper(&GraphSampleParser::Submit),
           nb::arg("serialized"),
           "Starts the parsing of a sequence of serialized graph samples in "
           "background threads, and returns a future.")
      .def("parse", ValueOrThrowWrapper(&GraphSampleParser::Parse),
           nb::arg("serialized"),
           "Parses a sequence of serialized graph samples into a list of "
           "InMemoryGraphs.");
}

}  // namespace
}  // namespace dgf::tf_graph_sample_ext

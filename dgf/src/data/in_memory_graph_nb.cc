#include "dgf/src/data/in_memory_graph_nb.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "dgf/src/data/in_memory_graph.h"
#include "dgf/src/util/nanobind_util.h"
#include "dgf/src/util/util.h"

namespace dgf::data {
namespace nb = nanobind;

// Typedef to avoid compiler errors when used in DGF_GET_ATTR_OR_RETURN macros.
// When using this type in a macro, we specify the explicit type rather than the
// alias for readability.
using AdjacencyPyT = nb::ndarray<int64_t, nb::numpy, nb::shape<2, -1>>;

template <typename T>
std::vector<size_t> GetShape(
    const nb::ndarray<T, nb::c_contig, nb::device::cpu>& arr) {
  std::vector<size_t> shape;
  shape.reserve(arr.ndim());
  for (size_t i = 0; i < arr.ndim(); ++i) {
    shape.push_back(arr.shape(i));
  }
  return shape;
}

template <typename T>
std::vector<int64_t> GetStrides(
    const nb::ndarray<T, nb::c_contig, nb::device::cpu>& arr) {
  std::vector<int64_t> strides;
  strides.reserve(arr.ndim());
  for (size_t i = 0; i < arr.ndim(); ++i) {
    strides.push_back(arr.stride(i));
  }
  return strides;
}

absl::StatusOr<TensorView<std::string_view>> SequenceToTensorView(
    nb::sequence seq, std::vector<nb::object>& refs) {
  TensorView<std::string_view> view;
  if (nb::len(seq) == 0 || seq.is_none()) {
    return view;
  }

  // Try 1D first.
  auto maybe_bytes = NumpyBytesArray::Create(seq);
  if (maybe_bytes.ok()) {
    refs.push_back(nb::borrow<nb::object>(seq));
    auto row_view = maybe_bytes->ToVectorNotOwned();
    view.data.insert(view.data.end(), row_view.begin(), row_view.end());
    view.shape.push_back(row_view.size());
    view.strides.push_back((int64_t)(maybe_bytes->stride_));
    view.itemsizes.push_back(maybe_bytes->itemsize_);
    return view;
  }

  // Try 2D
  auto& data = view.data;
  for (auto row : seq) {
    refs.push_back(nb::borrow<nb::object>(row));
    auto maybe_bytes = NumpyBytesArray::Create(refs.back());
    if (!maybe_bytes.ok()) {
      // nb::print("repr: ", nb::repr(refs.back()));
      LOG(INFO) << "typename: " << nb::type_name(row.type()).c_str();
      return absl::Status(
          maybe_bytes.status().code(),
          absl::StrCat("Only Ragged byte views are currently supported: ",
                       maybe_bytes.status().message()));
    }
    std::vector<std::string_view> row_view = maybe_bytes->ToVectorNotOwned();
    data.insert(data.end(), row_view.begin(), row_view.end());
    view.shape.push_back(row_view.size());
    view.strides.push_back((int64_t)(maybe_bytes->stride_));

    // We expect that the itemsize should be constant.
    view.strides.push_back(maybe_bytes->itemsize_);
  }
  return view;
}

// TODO(bmayer): Take schema as input arg to simplify logic.
absl::StatusOr<FeaturesView> CreateFeaturesView(nb::dict features,
                                                std::vector<nb::object>& refs) {
  FeaturesView features_view;
  for (auto [py_feature_name, py_feature] : features) {
    std::string feature_name = nb::cast<std::string>(py_feature_name);

    if (nb::isinstance<nb::ndarray<float, nb::c_contig, nb::device::cpu>>(
            py_feature)) {
      refs.push_back(nb::borrow<nb::object>(py_feature));
      auto arr = nb::cast<nb::ndarray<float, nb::c_contig, nb::device::cpu>>(
          py_feature);

      auto shape = GetShape<float>(arr);
      auto strides = GetStrides<float>(arr);

      features_view.float_features[feature_name] = TensorView<float>(
          absl::Span<float>(arr.data(), arr.size()), shape, strides);
    } else if (nb::isinstance<
                   nb::ndarray<int64_t, nb::c_contig, nb::device::cpu>>(
                   py_feature)) {
      refs.push_back(nb::borrow<nb::object>(py_feature));
      auto arr = nb::cast<nb::ndarray<int64_t, nb::c_contig, nb::device::cpu>>(
          py_feature);
      auto shape = GetShape<int64_t>(arr);
      auto strides = GetStrides<int64_t>(arr);
      features_view.int64_features[feature_name] = TensorView<int64_t>(
          absl::Span<int64_t>(arr.data(), arr.size()), shape, strides);
    } else if (nb::isinstance<nb::sequence>(py_feature)) {
      auto bytes_view =
          SequenceToTensorView(nb::borrow<nb::sequence>(py_feature), refs);
      if (!bytes_view.ok()) {
        return absl::Status(
            bytes_view.status().code(),
            absl::StrCat(
                "Only 1d or 2d byte sequences are currently supported.",
                bytes_view.status().message()));
      }
      features_view.bytes_features[feature_name] = *bytes_view;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported feature type: ",
                       nb::type_name(py_feature.type()).c_str()));
    }
  }

  return features_view;
}

absl::Status AddNodeSet(Graph& graph, const std::string& name, nb::object obj) {
  DGF_GET_ATTR_OR_RETURN(int64_t, num_nodes, obj, "num_nodes");
  DGF_GET_ATTR_OR_RETURN(nb::dict, features, obj, "features");
  absl::StatusOr<FeaturesView> features_view =
      CreateFeaturesView(features, graph.refs_);

  if (!features_view.ok()) {
    return features_view.status();
  }
  graph.view.node_sets.emplace_back(name, num_nodes, *features_view);
  return absl::OkStatus();
}

absl::StatusOr<AdjacencyView> CreateAdjacencyView(
    const nb::ndarray<int64_t, nb::numpy, nb::shape<2, -1>> arr) {
  AdjacencyView adjacency_view;

  const size_t num_edges = arr.shape(1);
  adjacency_view.source = absl::Span<int64_t>(arr.data(), num_edges);
  adjacency_view.target =
      absl::Span<int64_t>(arr.data() + num_edges, num_edges);

  return adjacency_view;
}

absl::Status AddEdgeSet(Graph& graph, const std::string& name, nb::object obj) {
  DGF_GET_ATTR_OR_RETURN(nb::dict, features, obj, "features");
  DGF_GET_ATTR_OR_RETURN(nb::object, adjacency_ref, obj, "adjacency");
  DGF_GET_ATTR_OR_RETURN(AdjacencyPyT, adjacency, obj, "adjacency");

  // Keep the adjacency array leaf ref.
  graph.refs_.push_back(nb::borrow<nb::object>(adjacency_ref));
  absl::StatusOr<AdjacencyView> adjacency_view = CreateAdjacencyView(adjacency);
  if (!adjacency_view.ok()) {
    return adjacency_view.status();
  }
  absl::StatusOr<FeaturesView> features_view =
      CreateFeaturesView(features, graph.refs_);
  if (!features_view.ok()) {
    return features_view.status();
  }
  graph.view.edge_sets.emplace_back(name, *adjacency_view, *features_view);
  return absl::OkStatus();
}

absl::StatusOr<Graph> Graph::Create(nb::object obj) {
  Graph graph;
  DGF_GET_ATTR_OR_RETURN(nb::dict, node_sets, obj, "node_sets");
  DGF_GET_ATTR_OR_RETURN(nb::dict, edge_sets, obj, "edge_sets");
  graph.view.node_sets.reserve(node_sets.size());
  graph.view.edge_sets.reserve(edge_sets.size());

  for (auto [py_nodeset_name, py_nodeset] : node_sets) {
    std::string nodeset_name = nb::cast<std::string>(py_nodeset_name);

    // We don't need to record the node set dictionary as a reference, only the
    // leaf array objects.
    const auto nodeset_status =
        AddNodeSet(graph, nodeset_name, nb::borrow<nb::object>(py_nodeset));
    if (!nodeset_status.ok()) {
      return absl::Status(
          nodeset_status.code(),
          absl::StrCat("Failed to add nodeset: ", nodeset_name,
                       " with error:", nodeset_status.message()));
    }
  }

  for (auto [py_edgeset_name, py_edgeset] : edge_sets) {
    std::string edgeset_name = nb::cast<std::string>(py_edgeset_name);
    auto status =
        AddEdgeSet(graph, edgeset_name, nb::borrow<nb::object>(py_edgeset));
    if (!status.ok()) {
      return absl::Status(status.code(),
                          absl::StrCat("Failed to add edgeset: ", edgeset_name,
                                       " with error: ", status.message()));
    }
  }

  return graph;
}

absl::StatusOr<std::vector<Graph>> CreateGraphs(nb::sequence seq) {
  std::vector<Graph> graphs;
  graphs.reserve(nb::len(seq));

  for (auto obj : seq) {
    DGF_ASSIGN_OR_RETURN(auto graph,
                         dgf::data::Graph::Create(nb::borrow<nb::object>(obj)));
    graphs.push_back(graph);
  }
  return graphs;
}

}  // namespace dgf::data

#include "dgf/src/io/tf_graph_sample.h"

#include <string>
#include <string_view>

#include "absl/functional/function_ref.h"
#include "absl/strings/str_cat.h"
#include "dgf/src/data/in_memory_graph.h"
#include "dgf/src/data/tensorflow.pb.h"
#include "dgf/src/util/tf_feature_util.h"

namespace dgf::tf_graph_sample {
using GraphView = dgf::data::GraphView;
using FeaturesView = dgf::data::FeaturesView;

std::string NodeKey(std::string_view node_set_name, std::string_view suffix) {
  return absl::StrCat("nodes/", node_set_name, ".", suffix);
}

std::string EdgeKey(std::string_view edge_set_name, std::string_view suffix) {
  return absl::StrCat("edges/", edge_set_name, ".", suffix);
}

std::string RaggedDimKey(std::string_view feature_key, int dim_idx) {
  return absl::StrCat(feature_key, ".d", dim_idx);
}

void AppendFeatureValues(
    const FeaturesView& features,
    absl::FunctionRef<std::string(std::string_view)> feature_key,
    data::tensorflow::Example& example) {
  auto& feature_map = *example.mutable_features()->mutable_feature();
  for (const auto& [node_feature_name, node_feature_value] :
       features.float_features) {
    data::tensorflow::Feature& tf_feature =
        feature_map[feature_key(node_feature_name)];
    dgf::util::tensorflow::AppendFeatureValues(node_feature_value.data.begin(),
                                               node_feature_value.data.end(),
                                               &tf_feature);
  }

  for (const auto& [node_feature_name, node_feature_value] :
       features.int64_features) {
    data::tensorflow::Feature& tf_feature =
        feature_map[feature_key(node_feature_name)];
    dgf::util::tensorflow::AppendFeatureValues(node_feature_value.data.begin(),
                                               node_feature_value.data.end(),
                                               &tf_feature);
  }

  for (const auto& [node_feature_name, node_feature_value] :
       features.bytes_features) {
    data::tensorflow::Feature& tf_feature =
        feature_map[feature_key(node_feature_name)];
    dgf::util::tensorflow::AppendFeatureValues(node_feature_value.data.begin(),
                                               node_feature_value.data.end(),
                                               &tf_feature);
  }
}

void GraphToTfgnnExample(const GraphView& graph,
                         data::tensorflow::Example* example) {
  for (const auto& node_set : graph.node_sets) {
    auto& feature_map = *(example->mutable_features()->mutable_feature());
    data::tensorflow::Feature& size_feature =
        feature_map[NodeKey(node_set.name, kSizeKey)];
    size_feature.mutable_int64_list()->add_value(node_set.num_nodes);
    AppendFeatureValues(
        node_set.features,
        [&](std::string_view name) { return NodeKey(node_set.name, name); },
        *example);
  }

  for (const auto& edge_set : graph.edge_sets) {
    auto& feature_map = *(example->mutable_features()->mutable_feature());
    data::tensorflow::Feature& size_feature =
        feature_map[EdgeKey(edge_set.name, kSizeKey)];
    size_feature.mutable_int64_list()->add_value(
        edge_set.adjacency.source.size());
    data::tensorflow::Feature& source_feature =
        feature_map[EdgeKey(edge_set.name, kSourceKey)];
    dgf::util::tensorflow::AppendFeatureValues(
        edge_set.adjacency.source.begin(), edge_set.adjacency.source.end(),
        &source_feature);
    data::tensorflow::Feature& target_feature =
        feature_map[EdgeKey(edge_set.name, kTargetKey)];
    dgf::util::tensorflow::AppendFeatureValues(
        edge_set.adjacency.target.begin(), edge_set.adjacency.target.end(),
        &target_feature);
    AppendFeatureValues(
        edge_set.features,
        [&](std::string_view name) { return EdgeKey(edge_set.name, name); },
        *example);
  }
}

}  // namespace dgf::tf_graph_sample

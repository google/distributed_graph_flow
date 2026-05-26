#include "dgf/src/data/schema_nb.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "dgf/src/data/schema.h"
#include "dgf/src/util/nanobind_util.h"
#include "dgf/src/util/util.h"

namespace dgf::data {

namespace {
namespace nb = nanobind;

absl::StatusOr<GraphSchema::Feature::eFormat> StringToFormat(
    const std::string& s) {
  if (s == "INTEGER_32") return GraphSchema::Feature::eFormat::INTEGER_32;
  if (s == "INTEGER_64") return GraphSchema::Feature::eFormat::INTEGER_64;
  if (s == "FLOAT_32") return GraphSchema::Feature::eFormat::FLOAT_32;
  if (s == "FLOAT_64") return GraphSchema::Feature::eFormat::FLOAT_64;
  if (s == "BYTES") return GraphSchema::Feature::eFormat::BYTES;
  if (s == "BOOL") return GraphSchema::Feature::eFormat::BOOL;
  return absl::InvalidArgumentError(absl::StrCat("Unknown FeatureFormat: ", s));
}

absl::StatusOr<GraphSchema::Feature> ParseFeatureSchema(
    const nb::object& py_feature_schema, const std::string& feature_name) {
  GraphSchema::Feature feature;
  feature.name = feature_name;

  DGF_GET_ATTR_OR_RETURN(nb::object, py_format, py_feature_schema, "format");
  DGF_GET_ATTR_OR_RETURN(std::string, format_str, py_format, "value");
  DGF_ASSIGN_OR_RETURN(feature.format, StringToFormat(format_str));

  DGF_GET_ATTR_OR_RETURN(nb::object, py_shape, py_feature_schema, "shape");
  if (py_shape.is_none()) {
    // Shape is []
  } else if (nb::isinstance<nb::tuple>(py_shape)) {
    nb::tuple shape_list = nb::cast<nb::tuple>(py_shape);
    for (const auto& dim : shape_list) {
      if (dim.is_none()) {
        feature.shape.push_back(-1);
      } else if (nb::isinstance<nb::int_>(dim)) {
        feature.shape.push_back(nb::cast<int>(dim));
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid shape dimension type for feature '", feature_name, "'"));
      }
    }
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid shape type for feature '", feature_name, "'"));
  }
  return feature;
}

}  // namespace

absl::StatusOr<GraphSchema::FeatureSet> ParseFeatures(
    const nb::dict& py_features) {
  std::vector<std::string> feature_names;
  for (auto const& [key, val] : py_features) {
    feature_names.push_back(nb::cast<std::string>(key));
  }
  std::sort(feature_names.begin(), feature_names.end());

  GraphSchema::FeatureSet featureset;
  featureset.features.reserve(feature_names.size());
  for (int i = 0; i < feature_names.size(); ++i) {
    const std::string& feature_name = feature_names[i];
    nb::object py_feature_schema = py_features[string_to_py_str(feature_name)];
    DGF_ASSIGN_OR_RETURN(GraphSchema::Feature feature,
                         ParseFeatureSchema(py_feature_schema, feature_name));
    featureset.feature_name_to_idx[feature_name] = i;
    featureset.features.push_back(std::move(feature));
  }
  return featureset;
}

absl::StatusOr<std::unique_ptr<GraphSchema>> CreateGraphSchema(
    const nanobind::object& py_schema) {
  auto schema = std::make_unique<GraphSchema>();

  DGF_GET_ATTR_OR_RETURN(nb::dict, py_node_sets, py_schema, "node_sets");
  DGF_GET_ATTR_OR_RETURN(nb::dict, py_edge_sets, py_schema, "edge_sets");

  // 1. Parse NodeSets
  std::vector<std::string> nodeset_names;
  for (auto const& [key, val] : py_node_sets) {
    nodeset_names.push_back(nb::cast<std::string>(key));
  }
  std::sort(nodeset_names.begin(), nodeset_names.end());

  for (int i = 0; i < nodeset_names.size(); ++i) {
    const std::string& nodeset_name = nodeset_names[i];
    schema->nodeset_name_to_idx[nodeset_name] = i;

    nb::object py_nodeset = py_node_sets[string_to_py_str(nodeset_name)];
    DGF_GET_ATTR_OR_RETURN(nb::dict, py_features, py_nodeset, "features");

    GraphSchema::Nodeset nodeset;
    nodeset.name = nodeset_name;
    DGF_ASSIGN_OR_RETURN(nodeset.featureset, ParseFeatures(py_features));
    schema->nodesets.push_back(std::move(nodeset));
  }

  // 2. Parse EdgeSets
  std::vector<std::string> edgeset_names;
  for (auto const& [key, val] : py_edge_sets) {
    edgeset_names.push_back(nb::cast<std::string>(key));
  }
  std::sort(edgeset_names.begin(), edgeset_names.end());

  for (int i = 0; i < edgeset_names.size(); ++i) {
    const std::string& edgeset_name = edgeset_names[i];
    nb::object py_edgeset = py_edge_sets[string_to_py_str(edgeset_name)];
    schema->edgeset_name_to_idx[edgeset_name] = i;

    DGF_GET_ATTR_OR_RETURN(std::string, source_name, py_edgeset, "source");
    DGF_GET_ATTR_OR_RETURN(std::string, target_name, py_edgeset, "target");
    DGF_GET_ATTR_OR_RETURN(nb::dict, py_features, py_edgeset, "features");

    GraphSchema::Edgeset edgeset;
    edgeset.name = edgeset_name;

    auto source_it = schema->nodeset_name_to_idx.find(source_name);
    if (source_it == schema->nodeset_name_to_idx.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Source nodeset '", source_name, "' not found in node_sets."));
    }
    edgeset.source_nodeset = source_it->second;

    auto target_it = schema->nodeset_name_to_idx.find(target_name);
    if (target_it == schema->nodeset_name_to_idx.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Target nodeset '", target_name, "' not found in node_sets."));
    }
    edgeset.target_nodeset = target_it->second;

    DGF_ASSIGN_OR_RETURN(edgeset.featureset, ParseFeatures(py_features));
    schema->edgesets.push_back(std::move(edgeset));
  }

  return schema;
}

}  // namespace dgf::data

#include "dgf/src/data/schema.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace dgf::data {

std::string GraphSchema::Feature::FormatToString(
    GraphSchema::Feature::eFormat format) {
  switch (format) {
    case eFormat::INTEGER_32:
      return "INTEGER_32";
    case eFormat::INTEGER_64:
      return "INTEGER_64";
    case eFormat::FLOAT_32:
      return "FLOAT_32";
    case eFormat::FLOAT_64:
      return "FLOAT_64";
    case eFormat::BYTES:
      return "BYTES";
    case eFormat::BOOL:
      return "BOOL";
    default:
      return "UNKNOWN";
  }
}

bool GraphSchema::Feature::fixed_size() const {
  for (int dim : shape) {
    if (dim < 0) {
      return false;
    }
  }
  return true;
}

std::string GraphSchema::Feature::to_string(int indent) const {
  std::string prefix(indent * 2, ' ');
  auto shape_formatter = [](std::string* out, int dim) {
    absl::StrAppend(out, dim == -1 ? "None" : std::to_string(dim));
  };
  return absl::StrCat(
      prefix, "Feature(name='", name, "', shape=[",
      absl::StrJoin(shape, ", ", shape_formatter),
      "], format=", FormatToString(format),
      is_timeseries ? ", is_timeseries=true" : "",
      timestamps.empty() ? "" : absl::StrCat(", timestamps='", timestamps, "'"),
      ")");
}

std::string GraphSchema::Nodeset::to_string(int indent) const {
  std::string prefix(indent * 2, ' ');
  auto feature_formatter = [&](std::string* out, const Feature& feature) {
    absl::StrAppend(out, feature.to_string(indent + 1));
  };
  std::string features_part =
      featureset.features.empty()
          ? "[]"
          : absl::StrCat(
                "[\n",
                absl::StrJoin(featureset.features, ",\n", feature_formatter),
                "\n", prefix, "]");
  return absl::StrCat(prefix, "Nodeset(name='", name,
                      "', features=", features_part, ")");
}

std::string GraphSchema::Edgeset::to_string(int indent) const {
  std::string prefix(indent * 2, ' ');
  auto feature_formatter = [&](std::string* out, const Feature& feature) {
    absl::StrAppend(out, feature.to_string(indent + 1));
  };
  std::string features_part =
      featureset.features.empty()
          ? "[]"
          : absl::StrCat(
                "[\n",
                absl::StrJoin(featureset.features, ",\n", feature_formatter),
                "\n", prefix, "]");
  return absl::StrCat(
      prefix, "Edgeset(name='", name, "', source_nodeset=", source_nodeset,
      ", target_nodeset=", target_nodeset, ", features=", features_part, ")");
}

std::string GraphSchema::to_string() const {
  auto nodeset_formatter = [&](std::string* out, const Nodeset& nodeset) {
    absl::StrAppend(out, nodeset.to_string(1));
  };
  auto edgeset_formatter = [&](std::string* out, const Edgeset& edgeset) {
    absl::StrAppend(out, edgeset.to_string(1));
  };
  std::string nodesets_part =
      nodesets.empty()
          ? "[]"
          : absl::StrCat("[\n",
                         absl::StrJoin(nodesets, ",\n", nodeset_formatter),
                         "\n]");
  std::string edgesets_part =
      edgesets.empty()
          ? "[]"
          : absl::StrCat("[\n",
                         absl::StrJoin(edgesets, ",\n", edgeset_formatter),
                         "\n]");
  return absl::StrCat("GraphSchema(nodesets=", nodesets_part,
                      ", edgesets=", edgesets_part, ")");
}

}  // namespace dgf::data

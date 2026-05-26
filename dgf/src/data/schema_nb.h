#ifndef DGF_SRC_DATA_SCHEMA_NB_H_
#define DGF_SRC_DATA_SCHEMA_NB_H_

#include <memory>

#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "dgf/src/data/schema.h"

namespace dgf::data {

// Convert a python schema into a c++ schema.
absl::StatusOr<std::unique_ptr<GraphSchema>> CreateGraphSchema(
    const nanobind::object& py_schema);

// Parse a feature set schema.
absl::StatusOr<GraphSchema::FeatureSet> ParseFeatures(
    const nanobind::dict& py_features);

}  // namespace dgf::data

#endif  // DGF_SRC_DATA_SCHEMA_NB_H_

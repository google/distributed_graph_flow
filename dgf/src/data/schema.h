#ifndef DGF_SRC_DATA_SCHEMA_H_
#define DGF_SRC_DATA_SCHEMA_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace dgf::data {

// Equivalent of the python graph schema.
//
// This class creates an implicit indexing of the nodesets/edgesets/features.
// This indexing is guaranteed to be the same each time the GraphSchema is
// built.
struct GraphSchema {
  struct Feature {
    enum class eFormat {
      INTEGER_32,
      INTEGER_64,
      FLOAT_32,
      FLOAT_64,
      BYTES,
      BOOL
    };

    std::string name;
    // Shape of the feature. -1 (in cc) is equivalent to None (in python).
    std::vector<int> shape;
    eFormat format;
    bool is_timeseries = false;
    bool is_creation_time = false;
    std::string group;

    std::string to_string(int indent) const;

    // Returns true if the feature has a fixed size.
    bool fixed_size() const;

   private:
    static std::string FormatToString(eFormat format);
  };

  struct FeatureSet {
    std::vector<Feature> features;
    absl::flat_hash_map<std::string, int> feature_name_to_idx;
  };

  struct Nodeset {
    std::string name;
    FeatureSet featureset;

    std::string to_string(int indent) const;
  };

  struct Edgeset {
    std::string name;
    int source_nodeset;
    int target_nodeset;
    FeatureSet featureset;

    std::string to_string(int indent) const;
  };

  std::vector<Nodeset> nodesets;
  std::vector<Edgeset> edgesets;

  absl::flat_hash_map<std::string, int> nodeset_name_to_idx;
  absl::flat_hash_map<std::string, int> edgeset_name_to_idx;

  // Debug string
  std::string to_string() const;
};

}  // namespace dgf::data

#endif  // DGF_SRC_DATA_SCHEMA_H_

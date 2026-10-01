// Fast parsing of serialized TF-GNN Graph Samples (i.e. `tf.train.Example`)
// into in-memory buffers, following the semantic of `tf.io.parse_example`.
// Uses the TensorFlow-free copy of the Example proto in
// `dgf/src/data/tensorflow.proto`.

#ifndef DGF_SRC_IO_TF_GRAPH_SAMPLE_PARSER_H_
#define DGF_SRC_IO_TF_GRAPH_SAMPLE_PARSER_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dgf/src/data/schema.h"
#include "dgf/src/data/tensorflow.pb.h"

namespace dgf::tf_graph_sample {

using Format = data::GraphSchema::Feature::eFormat;

// Owned, C-contiguous, n-dimensional array.
struct Array {
  // The data is allocated with `std::malloc`.
  struct Deleter {
    void operator()(char* p) const { std::free(p); }
  };

  Format format = Format::INTEGER_64;
  std::vector<int64_t> shape;
  // Size in bytes of each item (for BYTES: the length of the longest value).
  size_t itemsize = 0;
  std::unique_ptr<char, Deleter> data;

  size_t num_items() const;
  size_t num_bytes() const { return num_items() * itemsize; }
};

// A decoded feature.
struct DecodedFeature {
  // For static shape features: The final array of shape [num_items, *shape].
  // For ragged features: The flat values. For imported ids: The flat values
  // with their storage format.
  Array values;
  // For ragged features: The row lengths (int64) of each ragged dimension (in
  // order of the dimensions).
  std::vector<Array> row_lengths;
};

struct DecodedNodeSet {
  int64_t num_nodes = 0;
  std::vector<DecodedFeature> features;
};

struct DecodedEdgeSet {
  int64_t num_edges = 0;
  // Array of shape [2, num_edges] and format int64.
  Array adjacency;
  std::vector<DecodedFeature> features;
};

// The node sets, edge sets and features are indexed as in the schema.
struct DecodedGraph {
  std::vector<DecodedNodeSet> node_sets;
  std::vector<DecodedEdgeSet> edge_sets;
};

// Decodes serialized TF-GNN Graph Samples.
class GraphSampleDecoder {
 public:
  // If not empty, `import_node_ids` (resp. `import_edge_ids`) is the name of
  // the node (resp. edge) feature containing the ids. This feature is returned
  // "as is" i.e. flat and with its storage format.
  static absl::StatusOr<std::unique_ptr<GraphSampleDecoder>> Create(
      data::GraphSchema schema, std::string_view import_node_ids = {},
      std::string_view import_edge_ids = {});

  // Decodes a single serialized graph sample. Thread-safe.
  absl::Status Decode(std::string_view serialized, DecodedGraph* graph) const;

  // Decodes a set of serialized graph samples. Faster than calling `Decode` on
  // each of them. Thread-safe.
  absl::Status DecodeBatch(absl::Span<const std::string_view> serialized,
                           absl::Span<DecodedGraph> graphs) const;

  const data::GraphSchema& schema() const { return schema_; }

 private:
  // Index of a key in `keys_`.
  using KeyIdx = int;
  // The features of an example indexed by KeyIdx. Missing features are null.
  using FeaturesByKey = absl::Span<const data::tensorflow::Feature* const>;

  struct CompiledFeature {
    KeyIdx key_idx;
    // Output format.
    Format format;
    // If true, returns the values "as is" (for the imported ids).
    bool raw = false;
    // Shape of the feature, excluding the item dimension.
    std::vector<int64_t> shape;
    // Key idx of the row lengths of the ragged dims. Empty if the feature is
    // not ragged.
    std::vector<KeyIdx> ragged_key_idxs;
    // Product of the static dimensions.
    int64_t static_size = 1;
  };

  struct CompiledSet {
    KeyIdx size_key_idx;
    KeyIdx source_key_idx = -1;  // Only for edge sets.
    KeyIdx target_key_idx = -1;  // Only for edge sets.
    std::vector<CompiledFeature> features;
  };

  GraphSampleDecoder(data::GraphSchema schema, std::vector<std::string> keys,
                     absl::flat_hash_map<std::string, KeyIdx> key_to_idx,
                     std::vector<CompiledSet> node_sets,
                     std::vector<CompiledSet> edge_sets);

  // Decodes a parsed graph sample.
  absl::Status DecodeExample(const data::tensorflow::Example& example,
                             DecodedGraph* graph) const;

  absl::Status DecodeFeature(const CompiledFeature& compiled,
                             FeaturesByKey features, int64_t num_items,
                             DecodedFeature* output) const;

  absl::Status DecodeFeatures(const std::vector<CompiledFeature>& compiled,
                              FeaturesByKey features, int64_t num_items,
                              std::vector<DecodedFeature>* output) const;

  data::GraphSchema schema_;
  std::vector<std::string> keys_;
  absl::flat_hash_map<std::string, KeyIdx> key_to_idx_;
  std::vector<CompiledSet> node_sets_;
  std::vector<CompiledSet> edge_sets_;
};

}  // namespace dgf::tf_graph_sample

#endif  // DGF_SRC_IO_TF_GRAPH_SAMPLE_PARSER_H_

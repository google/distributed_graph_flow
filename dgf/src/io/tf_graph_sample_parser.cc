#include "dgf/src/io/tf_graph_sample_parser.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "google/protobuf/arena.h"
#include "dgf/src/data/schema.h"
#include "dgf/src/data/tensorflow.pb.h"
#include "dgf/src/io/tf_graph_sample.h"
#include "dgf/src/util/util.h"

namespace dgf::tf_graph_sample {
namespace {

using ::dgf::data::tensorflow::Example;
using ::dgf::data::tensorflow::Feature;

// Size of the memory blocks of the arena used to parse the examples.
constexpr size_t kArenaBlockSize = 1 << 20;

std::string_view KindName(Feature::KindCase kind) {
  switch (kind) {
    case Feature::KIND_NOT_SET:
      return "none";
    case Feature::kBytesList:
      return "bytes_list";
    case Feature::kFloatList:
      return "float_list";
    case Feature::kInt64List:
      return "int64_list";
  }
  return "unknown";
}

// Format used to store the values of a given format in a tf.train.Example. See
// `FEATURE_FORMAT_TO_TFGNN_STORAGE_FORMAT` in "dgf/src/io/feature_format.py".
Format StorageFormat(Format format) {
  switch (format) {
    case Format::INTEGER_32:
    case Format::INTEGER_64:
    case Format::BOOL:
      return Format::INTEGER_64;
    case Format::FLOAT_32:
    case Format::FLOAT_64:
      return Format::FLOAT_32;
    case Format::BYTES:
      return Format::BYTES;
  }
  return format;
}

// Kind of the tf.train.Feature storing values of a given format.
Feature::KindCase StorageKind(Format format) {
  switch (StorageFormat(format)) {
    case Format::INTEGER_64:
      return Feature::kInt64List;
    case Format::FLOAT_32:
      return Feature::kFloatList;
    case Format::BYTES:
      return Feature::kBytesList;
    default:
      return Feature::KIND_NOT_SET;
  }
}

size_t ItemSize(Format format) {
  switch (format) {
    case Format::INTEGER_64:
    case Format::FLOAT_64:
      return 8;
    case Format::INTEGER_32:
    case Format::FLOAT_32:
      return 4;
    case Format::BOOL:
      return 1;
    case Format::BYTES:
      return 0;  // Variable.
  }
  return 0;
}

std::string ShapeToString(absl::Span<const int64_t> shape) {
  return absl::StrCat("[", absl::StrJoin(shape, ", "), "]");
}

// Allocates the buffer of an array. Always allocates at least one byte (so
// that the data pointer is never null).
void AllocateArray(Array* array) {
  array->data.reset(
      static_cast<char*>(std::malloc(std::max<size_t>(1, array->num_bytes()))));
}

// Number of values in a feature. `feature` is null if the feature is missing.
size_t NumValues(const Feature* feature) {
  if (feature == nullptr) {
    return 0;
  }
  switch (feature->kind_case()) {
    case Feature::KIND_NOT_SET:
      return 0;
    case Feature::kBytesList:
      return feature->bytes_list().value_size();
    case Feature::kFloatList:
      return feature->float_list().value_size();
    case Feature::kInt64List:
      return feature->int64_list().value_size();
  }
  return 0;
}

// Copies (and casts) the values of a repeated field into `output`.
template <typename T, typename Values>
void CopyValues(const Values& values, T* output) {
  using V = typename Values::value_type;
  if constexpr (std::is_same_v<T, V>) {
    if (!values.empty()) {
      std::memcpy(output, values.data(), values.size() * sizeof(T));
    }
  } else if constexpr (std::is_same_v<T, bool>) {
    for (int i = 0; i < values.size(); i++) {
      output[i] = values[i] != 0;
    }
  } else {
    for (int i = 0; i < values.size(); i++) {
      output[i] = static_cast<T>(values[i]);
    }
  }
}

// Decodes the values of a feature into a newly allocated array. The kind of
// `feature` (if not null) should match `format`, and its number of values
// should match `shape`.
void DecodeValues(const Feature* feature, Format format,
                  std::vector<int64_t> shape, Array* output) {
  output->format = format;
  output->shape = std::move(shape);
  if (format == Format::BYTES) {
    size_t itemsize = 1;
    if (feature != nullptr) {
      for (const auto& value : feature->bytes_list().value()) {
        itemsize = std::max(itemsize, value.size());
      }
    }
    output->itemsize = itemsize;
    AllocateArray(output);
    char* data = output->data.get();
    // NumPy pads the fixed-length bytes with zeros.
    std::memset(data, 0, output->num_bytes());
    if (feature != nullptr) {
      for (const auto& value : feature->bytes_list().value()) {
        std::memcpy(data, value.data(), value.size());
        data += itemsize;
      }
    }
    return;
  }

  output->itemsize = ItemSize(format);
  AllocateArray(output);
  if (feature == nullptr) {
    return;
  }
  char* data = output->data.get();
  switch (format) {
    case Format::INTEGER_64:
      CopyValues(feature->int64_list().value(),
                 reinterpret_cast<int64_t*>(data));
      break;
    case Format::INTEGER_32:
      CopyValues(feature->int64_list().value(),
                 reinterpret_cast<int32_t*>(data));
      break;
    case Format::BOOL:
      CopyValues(feature->int64_list().value(), reinterpret_cast<bool*>(data));
      break;
    case Format::FLOAT_32:
      CopyValues(feature->float_list().value(), reinterpret_cast<float*>(data));
      break;
    case Format::FLOAT_64:
      CopyValues(feature->float_list().value(),
                 reinterpret_cast<double*>(data));
      break;
    case Format::BYTES:
      break;
  }
}

// Checks that a feature has the expected kind (or is missing), and counts its
// values.
absl::StatusOr<size_t> CheckKindAndCountValues(std::string_view key,
                                               const Feature* feature,
                                               Feature::KindCase expected) {
  if (feature != nullptr && feature->kind_case() != expected) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The feature \"", key, "\" of the tf.train.Example is a ",
        KindName(feature->kind_case()), " while a ", KindName(expected),
        " is expected. Make sure the graph sample matches the schema."));
  }
  return NumValues(feature);
}

// Gets the number of items of a node or edge set i.e. the first value of the
// "#size" feature.
absl::StatusOr<int64_t> GetNumItems(std::string_view key,
                                    const Feature* feature) {
  DGF_ASSIGN_OR_RETURN(
      const size_t count,
      CheckKindAndCountValues(key, feature, Feature::kInt64List));
  if (count == 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("The feature \"", key,
                     "\" is missing (or empty) in the graph sample."));
  }
  return feature->int64_list().value(0);
}

absl::Status ParseExample(std::string_view serialized, Example* example) {
  if (!example->ParseFromString(serialized)) {
    return absl::InvalidArgumentError(
        "Cannot parse serialized tf.train.Example");
  }
  return absl::OkStatus();
}

}  // namespace

size_t Array::num_items() const {
  size_t n = 1;
  for (const auto dim : shape) {
    n *= dim;
  }
  return n;
}

GraphSampleDecoder::GraphSampleDecoder(
    data::GraphSchema schema, std::vector<std::string> keys,
    absl::flat_hash_map<std::string, KeyIdx> key_to_idx,
    std::vector<CompiledSet> node_sets, std::vector<CompiledSet> edge_sets)
    : schema_(std::move(schema)),
      keys_(std::move(keys)),
      key_to_idx_(std::move(key_to_idx)),
      node_sets_(std::move(node_sets)),
      edge_sets_(std::move(edge_sets)) {}

absl::StatusOr<std::unique_ptr<GraphSampleDecoder>> GraphSampleDecoder::Create(
    data::GraphSchema schema, std::string_view import_node_ids,
    std::string_view import_edge_ids) {
  std::vector<std::string> keys;
  absl::flat_hash_map<std::string, KeyIdx> key_to_idx;
  const auto get_key_idx = [&](const std::string& key) -> KeyIdx {
    const KeyIdx idx = GetOrCreateIndex(key_to_idx, key);
    if (idx == static_cast<KeyIdx>(keys.size())) {
      keys.push_back(key);
    }
    return idx;
  };

  const auto compile_features =
      [&](const data::GraphSchema::FeatureSet& featureset,
          std::string_view import_ids,
          absl::FunctionRef<std::string(std::string_view)> feature_key,
          std::vector<CompiledFeature>* compiled) -> absl::Status {
    for (const auto& feature : featureset.features) {
      CompiledFeature compiled_feature;
      const std::string key = feature_key(feature.name);
      compiled_feature.key_idx = get_key_idx(key);
      if (!import_ids.empty() && feature.name == import_ids) {
        compiled_feature.raw = true;
        compiled_feature.format = StorageFormat(feature.format);
        compiled->push_back(std::move(compiled_feature));
        continue;
      }
      compiled_feature.format = feature.format;
      for (int dim_idx = 0; dim_idx < feature.shape.size(); dim_idx++) {
        const int dim = feature.shape[dim_idx];
        if (dim == -1) {
          // The 0-th dimension is the item dimension.
          compiled_feature.ragged_key_idxs.push_back(
              get_key_idx(RaggedDimKey(key, dim_idx + 1)));
        } else if (dim < 0) {
          return absl::InvalidArgumentError(
              absl::StrCat("Invalid shape for feature ", key));
        } else {
          compiled_feature.static_size *= dim;
        }
        compiled_feature.shape.push_back(dim);
      }
      compiled->push_back(std::move(compiled_feature));
    }
    return absl::OkStatus();
  };

  std::vector<CompiledSet> node_sets;
  for (const auto& node_set : schema.nodesets) {
    CompiledSet compiled;
    compiled.size_key_idx = get_key_idx(NodeKey(node_set.name, kSizeKey));
    DGF_RETURN_IF_ERROR(compile_features(
        node_set.featureset, import_node_ids,
        [&](std::string_view name) { return NodeKey(node_set.name, name); },
        &compiled.features));
    node_sets.push_back(std::move(compiled));
  }

  std::vector<CompiledSet> edge_sets;
  for (const auto& edge_set : schema.edgesets) {
    CompiledSet compiled;
    compiled.size_key_idx = get_key_idx(EdgeKey(edge_set.name, kSizeKey));
    compiled.source_key_idx = get_key_idx(EdgeKey(edge_set.name, kSourceKey));
    compiled.target_key_idx = get_key_idx(EdgeKey(edge_set.name, kTargetKey));
    DGF_RETURN_IF_ERROR(compile_features(
        edge_set.featureset, import_edge_ids,
        [&](std::string_view name) { return EdgeKey(edge_set.name, name); },
        &compiled.features));
    edge_sets.push_back(std::move(compiled));
  }

  return std::unique_ptr<GraphSampleDecoder>(new GraphSampleDecoder(
      std::move(schema), std::move(keys), std::move(key_to_idx),
      std::move(node_sets), std::move(edge_sets)));
}

absl::Status GraphSampleDecoder::DecodeFeature(const CompiledFeature& compiled,
                                               FeaturesByKey features,
                                               const int64_t num_items,
                                               DecodedFeature* output) const {
  const std::string& key = keys_[compiled.key_idx];
  const Feature* values = features[compiled.key_idx];
  DGF_ASSIGN_OR_RETURN(
      const size_t num_values,
      CheckKindAndCountValues(key, values, StorageKind(compiled.format)));

  if (compiled.raw) {
    DecodeValues(values, compiled.format, {static_cast<int64_t>(num_values)},
                 &output->values);
    return absl::OkStatus();
  }

  if (compiled.ragged_key_idxs.empty()) {
    std::vector<int64_t> shape;
    shape.reserve(compiled.shape.size() + 1);
    shape.push_back(num_items);
    shape.insert(shape.end(), compiled.shape.begin(), compiled.shape.end());
    if (num_items < 0 ||
        num_values != static_cast<size_t>(num_items * compiled.static_size)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The feature \"", key, "\" contains ", num_values,
          " values which cannot be reshaped into ", ShapeToString(shape),
          ". Make sure the graph sample matches the schema."));
    }
    DecodeValues(values, compiled.format, std::move(shape), &output->values);
    return absl::OkStatus();
  }

  // Ragged feature: Returns the flat values and the row lengths.
  DecodeValues(values, compiled.format, {static_cast<int64_t>(num_values)},
               &output->values);
  output->row_lengths.resize(compiled.ragged_key_idxs.size());
  for (int i = 0; i < compiled.ragged_key_idxs.size(); i++) {
    const KeyIdx key_idx = compiled.ragged_key_idxs[i];
    DGF_ASSIGN_OR_RETURN(
        const size_t num_row_lengths,
        CheckKindAndCountValues(keys_[key_idx], features[key_idx],
                                Feature::kInt64List));
    DecodeValues(features[key_idx], Format::INTEGER_64,
                 {static_cast<int64_t>(num_row_lengths)},
                 &output->row_lengths[i]);
  }
  return absl::OkStatus();
}

absl::Status GraphSampleDecoder::DecodeFeatures(
    const std::vector<CompiledFeature>& compiled, FeaturesByKey features,
    int64_t num_items, std::vector<DecodedFeature>* output) const {
  output->resize(compiled.size());
  for (int feature_idx = 0; feature_idx < compiled.size(); feature_idx++) {
    DGF_RETURN_IF_ERROR(DecodeFeature(compiled[feature_idx], features,
                                      num_items, &(*output)[feature_idx]));
  }
  return absl::OkStatus();
}

absl::Status GraphSampleDecoder::Decode(std::string_view serialized,
                                        DecodedGraph* graph) const {
  Example example;
  DGF_RETURN_IF_ERROR(ParseExample(serialized, &example));
  return DecodeExample(example, graph);
}

absl::Status GraphSampleDecoder::DecodeExample(const Example& example,
                                               DecodedGraph* graph) const {
  // The features indexed by key index. Missing features are null.
  std::vector<const Feature*> features(keys_.size(), nullptr);
  for (const auto& [key, feature] : example.features().feature()) {
    if (feature.kind_case() == Feature::KIND_NOT_SET) {
      continue;
    }
    if (const auto it = key_to_idx_.find(key); it != key_to_idx_.end()) {
      features[it->second] = &feature;
    }
  }

  graph->node_sets.resize(node_sets_.size());
  for (int set_idx = 0; set_idx < node_sets_.size(); set_idx++) {
    const auto& compiled = node_sets_[set_idx];
    auto& output = graph->node_sets[set_idx];
    DGF_ASSIGN_OR_RETURN(output.num_nodes,
                         GetNumItems(keys_[compiled.size_key_idx],
                                     features[compiled.size_key_idx]));
    DGF_RETURN_IF_ERROR(DecodeFeatures(compiled.features, features,
                                       output.num_nodes, &output.features));
  }

  graph->edge_sets.resize(edge_sets_.size());
  for (int set_idx = 0; set_idx < edge_sets_.size(); set_idx++) {
    const auto& compiled = edge_sets_[set_idx];
    auto& output = graph->edge_sets[set_idx];
    DGF_ASSIGN_OR_RETURN(output.num_edges,
                         GetNumItems(keys_[compiled.size_key_idx],
                                     features[compiled.size_key_idx]));

    const Feature* source = features[compiled.source_key_idx];
    const Feature* target = features[compiled.target_key_idx];
    DGF_ASSIGN_OR_RETURN(const size_t num_sources,
                         CheckKindAndCountValues(keys_[compiled.source_key_idx],
                                                 source, Feature::kInt64List));
    DGF_ASSIGN_OR_RETURN(const size_t num_targets,
                         CheckKindAndCountValues(keys_[compiled.target_key_idx],
                                                 target, Feature::kInt64List));
    if (num_sources != num_targets) {
      return absl::InvalidArgumentError(
          absl::StrCat("The number of sources (", num_sources,
                       ") and targets (", num_targets, ") of the edge set \"",
                       schema_.edgesets[set_idx].name, "\" differ."));
    }
    const size_t n = num_sources;
    Array& adjacency = output.adjacency;
    adjacency.format = Format::INTEGER_64;
    adjacency.shape = {2, static_cast<int64_t>(n)};
    adjacency.itemsize = ItemSize(Format::INTEGER_64);
    AllocateArray(&adjacency);
    int64_t* adjacency_data = reinterpret_cast<int64_t*>(adjacency.data.get());
    if (source != nullptr) {
      CopyValues(source->int64_list().value(), adjacency_data);
    }
    if (target != nullptr) {
      CopyValues(target->int64_list().value(), adjacency_data + n);
    }

    DGF_RETURN_IF_ERROR(DecodeFeatures(compiled.features, features,
                                       output.num_edges, &output.features));
  }
  return absl::OkStatus();
}

absl::Status GraphSampleDecoder::DecodeBatch(
    absl::Span<const std::string_view> serialized,
    absl::Span<DecodedGraph> graphs) const {
  if (serialized.size() != graphs.size()) {
    return absl::InvalidArgumentError("Unexpected number of graphs");
  }
  // Parses the examples in an arena whose memory is reused across samples
  // (~10% faster than heap allocations).
  const auto initial_block =
      std::make_unique_for_overwrite<char[]>(kArenaBlockSize);
  google::protobuf::ArenaOptions options;
  options.initial_block = initial_block.get();
  options.initial_block_size = kArenaBlockSize;
  options.max_block_size = kArenaBlockSize;
  google::protobuf::Arena arena(options);
  for (size_t i = 0; i < serialized.size(); i++) {
    auto* example = google::protobuf::Arena::Create<Example>(&arena);
    absl::Status status = ParseExample(serialized[i], example);
    if (status.ok()) {
      status = DecodeExample(*example, &graphs[i]);
    }
    arena.Reset();
    DGF_RETURN_IF_ERROR(status);
  }
  return absl::OkStatus();
}

}  // namespace dgf::tf_graph_sample

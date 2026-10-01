#include "dgf/src/io/tf_graph_sample_parser.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dgf/src/data/schema.h"
#include "dgf/src/data/tensorflow.pb.h"
#include "dgf/src/util/test_util.h"  // IWYU pragma: keep

namespace dgf::tf_graph_sample {
namespace {

using ::dgf::data::tensorflow::Example;
using Feature = ::dgf::data::GraphSchema::Feature;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

data::tensorflow::Feature& GetFeature(Example* example,
                                      const std::string& key) {
  return (*example->mutable_features()->mutable_feature())[key];
}

void AddInt64s(Example* example, const std::string& key,
               const std::vector<int64_t>& values) {
  GetFeature(example, key)
      .mutable_int64_list()
      ->mutable_value()
      ->Add(values.begin(), values.end());
}

void AddFloats(Example* example, const std::string& key,
               const std::vector<float>& values) {
  GetFeature(example, key)
      .mutable_float_list()
      ->mutable_value()
      ->Add(values.begin(), values.end());
}

void AddBytes(Example* example, const std::string& key,
              const std::vector<std::string>& values) {
  GetFeature(example, key)
      .mutable_bytes_list()
      ->mutable_value()
      ->Add(values.begin(), values.end());
}

template <typename T>
std::vector<T> Values(const Array& array) {
  std::vector<T> result;
  for (size_t i = 0; i < array.num_items(); i++) {
    T value;
    std::memcpy(&value, array.data.get() + i * sizeof(T), sizeof(T));
    result.push_back(value);
  }
  return result;
}

std::vector<std::string> BytesValues(const Array& array) {
  std::vector<std::string> result;
  for (size_t i = 0; i < array.num_items(); i++) {
    result.emplace_back(array.data.get() + i * array.itemsize, array.itemsize);
  }
  return result;
}

data::GraphSchema TestSchema() {
  std::vector<Feature> features = {
      {.name = "f", .shape = {2}, .format = Format::FLOAT_64},
      {.name = "i", .format = Format::INTEGER_32},
      {.name = "b", .format = Format::BOOL},
      {.name = "s", .format = Format::BYTES},
      {.name = "r", .shape = {-1}, .format = Format::INTEGER_64},
      {.name = "#id", .shape = {-1}, .format = Format::FLOAT_64},
  };
  data::GraphSchema::FeatureSet featureset;
  for (int i = 0; i < features.size(); i++) {
    featureset.feature_name_to_idx[features[i].name] = i;
  }
  featureset.features = std::move(features);

  data::GraphSchema schema;
  schema.nodesets.push_back({.name = "n", .featureset = std::move(featureset)});
  schema.edgesets.push_back(
      {.name = "e", .source_nodeset = 0, .target_nodeset = 0});
  return schema;
}

absl::StatusOr<std::unique_ptr<GraphSampleDecoder>> CreateTestDecoder() {
  return GraphSampleDecoder::Create(TestSchema(), /*import_node_ids=*/"#id");
}

Example TestExample() {
  Example example;
  AddInt64s(&example, "nodes/n.#size", {2});
  AddFloats(&example, "nodes/n.f", {1, 2, 3, 4});
  AddInt64s(&example, "nodes/n.i", {5, (int64_t{1} << 32) + 6});
  AddInt64s(&example, "nodes/n.b", {0, 3});
  AddBytes(&example, "nodes/n.s", {"a", "bcd"});
  AddInt64s(&example, "nodes/n.r", {7, 8, 9});
  AddInt64s(&example, "nodes/n.r.d1", {1, 2});
  AddFloats(&example, "nodes/n.#id", {10, 11});
  AddInt64s(&example, "edges/e.#size", {1});
  AddInt64s(&example, "edges/e.#source", {0});
  AddInt64s(&example, "edges/e.#target", {1});
  AddFloats(&example, "unused", {1});
  return example;
}

// Note: Most of the decoding logic is tested against the Python implementation
// in "tf_graph_sample_test.py".
TEST(GraphSampleDecoderTest, Decode) {
  ASSERT_OK_AND_ASSIGN(const auto decoder, CreateTestDecoder());
  DecodedGraph graph;
  ASSERT_OK(decoder->Decode(TestExample().SerializeAsString(), &graph));

  ASSERT_EQ(graph.node_sets.size(), 1);
  const auto& node_set = graph.node_sets[0];
  EXPECT_EQ(node_set.num_nodes, 2);
  ASSERT_EQ(node_set.features.size(), 6);

  const auto& f = node_set.features[0].values;
  EXPECT_THAT(f.shape, ElementsAre(2, 2));
  EXPECT_THAT(Values<double>(f), ElementsAre(1, 2, 3, 4));

  const auto& i = node_set.features[1].values;
  EXPECT_THAT(Values<int32_t>(i), ElementsAre(5, 6));  // Wrapping cast.

  const auto& b = node_set.features[2].values;
  EXPECT_THAT(Values<bool>(b), ElementsAre(false, true));

  const auto& s = node_set.features[3].values;
  EXPECT_EQ(s.itemsize, 3);
  EXPECT_THAT(BytesValues(s),
              ElementsAre(std::string("a\0\0", 3), std::string("bcd")));

  const auto& r = node_set.features[4];
  EXPECT_THAT(Values<int64_t>(r.values), ElementsAre(7, 8, 9));
  ASSERT_EQ(r.row_lengths.size(), 1);
  EXPECT_THAT(Values<int64_t>(r.row_lengths[0]), ElementsAre(1, 2));

  // The imported ids are returned "as is" i.e. with their storage format.
  const auto& id = node_set.features[5].values;
  EXPECT_EQ(id.format, Format::FLOAT_32);
  EXPECT_THAT(Values<float>(id), ElementsAre(10, 11));

  ASSERT_EQ(graph.edge_sets.size(), 1);
  const auto& edge_set = graph.edge_sets[0];
  EXPECT_EQ(edge_set.num_edges, 1);
  EXPECT_THAT(edge_set.adjacency.shape, ElementsAre(2, 1));
  EXPECT_THAT(Values<int64_t>(edge_set.adjacency), ElementsAre(0, 1));
}

TEST(GraphSampleDecoderTest, EmptySets) {
  ASSERT_OK_AND_ASSIGN(const auto decoder, CreateTestDecoder());
  Example example;
  AddInt64s(&example, "nodes/n.#size", {0});
  AddInt64s(&example, "edges/e.#size", {0});
  DecodedGraph graph;
  ASSERT_OK(decoder->Decode(example.SerializeAsString(), &graph));
  EXPECT_THAT(graph.node_sets[0].features[0].values.shape, ElementsAre(0, 2));
  // Empty bytes arrays have an item size of 1 (like NumPy).
  EXPECT_EQ(graph.node_sets[0].features[3].values.itemsize, 1);
  EXPECT_THAT(graph.edge_sets[0].adjacency.shape, ElementsAre(2, 0));
}

TEST(GraphSampleDecoderTest, DecodeBatch) {
  ASSERT_OK_AND_ASSIGN(const auto decoder, CreateTestDecoder());
  const std::string serialized = TestExample().SerializeAsString();
  const std::vector<std::string_view> inputs(10, serialized);
  std::vector<DecodedGraph> graphs(inputs.size());
  ASSERT_OK(decoder->DecodeBatch(inputs, absl::MakeSpan(graphs)));
  for (const auto& graph : graphs) {
    EXPECT_THAT(Values<double>(graph.node_sets[0].features[0].values),
                ElementsAre(1, 2, 3, 4));
  }
}

TEST(GraphSampleDecoderTest, InvalidExample) {
  ASSERT_OK_AND_ASSIGN(const auto decoder, CreateTestDecoder());
  const struct {
    std::function<void(Example*)> mutate;
    std::string_view error;
  } kCases[] = {
      {[](Example* e) { AddInt64s(e, "nodes/n.f", {1, 2, 3, 4}); },
       "float_list is expected"},
      {[](Example* e) { AddFloats(e, "nodes/n.f", {1, 2, 3}); },
       "cannot be reshaped"},
      {[](Example* e) {
         e->mutable_features()->mutable_feature()->erase("nodes/n.#size");
       },
       "missing"},
      {[](Example* e) { AddInt64s(e, "edges/e.#target", {1, 2}); }, "differ"},
  };
  for (const auto& [mutate, error] : kCases) {
    Example example = TestExample();
    mutate(&example);
    DecodedGraph graph;
    EXPECT_THAT(decoder->Decode(example.SerializeAsString(), &graph).message(),
                HasSubstr(error));
  }
}

TEST(GraphSampleDecoderTest, InvalidSchema) {
  data::GraphSchema schema = TestSchema();
  schema.nodesets[0].featureset.features[0].shape = {-2};
  EXPECT_FALSE(GraphSampleDecoder::Create(schema).ok());
}

}  // namespace
}  // namespace dgf::tf_graph_sample

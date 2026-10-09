#include "dgf/src/sampling/in_memory_sampler.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "dgf/src/util/test_util.h"

namespace dgf::sampling::in_memory_sampler {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::IsSubsetOf;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

AdjacencyIndex CreateTestIndex() {
  return {{0, 3, 3, 5}, {10, 11, 12, 20, 21}};
}

TEST(InMemorySamplerTest, Targets) {
  AdjacencyIndex index = CreateTestIndex();
  EXPECT_THAT(index.Targets(0), ElementsAre(10, 11, 12));
  EXPECT_THAT(index.Targets(1), ElementsAre());
  EXPECT_THAT(index.Targets(2), ElementsAre(20, 21));
}

TEST(InMemorySamplerTest, SampleFirst_LessThanAvailable) {
  AdjacencyIndex index = CreateTestIndex();
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleFirst(/*source_node=*/0, /*num_samples=*/2, &result));
  EXPECT_THAT(result, ElementsAre(10, 11));
}

TEST(InMemorySamplerTest, SampleFirst_MoreThanAvailable) {
  AdjacencyIndex index = CreateTestIndex();
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleFirst(/*source_node=*/0, /*num_samples=*/5, &result));
  EXPECT_THAT(result, ElementsAre(10, 11, 12));
}

TEST(InMemorySamplerTest, SampleFirst_NoNeighbors) {
  AdjacencyIndex index = CreateTestIndex();
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleFirst(/*source_node=*/1, /*num_samples=*/1, &result));
  EXPECT_THAT(result, IsEmpty());
}

TEST(InMemorySamplerTest, SampleRandomUniform_LessThanAvailable) {
  AdjacencyIndex index = CreateTestIndex();
  Rng rng = MakeRng(42);
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleRandomUniform(/*source_node=*/0, /*num_samples=*/2,
                                      &result, &rng));
  EXPECT_THAT(result, SizeIs(2));
  EXPECT_THAT(result, IsSubsetOf({10, 11, 12}));
}

TEST(InMemorySamplerTest, SampleRandomUniform_MoreThanAvailable) {
  AdjacencyIndex index = CreateTestIndex();
  Rng rng = MakeRng(42);
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleRandomUniform(/*source_node=*/0, /*num_samples=*/5,
                                      &result, &rng));
  EXPECT_THAT(result, UnorderedElementsAre(10, 11, 12));
}

TEST(InMemorySamplerTest, SampleRandomUniform_NoNeighbors) {
  AdjacencyIndex index = CreateTestIndex();
  Rng rng = MakeRng(42);
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleRandomUniform(/*source_node=*/1, /*num_samples=*/1,
                                      &result, &rng));
  EXPECT_THAT(result, IsEmpty());
}

TEST(InMemorySamplerTest, IndexStringValues) {
  // Test with unique values.
  EXPECT_THAT(IndexStringValues({"a", "b", "c"}),
              UnorderedElementsAre(Pair("a", 0), Pair("b", 1), Pair("c", 2)));

  // Test with empty input.
  EXPECT_THAT(IndexStringValues({}), IsEmpty());
}

TEST(InMemorySamplerTest, CreateFromEdgeList_BasicCase) {
  std::vector<Edge<false, false>> edges = {{.source = 0, .target = 0},
                                           {.source = 0, .target = 1},
                                           {.source = 2, .target = 3}};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<false, false>(
                           std::move(edges), 3, 4)));
  EXPECT_THAT(index.source_blocks, ElementsAre(0, 2, 2, 3));
  EXPECT_THAT(index.target_node_idxs, ElementsAre(0, 1, 3));
}

TEST(InMemorySamplerTest, CreateFromEdgeList_EmptyEdges) {
  std::vector<Edge<false, false>> edges = {};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<false, false>(
                           std::move(edges), 2, 2)));
  EXPECT_THAT(index.source_blocks, ElementsAre(0, 0, 0));
  EXPECT_THAT(index.target_node_idxs, IsEmpty());
}

TEST(InMemorySamplerTest, CreateFromEdgeList_SourceWithNoOutgoingEdges) {
  std::vector<Edge<false, false>> edges = {{.source = 0, .target = 1},
                                           {.source = 2, .target = 3}};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<false, false>(
                           std::move(edges), 4, 4)));
  EXPECT_THAT(index.source_blocks, ElementsAre(0, 1, 1, 2, 2));
  EXPECT_THAT(index.target_node_idxs, ElementsAre(1, 3));
}

TEST(InMemorySamplerTest, CreateFromEdgeList_DuplicateEdgesKept) {
  std::vector<Edge<false, false>> edges = {{.source = 0, .target = 1},
                                           {.source = 0, .target = 1},
                                           {.source = 0, .target = 2}};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<false, false>(
                           std::move(edges), 1, 3)));
  EXPECT_THAT(index.source_blocks, ElementsAre(0, 3));
  EXPECT_THAT(index.target_node_idxs, ElementsAre(1, 1, 2));
}

TEST(InMemorySamplerTest, CreateFromEdgeList_EdgesOutOfOrder) {
  std::vector<Edge<false, false>> edges = {{.source = 2, .target = 3},
                                           {.source = 0, .target = 1},
                                           {.source = 0, .target = 0}};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<false, false>(
                           std::move(edges), 3, 4)));
  EXPECT_THAT(index.source_blocks, ElementsAre(0, 2, 2, 3));
  EXPECT_THAT(index.target_node_idxs, ElementsAre(0, 1, 3));
}

TEST(SamplingPlanTest, ComputeStepIdxAndStepIdxToNode) {
  SamplingPlan plan;
  plan.root = std::make_unique<SamplingPlan::Node>();
  auto child1 = std::make_unique<SamplingPlan::Node>();
  plan.root->children.push_back(SamplingPlan::Edge{.node = std::move(child1)});
  auto child2 = std::make_unique<SamplingPlan::Node>();
  plan.root->children.push_back(SamplingPlan::Edge{.node = std::move(child2)});
  plan.ComputeStepIdx();

  // Test valid indices.
  ASSERT_OK_AND_ASSIGN(const SamplingPlan::Node& node0, plan.StepIdxToNode(0));
  EXPECT_EQ(node0.step_idx, 0);
  ASSERT_OK_AND_ASSIGN(const SamplingPlan::Node& node1, plan.StepIdxToNode(1));
  EXPECT_EQ(node1.step_idx, 1);
  ASSERT_OK_AND_ASSIGN(const SamplingPlan::Node& node2, plan.StepIdxToNode(2));
  EXPECT_EQ(node2.step_idx, 2);
}

TEST(InMemorySamplerTest, CreateFromEdgeList_Temporal) {
  std::vector<Edge<true, false>> edges = {
      {.source = 0, .target = 1, .timestamp = 15},
      {.source = 0, .target = 2, .timestamp = 25},
      {.source = 1, .target = 3, .timestamp = 35}};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<true, false>(
                           std::move(edges), 4, 4)));
  EXPECT_THAT(index.source_blocks, ElementsAre(0, 2, 3, 3, 3));
  EXPECT_THAT(index.target_node_idxs, ElementsAre(1, 2, 3));
  EXPECT_THAT(index.timestamps, ElementsAre(15, 25, 35));
}

TEST(InMemorySamplerTest, SampleWithTimestamp_FiltersFutureEdges) {
  std::vector<Edge<true, false>> edges = {
      {.source = 0, .target = 1, .timestamp = 15},
      {.source = 0, .target = 2, .timestamp = 25},
      {.source = 1, .target = 3, .timestamp = 35}};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<true, false>(
                           std::move(edges), 4, 4)));
  Rng rng = MakeRng(42);
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleRandomUniformWithTimestamp(
      /*source_node=*/0, /*seed_timestamp=*/20, /*num_samples=*/2, &result,
      &rng));
  EXPECT_THAT(result, ElementsAre(1));
}

// New tests for edge masking

TEST(InMemorySamplerTest, SampleFirst_WithMasking) {
  std::vector<Edge<false, true>> edges = {
      {.source = 0, .target = 1, .edge_idx = 0},
      {.source = 0, .target = 2, .edge_idx = 1},
      {.source = 0, .target = 3, .edge_idx = 2}};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<false, true>(
                           std::move(edges), 1, 4)));
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleFirst(/*source_node=*/0, /*num_samples=*/2, &result,
                              /*masked_edge_idx=*/1));
  EXPECT_THAT(result, ElementsAre(1, 3));  // Skips edge 1 (target 2)
}

TEST(InMemorySamplerTest, SampleRandomUniform_WithMasking) {
  std::vector<Edge<false, true>> edges = {
      {.source = 0, .target = 1, .edge_idx = 0},
      {.source = 0, .target = 2, .edge_idx = 1},
      {.source = 0, .target = 3, .edge_idx = 2}};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<false, true>(
                           std::move(edges), 1, 4)));
  Rng rng = MakeRng(42);
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleRandomUniform(/*source_node=*/0, /*num_samples=*/2,
                                      &result, &rng, /*masked_edge_idx=*/1));
  EXPECT_THAT(result, SizeIs(2));
  EXPECT_THAT(result, IsSubsetOf({1, 3}));  // Skips edge 1 (target 2)
}

TEST(InMemorySamplerTest, SampleRandomUniform_AllMasked) {
  std::vector<Edge<false, true>> edges = {
      {.source = 0, .target = 1, .edge_idx = 5},
      {.source = 0, .target = 2, .edge_idx = 5},
      {.source = 0, .target = 3, .edge_idx = 5}};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<false, true>(
                           std::move(edges), 1, 4)));
  Rng rng = MakeRng(42);
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleRandomUniform(/*source_node=*/0, /*num_samples=*/2,
                                      &result, &rng, /*masked_edge_idx=*/5));
  EXPECT_THAT(result, IsEmpty());
}

TEST(InMemorySamplerTest, SampleRandomUniform_NoneMasked) {
  std::vector<Edge<false, true>> edges = {
      {.source = 0, .target = 1, .edge_idx = 0},
      {.source = 0, .target = 2, .edge_idx = 1},
      {.source = 0, .target = 3, .edge_idx = 2}};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<false, true>(
                           std::move(edges), 1, 4)));
  Rng rng = MakeRng(42);
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleRandomUniform(/*source_node=*/0, /*num_samples=*/2,
                                      &result, &rng, /*masked_edge_idx=*/5));
  EXPECT_THAT(result, SizeIs(2));
  EXPECT_THAT(result, IsSubsetOf({1, 2, 3}));
}

TEST(InMemorySamplerTest, SampleRandomUniform_NumSamplesLargerThanAvailable) {
  std::vector<Edge<false, true>> edges = {
      {.source = 0, .target = 1, .edge_idx = 0},
      {.source = 0, .target = 2, .edge_idx = 1},
      {.source = 0, .target = 3, .edge_idx = 2}};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<false, true>(
                           std::move(edges), 1, 4)));
  Rng rng = MakeRng(42);
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleRandomUniform(/*source_node=*/0, /*num_samples=*/5,
                                      &result, &rng, /*masked_edge_idx=*/1));
  EXPECT_THAT(result, UnorderedElementsAre(1, 3));  // Skips edge 1 (target 2)
}

TEST(InMemorySamplerTest, SampleRandomUniform_MoreCandidates) {
  std::vector<Edge<false, true>> edges = {
      {.source = 0, .target = 1, .edge_idx = 0},
      {.source = 0, .target = 2, .edge_idx = 1},
      {.source = 0, .target = 3, .edge_idx = 2},
      {.source = 0, .target = 4, .edge_idx = 3},
      {.source = 0, .target = 5, .edge_idx = 4}};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<false, true>(
                           std::move(edges), 1, 6)));
  Rng rng = MakeRng(42);
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleRandomUniform(/*source_node=*/0, /*num_samples=*/2,
                                      &result, &rng,
                                      /*masked_edge_idx=*/2));  // Mask target 3
  EXPECT_THAT(result, SizeIs(2));
  EXPECT_THAT(result, IsSubsetOf({1, 2, 4, 5}));
}

TEST(InMemorySamplerTest, HasEdge) {
  AdjacencyIndex index = CreateTestIndex();
  EXPECT_TRUE(index.HasEdge(0, 10));
  EXPECT_TRUE(index.HasEdge(0, 11));
  EXPECT_TRUE(index.HasEdge(0, 12));
  EXPECT_FALSE(index.HasEdge(0, 13));
  EXPECT_FALSE(index.HasEdge(0, 20));

  EXPECT_FALSE(index.HasEdge(1, 10));

  EXPECT_TRUE(index.HasEdge(2, 20));
  EXPECT_TRUE(index.HasEdge(2, 21));
  EXPECT_FALSE(index.HasEdge(2, 10));
}

TEST(MergeTest, ComputeMergeLayout) {
  // 2 nodesets, 1 edgeset, 3 samples. The first nodeset is padded.
  const MergeLayout layout = ComputeMergeLayout(
      /*sample_num_nodes=*/{{1, 2, 3}, {0, 4, 1}},
      /*sample_num_edges=*/{{2, 0, 5}},
      MergePadding{/*num_nodes=*/{10, std::nullopt},
                   /*num_edges=*/{std::nullopt}});
  EXPECT_THAT(layout.node_offsets,
              ElementsAre(ElementsAre(0, 1, 3, 6), ElementsAre(0, 0, 4, 5)));
  EXPECT_THAT(layout.edge_offsets, ElementsAre(ElementsAre(0, 2, 2, 7)));
  EXPECT_THAT(layout.num_nodes, ElementsAre(10, 5));
  EXPECT_THAT(layout.num_edges, ElementsAre(7));
}

TEST(MergeTest, ComputeMergeLayout_NoSamples) {
  const MergeLayout layout = ComputeMergeLayout(
      {{}}, {{}},
      MergePadding{/*num_nodes=*/{4}, /*num_edges=*/{std::nullopt}});
  EXPECT_THAT(layout.node_offsets, ElementsAre(ElementsAre(0)));
  EXPECT_THAT(layout.num_nodes, ElementsAre(4));
  EXPECT_THAT(layout.num_edges, ElementsAre(0));
}

TEST(MergeTest, FindPaddingOverflow) {
  const auto overflow = [](std::optional<std::size_t> padded_nodes,
                           std::optional<std::size_t> padded_edges) {
    const MergePadding padding{{padded_nodes}, {padded_edges}};
    return FindPaddingOverflow(ComputeMergeLayout({{2, 3}}, {{4}}, padding),
                               padding);
  };
  // 5 nodes + 1 sentinel node, and 4 edges.
  EXPECT_FALSE(overflow(6, 4).has_value());
  EXPECT_FALSE(overflow(std::nullopt, std::nullopt).has_value());

  const auto node_overflow = overflow(5, 4);
  ASSERT_TRUE(node_overflow.has_value());
  EXPECT_TRUE(node_overflow->is_nodeset);
  EXPECT_EQ(node_overflow->set_idx, 0);
  EXPECT_EQ(node_overflow->required, 6);
  EXPECT_EQ(node_overflow->padded, 5);

  const auto edge_overflow = overflow(6, 3);
  ASSERT_TRUE(edge_overflow.has_value());
  EXPECT_FALSE(edge_overflow->is_nodeset);
  EXPECT_EQ(edge_overflow->required, 4);
  EXPECT_EQ(edge_overflow->padded, 3);
}

// Gathers the rows `idxs` of `src` (with `row_size` values per row) into a
// `num_dst_rows` rows output initialized with -1.
template <typename T>
std::vector<T> TestGather(const std::vector<T>& src, std::size_t row_size,
                          const std::vector<InputIdx>& idxs,
                          std::size_t num_dst_rows, bool* ok) {
  std::vector<T> dst(num_dst_rows * row_size, -1);
  std::vector<GatherRowsTask> tasks;
  AppendGatherRowsTasks(reinterpret_cast<const char*>(src.data()),
                        src.size() / row_size, idxs.data(), idxs.size(),
                        reinterpret_cast<char*>(dst.data()), num_dst_rows,
                        row_size * sizeof(T), /*rows_per_task=*/2, &tasks);
  *ok = true;
  for (const auto& task : tasks) {
    *ok &= GatherRows(task);
  }
  return dst;
}

TEST(MergeTest, GatherRows) {
  bool ok;
  // Typed copies (1, 2, 4, and 8 bytes rows).
  EXPECT_THAT(TestGather<int8_t>({10, 11, 12}, 1, {2, 0, 2}, 4, &ok),
              ElementsAre(12, 10, 12, 0));
  EXPECT_TRUE(ok);
  EXPECT_THAT(TestGather<int16_t>({10, 11, 12}, 1, {1}, 1, &ok),
              ElementsAre(11));
  EXPECT_TRUE(ok);
  EXPECT_THAT(TestGather<float>({1.f, 2.f}, 1, {1, 1, 0}, 3, &ok),
              ElementsAre(2.f, 2.f, 1.f));
  EXPECT_TRUE(ok);
  EXPECT_THAT(TestGather<double>({1., 2.}, 1, {}, 2, &ok), ElementsAre(0., 0.));
  EXPECT_TRUE(ok);
  // Generic copies (rows of 3 x 4 bytes).
  EXPECT_THAT(TestGather<int32_t>({1, 2, 3, 4, 5, 6}, 3, {1, 0}, 3, &ok),
              ElementsAre(4, 5, 6, 1, 2, 3, 0, 0, 0));
  EXPECT_TRUE(ok);
}

TEST(MergeTest, GatherRows_OutOfBounds) {
  bool ok;
  TestGather<int64_t>({1, 2}, 1, {0, 2}, 2, &ok);
  EXPECT_FALSE(ok);
  TestGather<int64_t>({1, 2}, 1, {std::numeric_limits<InputIdx>::max()}, 1,
                      &ok);
  EXPECT_FALSE(ok);
  TestGather<int32_t>({1, 2, 3, 4, 5, 6}, 3, {2}, 1, &ok);
  EXPECT_FALSE(ok);
}

TEST(MergeTest, AppendGatherRowsTasks) {
  std::vector<GatherRowsTask> tasks;
  const char src[1] = {};
  const InputIdx idxs[5] = {};
  char dst[1] = {};
  AppendGatherRowsTasks(src, /*src_num_rows=*/1, idxs, /*num_idxs=*/5, dst,
                        /*num_dst_rows=*/8, /*row_bytes=*/4,
                        /*rows_per_task=*/2, &tasks);
  ASSERT_THAT(tasks, SizeIs(5));
  const auto range = [](const GatherRowsTask& task) {
    return std::make_tuple(task.begin, task.end, task.idxs != nullptr);
  };
  EXPECT_EQ(range(tasks[0]), std::make_tuple(0, 2, true));
  EXPECT_EQ(range(tasks[1]), std::make_tuple(2, 4, true));
  EXPECT_EQ(range(tasks[2]), std::make_tuple(4, 5, true));
  EXPECT_EQ(range(tasks[3]), std::make_tuple(5, 7, false));
  EXPECT_EQ(range(tasks[4]), std::make_tuple(7, 8, false));
}

}  // namespace
}  // namespace dgf::sampling::in_memory_sampler

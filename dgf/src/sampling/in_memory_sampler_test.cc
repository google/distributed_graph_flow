#include "dgf/src/sampling/in_memory_sampler.h"

#include <algorithm>
#include <cmath>
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
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "dgf/src/util/test_util.h"
#include "dgf/src/util/util.h"

namespace dgf::sampling::in_memory_sampler {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::IsSubsetOf;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

// Number of nodes of the graph of `CreateTestEdges`.
constexpr std::size_t kNumTestNodes = 30;

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

// Creates an index where the single source node 0 is connected to the targets
// [0, num_targets).
AdjacencyIndex CreateHighDegreeIndex(std::size_t num_targets) {
  AdjacencyIndex index;
  index.source_blocks = {0, num_targets};
  for (std::size_t i = 0; i < num_targets; i++) {
    index.target_node_idxs.push_back(i);
  }
  return index;
}

// Covers the different sampling algorithms used depending on the ratio
// between the number of samples and the number of neighbors.
TEST(InMemorySamplerTest, SampleRandomUniform_HighDegree) {
  constexpr std::size_t kNumTargets = 100;
  const AdjacencyIndex index = CreateHighDegreeIndex(kNumTargets);
  for (const std::size_t num_samples : {1, 5, 15, 16, 20, 49, 50, 70, 99}) {
    SCOPED_TRACE(num_samples);
    Rng rng = MakeRng(42);
    std::vector<int> counts(kNumTargets, 0);
    constexpr int kNumRuns = 2000;
    std::vector<std::size_t> result;
    for (int run = 0; run < kNumRuns; run++) {
      EXPECT_OK(index.SampleRandomUniform(/*source_node=*/0, num_samples,
                                          &result, &rng));
      ASSERT_THAT(result, SizeIs(num_samples));
      const absl::flat_hash_set<std::size_t> unique(result.begin(),
                                                    result.end());
      ASSERT_THAT(unique, SizeIs(num_samples));
      for (const auto target : result) {
        ASSERT_LT(target, kNumTargets);
        counts[target]++;
      }
    }
    // Each target is selected with probability num_samples / kNumTargets.
    const double expected =
        static_cast<double>(kNumRuns) * num_samples / kNumTargets;
    for (const int count : counts) {
      EXPECT_NEAR(count, expected, 5 * std::sqrt(expected) + 1);
    }
  }
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

// With timestamps, the targets are sorted by timestamp first.
TEST(InMemorySamplerTest, HasEdge_Temporal) {
  std::vector<Edge<true, false>> edges = {
      {.source = 0, .target = 1, .timestamp = 30},
      {.source = 0, .target = 2, .timestamp = 20},
      {.source = 0, .target = 3, .timestamp = 10}};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<true, false>(
                           std::move(edges), 1, 4)));
  EXPECT_THAT(index.Targets(0), ElementsAre(3, 2, 1));
  EXPECT_FALSE(index.HasEdge(0, 0));
  EXPECT_TRUE(index.HasEdge(0, 1));
  EXPECT_TRUE(index.HasEdge(0, 2));
  EXPECT_TRUE(index.HasEdge(0, 3));
}

// The sampling without seed timestamp (e.g., sub-graph extraction) ignores the
// timestamps.
TEST(InMemorySamplerTest, SampleWithoutTimestamp_Temporal) {
  std::vector<Edge<true, false>> edges = {
      {.source = 0, .target = 1, .timestamp = 30},
      {.source = 0, .target = 2, .timestamp = 20}};
  ASSERT_OK_AND_ASSIGN(AdjacencyIndex index,
                       (AdjacencyIndex::CreateFromEdgeList<true, false>(
                           std::move(edges), 1, 3)));
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleFirst(/*source_node=*/0, /*num_samples=*/5, &result));
  EXPECT_THAT(result, ElementsAre(2, 1));
  Rng rng = MakeRng(42);
  EXPECT_OK(index.SampleRandomUniform(/*source_node=*/0, /*num_samples=*/5,
                                      &result, &rng));
  EXPECT_THAT(result, UnorderedElementsAre(1, 2));
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

using NodePair = std::pair<InputIdx, InputIdx>;

// Creates a plan node expanding the edgeset 0 in both directions for
// `num_hops` hops.
std::unique_ptr<SamplingPlan::Node> CreatePlanNode(const int num_hops,
                                                   const int hop_width) {
  auto node = std::make_unique<SamplingPlan::Node>();
  node->nodeset_idx = 0;
  if (num_hops > 0) {
    for (const bool reversed : {false, true}) {
      node->children.push_back({.edgeset_idx = 0,
                                .reversed = reversed,
                                .node = CreatePlanNode(num_hops - 1, hop_width),
                                .hop_width = hop_width});
    }
  }
  return node;
}

// Creates a homogeneous graph with the given directed edges, sampled with a
// `num_hops` plan following the edges in both directions.
SamplerIndex CreateSamplerIndex(const std::size_t num_nodes,
                                const std::vector<NodePair>& edges,
                                const int num_hops, const int hop_width,
                                const bool multi_visit) {
  std::vector<Edge<false, false>> forward;
  std::vector<Edge<false, false>> backward;
  for (const auto& [source, target] : edges) {
    forward.push_back({.source = source, .target = target});
    backward.push_back({.source = target, .target = source});
  }
  SamplerIndex index;
  index.nodesets_.push_back({.name = "n", .num_nodes = num_nodes});
  SamplerIndex::EdgeSet edgeset{.name = "e",
                                .source_nodeset = 0,
                                .target_nodeset = 0,
                                .need_forward = true,
                                .need_backward = true};
  edgeset.forward_index = AdjacencyIndex::CreateFromEdgeList<false, false>(
                              std::move(forward), num_nodes, num_nodes)
                              .value();
  edgeset.backward_index = AdjacencyIndex::CreateFromEdgeList<false, false>(
                               std::move(backward), num_nodes, num_nodes)
                               .value();
  index.edgesets_.push_back(std::move(edgeset));
  index.plan_.root = CreatePlanNode(num_hops, hop_width);
  index.plan_.with_replacement = false;
  index.plan_.multi_visit = multi_visit;
  index.plan_.ComputeStepIdx();
  index.ComputeStepSlots();
  return index;
}

// Returns the edges of a small graph. If `with_hub`, node 0 is connected to all
// the other nodes. The max degree is less than 2 * kNumTestNodes.
std::vector<NodePair> CreateTestEdges(const bool with_hub) {
  std::vector<NodePair> edges;
  for (InputIdx i = 0; i < kNumTestNodes; i++) {
    edges.push_back({i, (i * 7 + 3) % kNumTestNodes});
    edges.push_back({i, (i * 11 + 5) % kNumTestNodes});
    if (with_hub && i > 0) {
      edges.push_back({0, i});
    }
  }
  return edges;
}

// Grows a sample from `seeds`.
absl::StatusOr<SampleBuilder> GrowSample(const SamplerIndex& index,
                                         const std::vector<InputIdx>& seeds,
                                         const GrowMode mode,
                                         const uint64_t rng_seed) {
  SampleBuilder builder;
  builder.rng = MakeRng(rng_seed);
  DGF_RETURN_IF_ERROR(builder.Grow(index, seeds, std::nullopt,
                                   /*masked_edge_idx=*/-1, mode));
  return builder;
}

// Returns the sorted original node idxs of a sample.
std::vector<InputIdx> OriginalNodes(const SampleBuilder& builder) {
  std::vector<InputIdx> nodes =
      builder.nodesets[0].sampled_node_idx_to_node_idx;
  std::sort(nodes.begin(), nodes.end());
  return nodes;
}

// Returns the sorted edges of a sample, as pairs of original node idxs.
std::vector<NodePair> OriginalEdges(const SampleBuilder& builder) {
  const auto& node_idxs = builder.nodesets[0].sampled_node_idx_to_node_idx;
  std::vector<NodePair> edges;
  for (const auto& [source, target] : builder.edgesets[0].edges) {
    edges.push_back({node_idxs[source], node_idxs[target]});
  }
  std::sort(edges.begin(), edges.end());
  return edges;
}

TEST(SampleBuilderTest, Grow_SeedNodesFirst) {
  const SamplerIndex index =
      CreateSamplerIndex(kNumTestNodes, CreateTestEdges(/*with_hub=*/true),
                         /*num_hops=*/2, /*hop_width=*/3,
                         /*multi_visit=*/true);
  ASSERT_OK_AND_ASSIGN(const SampleBuilder builder,
                       GrowSample(index, {5, 2}, GrowMode::kSubGraph, 1));
  const auto& node_idxs = builder.nodesets[0].sampled_node_idx_to_node_idx;
  ASSERT_GE(node_idxs.size(), 2);
  EXPECT_EQ(node_idxs[0], 5);
  EXPECT_EQ(node_idxs[1], 2);
}

TEST(SampleBuilderTest, Grow_OutOfBoundsSeed) {
  const SamplerIndex index =
      CreateSamplerIndex(kNumTestNodes, CreateTestEdges(/*with_hub=*/true),
                         /*num_hops=*/1, /*hop_width=*/3,
                         /*multi_visit=*/true);
  EXPECT_EQ(GrowSample(index, {kNumTestNodes}, GrowMode::kSubGraph, 1)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(SampleBuilderTest, Grow_NoDuplicateEdgesWithoutReplacement) {
  const std::vector<NodePair> graph_edges = CreateTestEdges(/*with_hub=*/true);
  const SamplerIndex index =
      CreateSamplerIndex(kNumTestNodes, graph_edges, /*num_hops=*/3,
                         /*hop_width=*/3, /*multi_visit=*/false);
  for (const GrowMode mode : {GrowMode::kPlan, GrowMode::kSubGraph}) {
    for (uint64_t rng_seed = 0; rng_seed < 20; rng_seed++) {
      ASSERT_OK_AND_ASSIGN(
          const SampleBuilder builder,
          GrowSample(index, {rng_seed % kNumTestNodes}, mode, rng_seed));
      const std::vector<NodePair> edges = OriginalEdges(builder);
      EXPECT_EQ(std::adjacent_find(edges.begin(), edges.end()), edges.end());
      EXPECT_THAT(edges, IsSubsetOf(graph_edges));
    }
  }
}

TEST(SampleBuilderTest, Grow_SubGraphEqualsPlanWithoutMultiVisit) {
  const std::vector<NodePair> graph_edges = CreateTestEdges(/*with_hub=*/true);
  const SamplerIndex index_no_multi_visit =
      CreateSamplerIndex(kNumTestNodes, graph_edges, /*num_hops=*/3,
                         /*hop_width=*/3, /*multi_visit=*/false);
  // `kSubGraph` ignores the `multi_visit` option of the plan.
  const SamplerIndex index_multi_visit =
      CreateSamplerIndex(kNumTestNodes, graph_edges, /*num_hops=*/3,
                         /*hop_width=*/3, /*multi_visit=*/true);
  for (uint64_t rng_seed = 0; rng_seed < 20; rng_seed++) {
    const std::vector<InputIdx> seeds = {rng_seed % kNumTestNodes};
    ASSERT_OK_AND_ASSIGN(
        const SampleBuilder plan,
        GrowSample(index_no_multi_visit, seeds, GrowMode::kPlan, rng_seed));
    ASSERT_OK_AND_ASSIGN(
        const SampleBuilder subgraph,
        GrowSample(index_multi_visit, seeds, GrowMode::kSubGraph, rng_seed));
    EXPECT_EQ(plan.nodesets[0].sampled_node_idx_to_node_idx,
              subgraph.nodesets[0].sampled_node_idx_to_node_idx);
    EXPECT_EQ(plan.edgesets[0].edges, subgraph.edgesets[0].edges);
  }
}

// With a hop width larger than the max degree, there is no randomness: The
// result does not depend on the random seed nor on the order of the seeds, and
// the sub-graph of several seeds is the union of their sub-graphs.
TEST(SampleBuilderTest, Grow_SubGraphWithLargeHopWidth) {
  const SamplerIndex index = CreateSamplerIndex(
      kNumTestNodes, CreateTestEdges(/*with_hub=*/false), /*num_hops=*/2,
      /*hop_width=*/2 * kNumTestNodes, /*multi_visit=*/true);
  ASSERT_OK_AND_ASSIGN(const SampleBuilder a,
                       GrowSample(index, {1}, GrowMode::kSubGraph, 1));
  ASSERT_OK_AND_ASSIGN(const SampleBuilder a_other_rng,
                       GrowSample(index, {1}, GrowMode::kSubGraph, 2));
  ASSERT_OK_AND_ASSIGN(const SampleBuilder b,
                       GrowSample(index, {17}, GrowMode::kSubGraph, 1));
  ASSERT_OK_AND_ASSIGN(const SampleBuilder ab,
                       GrowSample(index, {1, 17}, GrowMode::kSubGraph, 1));
  ASSERT_OK_AND_ASSIGN(const SampleBuilder ba,
                       GrowSample(index, {17, 1}, GrowMode::kSubGraph, 2));

  EXPECT_EQ(OriginalNodes(a), OriginalNodes(a_other_rng));
  EXPECT_EQ(OriginalEdges(a), OriginalEdges(a_other_rng));
  EXPECT_EQ(OriginalNodes(ab), OriginalNodes(ba));
  EXPECT_EQ(OriginalEdges(ab), OriginalEdges(ba));

  std::vector<InputIdx> union_nodes = OriginalNodes(a);
  const std::vector<InputIdx> b_nodes = OriginalNodes(b);
  union_nodes.insert(union_nodes.end(), b_nodes.begin(), b_nodes.end());
  std::sort(union_nodes.begin(), union_nodes.end());
  union_nodes.erase(std::unique(union_nodes.begin(), union_nodes.end()),
                    union_nodes.end());
  EXPECT_EQ(OriginalNodes(ab), union_nodes);
  EXPECT_LT(union_nodes.size(), kNumTestNodes);
}

}  // namespace
}  // namespace dgf::sampling::in_memory_sampler

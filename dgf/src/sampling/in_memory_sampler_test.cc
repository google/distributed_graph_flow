#include "dgf/src/sampling/in_memory_sampler.h"

#include <cstddef>
#include <memory>
#include <random>
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
  std::mt19937_64 rng(42);
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleRandomUniform(/*source_node=*/0, /*num_samples=*/2,
                                      &result, &rng));
  EXPECT_THAT(result, SizeIs(2));
  EXPECT_THAT(result, IsSubsetOf({10, 11, 12}));
}

TEST(InMemorySamplerTest, SampleRandomUniform_MoreThanAvailable) {
  AdjacencyIndex index = CreateTestIndex();
  std::mt19937_64 rng(42);
  std::vector<std::size_t> result;
  EXPECT_OK(index.SampleRandomUniform(/*source_node=*/0, /*num_samples=*/5,
                                      &result, &rng));
  EXPECT_THAT(result, UnorderedElementsAre(10, 11, 12));
}

TEST(InMemorySamplerTest, SampleRandomUniform_NoNeighbors) {
  AdjacencyIndex index = CreateTestIndex();
  std::mt19937_64 rng(42);
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
  std::mt19937_64 rng(42);
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
  std::mt19937_64 rng(42);
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
  std::mt19937_64 rng(42);
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
  std::mt19937_64 rng(42);
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
  std::mt19937_64 rng(42);
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
  std::mt19937_64 rng(42);
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
}  // namespace
}  // namespace dgf::sampling::in_memory_sampler

#include "dgf/src/util/concurrency.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "dgf/src/util/test_util.h"  // IWYU pragma: keep

namespace dgf::util::concurrency {
namespace {

TEST(StatusThreadPool, Basic) {
  std::atomic<int> counter{0};
  const int n = 100;
  {
    StatusThreadPool pool(5);
    for (int i = 1; i <= n; i++) {
      pool.Schedule([&, i]() {
        counter += i;
        return absl::OkStatus();
      });
    }
    EXPECT_OK(pool.Join());
  }
  EXPECT_EQ(counter, n * (n + 1) / 2);
}

TEST(StatusThreadPool, Failure) {
  StatusThreadPool pool(5);
  pool.Schedule([]() { return absl::InvalidArgumentError("fail"); });
  pool.Schedule([]() { return absl::OkStatus(); });
  absl::Status status = pool.Join();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "fail");
}

TEST(StatusThreadPool, SyncMode) {
  StatusThreadPool pool(0);
  EXPECT_OK(pool.Join());
  pool.Schedule([]() { return absl::InternalError("sync fail"); });
  absl::Status status = pool.Join();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
}

TEST(ThreadPool, Empty) { ThreadPool pool(5); }

TEST(ThreadPool, Basic) {
  std::atomic<int> counter{0};
  const int n = 100;
  {
    ThreadPool pool(5);
    for (int i = 1; i <= n; i++) {
      pool.Schedule([&, i]() { counter += i; });
    }
  }
  EXPECT_EQ(counter, n * (n + 1) / 2);
}

TEST(Channel, Basic) {
  Channel<int> channel;
  channel.Push(10);
  channel.Push(20);
  EXPECT_EQ(channel.Pop(), 10);
  channel.Close();
  EXPECT_EQ(channel.Pop(), 20);
  EXPECT_EQ(channel.Pop(), std::nullopt);
}

TEST(Channel, WithCapacity) {
  Channel<int> channel(/*max_items=*/1);
  std::atomic<bool> push_blocked{false};
  std::atomic<bool> push_finished{false};

  // Push one item, filling the capacity.
  channel.Push(1);

  // Schedule a push that should block.
  std::thread t([&]() {
    push_blocked = true;
    channel.Push(2);
    push_finished = true;
  });

  // Wait a bit to ensure the thread has started and is likely blocked.
  absl::SleepFor(absl::Milliseconds(100));
  EXPECT_TRUE(push_blocked);
  EXPECT_FALSE(push_finished);

  // Pop the first item, unblocking the second push.
  EXPECT_EQ(channel.Pop(), 1);

  // Wait for the second push to complete.
  t.join();
  EXPECT_TRUE(push_finished);

  // Pop the second item.
  EXPECT_EQ(channel.Pop(), 2);

  channel.Close();
  EXPECT_EQ(channel.Pop(), std::nullopt);
}

TEST(Utils, ConcurrentForLoop) {
  std::atomic<int> sum{0};
  std::vector<int> items(500, 2);
  {
    ThreadPool pool(5);
    ConcurrentForLoop(
        4, &pool, items.size(),
        [&sum, &items](size_t block_idx, size_t begin_idx, size_t end_idx) {
          int a = 0;
          for (int i = begin_idx; i < end_idx; i++) {
            a += items[i];
          }
          sum += a;
        });
  }
  EXPECT_EQ(sum, items.size() * 2);
}

TEST(Utils, ScheduleConcurrentForLoop) {
  for (const int num_items : {0, 1, 7, 500}) {
    std::atomic<int> sum{0};
    absl::Notification done;
    ThreadPool pool(5);
    ScheduleConcurrentForLoop(
        4, &pool, num_items,
        [&sum](size_t block_idx, size_t begin_idx, size_t end_idx) {
          sum += end_idx - begin_idx;
        },
        [&done]() { done.Notify(); });
    done.WaitForNotification();
    EXPECT_EQ(sum.load(), num_items);
  }
}

struct Block {
  size_t block_idx;
  size_t begin_idx;
  size_t end_idx;
};

void SortByBlockIdx(std::vector<Block>& blocks) {
  std::sort(blocks.begin(), blocks.end(), [](const Block& a, const Block& b) {
    return a.block_idx < b.block_idx;
  });
}

// Returns the blocks sorted by index. Checks that `done` is called once, last.
std::vector<Block> RunScheduleConcurrentForLoop(const size_t num_blocks,
                                                const size_t num_items,
                                                const int num_threads) {
  absl::Mutex mutex;
  std::vector<Block> blocks;
  std::atomic<int> num_done_calls{0};
  size_t num_blocks_at_done = 0;
  absl::Notification done;
  {
    ThreadPool pool(num_threads);
    ScheduleConcurrentForLoop(
        num_blocks, &pool, num_items,
        [&](size_t block_idx, size_t begin_idx, size_t end_idx) {
          absl::MutexLock l(&mutex);
          blocks.push_back({block_idx, begin_idx, end_idx});
        },
        [&]() {
          {
            absl::MutexLock l(&mutex);
            num_blocks_at_done = blocks.size();
          }
          num_done_calls++;
          done.Notify();
        });
    done.WaitForNotification();
  }
  EXPECT_EQ(num_done_calls.load(), 1);
  EXPECT_EQ(num_blocks_at_done, blocks.size());
  SortByBlockIdx(blocks);
  return blocks;
}

void CheckIsBalancedPartition(const std::vector<Block>& blocks,
                              const size_t num_blocks, const size_t num_items) {
  const size_t expected_num_blocks = std::min(num_blocks, num_items);
  ASSERT_EQ(blocks.size(), expected_num_blocks);
  if (expected_num_blocks == 0) {
    return;
  }
  const size_t min_block_size = num_items / expected_num_blocks;
  size_t next_begin_idx = 0;
  for (size_t i = 0; i < blocks.size(); i++) {
    SCOPED_TRACE(testing::Message() << "block " << i);
    EXPECT_EQ(blocks[i].block_idx, i);
    EXPECT_EQ(blocks[i].begin_idx, next_begin_idx);
    const size_t block_size = blocks[i].end_idx - blocks[i].begin_idx;
    EXPECT_GE(block_size, 1);
    EXPECT_GE(block_size, min_block_size);
    EXPECT_LE(block_size, min_block_size + 1);
    next_begin_idx = blocks[i].end_idx;
  }
  EXPECT_EQ(next_begin_idx, num_items);
}

TEST(Utils, ScheduleConcurrentForLoopIsBalancedPartition) {
  for (const int num_threads : {0, 1, 5}) {
    for (size_t num_blocks = 0; num_blocks <= 12; num_blocks++) {
      for (size_t num_items = 0; num_items <= 30; num_items++) {
        SCOPED_TRACE(testing::Message()
                     << "num_threads=" << num_threads
                     << " num_blocks=" << num_blocks
                     << " num_items=" << num_items);
        CheckIsBalancedPartition(
            RunScheduleConcurrentForLoop(num_blocks, num_items, num_threads),
            num_blocks, num_items);
      }
    }
  }
}

TEST(Utils, ScheduleConcurrentForLoopNumBlocksCloseToNumItems) {
  const auto blocks = RunScheduleConcurrentForLoop(
      /*num_blocks=*/7, /*num_items=*/10, /*num_threads=*/5);
  CheckIsBalancedPartition(blocks, 7, 10);
  const std::vector<std::pair<size_t, size_t>> expected_ranges = {
      {0, 2}, {2, 4}, {4, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 10}};
  ASSERT_EQ(blocks.size(), expected_ranges.size());
  for (size_t i = 0; i < blocks.size(); i++) {
    EXPECT_EQ(blocks[i].begin_idx, expected_ranges[i].first);
    EXPECT_EQ(blocks[i].end_idx, expected_ranges[i].second);
  }
}

TEST(Utils, ScheduleConcurrentForLoopMoreBlocksThanItems) {
  CheckIsBalancedPartition(RunScheduleConcurrentForLoop(
                               /*num_blocks=*/100, /*num_items=*/3,
                               /*num_threads=*/5),
                           100, 3);
}

TEST(Utils, ScheduleConcurrentForLoopHugeNumItems) {
  constexpr size_t kNumItems = std::numeric_limits<size_t>::max();
  CheckIsBalancedPartition(RunScheduleConcurrentForLoop(
                               /*num_blocks=*/7, kNumItems,
                               /*num_threads=*/0),
                           7, kNumItems);
}

TEST(Utils, ScheduleConcurrentForLoopReleasesFunctionBeforeDone) {
  for (const int num_threads : {0, 5}) {
    auto token = std::make_shared<int>(0);
    const std::weak_ptr<int> weak_token = token;
    bool function_released_at_done = false;
    absl::Notification done;
    ThreadPool pool(num_threads);
    ScheduleConcurrentForLoop(
        4, &pool, 100,
        [token = std::move(token)](size_t, size_t, size_t) {},
        [&]() {
          function_released_at_done = weak_token.expired();
          done.Notify();
        });
    done.WaitForNotification();
    EXPECT_TRUE(function_released_at_done);
  }
}

TEST(Utils, ConcurrentForLoopIsBalancedPartition) {
  ThreadPool pool(5);
  for (size_t num_blocks = 2; num_blocks <= 12; num_blocks++) {
    for (size_t num_items = 0; num_items <= 30; num_items++) {
      SCOPED_TRACE(testing::Message() << "num_blocks=" << num_blocks
                                      << " num_items=" << num_items);
      absl::Mutex mutex;
      std::vector<Block> blocks;
      ConcurrentForLoop(
          num_blocks, &pool, num_items,
          [&](size_t block_idx, size_t begin_idx, size_t end_idx) {
            absl::MutexLock l(&mutex);
            blocks.push_back({block_idx, begin_idx, end_idx});
          });
      SortByBlockIdx(blocks);
      CheckIsBalancedPartition(blocks, num_blocks, num_items);
    }
  }
}

TEST(Utils, ConcurrentForLoopSingleBlock) {
  ThreadPool pool(5);
  for (const size_t num_blocks : {0, 1}) {
    for (const size_t num_items : {0, 1, 10}) {
      std::vector<Block> blocks;
      ConcurrentForLoop(
          num_blocks, &pool, num_items,
          [&](size_t block_idx, size_t begin_idx, size_t end_idx) {
            blocks.push_back({block_idx, begin_idx, end_idx});
          });
      ASSERT_EQ(blocks.size(), 1);
      EXPECT_EQ(blocks[0].block_idx, 0);
      EXPECT_EQ(blocks[0].begin_idx, 0);
      EXPECT_EQ(blocks[0].end_idx, num_items);
    }
  }
}

}  // namespace
}  // namespace dgf::util::concurrency

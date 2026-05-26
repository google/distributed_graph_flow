#include "dgf/src/util/concurrency.h"

#include <atomic>
#include <cstddef>
#include <optional>
#include <thread>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "dgf/src/util/test_util.h"

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

}  // namespace
}  // namespace dgf::util::concurrency

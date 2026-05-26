// Utilities for concurrent computation.
// Should be opensourcable.
#ifndef DGF_SRC_UTIL_CONCURRENCY_H_
#define DGF_SRC_UTIL_CONCURRENCY_H_

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"

namespace dgf::util::concurrency {

// TODO(gbm): Use XYZ's thread.
typedef std::unique_lock<std::mutex> MutexLock;
typedef std::thread Thread;
typedef std::mutex Mutex;

// A basic channel.
template <typename Input>
class Channel {
 public:
  Channel(const std::optional<size_t> max_items = {}) : max_items_(max_items) {}

  ~Channel() {
    if (!channel_closed_) {
      Close();
    }
  }

  // Close the channel. No new items can be push in the channel. Any "Pop" will
  // return immediately with an empty optional.
  void Close() {
    MutexLock l(mutex_);
    if (channel_closed_) {
      return;
    }
    channel_closed_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  // Push an item in the channel.
  void Push(Input item) {
    MutexLock l(mutex_);
    if (max_items_.has_value()) {
      while (content_.size() >= *max_items_ && !channel_closed_) {
        not_full_.wait(l);
      }
    }
    if (channel_closed_) {
      return;  // Cannot push to a closed channel.
    }
    content_.push(std::move(item));
    not_empty_.notify_one();
  }

  // Pops a value from the channel. If the channel is closed and empty, returns
  // {}. If the channel is empty but not closed, blocks. If the channel is not
  // empty, returns the first added element.
  std::optional<Input> Pop() {
    MutexLock l(mutex_);
    while (content_.empty() && !channel_closed_) {
      not_empty_.wait(l);
    }
    if (channel_closed_ && content_.empty()) {
      return {};
    }
    Input input{std::move(content_.front())};
    content_.pop();
    if (max_items_.has_value()) {
      not_full_.notify_one();
    }
    return std::move(input);
  }

 private:
  std::queue<Input> content_;
  std::atomic<bool> channel_closed_ = false;
  std::condition_variable not_empty_;  // Signaled when an item is added.
  std::condition_variable not_full_;   // Signaled when an item is removed.
  std::mutex mutex_;
  std::optional<size_t> max_items_;
};

// A basic thread pool.
class ThreadPool {
 public:
  // Start the threads in the thread pool.
  //  If "num_threads==0", jobs are run synchronously.
  ThreadPool(int num_threads);

  // Join all the threads.
  ~ThreadPool();

  void Schedule(std::function<void()> callback);

  // Number of threads configured in the constructor.
  int num_threads() const { return threads_.size(); }

 private:
  // Ensure all the jobs are done and all the threads have been joined.
  void JoinAllAndStopThreads();

  // Running loop for the threads.
  void ThreadLoop();

  // Number of threads.
  int num_threads_;

  // Active threads.
  std::vector<Thread> threads_;

  // Scheduled jobs.
  Channel<std::function<void()>> jobs_;
};

// A thread pool where scheduled jobs returns a status.
// If any of the job fails, "Join" will return the first error.
class StatusThreadPool {
 public:
  StatusThreadPool(int num_threads);

  ~StatusThreadPool();

  void Schedule(std::function<absl::Status()> callback);

  // Wait for all the jobs to finish and return the aggregated status.
  absl::Status Join();

 private:
  void ThreadLoop();

  int num_threads_;
  bool joined_ = false;
  absl::Status status_;
  mutable std::mutex status_mutex_;
  std::vector<Thread> threads_;
  Channel<std::function<absl::Status()>> jobs_;
};

// A list of threads with some utility methods.
class ThreadVector {
 public:
  void Start(int num_threads, std::function<void()> callback);

  void JoinAndClear();

 private:
  std::vector<std::unique_ptr<Thread>> threads_;
};

// Applies "function" over a range of elements using multi-threading.
void ConcurrentForLoop(
    size_t num_blocks, ThreadPool* thread_pool, size_t num_items,
    const std::function<void(size_t block_idx, size_t begin_item_idx,
                             size_t end_item_idx)>& function);

}  // namespace dgf::util::concurrency

#endif  // DGF_SRC_UTIL_CONCURRENCY_H_

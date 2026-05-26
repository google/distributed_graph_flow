#include "dgf/src/util/concurrency.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/synchronization/blocking_counter.h"

namespace dgf::util::concurrency {

ThreadPool::ThreadPool(int num_threads) : num_threads_(num_threads) {
  while (threads_.size() < num_threads_) {
    threads_.emplace_back(&ThreadPool::ThreadLoop, this);
  }
}

ThreadPool::~ThreadPool() { JoinAllAndStopThreads(); }

void ThreadPool::JoinAllAndStopThreads() {
  if (num_threads_ == 0) {
    return;
  }
  jobs_.Close();
  for (auto& thread : threads_) {
    thread.join();
  }
  threads_.clear();
}

void ThreadPool::Schedule(std::function<void()> callback) {
  if (num_threads_ == 0) {
    callback();
  } else {
    jobs_.Push(std::move(callback));
  }
}

void ThreadPool::ThreadLoop() {
  while (true) {
    // Get a job.
    auto optional_input = jobs_.Pop();
    if (!optional_input.has_value()) {
      break;
    }
    // Run the job.
    std::move(optional_input).value()();
  }
}

StatusThreadPool::StatusThreadPool(int num_threads)
    : num_threads_(num_threads) {
  while (threads_.size() < num_threads_) {
    threads_.emplace_back(&StatusThreadPool::ThreadLoop, this);
  }
}

StatusThreadPool::~StatusThreadPool() { Join().IgnoreError(); }

void StatusThreadPool::Schedule(std::function<absl::Status()> callback) {
  if (num_threads_ == 0) {
    const auto status = callback();
    if (!status.ok()) {
      MutexLock l(status_mutex_);
      status_.Update(status);
    }
  } else {
    jobs_.Push(std::move(callback));
  }
}

absl::Status StatusThreadPool::Join() {
  if (num_threads_ == 0) {
    return status_;
  }
  std::vector<Thread> threads_to_join;
  {
    MutexLock l(status_mutex_);
    if (!joined_) {
      jobs_.Close();
      threads_to_join = std::move(threads_);
      joined_ = true;
    }
  }
  for (auto& thread : threads_to_join) {
    thread.join();
  }
  MutexLock l(status_mutex_);
  return status_;
}

void StatusThreadPool::ThreadLoop() {
  while (true) {
    // Get a job.
    auto optional_input = jobs_.Pop();
    if (!optional_input.has_value()) {
      break;
    }
    // Run the job.
    const auto status = std::move(optional_input).value()();
    if (!status.ok()) {
      MutexLock l(status_mutex_);
      status_.Update(status);
    }
  }
}

void ThreadVector::Start(int num_threads, std::function<void()> callback) {
  threads_.resize(num_threads);
  for (int i = 0; i < num_threads; i++) {
    threads_[i] = std::make_unique<Thread>(callback);
  }
}

void ThreadVector::JoinAndClear() {
  for (auto& thread : threads_) {
    thread->join();
  }
  threads_.clear();
}

void ConcurrentForLoop(
    const size_t num_blocks, ThreadPool* thread_pool, const size_t num_items,
    const std::function<void(size_t block_idx, size_t begin_item_idx,
                             size_t end_item_idx)>& function) {
  if (num_blocks <= 1) {
    function(0, 0, num_items);
    return;
  }
  const size_t effective_num_blocks = std::min(num_blocks, num_items);
  absl::BlockingCounter blocker(effective_num_blocks);
  size_t begin_idx = 0;
  const size_t block_size =
      (num_items + effective_num_blocks - 1) / effective_num_blocks;
  for (size_t block_idx = 0; block_idx < effective_num_blocks; block_idx++) {
    const auto end_idx = std::min(begin_idx + block_size, num_items);
    if (begin_idx <= end_idx) {
      thread_pool->Schedule(
          [block_idx, begin_idx, end_idx, &blocker, &function]() -> void {
            function(block_idx, begin_idx, end_idx);
            blocker.DecrementCount();
          });
      begin_idx += block_size;
    } else {
      blocker.DecrementCount();
    }
  }
  blocker.Wait();
}

}  // namespace dgf::util::concurrency

#ifndef DGF_SRC_UTIL_STATUS_CASTER_H_
#define DGF_SRC_UTIL_STATUS_CASTER_H_

// Status caster utilities.
// Copied + adapted from TF/XLA codebase:
// tensorflow/compiler/xla/pjrt/status_casters.h

#include <stdexcept>  // IWYU pragma: keep
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace dgf {

inline void ThrowIfError(absl::Status src) {
  if (!src.ok()) {
    throw std::invalid_argument(std::string(src.message()));
  }
}

template <typename Sig, typename F>
struct ThrowIfErrorWrapper;

template <typename F>
ThrowIfErrorWrapper(F) -> ThrowIfErrorWrapper<decltype(&F::operator()), F>;

template <typename... Args>
ThrowIfErrorWrapper(absl::Status (&)(Args...))
    -> ThrowIfErrorWrapper<absl::Status(Args...), absl::Status (&)(Args...)>;

template <typename C, typename... Args>
ThrowIfErrorWrapper(absl::Status (C::*)(Args...))
    -> ThrowIfErrorWrapper<absl::Status(Args...), C>;

template <typename... Args>
struct ThrowIfErrorWrapper<absl::Status(Args...), absl::Status (&)(Args...)> {
  explicit ThrowIfErrorWrapper(absl::Status (&f)(Args...)) : func(f) {}
  void operator()(Args... args) const {
    ThrowIfError(func(std::forward<Args>(args)...));
  }
  absl::Status (&func)(Args...);
};

template <typename C, typename... Args, typename F>
struct ThrowIfErrorWrapper<absl::Status (C::*)(Args...), F> {
  explicit ThrowIfErrorWrapper(F&& f) : func(std::move(f)) {}
  void operator()(Args... args) const {
    ThrowIfError(func(std::forward<Args>(args)...));
  }
  F func;
};
template <typename C, typename... Args, typename F>
struct ThrowIfErrorWrapper<absl::Status (C::*)(Args...) const, F> {
  explicit ThrowIfErrorWrapper(F&& f) : func(std::move(f)) {}
  void operator()(Args... args) const {
    ThrowIfError(func(std::forward<Args>(args)...));
  }
  F func;
};

template <typename C, typename... Args>
struct ThrowIfErrorWrapper<absl::Status(Args...), C> {
  explicit ThrowIfErrorWrapper(absl::Status (C::*ptmf)(Args...)) : ptmf(ptmf) {}
  void operator()(C& instance, Args... args) const {
    ThrowIfError((instance.*ptmf)(std::forward<Args>(args)...));
  }
  absl::Status (C::*ptmf)(Args...);
};

template <typename C, typename... Args>
struct ThrowIfErrorWrapper<absl::Status(Args...) const, C> {
  explicit ThrowIfErrorWrapper(absl::Status (C::*ptmf)(Args...) const)
      : ptmf(ptmf) {}
  void operator()(const C& instance, Args... args) const {
    ThrowIfError((instance.*ptmf)(std::forward<Args>(args)...));
  }
  absl::Status (C::*ptmf)(Args...) const;
};

template <typename T>
T ValueOrThrow(absl::StatusOr<T> v) {
  if (!v.ok()) {
    throw std::invalid_argument(std::string(v.status().message()));
  }
  return std::move(v).value();
}

template <typename Sig, typename F>
struct ValueOrThrowWrapper;

template <typename F>
ValueOrThrowWrapper(F) -> ValueOrThrowWrapper<decltype(&F::operator()), F>;

template <typename R, typename... Args>
ValueOrThrowWrapper(absl::StatusOr<R> (&)(Args...))
    -> ValueOrThrowWrapper<absl::StatusOr<R>(Args...),
                           absl::StatusOr<R> (&)(Args...)>;

template <typename C, typename R, typename... Args>
ValueOrThrowWrapper(absl::StatusOr<R> (C::*)(Args...))
    -> ValueOrThrowWrapper<absl::StatusOr<R>(Args...), C>;

// Deduction guide for const methods.
template <typename C, typename R, typename... Args>
ValueOrThrowWrapper(absl::StatusOr<R> (C::*)(Args...) const)
    -> ValueOrThrowWrapper<absl::StatusOr<R>(Args...) const, C>;

template <typename R, typename... Args>
struct ValueOrThrowWrapper<absl::StatusOr<R>(Args...),
                           absl::StatusOr<R> (&)(Args...)> {
  explicit ValueOrThrowWrapper(absl::StatusOr<R> (&f)(Args...)) : func(f) {}
  R operator()(Args... args) const {
    return ValueOrThrow(func(std::forward<Args>(args)...));
  }
  absl::StatusOr<R> (&func)(Args...);
};
template <typename R, typename C, typename... Args, typename F>
struct ValueOrThrowWrapper<absl::StatusOr<R> (C::*)(Args...), F> {
  explicit ValueOrThrowWrapper(F&& f) : func(std::move(f)) {}
  R operator()(Args... args) const {
    return ValueOrThrow(func(std::forward<Args>(args)...));
  }
  F func;
};
template <typename R, typename C, typename... Args, typename F>
struct ValueOrThrowWrapper<absl::StatusOr<R> (C::*)(Args...) const, F> {
  explicit ValueOrThrowWrapper(F&& f) : func(std::move(f)) {}
  R operator()(Args... args) const {
    return ValueOrThrow(func(std::forward<Args>(args)...));
  }
  F func;
};
// For unbound nonstatic member functions, non-const and const versions.
// `ptmf` stands for "pointer to member function".
template <typename R, typename C, typename... Args>
struct ValueOrThrowWrapper<absl::StatusOr<R>(Args...), C> {
  explicit ValueOrThrowWrapper(absl::StatusOr<R> (C::*ptmf)(Args...))
      : ptmf(ptmf) {}
  R operator()(C& instance, Args... args) const {
    return ValueOrThrow((instance.*ptmf)(std::forward<Args>(args)...));
  }
  absl::StatusOr<R> (C::*ptmf)(Args...);
};
template <typename R, typename C, typename... Args>
struct ValueOrThrowWrapper<absl::StatusOr<R>(Args...) const, C> {
  explicit ValueOrThrowWrapper(absl::StatusOr<R> (C::*ptmf)(Args...) const)
      : ptmf(ptmf) {}
  R operator()(const C& instance, Args... args) const {
    return ValueOrThrow((instance.*ptmf)(std::forward<Args>(args)...));
  }
  absl::StatusOr<R> (C::*ptmf)(Args...) const;
};

}  // namespace dgf

#endif  // DGF_SRC_UTIL_STATUS_CASTER_H_

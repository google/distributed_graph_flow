

#ifndef DGF_SRC_UTIL_TEST_UTIL_H_
#define DGF_SRC_UTIL_TEST_UTIL_H_

#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace dgf {

MATCHER(StatusIsOk, "Status is OK") { return arg.ok(); }

#ifndef EXPECT_OK
#define EXPECT_OK(expr) EXPECT_THAT(expr, ::dgf::StatusIsOk())
#endif
#ifndef ASSERT_OK
#define ASSERT_OK(expr) ASSERT_THAT(expr, ::dgf::StatusIsOk())
#endif

// Concatenation helpers (defined outside guard as they are safe)
#define ASSERT_OK_AND_ASSIGN_CONCAT_IMPL(x, y) x##y
#define ASSERT_OK_AND_ASSIGN_CONCAT(x, y) ASSERT_OK_AND_ASSIGN_CONCAT_IMPL(x, y)

#ifndef ASSERT_OK_AND_ASSIGN
#define ASSERT_OK_AND_ASSIGN_IMPL(lhs, rexpr, status_name) \
  auto status_name = (rexpr);                              \
  ASSERT_THAT(status_name.status(), ::dgf::StatusIsOk());  \
  lhs = std::move(status_name).value();

#define ASSERT_OK_AND_ASSIGN(lhs, rexpr) \
  ASSERT_OK_AND_ASSIGN_IMPL(             \
      lhs, rexpr, ASSERT_OK_AND_ASSIGN_CONCAT(_status_or_, __LINE__))
#endif

}  // namespace dgf

#endif  // DGF_SRC_UTIL_TEST_UTIL_H_

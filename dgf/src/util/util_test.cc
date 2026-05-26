#include "dgf/src/util/util.h"

#include "gtest/gtest.h"

namespace dgf {
namespace {

TEST(UtilTest, ShardedFilename) {
  EXPECT_EQ(ShardedFilename("ab", 0, 10, ".cd"), "ab-00000-of-00010.cd");
  EXPECT_EQ(ShardedFilename("ab", 0, 10, ""), "ab-00000-of-00010");
}

}  // namespace
}  // namespace dgf

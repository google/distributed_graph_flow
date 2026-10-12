#include "dgf/src/util/random.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "absl/random/distributions.h"

namespace dgf {
namespace {

// The expected values are produced by a reference implementation of
// xoshiro256++ seeded with splitmix64. They must never change: changing them
// would change all the seeded samples.
TEST(DeterministicRngTest, ReferenceValues) {
  DeterministicRng rng0(0);
  EXPECT_EQ(rng0(), 0x53175d61490b23dfULL);
  EXPECT_EQ(rng0(), 0x61da6f3dc380d507ULL);
  EXPECT_EQ(rng0(), 0x5c0fdf91ec9a7bfcULL);

  DeterministicRng rng1(1);
  EXPECT_EQ(rng1(), 0xcfc5d07f6f03c29bULL);
  EXPECT_EQ(rng1(), 0xbf424132963fe08dULL);
  EXPECT_EQ(rng1(), 0x19a37d5757aaf520ULL);

  DeterministicRng rng42(42);
  EXPECT_EQ(rng42(), 0xd0764d4f4476689fULL);
  EXPECT_EQ(rng42(), 0x519e4174576f3791ULL);
  EXPECT_EQ(rng42(), 0xfbe07cfb0c24ed8cULL);

  DeterministicRng rng_max(0xffffffffffffffffULL);
  EXPECT_EQ(rng_max(), 0x56ccf8ce948e27b2ULL);
  EXPECT_EQ(rng_max(), 0xe68588432e5a5b90ULL);
  EXPECT_EQ(rng_max(), 0xe3e9b5a48119ca8bULL);
}

TEST(DeterministicRngTest, SameSeedSameSequence) {
  DeterministicRng a(123);
  DeterministicRng b(123);
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(a(), b());
  }
}

TEST(DeterministicRngTest, DifferentSeedsDifferentSequences) {
  DeterministicRng a(123);
  DeterministicRng b(124);
  int num_equal = 0;
  for (int i = 0; i < 1000; ++i) {
    num_equal += (a() == b());
  }
  EXPECT_EQ(num_equal, 0);
}

TEST(DeterministicRngTest, DefaultConstructorIsNonDeterministic) {
  DeterministicRng a;
  DeterministicRng b;
  int num_equal = 0;
  for (int i = 0; i < 1000; ++i) {
    num_equal += (a() == b());
  }
  EXPECT_EQ(num_equal, 0);
}

TEST(DeterministicRngTest, SatisfiesUniformRandomBitGenerator) {
  static_assert(std::is_same_v<DeterministicRng::result_type, uint64_t>);
  EXPECT_EQ(DeterministicRng::min(), 0);
  EXPECT_EQ(DeterministicRng::max(), std::numeric_limits<uint64_t>::max());
  // Usable with the std algorithms and the absl distributions.
  DeterministicRng rng(0);
  std::vector<int> values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::shuffle(values.begin(), values.end(), rng);
  for (int i = 0; i < 100; ++i) {
    const int x = absl::Uniform<int>(rng, 0, 10);
    EXPECT_GE(x, 0);
    EXPECT_LT(x, 10);
  }
}

}  // namespace
}  // namespace dgf

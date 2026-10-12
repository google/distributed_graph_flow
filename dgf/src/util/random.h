// Random number generation utilities.

#ifndef DGF_SRC_UTIL_RANDOM_H_
#define DGF_SRC_UTIL_RANDOM_H_

#include <cstdint>

#include "absl/random/distributions.h"
#include "absl/random/random.h"

namespace dgf {

// Fast, deterministic random number generator (xoshiro256++, seeded with
// splitmix64; see https://prng.di.unimi.it/).
//
// A given seed always produces the same sequence of numbers, on every
// platform, in every build mode, and in every process. This is a deliberate
// difference from the absl generators (e.g. `absl::BitGen`,
// `absl::InsecureBitGen`), which mix a per-process (or per-build) salt into the
// seed and are thus not reproducible across runs, even when seeded.
//
// This class satisfies the C++ UniformRandomBitGenerator requirements and can
// be used with `absl::Uniform`, `std::sample`, `std::shuffle`, etc.
//
// Not suitable for cryptographic purposes.
class DeterministicRng {
 public:
  using result_type = std::uint64_t;

  // Seeds the generator with `seed`. A given seed always produces the same
  // sequence.
  explicit DeterministicRng(std::uint64_t seed) { Seed(seed); }

  // Seeds the generator non-deterministically (from the OS entropy). Use when
  // reproducibility is not required (e.g. when the user does not provide a
  // seed).
  DeterministicRng() {
    absl::BitGen entropy;
    Seed(absl::Uniform<std::uint64_t>(entropy));
  }

  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return ~result_type{0}; }

  result_type operator()() {
    const std::uint64_t result = Rotl(state_[0] + state_[3], 23) + state_[0];
    const std::uint64_t t = state_[1] << 17;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= t;
    state_[3] = Rotl(state_[3], 45);
    return result;
  }

 private:
  void Seed(std::uint64_t seed) {
    // splitmix64 expands the seed into the 4 state words. This also ensures
    // that the state is never all-zeros (xoshiro's single invalid state).
    for (std::uint64_t& s : state_) {
      seed += 0x9E3779B97F4A7C15ULL;
      std::uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      s = z ^ (z >> 31);
    }
  }

  static constexpr std::uint64_t Rotl(std::uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }

  std::uint64_t state_[4];
};

}  // namespace dgf

#endif  // DGF_SRC_UTIL_RANDOM_H_

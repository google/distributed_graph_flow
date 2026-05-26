#include <string>
#include <string_view>

#include "absl/strings/str_format.h"

namespace dgf {

std::string ShardedFilename(std::string_view filename, int shard,
                            int num_shards, std::string_view extension) {
  return absl::StrFormat("%s-%05d-of-%05d%s", filename, shard, num_shards,
                         extension);
}

}  // namespace dgf

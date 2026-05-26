// General utilities for c++ code

#ifndef DGF_SRC_UTIL_UTIL_H_
#define DGF_SRC_UTIL_UTIL_H_

#include <string>
#include <string_view>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace dgf {

// Evaluates an expression returning a absl::Status. Returns with the status
// if the status is not "OK".
//
// Usage example:
//   absl::Status f() {
//     auto g = []() -> absl::Status { ... };
//     DGF_RETURN_IF_ERROR(g());
//     return absl::OKStatus();
//   }
#ifndef DGF_RETURN_IF_ERROR
#define DGF_RETURN_IF_ERROR(expr)            \
  {                                          \
    auto _status = (expr);                   \
    if (ABSL_PREDICT_FALSE(!_status.ok())) { \
      return _status;                        \
    }                                        \
  }
#endif

#define TOKEN_PASTE(x, y) x##y
#define CONCATENATE(x, y) TOKEN_PASTE(x, y)

// Evaluates an expression returning a absl::StatusOr. Returns with the status
// if the status is not "OK". Move the value to "lhs" and continue the execution
// otherwise.
//
// Usage example:
//   absl::Status f() {
//     auto g = []() -> absl::StatusOr<int> { ... };
//     DGF_ASSIGN_OR_RETURN(const auto x, g());
//     return absl::OKStatus();
//   }
//
// A third argument containing a extra error message is possible.
//
// Usage example:
//   absl::Status f() {
//     auto g = []() -> absl::StatusOr<int> { ... };
//     DGF_ASSIGN_OR_RETURN(const auto x, g(), _  << "Extra information");
//     return absl::OKStatus();
//   }

#define DGF_ASSIGN_OR_RETURN(...)                                            \
  SELECT_FOURTH_ARGUMENT_FROM_LIST(                                          \
      (__VA_ARGS__, DGF_ASSIGN_OR_RETURN_3ARGS, DGF_ASSIGN_OR_RETURN_2ARGS)) \
  (__VA_ARGS__)

#define DGF_ASSIGN_OR_RETURN_2ARGS(lhs, rexpr) \
  DGF_ASSIGN_OR_RETURN_2ARGS_IMP(lhs, rexpr,   \
                                 CONCATENATE(_status_or_value, __LINE__))

#define DGF_ASSIGN_OR_RETURN_3ARGS(lhs, rexpr, message) \
  DGF_ASSIGN_OR_RETURN_3ARGS_IMP(lhs, rexpr, message,   \
                                 CONCATENATE(_status_or_value, __LINE__))

#define SELECT_FOURTH_ARGUMENT(_1, _2, _3, _4, ...) _4
#define SELECT_FOURTH_ARGUMENT_FROM_LIST(args) SELECT_FOURTH_ARGUMENT args

#define DGF_ASSIGN_OR_RETURN_2ARGS_IMP(lhs, rexpr, tmpvar) \
  auto tmpvar = (rexpr);                                   \
  if (ABSL_PREDICT_FALSE(!tmpvar.ok())) {                  \
    return tmpvar.status();                                \
  }                                                        \
  lhs = std::move(tmpvar).value()

#define DGF_ASSIGN_OR_RETURN_3ARGS_IMP(lhs, rexpr, message, tmpvar) \
  auto tmpvar = (rexpr);                                            \
  if (ABSL_PREDICT_FALSE(!tmpvar.ok())) {                           \
    std::string _;                                                  \
    LOG(WARNING) << message;                                        \
    return tmpvar.status();                                         \
  }                                                                 \
  lhs = std::move(tmpvar).value()

// Maps a string `value` to a unique integer index. If `value` is already in the
// `index`, returns its existing integer mapping. Otherwise, assigns `value` the
// next available integer index (equal to the current size of the map) and
// returns this new index.
inline int GetOrCreateIndex(absl::flat_hash_map<std::string, int>& index,
                            const std::string_view& value) {
  // Attempt to insert the value with the current size of the map.
  // try_emplace will only insert if the key does not already exist.
  auto [it, inserted] = index.try_emplace(value, index.size());
  return it->second;
}

// Gets an item from a map (e.g., flat_hash_map). If the item does not exist,
// return an error.
template <typename Map>
absl::StatusOr<typename Map::mapped_type> GetItem(
    const Map& map, const typename Map::key_type& key) {
  auto it = map.find(key);
  if (it == map.end()) {
    return absl::NotFoundError(absl::StrCat("Key not found in map: ", key));
  }
  return it->second;
}

// Checks than an expression is true, otherwise return an error.
#define DGF_STATUS_CHECK(expr) \
  if (!(expr)) return absl::InvalidArgumentError("Check failed " #expr)

// Generates a sharded filename. Similar to "sharded_filename" function in
// "shard.py".
std::string ShardedFilename(std::string_view filename, int shard,
                            int num_shards, std::string_view extension);

}  // namespace dgf

#endif  // DGF_SRC_UTIL_UTIL_H_

// Utilities to use nano bind objects.

#ifndef DGF_SRC_UTIL_NANOBIND_UTIL_H_
#define DGF_SRC_UTIL_NANOBIND_UTIL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"  // IWYU pragma: keep

namespace dgf {

// Gets an attribute from a python objet. If the attributes does not exist or is
// of the wrong type, return an absl error.
#define DGF_GET_ATTR_OR_RETURN(TYPE, DST, OBJECT, KEY)               \
  if (!nanobind::hasattr(OBJECT, KEY)) {                             \
    return absl::InvalidArgumentError("Cannot find attribute " KEY); \
  }                                                                  \
  auto _tmpvar_##DST = OBJECT.attr(KEY);                             \
  if (!nanobind::isinstance<TYPE>(_tmpvar_##DST)) {                  \
    return absl::InvalidArgumentError(                               \
        absl::StrCat(KEY " is not of type " #TYPE));                 \
  }                                                                  \
  TYPE DST = nanobind::cast<TYPE>(_tmpvar_##DST)

// Gets an item from a dict. If the item does not exist, return an error.
template <typename T, typename K>
inline absl::StatusOr<T> GetItemFromPyDict(const nanobind::dict& dict,
                                           const K& key) {
  if (!dict.contains(key)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Key \"", nanobind::cast<std::string>(key), "\" not found in map"));
  }
  const nanobind::object& value = dict[key];
  if constexpr (std::is_same_v<T, nanobind::object>) {
    return value;
  } else {
    if (!nanobind::isinstance<T>(value)) {
      std::string actual_type = nanobind::str(value.type()).c_str();
      return absl::InvalidArgumentError(
          absl::StrCat("Value is of the wrong type. Found type:", actual_type));
    }
    return nanobind::cast<T>(value);
  }
}

inline nanobind::str string_to_py_str(const std::string& v) {
  // Note: NB removed native support for strings.
  return nanobind::str(v.c_str(), v.size());
}

#define DGF_BEGIN_EXCEPTION_TO_STATUS try {
#define DGF_END_EXCEPTION_TO_STATUS              \
  }                                              \
  catch (const std::exception& e) {              \
    return absl::InvalidArgumentError(e.what()); \
  }

// View to Numpy multi-bytes array (e.g. dtype=S5) with 1 dimensions.
// The creation and destruction of "NumpyBytesArray" requires being in the GIL.
//
// Avoid the limitation of nanobind with bytes arrays.
// https://github.com/wjakob/nanobind/issues/1217
struct NumpyBytesArray {
  // Wraps a single dimensional np::array of bytes.
  static absl::StatusOr<NumpyBytesArray> Create(const nanobind::object& data);

  // Number of items.
  size_t size() const { return size_; }

  // Value accessor.
  std::string_view operator[](size_t i) const;

  // Extracts the content of the numpy array into a string_view vector.
  std::vector<std::string_view> ToVectorNotOwned() const;

  // String with all the values.
  std::string ToString() const;

  char* data_;
  size_t stride_;
  size_t itemsize_;
  size_t size_;
};

// Converts a list of python bytes into a c++ vector of string.
// Note: Nanobind does not implement this conversion automatically.
absl::StatusOr<std::vector<std::string>> ListOfBytesToVectorOfStrings(
    const nanobind::object& list);

// Converts a c++ list of string values into a numpy array of multi-bytes.
// The array can be: {span, vector}<{string, string_view}> and
// ProtoRepeatedField<string>.
template <typename Source>
nanobind::object CCVectorOfStringToNumpyArray(const Source& src) {
  // Determine the maximum len of the strings.
  size_t itemsize = 1;
  for (const auto& value : src) {
    itemsize = std::max(itemsize, value.size());
  }

  // Create the buffer.
  const size_t n = src.size();
  const size_t buffer_size = n * itemsize;
  char* data = new char[buffer_size];
  std::memset(data, 0, buffer_size);  // Padding
  for (size_t i = 0; i < src.size(); i++) {
    std::memcpy(data + i * itemsize, src[i].c_str(), src[i].size());
  }

  // Create an array of uint8.
  nanobind::capsule owner(data, [](void* p) noexcept { delete[] (char*)p; });
  nanobind::ndarray<uint8_t, nanobind::numpy, nanobind::shape<-1>,
                    nanobind::c_contig>
      byte_array(
          /* data = */ data,
          /* shape = */ {buffer_size},
          /* owner = */ owner);

  // Cast to an array of stings.
  nanobind::object generic_arr = nanobind::cast(byte_array);
  return generic_arr.attr("view")(
      string_to_py_str(absl::StrCat("S", itemsize)));
}

// Converts a c++ list of scalar values (e.g., float, integer, bool) into a
// numpy array. Does not support strings.
template <typename NPType, typename CCType>
nanobind::ndarray<NPType, nanobind::numpy, nanobind::shape<-1>>
CCVectorToNumpyArray(std::span<const CCType> src) {
  const std::size_t n = src.size();
  NPType* data = new NPType[n];
  for (std::size_t i = 0; i < n; i++) {
    data[i] = static_cast<NPType>(src[i]);
  }
  nanobind::capsule owner(data, [](void* p) noexcept { delete[] (NPType*)p; });
  return nanobind::ndarray<NPType, nanobind::numpy, nanobind::shape<-1>>(
      /* data = */ data,
      /* shape = */ {n},
      /* owner = */ owner);
}

}  // namespace dgf

#endif  // DGF_SRC_UTIL_NANOBIND_UTIL_H_

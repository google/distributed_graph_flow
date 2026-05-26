#include "dgf/src/util/nanobind_util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"  // IWYU pragma: keep

namespace dgf {

namespace {

namespace nb = nanobind;

// Removes '\0' at the end of a string_view. Returns a string_view without the
// zeroes.
std::string_view remove_tailing_zeros(std::string_view src) {
  size_t i = src.size();
  while (i > 0 && src[i - 1] == 0) {
    i--;
  }
  return src.substr(0, i);
}

}  // namespace

absl::StatusOr<NumpyBytesArray> NumpyBytesArray::Create(
    const nb::object& data) {
  PyObject* py_obj = data.ptr();
  Py_buffer view;
  if (PyObject_GetBuffer(py_obj, &view, PyBUF_FULL) != 0) {
    return absl::InvalidArgumentError("Not a buffer object");
  }
  if (view.ndim != 1) {
    return absl::InvalidArgumentError("Wrong shape");
  }
  std::string_view type(view.format);
  if (type.back() != 's') {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expecting a buffer of np.bytes (i.e. 's') array. Got a buffer of \'",
        type, "\' instead"));
  }
  auto ret = NumpyBytesArray{
      .data_ = (char*)view.buf,
      .stride_ = (size_t)view.strides[0],
      .itemsize_ = (size_t)view.itemsize,
      .size_ = (size_t)view.shape[0],
  };
  // We explicitly do NOT call PyBuffer_Release(&view) here. NumpyBytesArray is
  // a non-owning view that directly references `view.buf`. Releasing the
  // buffer view inside this factory method would immediately invalidate the
  // memory pointer, causing use-after-free crashes when accessing elements
  // later. The Python object memory is guaranteed to remain locked as long as
  // the parent Python array is kept alive by the caller.
  // PyBuffer_Release(&view);
  return ret;
}

std::string_view NumpyBytesArray::operator[](size_t i) const {
  DCHECK_LT(i, size_);
  return remove_tailing_zeros({data_ + i * stride_, itemsize_});
}

std::vector<std::string_view> NumpyBytesArray::ToVectorNotOwned() const {
  std::vector<std::string_view> dst(size_);
  for (size_t i = 0; i < size_; i++) {
    dst[i] = (*this)[i];
  }
  return dst;
}

std::string NumpyBytesArray::ToString() const {
  return absl::StrCat("[", absl::StrJoin(ToVectorNotOwned(), ", "), "]");
}

absl::StatusOr<std::vector<std::string>> ListOfBytesToVectorOfStrings(
    const nb::object& list) {
  if (!nb::isinstance<nb::list>(list)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input is not a nb::list. Found type: ", nb::str(list.type()).c_str()));
  }
  nb::list nb_list = nb::cast<nb::list>(list);
  std::vector<std::string> result;
  result.reserve(nb_list.size());
  for (size_t i = 0; i < nb_list.size(); ++i) {
    nb::object item = nb_list[i];
    if (!nb::isinstance<nb::bytes>(item)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "List element at index ", i,
          " is not bytes. Found type: ", nb::str(item.type()).c_str()));
    }
    nb::bytes nb_bytes = nb::cast<nb::bytes>(item);
    result.push_back(std::string(static_cast<const char*>(nb_bytes.c_str()),
                                 nb_bytes.size()));
  }
  return result;
}

}  // namespace dgf

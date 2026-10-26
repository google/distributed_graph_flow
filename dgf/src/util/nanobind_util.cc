#include "dgf/src/util/nanobind_util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <utility>
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
  NumpyBytesArray ret;
  if (PyObject_GetBuffer(data.ptr(), &ret.view_, PyBUF_FULL) != 0) {
    PyErr_Clear();
    return absl::InvalidArgumentError("Not a buffer object");
  }
  // From here, `ret` owns the buffer export: it is released when `ret` is
  // destroyed, including on the error paths below. The data pointer stays
  // valid while the caller keeps the Python array alive (as before).
  ret.owns_view_ = true;
  const Py_buffer& view = ret.view_;
  if (view.ndim != 1) {
    return absl::InvalidArgumentError("Wrong shape");
  }
  std::string_view type(view.format);
  if (type.back() != 's') {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expecting a buffer of np.bytes (i.e. 's') array. Got a buffer of \'",
        type, "\' instead"));
  }
  ret.data_ = (char*)view.buf;
  ret.stride_ = (size_t)view.strides[0];
  ret.itemsize_ = (size_t)view.itemsize;
  ret.size_ = (size_t)view.shape[0];
  return ret;
}

NumpyBytesArray::~NumpyBytesArray() { Release(); }

NumpyBytesArray::NumpyBytesArray(NumpyBytesArray&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      stride_(std::exchange(other.stride_, 0)),
      itemsize_(std::exchange(other.itemsize_, 0)),
      size_(std::exchange(other.size_, 0)),
      view_(std::exchange(other.view_, Py_buffer{})),
      owns_view_(std::exchange(other.owns_view_, false)) {}

NumpyBytesArray& NumpyBytesArray::operator=(NumpyBytesArray&& other) noexcept {
  if (this != &other) {
    Release();
    // Leave `other` in a valid empty state: it must not expose the data of a
    // buffer export it no longer owns.
    data_ = std::exchange(other.data_, nullptr);
    stride_ = std::exchange(other.stride_, 0);
    itemsize_ = std::exchange(other.itemsize_, 0);
    size_ = std::exchange(other.size_, 0);
    view_ = std::exchange(other.view_, Py_buffer{});
    owns_view_ = std::exchange(other.owns_view_, false);
  }
  return *this;
}

void NumpyBytesArray::Release() {
  if (owns_view_) {
    PyBuffer_Release(&view_);
    owns_view_ = false;
  }
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

absl::StatusOr<std::vector<std::string_view>> SequenceOfBytesToStringViews(
    const nb::handle& sequence) {
  std::vector<std::string_view> views;
  if (PyList_Check(sequence.ptr()) || PyTuple_Check(sequence.ptr())) {
    views.reserve(nb::len(sequence));
  }
  for (nb::handle item : sequence) {
    char* buffer;
    Py_ssize_t length;
    if (PyBytes_AsStringAndSize(item.ptr(), &buffer, &length) != 0) {
      PyErr_Clear();
      return absl::InvalidArgumentError(
          absl::StrCat("Expecting a sequence of bytes. Got an item of type ",
                       nb::type_name(item.type()).c_str()));
    }
    views.emplace_back(buffer, length);
  }
  return views;
}

}  // namespace dgf

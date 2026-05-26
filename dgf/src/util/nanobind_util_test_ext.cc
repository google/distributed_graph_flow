
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "dgf/src/util/nanobind_util.h"
#include "dgf/src/util/status_caster.h"
#include "dgf/src/util/util.h"

namespace nb = nanobind;

namespace dgf {
namespace {

#define ASSERT(X)                                             \
  if (!(X)) {                                                 \
    return absl::InvalidArgumentError("Something is wrong."); \
  }

absl::Status TestNumpyBytesArray(const nb::object& array) {
  DGF_ASSIGN_OR_RETURN(const auto x, NumpyBytesArray::Create(array));
  ASSERT(x.size() == 4);
  ASSERT(x[0] == "a");
  ASSERT(x[1] == "bcd");
  ASSERT(x[2] == "");
  ASSERT(x[3] == "f");
  auto view = x.ToVectorNotOwned();
  ASSERT(view.size() == 4);
  ASSERT(view[0] == "a");
  ASSERT(view[1] == "bcd");
  ASSERT(view[2] == "");
  ASSERT(view[3] == "f");
  return absl::OkStatus();
}

nanobind::object TestCCArrayToNumpyArray(
    const std::vector<std::string>& input) {
  return CCVectorOfStringToNumpyArray(input);
}

nanobind::ndarray<int32_t, nanobind::numpy, nanobind::shape<-1>>
CCVectorToNumpyArrayInt32ToInt32(const std::vector<int32_t>& input) {
  return CCVectorToNumpyArray<int32_t, int32_t>(input);
}

nanobind::ndarray<int32_t, nanobind::numpy, nanobind::shape<-1>>
CCVectorToNumpyArrayInt64ToInt32(const std::vector<int64_t>& input) {
  return CCVectorToNumpyArray<int32_t, int64_t>(input);
}

}  // namespace

// Exposes methods in python.
NB_MODULE(nanobind_util_test_ext, m) {
  m.def("TestNumpyBytesArray", ThrowIfErrorWrapper(TestNumpyBytesArray));

  m.def("CCVectorToNumpyArrayInt32ToInt32", CCVectorToNumpyArrayInt32ToInt32);
  m.def("CCVectorToNumpyArrayInt64ToInt32", CCVectorToNumpyArrayInt64ToInt32);

  m.def("ListOfBytesToVectorOfStrings",
        ValueOrThrowWrapper(ListOfBytesToVectorOfStrings));

  m.def("TestCCArrayToNumpyArray", TestCCArrayToNumpyArray);
}

}  // namespace dgf

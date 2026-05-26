#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "dgf/src/data/schema.h"
#include "dgf/src/data/schema_nb.h"
#include "dgf/src/util/nanobind_util.h"
#include "dgf/src/util/status_caster.h"
#include "dgf/src/util/util.h"

namespace nb = nanobind;

namespace dgf::data {

absl::StatusOr<std::string> ParseAndDebugPrintSchema(
    const nb::object& py_schema) {
  DGF_ASSIGN_OR_RETURN(auto schema, CreateGraphSchema(py_schema));
  return schema->to_string();
}

NB_MODULE(schema_ext, m) {
  m.def("ParseAndDebugPrintSchema",
        ValueOrThrowWrapper(ParseAndDebugPrintSchema));
}

}  // namespace dgf::data

#include "dgf/src/sampling/in_memory_sampler_nb.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "dgf/src/data/schema.h"
#include "dgf/src/sampling/in_memory_sampler.h"
#include "dgf/src/util/nanobind_util.h"
#include "dgf/src/util/util.h"

namespace dgf::sampling::in_memory_sampler {

NodeIdxs NodeIdxsToNumpyArray(const std::vector<InputIdx>& src) {
  return CCVectorToNumpyArray<int64_t, InputIdx>(src);
}

ModuleIndex::ModuleIndex()
    : in_memory_graph_mod(
          nanobind::module_::import_("dgf.src.data.in_memory_graph")),
      graph_cls(in_memory_graph_mod.attr("InMemoryGraph")),
      nodeset_cls(in_memory_graph_mod.attr("InMemoryNodeSet")),
      edgeset_cls(in_memory_graph_mod.attr("InMemoryEdgeSet")),
      key_idx_feature("#idx"),
      key_id_feature("#id") {}

absl::StatusOr<SamplingPlan> CreateSamplingPlan(
    const nb::object& py_plan, const data::GraphSchema& schema) {
  SamplingPlan plan;

  DGF_GET_ATTR_OR_RETURN(nb::object, py_root, py_plan, "root");
  DGF_GET_ATTR_OR_RETURN(bool, with_replacement, py_plan, "with_replacement");

  plan.with_replacement = with_replacement;

  std::function<absl::StatusOr<std::unique_ptr<SamplingPlan::Node>>(
      const nb::object&)>
      parse_plan = [&](const nb::object& py_node)
      -> absl::StatusOr<std::unique_ptr<SamplingPlan::Node>> {
    DGF_GET_ATTR_OR_RETURN(std::string, nodeset, py_node, "nodeset");

    auto plannode = std::make_unique<SamplingPlan::Node>();
    DGF_ASSIGN_OR_RETURN(plannode->nodeset_idx,
                         GetItem(schema.nodeset_name_to_idx, nodeset));

    DGF_GET_ATTR_OR_RETURN(nb::list, py_children, py_node, "children");
    for (const auto& py_child : py_children) {
      SamplingPlan::Edge planedge;
      DGF_GET_ATTR_OR_RETURN(std::string, edgeset, py_child, "edgeset");
      DGF_ASSIGN_OR_RETURN(planedge.edgeset_idx,
                           GetItem(schema.edgeset_name_to_idx, edgeset));

      DGF_GET_ATTR_OR_RETURN(bool, reversed, py_child, "reversed");
      planedge.reversed = reversed;

      DGF_GET_ATTR_OR_RETURN(int, hop_width, py_child, "hop_width");
      planedge.hop_width = hop_width;

      DGF_GET_ATTR_OR_RETURN(nb::object, py_node, py_child, "node");
      DGF_ASSIGN_OR_RETURN(planedge.node, parse_plan(py_node));

      plannode->children.push_back(std::move(planedge));
    }

    return plannode;
  };
  DGF_ASSIGN_OR_RETURN(plan.root, parse_plan(py_root));

  plan.ComputeStepIdx();
  return plan;
}

}  // namespace dgf::sampling::in_memory_sampler

#ifndef DGF_SRC_IO_TF_GRAPH_SAMPLE_H_
#define DGF_SRC_IO_TF_GRAPH_SAMPLE_H_

#include <string>
#include <string_view>

#include "dgf/src/data/in_memory_graph.h"
#include "dgf/src/data/tensorflow.pb.h"

namespace dgf::tf_graph_sample {

// Key naming of the TF-GNN Graph Samples. See "dgf/src/io/tf.py".
inline constexpr std::string_view kSizeKey = "#size";
inline constexpr std::string_view kSourceKey = "#source";
inline constexpr std::string_view kTargetKey = "#target";

// Key of a node feature (or of "#size").
std::string NodeKey(std::string_view node_set_name, std::string_view suffix);

// Key of an edge feature (or of "#size" / "#source" / "#target").
std::string EdgeKey(std::string_view edge_set_name, std::string_view suffix);

// Key of the row lengths of the `dim_idx`-th dimension of a feature.
std::string RaggedDimKey(std::string_view feature_key, int dim_idx);

void GraphToTfgnnExample(const dgf::data::GraphView& graph,
                         dgf::data::tensorflow::Example* example);

}  // namespace dgf::tf_graph_sample

#endif  // DGF_SRC_IO_TF_GRAPH_SAMPLE_H_

#ifndef DGF_SRC_IO_TF_GRAPH_SAMPLE_H_
#define DGF_SRC_IO_TF_GRAPH_SAMPLE_H_

#include "dgf/src/data/in_memory_graph.h"
#include "dgf/src/data/tensorflow.pb.h"

namespace dgf::tf_graph_sample {

void GraphToTfgnnExample(const dgf::data::GraphView& graph,
                         dgf::data::tensorflow::Example* example);

}  // namespace dgf::tf_graph_sample

#endif  // DGF_SRC_IO_TF_GRAPH_SAMPLE_H_

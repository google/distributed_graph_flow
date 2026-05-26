#include "dgf/src/util/tf_feature_util.h"

#include <cstdint>
#include <string>

#include "google/protobuf/repeated_field.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "dgf/src/data/tensorflow.pb.h"

namespace dgf::util::tensorflow {
template <>
google::protobuf::RepeatedField<int64_t>* GetFeatureValues<int64_t>(
    dgf::data::tensorflow::Feature* feature) {
  return feature->mutable_int64_list()->mutable_value();
}

template <>
google::protobuf::RepeatedField<float>* GetFeatureValues<float>(
    dgf::data::tensorflow::Feature* feature) {
  return feature->mutable_float_list()->mutable_value();
}

template <>
google::protobuf::RepeatedPtrField<std::string>* GetFeatureValues<std::string>(
    dgf::data::tensorflow::Feature* feature) {
  return feature->mutable_bytes_list()->mutable_value();
}
}  // namespace dgf::util::tensorflow

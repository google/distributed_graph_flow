// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Clone of tensorflow's feature_util.h library to avoid compiling all of
// tensorflow.

#ifndef DGF_SRC_UTIL_TF_FEATURE_UTIL_H_
#define DGF_SRC_UTIL_TF_FEATURE_UTIL_H_

#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>

#include "absl/strings/string_view.h"
#include "google/protobuf/repeated_field.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "dgf/src/data/tensorflow.pb.h"

namespace dgf::util::tensorflow {

// FeatureTrait map iterator types to one of [bytes, float, int64].
template <typename ValueType, class Enable = void>
struct FeatureTrait;

template <typename ValueType>
struct FeatureTrait<ValueType, typename std::enable_if<
                                   std::is_integral<ValueType>::value>::type> {
  using Type = int64_t;
};

template <typename ValueType>
struct FeatureTrait<
    ValueType,
    typename std::enable_if<std::is_floating_point<ValueType>::value>::type> {
  using Type = float;
};

template <typename T>
struct is_string
    : public std::integral_constant<
          bool,
          std::is_same<char*, typename std::decay<T>::type>::value ||
              std::is_same<const char*, typename std::decay<T>::type>::value> {
};

template <>
struct is_string<std::string> : std::true_type {};

template <>
struct is_string<absl::string_view> : std::true_type {};

template <typename ValueType>
struct FeatureTrait<
    ValueType, typename std::enable_if<is_string<ValueType>::value>::type> {
  using Type = std::string;
};

// Generic template to get mutable pointers to a float, int64 or byte field.
template <typename FeatureType>
auto GetFeatureValues(dgf::data::tensorflow::Feature* feature) ->
    typename std::enable_if<!is_string<FeatureType>::value,
                            google::protobuf::RepeatedField<FeatureType>*>::type;

template <>
google::protobuf::RepeatedField<int64_t>* GetFeatureValues<int64_t>(
    dgf::data::tensorflow::Feature* feature);

template <>
google::protobuf::RepeatedField<float>* GetFeatureValues<float>(
    dgf::data::tensorflow::Feature* feature);

template <typename FeatureType>
auto GetFeatureValues(dgf::data::tensorflow::Feature* feature) ->
    typename std::enable_if<is_string<FeatureType>::value,
                            google::protobuf::RepeatedPtrField<FeatureType>*>::type;

template <>
google::protobuf::RepeatedPtrField<std::string>* GetFeatureValues<std::string>(
    dgf::data::tensorflow::Feature* feature);

template <typename IteratorType>
void AppendFeatureValues(IteratorType first, IteratorType last,
                         dgf::data::tensorflow::Feature* feature) {
  // The real value type of the iterator.
  using ValueType = typename std::iterator_traits<IteratorType>::value_type;

  // Resolve the real type to bytes, float or int64.
  using FeatureType = typename FeatureTrait<ValueType>::Type;

  // Get the mutable feature and append
  auto& values = *GetFeatureValues<FeatureType>(feature);
  values.Reserve(values.size() + std::distance(first, last));
  for (auto it = first; it != last; ++it) {
    *values.Add() = *it;
  }
}

}  // namespace dgf::util::tensorflow

#endif  // DGF_SRC_UTIL_TF_FEATURE_UTIL_H_

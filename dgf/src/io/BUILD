load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@nanobind_bazel//:build_defs.bzl", "nanobind_extension")

package(
    
    default_visibility = [
        "//:internal_users",
        "//dgf/src:__subpackages__",
    ],
)

# Libraries
# =========

py_library(
    name = "hgraph_in_avro",
    srcs = ["hgraph_in_avro.py"],
    type_checking = True,
    deps = [
        ":feature_format",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:shard",
        # fastavro dep,
        # numpy dep,
        # tqdm dep,
    ],
)

py_library(
    name = "hgraph_in_memory",
    srcs = ["hgraph_in_memory.py"],
    deps = [
        ":feature_format",
        ":graph_in_memory",
        ":hgraph_in_avro",
        ":io_ext",
        ":tf_graph_common",
        ":tfexample",
        # apache_beam dep,
        "//dgf/src/analyse:schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        "//dgf/src/util:proto",
        "//dgf/src/util:shard",
        "//dgf/src/util:util_py",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow_gnn",
        # numpy dep,
    ],
)

py_library(
    name = "hgraph_in_beam",
    srcs = ["hgraph_in_beam.py"],
    type_checking = True,
    deps = [
        ":beam_tf_graph_common",
        ":graph_in_memory",
        ":hgraph_in_memory",
        ":tf_graph_common",
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:proto",
        "//dgf/src/util:shard",
        "//dgf/src/util/weak_dep:weak_dep_apache_beam",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow_gnn",
        # numpy dep,
    ],
)

py_library(
    name = "graph_in_memory",
    srcs = ["graph_in_memory.py"],
    type_checking = True,
    deps = [
        ":feature_format",
        ":graph_constants",
        ":io_ext",
        ":parquet",
        ":schema",
        ":tf_graph_common",
        ":tfexample",
        "//dgf/src/analyse:schema",
        "//dgf/src/data:gf_metadata",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/transform:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        "//dgf/src/util:shard",
        "//dgf/src/util:util_py",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        # numpy dep,
    ],
)

py_library(
    name = "tf_graph_sample",
    srcs = ["tf_graph_sample.py"],
    type_checking = True,
    deps = [
        ":feature_format",
        ":tf",
        ":tf_graph_sample_ext",
        # apache_beam dep,
        # bagz dep,
        # bagz/beam:bag_io dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:shard",
        "//dgf/src/util/weak_dep:weak_dep_apache_beam",
        "//dgf/src/util/weak_dep:weak_dep_bagz",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        # numpy dep,
    ],
)

# C++ library that can be used to compse extensions for working with subgraphs.
cc_library(
    name = "tf_graph_sample_cc",
    srcs = ["tf_graph_sample.cc"],
    hdrs = ["tf_graph_sample.h"],
    deps = [
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/strings",
        "//dgf/src/data:in_memory_graph_cc",
        "//dgf/src/data:tensorflow_cc_proto",
        "//dgf/src/util:tf_feature_util",
    ],
)

# Fast parsing of serialized TF-GNN Graph Samples. Does not depend on TensorFlow.
cc_library(
    name = "tf_graph_sample_parser",
    srcs = ["tf_graph_sample_parser.cc"],
    hdrs = ["tf_graph_sample_parser.h"],
    deps = [
        ":tf_graph_sample_cc",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@protobuf//:protobuf" , #:arena",
        "//dgf/src/data:schema_cc",
        "//dgf/src/data:tensorflow_cc_proto",
        "//dgf/src/util",
    ],
)

cc_test(
    name = "tf_graph_sample_parser_test",
    srcs = ["tf_graph_sample_parser_test.cc"],
    deps = [
        ":tf_graph_sample_parser",
        "@googletest//:gtest_main",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "//dgf/src/data:schema_cc",
        "//dgf/src/data:tensorflow_cc_proto",
        "//dgf/src/util:test_util_cc",
    ],
)

py_library(
    name = "statistics",
    srcs = ["statistics.py"],
    type_checking = True,
    deps = [
        # apache_beam dep,
        # dacite dep,
        "//dgf/src/data:statistics",
        "//dgf/src/util:json",
        "//dgf/src/util/weak_dep:weak_dep_apache_beam",
    ],
)

py_library(
    name = "graph_snapshots_metadata",
    srcs = ["graph_snapshots_metadata.py"],
    type_checking = True,
    deps = [
        "//dgf/src/data:graph_snapshots_metadata",
        "//dgf/src/util:filesystem",
    ],
)

py_test(
    name = "graph_snapshots_metadata_test",
    srcs = ["graph_snapshots_metadata_test.py"],
    type_checking = True,
    deps = [
        ":graph_snapshots_metadata",
        # absl/testing:absltest dep,
        "//dgf/src/data:graph_snapshots_metadata",
    ],
)

py_library(
    name = "graph_snapshots",
    srcs = ["graph_snapshots.py"],
    type_checking = True,
    deps = [
        ":graph_constants",
        ":graph_in_memory",
        ":schema",
        "//dgf/src/data:gf_metadata",
        "//dgf/src/data:graph_snapshots",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/transform:combine_graph_snapshots",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        "//dgf/src/validate:graph_snapshots",
        # numpy dep,
    ],
)

py_test(
    name = "graph_snapshots_test",
    srcs = ["graph_snapshots_test.py"],
    type_checking = True,
    deps = [
        ":graph_constants",
        ":graph_in_memory",
        ":graph_snapshots",
        ":graph_snapshots_metadata",
        ":schema",
        # absl/testing:absltest dep,
        "//dgf/src/data:graph_snapshots",
        "//dgf/src/data:graph_snapshots_metadata",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/transform:combine_graph_snapshots",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)

py_library(
    name = "schema",
    srcs = ["schema.py"],
    type_checking = True,
    deps = [
        "//dgf/src/data:schema",
        "//dgf/src/util:filesystem",
    ],
)

py_library(
    name = "spanner",
    srcs = ["spanner.py"],
    type_checking = True,
    deps = [
        # absl/logging dep,
        # apache_beam dep,
        "//dgf/src/analyse:schema",
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util/weak_dep:weak_dep_apache_beam",
        # google/cloud/spanner dep,
        # numpy dep,
    ],
)

py_library(
    name = "spanner_graph",
    srcs = ["spanner_graph.py"],
    type_checking = True,
    deps = [
        ":feature_format",
        ":io_ext",
        "//dgf/src/analyse:print_schema",
        "//dgf/src/analyse:schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io/gcp:spanner_graph_metadata",
        "//dgf/src/util:log",
        "//dgf/src/util:util_py",
        # numpy dep,
        # tqdm dep,
    ],
)

py_test(
    name = "spanner_graph_test",
    srcs = ["spanner_graph_test.py"],
    type_checking = True,
    deps = [
        ":spanner_graph",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/io/gcp:spanner_graph_metadata",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
    ],
)

py_library(
    name = "spanner_graph_metadata",
    type_checking = True,
)

py_library(
    name = "spanner_graph_in_memory",
    type_checking = True,
)

py_library(
    name = "spanner_graph_utils",
    type_checking = True,
)

py_library(
    name = "tfexample",
    srcs = ["tfexample.py"],
    type_checking = True,
    deps = [
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        # numpy dep,
        # tqdm dep,
    ],
)

py_library(
    name = "tf_graph_common",
    srcs = ["tf_graph_common.py"],
    type_checking = True,
    deps = [
        "//dgf/src/data:schema",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        # numpy dep,
    ],
)

py_library(
    name = "sparse_deferred",
    srcs = ["sparse_deferred.py"],
    type_checking = True,
    deps = [
        # sparse_deferred",
        # sparse_deferred/structs:graph_struct",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:schema",
        # jax dep,
        # numpy dep,
    ],
)

py_library(
    name = "jax",
    srcs = ["jax.py"],
    type_checking = True,
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:tf_in_memory_graph",
        # jax dep,
        # numpy dep,
    ],
)

py_library(
    name = "pyg",
    srcs = ["pyg.py"],
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util/weak_dep:weak_dep_base",
        # numpy dep,
    ],
)

py_library(
    name = "tf",
    srcs = ["tf.py"],
    type_checking = True,
    deps = [
        ":feature_format",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/data:tf_in_memory_graph",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
    ],
)

py_library(
    name = "dataset_loader",
    srcs = ["dataset_loader.py"],
    type_checking = True,
    deps = [
        ":cache",
        ":feature_format",
        ":graph_in_memory",
        # absl/logging dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util/weak_dep:weak_dep_ogb",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        # numpy dep,
        # pandas dep,
        # yaml dep,
        #  # ogb dep should not be a dep.
    ],
)

py_library(
    name = "dataset_loader_traffic",
    srcs = ["dataset_loader_traffic.py"],
    type_checking = True,
    deps = [
        ":cache",
        ":dataset_loader",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        # numpy dep,
        # pandas dep,
    ],
)

py_library(
    name = "dataset_loader_relbench",
    srcs = ["dataset_loader_relbench.py"],
    type_checking = True,
    deps = [
        ":cache",
        ":dataset_loader",
        "//dgf/src/analyse:text_detection",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        # numpy dep,
        # pandas dep,
    ],
)

py_library(
    name = "cache",
    srcs = ["cache.py"],
    type_checking = True,
    deps = [
        "//dgf/src/util:filesystem",
    ],
)

py_library(
    name = "graph_in_beam",
    srcs = ["graph_in_beam.py"],
    type_checking = True,
    deps = [
        ":beam_tf_graph_common",
        ":feature_format",
        ":graph_constants",
        ":graph_in_memory",
        ":hgraph_in_memory",
        ":schema",
        ":tf_graph_common",
        "//dgf/src/analyse:schema",
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:gf_metadata",
        "//dgf/src/data:schema",
        "//dgf/src/transform:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:shard",
        "//dgf/src/util/weak_dep:weak_dep_apache_beam",
        # numpy dep,
        # pyarrow:pyarrow_modules dep,
    ],
)

py_library(
    name = "feature_format",
    srcs = ["feature_format.py"],
    type_checking = True,
    deps = [
        "//dgf/src/data:schema",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        # numpy dep,
        # pyarrow:pyarrow_modules dep,
    ],
)

py_library(
    name = "parquet",
    srcs = ["parquet.py"],
    type_checking = True,
    deps = [
        ":feature_format",
        "//dgf/src/data:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        "//dgf/src/util:shard",
        # numba dep,
        # numpy dep,
        # pyarrow:pyarrow_modules dep,
        # pyarrow:pyarrow_pyx dep,
    ],
)

py_library(
    name = "networkx",
    srcs = ["networkx.py"],
    type_checking = True,
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        # networkx dep,
        # numpy dep,
    ],
)

py_library(
    name = "graph_constants",
    srcs = ["graph_constants.py"],
    type_checking = True,
    deps = [
    ],
)

# Nanobind
# =========

nanobind_extension(
    name = "io_ext",
    srcs = ["io_ext.cc"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "//dgf/src/util",
        "//dgf/src/util:concurrency",
        "//dgf/src/util:nanobind_util",
        "//dgf/src/util:status_caster",
    ],
)

nanobind_extension(
    name = "tf_graph_sample_ext",
    srcs = ["tf_graph_sample_ext.cc"],
    # Avoid the dynamic library ABI symbol collision between protobuf compiled in TensorFlow and
    # protobuff compiled by this rule. On macOS, `-Bsymbolic` is an ELF-only flag, so we restrict
    # the exported symbol list to `_PyInit_tf_graph_sample_ext`.
    linkopts = select({"@platforms//os:linux": ["-Wl,-Bsymbolic"], "//conditions:default": ["-Wl,-exported_symbol,_PyInit_tf_graph_sample_ext"]}),
    deps = [
        ":tf_graph_sample_cc",
        ":tf_graph_sample_parser",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "//dgf/src/data:in_memory_graph_cc",
        "//dgf/src/data:in_memory_graph_nb",
        "//dgf/src/data:schema_cc",
        "//dgf/src/data:schema_nb",
        "//dgf/src/util",
        "//dgf/src/util:concurrency",
        "//dgf/src/util:nanobind_util",
        "//dgf/src/util:status_caster",
        "@numpy_headers//:headers",
    ],
)
# Test
# =========

py_test(
    name = "hgraph_in_avro_test",
    srcs = ["hgraph_in_avro_test.py"],
    type_checking = True,
    deps = [
        ":hgraph_in_avro",
        # absl/testing:absltest dep,
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        # fastavro dep,
        # numpy dep,
    ],
)

py_test(
    name = "hgraph_in_memory_test",
    srcs = ["hgraph_in_memory_test.py"],
    type_checking = True,
    deps = [
        ":hgraph_in_memory",
        ":tfexample",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:proto",
        "//dgf/src/util:test_util",
        # google/protobuf:use_fast_cpp_protos dep,
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
        # tensorflow_gnn/proto dep,
    ],
)

py_test(
    name = "hgraph_in_beam_test",
    srcs = ["hgraph_in_beam_test.py"],
    type_checking = True,
    deps = [
        ":hgraph_in_beam",
        ":tfexample",
        # absl/testing:absltest dep,
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # google/protobuf:use_fast_cpp_protos dep,
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "graph_in_memory_test",
    srcs = ["graph_in_memory_test.py"],
    type_checking = True,
    deps = [
        ":graph_in_memory",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:gf_metadata",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/transform:canonicalize",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
    ],
)

py_test(
    name = "tf_graph_sample_test",
    srcs = ["tf_graph_sample_test.py"],
    shard_count = 4,
    type_checking = True,
    deps = [
        ":feature_format",
        ":tf_graph_sample",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        "//dgf/src/util/weak_dep:weak_dep_bagz",
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "spanner_test",
    srcs = ["spanner_test.py"],
    deps = [
        ":spanner",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
        "//dgf/src/util:gen_test_graph",
        # google/cloud/spanner dep,
        # google/cloud/spanner_v1 dep,
    ],
)

py_test(
    name = "tfexample_test",
    srcs = ["tfexample_test.py"],
    type_checking = True,
    deps = [
        ":tfexample",
        # absl/testing:absltest dep,
        # google/protobuf:use_fast_cpp_protos dep,
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "sparse_deferred_test",
    srcs = ["sparse_deferred_test.py"],
    type_checking = True,
    deps = [
        ":jax",
        ":sparse_deferred",
        # sparse_deferred/structs:graph_struct",
        # absl/testing:absltest dep,
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # jax dep,
        # numpy dep,
    ],
)

py_test(
    name = "jax_test",
    srcs = ["jax_test.py"],
    type_checking = True,
    deps = [
        ":jax",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:tf_in_memory_graph",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # jax dep,
        # numpy dep,
    ],
)

py_test(
    name = "tf_test",
    srcs = ["tf_test.py"],
    type_checking = True,
    deps = [
        ":tf",
        ":tf_graph_sample",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/data:tf_in_memory_graph",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "cache_test",
    srcs = ["cache_test.py"],
    type_checking = True,
    deps = [
        ":cache",
        # absl/testing:absltest dep,
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
    ],
)

py_test(
    name = "dataset_loader_test",
    srcs = ["dataset_loader_test.py"],
    type_checking = True,
    deps = [
        ":dataset_loader",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/analyse:print_schema",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
        # pandas dep,
    ],
)

py_test(
    name = "dataset_loader_traffic_test",
    srcs = ["dataset_loader_traffic_test.py"],
    type_checking = True,
    deps = [
        ":dataset_loader",
        ":dataset_loader_traffic",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/analyse:print_schema",
        "//dgf/src/util:log",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
        # pandas dep,
    ],
)

py_test(
    name = "dataset_loader_relbench_test",
    srcs = ["dataset_loader_relbench_test.py"],
    type_checking = True,
    deps = [
        ":dataset_loader",
        ":dataset_loader_relbench",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/analyse:print_schema",
        "//dgf/src/data:schema",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
        # pandas dep,
    ],
)

py_test(
    name = "graph_in_beam_test",
    srcs = ["graph_in_beam_test.py"],
    shard_count = 4,
    type_checking = True,
    deps = [
        ":graph_in_beam",
        ":graph_in_memory",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
        "//dgf/src/beam:runners",
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:gf_metadata",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/transform:canonicalize",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)

py_test(
    name = "parquet_test",
    srcs = ["parquet_test.py"],
    type_checking = True,
    deps = [
        ":parquet",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:shard",
        "//dgf/src/util:test_util",
        # numpy dep,
        # pyarrow:pyarrow_modules dep,
    ],
)

py_test(
    name = "io_cc_test",
    srcs = ["io_cc_test.py"],
    type_checking = True,
    deps = [
        ":io_ext",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)

py_test(
    name = "tf_graph_sample_ext_test",
    srcs = ["tf_graph_sample_ext_test.py"],
    deps = [
        ":tf_graph_sample",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "networkx_test",
    srcs = ["networkx_test.py"],
    deps = [
        ":networkx",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # networkx dep,
        # numpy dep,
    ],
)

py_test(
    name = "statistics_test",
    srcs = ["statistics_test.py"],
    type_checking = True,
    deps = [
        ":statistics",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:histogram",
        "//dgf/src/data:statistics",
        "//dgf/src/util:test_util",
    ],
)

py_library(
    name = "beam_tf_graph_common",
    srcs = ["beam_tf_graph_common.py"],
    type_checking = True,
    deps = [
        ":tf_graph_common",
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util/weak_dep:weak_dep_apache_beam",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        # numpy dep,
    ],
)

py_test(
    name = "tf_graph_common_test",
    srcs = ["tf_graph_common_test.py"],
    deps = [
        ":tf_graph_common",
        # absl/testing:absltest dep,
        "//dgf/src/data:schema",
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "beam_tf_graph_common_test",
    srcs = ["beam_tf_graph_common_test.py"],
    type_checking = True,
    deps = [
        ":beam_tf_graph_common",
        ":tf_graph_common",
        # absl/testing:absltest dep,
        # apache_beam dep,
        "//dgf/src/beam:runners",
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util/weak_dep:weak_dep_apache_beam",
        "//dgf/src/util/weak_dep:weak_dep_tensorflow",
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

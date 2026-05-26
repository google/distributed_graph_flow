load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
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
        ":hgraph_in_avro",
        ":tfrecord",
        # apache_beam dep,
        "//dgf/src/analyse:schema",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:proto",
        "//dgf/src/util:shard",
        # numpy dep,
        # tensorflow dep,
        # tensorflow_gnn/proto dep,
    ],
)

py_library(
    name = "hgraph_in_beam",
    srcs = ["hgraph_in_beam.py"],
    deps = [
        ":hgraph_in_memory",
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:proto",
        "//dgf/src/util:shard",
        # numpy dep,
        # tensorflow dep,
        # tensorflow_gnn/proto dep,
    ],
)

py_library(
    name = "graph_in_memory",
    srcs = ["graph_in_memory.py"],
    deps = [
        ":graph_constants",
        ":io_ext",
        ":parquet",
        ":schema",
        "//dgf/src/analyse:schema",
        "//dgf/src/data:gf_metadata",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/transform:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        "//dgf/src/util:shard",
        "//dgf/src/util:util_py",
        # numpy dep,
    ],
)

py_library(
    name = "tf_graph_sample",
    srcs = ["tf_graph_sample.py"],
    deps = [
        ":feature_format",
        ":tf_graph_sample_ext",
        # apache_beam dep,
        # bagz dep,
        # bagz/beam:bag_io dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:shard",
        # numpy dep,
        # tensorflow dep,
    ],
)

# C++ library that can be used to compse extensions for working with subgraphs.
cc_library(
    name = "tf_graph_sample_cc",
    srcs = ["tf_graph_sample.cc"],
    hdrs = ["tf_graph_sample.h"],
    deps = [
        "@abseil-cpp//absl/strings",
        "//dgf/src/data:in_memory_graph_cc",
        "//dgf/src/data:tensorflow_cc_proto",
        "//dgf/src/util:tf_feature_util",
    ],
)

py_library(
    name = "statistics",
    srcs = ["statistics.py"],
    deps = [
        # apache_beam dep,
        # dacite dep,
        "//dgf/src/data:statistics",
        "//dgf/src/util:json",
    ],
)

py_library(
    name = "schema",
    srcs = ["schema.py"],
    deps = [
        "//dgf/src/data:schema",
        "//dgf/src/util:filesystem",
    ],
)

py_library(
    name = "spanner",
    srcs = ["spanner.py"],
    deps = [
        # absl/logging dep,
        # apache_beam dep,
        "//dgf/src/analyse:schema",
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        # google/cloud/spanner dep,
        # numpy dep,
    ],
)

py_library(name = "spanner_graph")

py_library(name = "spanner_graph_metadata")

py_library(name = "spanner_graph_in_memory")

py_library(name = "spanner_graph_utils")

py_library(
    name = "tfrecord",
    srcs = ["tfrecord.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "sparse_deferred",
    srcs = ["sparse_deferred.py"],
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
    deps = [
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:jax_in_memory_graph",
        "//dgf/src/data:tf_in_memory_graph",
        # jax dep,
        # numpy dep,
    ],
)

py_library(
    name = "tf",
    srcs = ["tf.py"],
    deps = [
        ":feature_format",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:padding",
        "//dgf/src/data:schema",
        "//dgf/src/data:tf_in_memory_graph",
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "dataset_loader",
    srcs = ["dataset_loader.py"],
    deps = [
        ":cache",
        ":feature_format",
        ":graph_in_memory",
        # absl/logging dep,
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:log",
        # numpy dep,
        # pandas dep,
        # yaml dep,
        #  # ogb dep should not be a dep.
    ],
)

py_library(
    name = "cache",
    srcs = ["cache.py"],
    deps = [
        "//dgf/src/util:filesystem",
    ],
)

py_library(
    name = "graph_in_beam",
    srcs = ["graph_in_beam.py"],
    deps = [
        ":feature_format",
        ":graph_constants",
        ":schema",
        # apache_beam dep,
        "//dgf/src/analyse:schema",
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:gf_metadata",
        "//dgf/src/data:schema",
        "//dgf/src/transform:schema",
        "//dgf/src/util:filesystem",
        "//dgf/src/util:shard",
        # numpy dep,
        # pyarrow:pyarrow_modules dep,
    ],
)

py_library(
    name = "feature_format",
    srcs = ["feature_format.py"],
    deps = [
        "//dgf/src/data:schema",
        # numpy dep,
        # pyarrow:pyarrow_modules dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "parquet",
    srcs = ["parquet.py"],
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
    # protobuff compiled by this rule.
    linkopts = ["-Wl,-Bsymbolic"],
    deps = [
        ":tf_graph_sample_cc",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "//dgf/src/data:in_memory_graph_cc",
        "//dgf/src/data:in_memory_graph_nb",
        "//dgf/src/util",
        "//dgf/src/util:concurrency",
        "//dgf/src/util:nanobind_util",
        "//dgf/src/util:status_caster",
    ],
)
# Test
# =========

py_test(
    name = "hgraph_in_avro_test",
    srcs = ["hgraph_in_avro_test.py"],
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
    deps = [
        ":hgraph_in_memory",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:proto",
        "//dgf/src/util:test_util",
        # tensorflow_gnn/proto dep,
    ],
)

py_test(
    name = "hgraph_in_beam_test",
    srcs = ["hgraph_in_beam_test.py"],
    deps = [
        ":hgraph_in_beam",
        # absl/testing:absltest dep,
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)

py_test(
    name = "graph_in_memory_test",
    srcs = ["graph_in_memory_test.py"],
    deps = [
        ":graph_in_memory",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:in_memory_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
    ],
)

py_test(
    name = "tf_graph_sample_test",
    srcs = ["tf_graph_sample_test.py"],
    shard_count = 4,
    deps = [
        ":tf_graph_sample",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
        # bagz/beam:bag_io dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # tensorflow dep,
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
        # google/cloud/spanner_v1 dep,
    ],
)

py_test(
    name = "tfrecord_test",
    srcs = ["tfrecord_test.py"],
    deps = [
        ":tfrecord",
        # absl/testing:absltest dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "sparse_deferred_test",
    srcs = ["sparse_deferred_test.py"],
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
    deps = [
        ":tf",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/data:schema",
        "//dgf/src/data:tf_in_memory_graph",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "cache_test",
    srcs = ["cache_test.py"],
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
    deps = [
        ":dataset_loader",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "//dgf/src/analyse:print_schema",
        "//dgf/src/util:test_util",
        "//dgf/src/validate:in_memory_graph",
        # numpy dep,
    ],
)

py_test(
    name = "graph_in_beam_test",
    srcs = ["graph_in_beam_test.py"],
    deps = [
        ":graph_in_beam",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # apache_beam dep,
        "//dgf/src/data:distributed_graph",
        "//dgf/src/data:schema",
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
        # numpy dep,
    ],
)

py_test(
    name = "parquet_test",
    srcs = ["parquet_test.py"],
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
        "//dgf/src/util:gen_test_graph",
        "//dgf/src/util:test_util",
    ],
)

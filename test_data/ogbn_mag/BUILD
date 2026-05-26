package(
    
    default_visibility = ["//:users"],
    licenses = ["notice"],
)

exports_files([
    "ogbn_mag_sample_1.tfrecord",
    "ogbn_mag_sample_2.tfrecord",
    "ogbn_mag_schema.pbtxt",
])

filegroup(
    name = "test_data",
    srcs = glob(["**"]),
)

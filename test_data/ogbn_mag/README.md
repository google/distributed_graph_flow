# Datasets for unit testing

# Test Data for DGF Examples

This directory contains test data used by the example scripts in the parent
directory (`//third_party/py/dgf/examples/e2e_ogbn_mag`), particularly for the OGBN-MAG
dataset example.

## Files

*   `obgn_mag_schema.pbtxt`: The graph schema for the OGBN-MAG dataset, formatted
    as a text-serialized `tensorflow_gnn.GraphSchema` proto.
*   `obgn_mag_sample_1.textproto`: A text proto file containing a single
    `tensorflow.Example` representing a subgraph sample from OGBN-MAG.
*   `obgn_mag_sample_1.tfrecord`: The TFRecord version of
    `obgn_mag_sample_1.textproto`.
*   `obgn_mag_sample_2.textproto`: A text proto file containing three
    `tensorflow.Example` messages representing subgraph samples from OGBN-MAG,
    separated by blank lines.
*   `obgn_mag_sample_2.tfrecord`: The TFRecord version of
    `obgn_mag_sample_2.textproto`.

## Generating TFRecord Files

The `.tfrecord` files are generated from the corresponding `.textproto` files.
The text proto files contain `tensorflow.Example` messages in a human-readable
format. To convert them to the binary TFRecord format used by the training
pipeline, the following `gqui` command can be used:

**Example for a single sample:**

```bash
gqui from textproto:third_party/py/dgf/test_data/ogbn_mag/obgn_mag_sample_1.textproto \
     proto tensorflow.Example \
     --outfile=tfrecords:third_party/py/dgf/test_data/ogbn_mag/obgn_mag_sample_1.tfrecord

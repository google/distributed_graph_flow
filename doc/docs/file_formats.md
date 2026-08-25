# Graph File Format

This document presents the various on-disk file formats supported by GF. For the
in-memory graph formats, read [this doc instead](graph_formats.md).

## GF Graph

The GF Graph format is the primary and most efficient way to serialize small and
large graphs. The format is efficient, compatible with open-source / cloud,
suited for distributed reading & writing, and easy to create manually or with
other tools.

### Usage

Reading and writing a GF Graph in memory is done respectively with
`dgf.io.read_graph` and `dgf.io.write_graph`.

```python
graph, schema = dgf.io.read_graph("/path/to/ogb_arxiv")

dgf.io.write_graph(graph, schema, path="/tmp/my_graph_parquet")

# Write a graph using TF_RECORD container. By default, DGF uses Parquet files.
dgf.io.write_graph(
    graph, schema, path="/tmp/my_graph_tfrecord", container="TF_RECORD")
```

Reading and writing a GF Graph as a Beam distributed graph is done respectively
with `dgf.beam.io.read_graph` and `dgf.beam.io.write_graph`.

### Directory structure

A GF graph is structured as follows:

-   `metadata.json`: A JSON specifying details about the graph encoding:
    -   `version`: Version of the format as an integer.
    -   `container`: Container format used to store node and edge data:
        `"PARQUET"` (default) or `"TF_RECORD"`.
    -   `timestamp`: Optional integer creation timestamp.
    -   Optionally, contains the number of nodes and edges in each
        nodesets/edgesets.
-   `schema.json`: The `dgf.data.GraphSchema` graph schema in JSON.
    -   The nodesets are required to have a feature of type bytes or
        int (int32, int64) with semantic `PRIMARY_KEY`.
    -   The edgesets can optionally have a feature of type bytes or
        int (int32, int64) with semantic `PRIMARY_KEY`. This specifies if the
        edges have ids.
-   `nodesets` directory:
    -   `<nodeset name>@*.<ext>`: Feature values for each nodeset stored as
        sharded files (`.parquet` for `PARQUET` or `.tfrecord.gz` for
        `TF_RECORD`). The primary key column (e.g., `#id`) specifies the ID of
        the node.
-   `edgesets` directory:
    -   `<edgeset name>@*.<ext>`: Source and target node ID, and feature
        values, of each edgeset, stored as sharded files (`.parquet` for
        `PARQUET` or `.tfrecord.gz` for `TF_RECORD`). The optional `#id` column
        specifies the ID of the edge. The ID of the source and target nodes are
        stored in the `#source` and `#target` features.

### Supported Containers

GF Graph supports the following container formats:

#### Parquet (`PARQUET`)

Parquet has become the de facto format for tabular data. It provides efficient
columnar storage and compression. Many internal and external tools can produce
parquet files natively (e.g., `EXPORT DATA` in Google SQL / BigQuery with
`format = 'PARQUET'`, or Beam pipelines). You can use `parquet-tools` to
inspect parquet files directly.

Files are stored with the `.parquet` extension.

#### TFRecord (`TF_RECORD`)

TFRecord containers store serialized `tf.train.Example` protos compressed with
GZIP (`.tfrecord.gz` extension). Each record corresponds to a node or an edge.
This format is well-suited for integration with TensorFlow and TF-GNN pipelines.

## Graph AI HGraph

A Graph AI HGraph (or just HGraph) is another directory-based format for
defining heterogeneous graphs.

The HGraph format is similar to the GF format, with two major differences:

-   Nodes and edges are stored as `tf.train.Example` or other proprietary
    protos (instead of parquet files).
-   The protos are encapsulated within SSTables, RecordIO or TF Records.
-   The schema is defined by a TF-GNN schema proto (instead of a GF JSON
    Schema). This schema is more lenient (e.g., ID features are optional) and
    does not carry semantic information.

**Recommendation**

For efficiency reasons (up to 100x speed difference), Cloud / open-source
compatibility, and ease of creation / consumption, using the GF format is always
recommended over the Graph AI HGraph format.

### Usage

Reading and writing a Graph AI HGraph in memory is done respectively with
`dgf.io.read_graphai_hgraph` and `dgf.io.write_graphai_hgraph`.

Reading and writing a Graph AI HGraph as a Beam distributed graph is done respectively
with `dgf.beam.io.read_graphai_hgraph` and `dgf.beam.io.write_graphai_hgraph`.

Combining `dgf.io.read_graphai_hgraph` with `dgf.io.write_graph` allows you to
convert small (e.g. <100 edges) HGraph into GF graphs.

The
"convert_hgraph_to_gf_graph" example 
Beam CLI allows you to convert large HGraph to GF Graph.

## TF Graph Samples

A TF Graph Samples file defines one or several small heterogeneous graphs, where
each graph fits into memory of a single computer. TF Graph Samples files are well
suited to represent a large amount of small graphs e.g., graph samples.

A TF Graph Samples file (or sharded file) is a TensorFlow Record file containing
serialized `tensorflow.Example` protos, following the conventions of TF GNN
Graph.

Note: Unlike for Graph AI HGraph, the use of `tf.train.Example` in TF Graph
Samples is mostly efficient.

### Usage

TF Graph Samples can be read/written in memory using `dgf.io.read_tfgnn_graphs` and
`dgf.io.write_tfgnn_graphs`.

TF Graph Samples can be read/written in Beam using
`dgf.beam.io.read_tfgnn_graphs` and `dgf.beam.io.write_tfgnn_graphs`.

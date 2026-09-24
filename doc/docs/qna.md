# Q&A

## What are Graph Neural Networks?

Graph Neural Networks (GNNs) are a class of machine learning methods designed to
perform inference on data described by graphs, or on relational data in general.

## Is Graph Flow right for my problem?

If your data contains multiple interconnected tables, entities, transactions, or
time-series events, standard tabular ML (e.g., gradient boosted trees) ignores
the graph topology, while LLMs can be expensive and ungrounded. Graph Flow is
well suited for node classification (e.g., fraud or abuse detection), node
regression (e.g., failure prediction, churn), and link prediction (e.g.,
recommendation, entity resolution).

## What about temporal data?

While academic GNNs mostly focus on static graphs, Graph Flow extends GNNs to
full time-aware modeling. For example, Graph Flow supports dynamic graphs,
time-series features, and time-aware training and sampling (to avoid future
leakage).

## Should I use Graph Flow or TensorFlow GNN?

**You should use Graph Flow.** Both products are developed by the same team.
Graph Flow is a significant rewrite and improvement of TF-GNN for JAX, with
fundamental usability, efficiency, and quality improvements. It offers a
Simple API that makes GNNs accessible to both experts and ML novices, and an
Advanced API of composable building blocks for researchers.

TensorFlow GNN is deprecated, and Graph Flow should be used for all new
projects. Graph Flow can read and write TF-GNN formats (graphs and graph
samples), and can export models as TensorFlow SavedModels, so migrating
pipelines is smooth.

## What are the supported graph file formats?

Graph Flow supports several file formats for reading and writing graphs:

1.  **GF Graph (recommended)**: The primary and most efficient format. It uses a
    directory with `metadata.json`, `schema.json`, and Parquet files for the
    nodesets and edgesets. Parquet is efficient and compatible with most cloud
    and open-source tools. See the [Graph file format](file_formats.md) guide.
2.  **TF Graph Samples**: TFRecord files containing serialized
    `tensorflow.Example` protos following the TF-GNN conventions. This format is
    well suited to represent a large collection of small graph samples.

Graph Flow can also read graphs from Spanner Graph, BigQuery Graph, and
NetworkX.

## What are the supported in-memory graph formats?

Graph Flow provides several in-memory representations in the `dgf.data` module:

*   `dgf.data.InMemoryGraph`: The primary representation, storing data in NumPy
    arrays. If your graph is small (less than 1B edges), you can directly create
    it in this format.
*   `dgf.data.JaxInMemoryGraph`: Stores data in JAX arrays, supporting JAX
    compilation (JIT) for message-passing operations.
*   `dgf.data.TFInMemoryGraph`: Stores data in TensorFlow tensors, used for
    SavedModel serialization.
*   `dgf.beam.data.Graph`: A distributed representation using Apache Beam for
    large-scale graph processing.

See the [Graph in-memory objects](graph_formats.md) guide for details.

## How does Graph Flow handle large-scale graphs?

Graphs with up to ~1B edges can be sampled in memory, during training, on a
single machine.

For larger graphs, it is more efficient to compute graph samples before
training rather than during training. Once computed, the samples can directly
be consumed by the Simple API with
`graph_format="PATH_TF_SAMPLE_TF_RECORD"`. Graph Flow provides two tools to
compute these samples:

*   **Offline distributed sampler**: Distributes both the compute and the graph
    topology, scaling to trillions of edges. It runs as an Apache Beam pipeline
    on Google Cloud Dataflow. See the
    [Offline Distributed Sampler (GCP)](tutorial/gcp_offline_distributed_sampler.ipynb)
    tutorial.
*   **Offline semi-distributed sampler**: Runs the in-process sampling algorithm
    on the graph topology (in parallel across workers) with distributed feature
    aggregation. Scales up to ~100B edges. Implemented with Apache Beam (Python)
    and C++. See the
    [example](https://github.com/google/distributed_graph_flow/blob/main/examples/create_graph_samples_semi_distributed_v2.py).

For online inference, Graph Flow can sample graphs directly from Spanner Graph.

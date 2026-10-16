# Q&A

## What are Graph Neural Networks?

Graph Neural Networks (GNNs) are deep learning models that consume and make
predictions on graph and relational data (e.g., multiple tables of users,
accounts, and transactions connected by keys).

Unlike classic tabular models, GNNs consume relational data natively. There is
no need for lossy and hard-to-maintain feature engineering, such as manually
flattening tables with joins and aggregations.

GNNs are widely used for abuse, spam, and fraud detection, as well as in many
other domains. For example, they power weather forecasting (GraphCast) and
protein structure prediction (AlphaFold 2).

## Why use Graph Flow with coding agents? { #coding-agents }

Graph Flow is built for both engineers and coding agents (e.g., Gemini CLI,
Claude Code, Cursor).

Without Graph Flow, a coding agent asked to "train a GNN on my data" typically
generates thousands of lines of custom code (data loading, sampling, feature
normalization, model architecture, training loop, export). This code:

-   has to be reviewed and maintained by your team,
-   is often inefficient (e.g., naive sampling, non-optimized training loops),
-   can contain subtle bugs (e.g., future-data leakage, wrong normalization)
    that silently degrade model quality.

With Graph Flow, the agent writes ~10 lines of code calling a tested and
benchmarked API instead:

-   **Minimal code to maintain:** Your product owns a few lines of code, not a
    custom ML stack.
-   **Efficient:** Sampling, training, and serving are optimized and
    benchmarked by the Graph Flow team.
-   **Correct:** The code does what it is supposed to do. Graph Flow is unit
    tested and continuously evaluated on quality benchmarks.
-   **Free improvements:** New models, speedups, and fixes come with library
    updates, without re-generating code.

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

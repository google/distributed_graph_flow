# Changelog

## 0.2.0

### Features

- Add macOS (Apple Silicon / `arm64`, macOS 12+) pip wheels for `dgf` and `dgf-nightly`.
- Add `dgf-nightly` package published daily to PyPI.
- Add `dgf.learning.train_node_embedding_model` and `NodeEmbeddingModel` (GraphMAE self-supervised node embeddings).
- Add GraphSAGE and GCN architectures (`architecture="graphsage"` and `architecture="gcn"`).
- Add `model.analyze()` for feature importance and structural sensitivity analysis.
- Add `dgf.sampling.BigQueryGraphSampler` and `dgf.sampling.create_bigquery_view_from_spanner` for BigQuery and Spanner Graph sampling.
- Add `dgf.transform.canonicalize_graph` and `dgf.transform.perturb`.
- Add TensorFlow Hub, TensorFlow Text, and Keras Hub text embedding normalizers.
- Speed up in-memory subgraph sampling, feature gathering, and GATv2 backward pass.

## 0.0.4

### Features

- Speed-up in-process graph neighbor sampler by ~4x.
- Add support for Graph Transformer layers `dgf.jax.layers.HeterogeneousGraphTransformerConfig`. Enable them in the high-level API with `architecture="heterogeneous_graph_transformer",`.
- Show the model architecture in the `model.describe()` widget.
- Make the TG-GNN dependency optional (weak dependency).
- Speed-up semi-distributed graph neighbor sampler.

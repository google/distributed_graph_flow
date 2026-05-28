# Changelog

## Head

### Features

- Speed-up in-process graph neighbor sampler by ~4x.
- Add support for Graph Transformer layers `dgf.jax.layers.HeterogeneousGraphTransformerConfig`. Enable them in the high-level API with `architecture="heterogeneous_graph_transformer",`.
- Show the model architecture in the `model.describe()` widget.
- Make the TG-GNN dependency optional (weak dependency).

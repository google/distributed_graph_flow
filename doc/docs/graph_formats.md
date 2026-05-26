# In-Memory Graph Format

This document describes the various in-memory graph formats supported by Graph
Flow (GF). For information on disk-based graph formats, please refer to
[file_formats.md](file_formats.md).

These objects are available in the `dgf.data` module. Conversion functions
between these formats are located in the `dgf.convert` module.

## `dgf.data.InMemoryGraph`

Represents a graph stored as a dataclass containing NumPy arrays. This is the
primary in-memory graph representation in GF.

## `dgf.data.JaxInMemoryGraph`

Represents a graph stored as a dataclass of JAX arrays, mirroring the structure
of `InMemoryGraph`. It is augmented with JAX annotations to support JAX
compilation (JIT).

This format is typically used for JAX-implemented GNN message-passing
operations.

## `dgf.data.TFInMemoryGraph`

Represents a graph stored as a TensorFlow `ExtensionType` containing TensorFlow
tensors, following the same structure as `InMemoryGraph`.

It is primarily used to serialize GNN data normalization into a TensorFlow
SavedModel, as JAX does not support string-based features.

## `dgf.data.TFInMemoryGraphDict`

A flattened dict of `TFInMemoryGraph` with string keys and Tensor values, using
double underscores (`__`) as delimiters. Used for integration with systems like
Vertex AI.

*   **Nodes**: `nodes__{nodeset}__reserved_size` (scalar int32) and
    `nodes__{nodeset}__{feat}`.
*   **Edges**: `edges__{edgeset}__reserved_adjacency` (`[2, None]` int64) and
    `edges__{edgeset}__{feat}`.

*Note: `#` in feature names is replaced with `_hash_` (e.g., `#feat` ->
`_hash_feat`).*

## `dgf.beam.data.Graph`

A distributed representation of `InMemoryGraph` using Apache Beam, designed for
processing graphs that exceed the memory capacity of a single machine.

## Sparse Deferred Struct

The graph format used by the Sparse Deferred library. Unlike GF, which utilizes
distinct objects for different backends (`dgf.data.InMemoryGraph` (Numpy),
`dgf.data.JaxInMemoryGraph`, `dgf.data.TFInMemoryGraph`), a Sparse Deferred
Struct can encapsulate NumPy, TensorFlow, or JAX arrays.

It is used to leverage existing Sparse Deferred GNN message-passing
implementations.

## TF GNN graphs stored as `tf.train.Example`

A `tf.train.Example` protocol buffer containing graph data. This format
represents a flattened dictionary of tensors, similar to `TFInMemoryGraphDict`.
However, unlike `TFInMemoryGraphDict`, which supports tensors of arbitrary
shapes (typically `[num_nodes, num_dimensions]`), this format only supports
one-dimensional arrays. The original shape information is stored in separate
fields and must be used to reshape the arrays prior to use.

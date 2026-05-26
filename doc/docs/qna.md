# Q&A

## What are Graph Neural Networks?

Graph Neural Networks (GNNs) are a class of machine learning methods designed to perform inference on data described by graphs, or on relational data in general.

## Should I use Graph Flow or TensorFlow GNN?

**You should use Graph Flow:** Graph Flow (GF) is the recommended toolkit from the Google GNN team for developing and deploying GNN models.

Graph Flow is designed to simplify GNN development. It is JAX-first but library-agnostic, offering high-level APIs that make GNNs accessible to both experts and ML novices. If you have legacy TF-GNN data or models, Graph Flow provides converters to import/export TF-GNN formats, ensuring a smooth transition.

## How does Graph Flow handle large-scale graphs?

For graphs that exceed the memory of a single machine, Graph Flow provides:

*   **Semi-distributed sampling**: Using Apache Beam, the graph topology is loaded into memory, and feature aggregation is distributed via a MapReduce-like pipeline. Allows scaling up to 100B edges.
*   **Distributed sampling**: Not yet available in public package.

# Distributed Graph Flow

<p align="center">
<img src="https://raw.githubusercontent.com/google/distributed_graph_flow/main/doc/docs/image/logo.png" width="300" />
</p>

<p align="center">
<a href="https://pypi.org/project/dgf/"><img src="https://img.shields.io/pypi/v/dgf" alt="PyPI"></a>
<a href="https://pypi.org/project/dgf/"><img src="https://img.shields.io/pypi/pyversions/dgf" alt="Python"></a>
<a href="https://github.com/google/distributed_graph_flow/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
<a href="https://dgf.readthedocs.io/"><img src="https://img.shields.io/badge/docs-dgf.readthedocs.io-blue" alt="Documentation"></a>
</p>

**Graph Flow** (DGF) is an open-source Python library to train, evaluate, and
deploy Graph Neural Networks (**GNNs**) on tabular, relational, and temporal
data. It is developed by the Google GNN team, the team behind
[TensorFlow GNN](https://github.com/tensorflow/gnn), and is its successor.

Graph Flow has two APIs:

-   The **Simple API** is as easy to use as scikit-learn: one function call
    trains a GNN model from a graph and a target column, and the resulting
    model can be evaluated, used for predictions, and saved.
-   The **Advanced API** is a set of composable building blocks (graph IO,
    samplers, feature normalizers, and JAX/Flax GNN layers). Like Lego bricks,
    you pick the ones you need and assemble them with your own code, including
    PyTorch Geometric models.

📖 **Documentation:** https://dgf.readthedocs.io/

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

## 📦 Installation

To install DGF from [PyPI](https://pypi.org/project/dgf/), run:

```shell
pip install dgf -U
```

Graph Flow is available for Python 3.11–3.13 on Linux x86-64.

## 🔥 Why Graph Flow?

-   **Simple API:** Train, evaluate, and save a GNN model in about 10 lines of
    Python, without prior GNN experience.
-   **Temporal graphs:** Dynamic graphs and time-series features are supported.
    Time-aware sampling ensures that a model only sees information available
    before the prediction time.
-   **Scale:** Train in memory on graphs with up to 1B edges on a single
    machine. For larger graphs, the distributed sampler (Apache Beam on Google
    Cloud Dataflow) scales to trillions of edges.
-   **Advanced API:** Message-passing layers (MPNN, GAT, Graph Transformer),
    samplers, and normalizers are independent JAX/Flax building blocks for
    custom models.
-   **Interoperability:** Convert graphs to PyTorch Geometric, TensorFlow,
    TF-GNN, and NetworkX.
-   **Deployment:** Run inference in-process in Python, or export models to
    TensorFlow SavedModel (e.g., for Vertex AI).

## 😎 Minimal usage example

```python
import dgf

# Download an example graph
graph, schema = dgf.io.fetch_ogb_graph("arxiv")

# Train a GNN model to predict the "labels" feature on the "nodes" nodeset
model = dgf.learning.train_node_model(
    graph=graph,
    schema=schema,
    target_column="labels",
    target_nodeset="nodes",
)

# Inspect training statistics, architecture, and schema
model.describe()

# Evaluate quality metrics
model.evaluate()

# Make low-latency predictions in-process
model.predict(graph, seed_node_idxs=[0, 1, 2])

# Save the model (architecture, weights, sampling config, training logs, etc.)
model.save("/tmp/model")
```

<details>
<summary>Output of <code>model.describe()</code></summary>

<img src="https://raw.githubusercontent.com/google/distributed_graph_flow/main/doc/docs/image/usage_example_describe.png" alt="Model description" />

</details>

<details>
<summary>Output of <code>model.evaluate()</code></summary>

<img src="https://raw.githubusercontent.com/google/distributed_graph_flow/main/doc/docs/image/usage_example_evaluate.png" alt="Model evaluation" />

</details>

See the
[Getting Started tutorial](https://dgf.readthedocs.io/en/latest/tutorial/getting_started_simple_api.html)
for the complete walkthrough.

## 🤗 Need help?

-   Read the [documentation](https://dgf.readthedocs.io/) and the
    [Q&A](https://dgf.readthedocs.io/en/latest/qna.html).
-   **Bugs & feature requests:** Open a
    [GitHub issue](https://github.com/google/distributed_graph_flow/issues).
-   **Contact the team:** Email us at
    [distributed-graph-flow-contact@google.com](mailto:distributed-graph-flow-contact@google.com).

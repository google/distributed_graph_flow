# Distributed Graph Flow

<p align="center">
<img src="doc/docs/image/logo.png" width="300" />
</p>

**(Distributed) Graph Flow** (GF) is a Python toolkit to develop and deploy
Graph Neural Network (**GNN**) models.

For more information, check the documentation at https://dgf.readthedocs.io/

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

## Installation

To install DGF from [PyPI](https://pypi.org/project/dgf/), run:

```shell
pip install dgf -U
```

Currently, DGF is available on Python 3.11-13, on Linux x86-64.

## 😎 Minimal Usage example

```python
# Temporary fix for Keras dependency.
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Import (distributed) graph flow
import dgf

# Fetch an example graph
graph, schema = dgf.io.fetch_ogb_graph("arxiv")

# Train a model
model = dgf.learning.train_node_model(graph=graph, schema=schema, target_column="labels")

# Look at the model
model.describe()

# Evaluate the model
model.evaluate()

# Make predictions
model.predict(graph, seed_node_idxs=[0, 1, 2])

# Save the model for later
model.save("/tmp/model")
```

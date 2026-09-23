# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IO to load an OGB dataset into an in-memory graph."""

from __future__ import annotations

import contextlib
import enum
import functools
import io
import os
import tempfile
from typing import Any
import urllib.request
import zipfile

from absl import logging
from dgf.src.data import in_memory_graph as in_memory_graph_lib
from dgf.src.data import schema as schema_lib
from dgf.src.io import cache as cache_lib
from dgf.src.io import feature_format as feature_format_lib
from dgf.src.io import graph_in_memory as gf_graph_in_memory
from dgf.src.util import log
import dgf.src.util.filesystem as fs
from dgf.src.util.weak_dep import weak_dep_ogb
import numpy as np
import pandas as pd
import yaml

GRAPHLAND_DATASET_NAMES = [
    "web-fraud",
    "web-traffic",
    "web-topics",
    "avazu-ctr",
    "city-roads-M",
    "hm-categories",
    "tolokers-2",
    "city-reviews",
    "twitch-views",
    "city-roads-L",
    "artnet-views",
    "hm-prices",
    "pokec-regions",
    "artnet-exp",
]


class Repo(str, enum.Enum):
  """Where does the data comes from."""

  AUTO = "AUTO"
  OGB = "OGB"
  CNS = "CNS"
  ZENODO = "ZENODO"
  WEB = "WEB"


def download_ogb_graph(
    name: str, cache_dir: str | None = None
) -> tuple[Any, Any, Any]:
  """Downloads an OGB graph dataset.

  Args:
    name: The name of the OGB dataset.
    cache_dir: The directory to cache the downloaded dataset.

  Returns:
    A tuple containing the graph data, labels, and index splits.
  """

  nodeproppred = weak_dep_ogb.ogb_nodeproppred

  # Download dataset using OGB's Library-Agnostic Loader.
  # TODO: b/449224186 - Temporarily, always clean up the cache directory
  # because the library has trouble loading the dataset from cache.
  # The problem might need to be fixed in the open-source library, and
  # sync back into /third_party.
  if cache_dir is None:
    cache_dir = "/tmp/ogb_cache_dir"
  if fs.exists(cache_dir):
    logging.info("Clearing OGB dataset cache directory %s", cache_dir)
    fs.rmtree(cache_dir)

  # TODO: b/449202059 - NodePropPredDataset by default caches the datasets to
  # datasets/<dataset_name>. When running on a local cloudtop, the
  # library gets a permission denied error when trying to create the "dataset"
  # directory.
  # As a workaround, we set the root to cache_dir.
  # This logic needs to be revisited when moving to Borg / GCP.
  dataset = nodeproppred.NodePropPredDataset(name=name, root=cache_dir)
  graph, label = dataset[0]
  idx_split = dataset.get_idx_split()
  fs.rmtree(cache_dir)
  return graph, label, idx_split


def generate_ids(prefix: str, num_nodes: int) -> np.ndarray:
  """Generates an array of unique IDs with a given prefix."""
  return np.array([f"{prefix}{i}" for i in range(num_nodes)], dtype=np.bytes_)


def build_split_idx(
    num_nodes: int, ogb_splits: Any, subdict: str | None = None
) -> np.ndarray:
  """Given an OGB idx_split, generate the content of the #split feature."""
  splits = np.full(num_nodes, "n/a", dtype="S5")
  for name, idxs in ogb_splits.items():
    if subdict is not None:
      idxs = idxs[subdict]
    splits[idxs] = name
  return splits


def load_ogbn_arxiv() -> (
    tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]
):
  """Loads the OGBN-Arxiv dataset."""

  raw_graph, label, splits = download_ogb_graph("ogbn-arxiv")

  schema = schema_lib.GraphSchema(
      node_sets={
          "nodes": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "#split": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                  ),
                  "labels": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                      num_categorical_values=get_num_classes(label),
                  ),
                  "year": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
                  "feat": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.EMBEDDING,
                      shape=(128,),
                  ),
              }
          )
      },
      edge_sets={
          "edges": schema_lib.EdgeSchema(source="nodes", target="nodes")
      },
  )

  num_nodes = len(label)
  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "nodes": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=num_nodes,
              features={
                  "#id": generate_ids("n", num_nodes),
                  "#split": build_split_idx(num_nodes, splits),
                  "labels": label[:, 0],
                  "year": raw_graph["node_year"][:, 0],
                  "feat": raw_graph["node_feat"],
              },
          )
      },
      edge_sets={
          "edges": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.asarray(raw_graph["edge_index"]), features={}
          )
      },
  )
  return graph, schema


def load_ogbn_mag() -> (
    tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]
):
  """Loads the OGBN-Mag dataset."""

  raw_graph, label, splits = download_ogb_graph("ogbn-mag")

  schema = schema_lib.GraphSchema(
      node_sets={
          "author": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  )
              }
          ),
          "paper": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "#split": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                  ),
                  "labels": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                  ),
                  "year": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  ),
                  "feat": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.EMBEDDING,
                      shape=(128,),
                  ),
              }
          ),
          "field_of_study": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  )
              }
          ),
          "institution": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  )
              }
          ),
      },
      edge_sets={
          "has_topic": schema_lib.EdgeSchema(
              source="paper", target="field_of_study"
          ),
          "affiliated_with": schema_lib.EdgeSchema(
              source="author", target="institution"
          ),
          "cites": schema_lib.EdgeSchema(source="paper", target="paper"),
          "writes": schema_lib.EdgeSchema(source="author", target="paper"),
      },
  )

  num_papers = raw_graph["num_nodes_dict"]["paper"]
  num_authors = raw_graph["num_nodes_dict"]["author"]
  num_institutions = raw_graph["num_nodes_dict"]["institution"]
  num_fields_of_study = raw_graph["num_nodes_dict"]["field_of_study"]

  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "paper": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=num_papers,
              features={
                  "#id": generate_ids("p", num_papers),
                  "#split": build_split_idx(
                      num_papers, splits, subdict="paper"
                  ),
                  "labels": label["paper"][:, 0],
                  "year": raw_graph["node_year"]["paper"][:, 0],
                  "feat": raw_graph["node_feat_dict"]["paper"],
              },
          ),
          "author": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=num_authors,
              features={
                  "#id": generate_ids("a", num_authors),
              },
          ),
          "institution": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=num_institutions,
              features={
                  "#id": generate_ids("i", num_institutions),
              },
          ),
          "field_of_study": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=num_fields_of_study,
              features={
                  "#id": generate_ids("f", num_fields_of_study),
              },
          ),
      },
      edge_sets={
          "cites": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.asarray(
                  raw_graph["edge_index_dict"]["paper", "cites", "paper"]
              )
          ),
          "writes": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.asarray(
                  raw_graph["edge_index_dict"]["author", "writes", "paper"]
              )
          ),
          "affiliated_with": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.asarray(
                  raw_graph["edge_index_dict"][
                      "author", "affiliated_with", "institution"
                  ]
              )
          ),
          "has_topic": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.asarray(
                  raw_graph["edge_index_dict"][
                      "paper", "has_topic", "field_of_study"
                  ]
              )
          ),
      },
  )
  return graph, schema


def load_ogbn_products() -> (
    tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]
):
  """Loads the OGBN-Products dataset."""

  raw_graph, label, splits = download_ogb_graph("ogbn-products")

  schema = schema_lib.GraphSchema(
      node_sets={
          "nodes": schema_lib.NodeSchema(
              features={
                  "#id": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
                  ),
                  "#split": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.BYTES,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                  ),
                  "labels": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.INTEGER_64,
                      semantic=schema_lib.FeatureSemantic.CATEGORICAL,
                      num_categorical_values=get_num_classes(label),
                  ),
                  "feat": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.EMBEDDING,
                      shape=(100,),
                  ),
              }
          )
      },
      edge_sets={
          "edges": schema_lib.EdgeSchema(source="nodes", target="nodes")
      },
  )

  num_nodes = len(label)
  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "nodes": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=num_nodes,
              features={
                  "#id": generate_ids("n", num_nodes),
                  "#split": build_split_idx(num_nodes, splits),
                  "labels": label[:, 0],
                  "feat": raw_graph["node_feat"],
              },
          )
      },
      edge_sets={
          "edges": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.asarray(raw_graph["edge_index"]), features={}
          )
      },
  )
  return graph, schema


def load_from_cns(
    name: str,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Loads the data from CNS."""
  path = os.path.join(CNS_GF_REPO, name)
  return gf_graph_in_memory.read_graph(path)


def fetch_ogb_graph(
    name: str,
    cache_dir: str | None = "AUTO",
    verbose: bool = True,
    repo: Repo | str = Repo.AUTO,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Downloads and loads an OGB node property prediction dataset into memory.

  This function fetches datasets from the Open Graph Benchmark (OGB)
  (https://ogb.stanford.edu/docs/nodeprop/), converts them into an
  `InMemoryGraph` representation, and returns the graph along with its schema.

  The available graphs are:
    arxiv
    mag

  Usage example:

  ```python
  graph, schema = dgf.io.fetch_ogb_graph("arxiv")
  dgf.analyse.print_schema(schema)
  ```

  Args:
    name: The name of the OGB dataset.
    cache_dir: Optional. The directory to cache the graph in order to avoid to
      re-download it each time. If equal to "AUTO, use the OS default temporary
      directory. If None, does not cache the graph.
    verbose: Optional. Whether to print cache path information.
    repo: Define the source of the data.

  Returns:
    An InMemoryGraph instance representing the loaded dataset and the schema.
  """

  if cache_dir == "AUTO":
    cache_dir = os.path.join(tempfile.gettempdir(), "gf_fetch")

  if cache_dir is not None:
    fs.makedirs(cache_dir)
    cache_graph_path = os.path.join(cache_dir, f"{name}.cache")
    if verbose:
      log.info("Caching %s graph at %s", name, cache_graph_path)
  else:
    cache_graph_path = None

  if isinstance(repo, str):
    repo = Repo(repo)

  # Select the right repo.
  if repo == Repo.AUTO:
    repo = Repo.OGB

  if repo == Repo.OGB:
    loaders = {
        "arxiv": load_ogbn_arxiv,
        "mag": load_ogbn_mag,
        "products": load_ogbn_products,
    }
  elif repo == Repo.CNS:
    names = ["arxiv", "mag", "products"]
    loaders = {
        name: functools.partial(load_from_cns, name=f"ogb_{name}")
        for name in names
    }

  else:
    assert False

  if name not in loaders:
    raise ValueError(
        f"Unknown graph: {name}. The available graph names are: {list(loaders)}"
    )

  def load_graph():
    return loaders[name]()

  if cache_graph_path is None:
    return load_graph()
  else:
    return cache_lib.cache(cache_graph_path, load_graph)


def download_graphland_graph(
    name: str, mask_name: str, repo: Repo
) -> tuple[Any, Any, Any, Any, Any]:
  """Downloads a Graphland graph dataset from Zenodo or CNS.

  Args:
      name: The name of the Graphland dataset.
      mask_name: The name of the mask to use for splitting.
      repo: The repository to fetch the data from (Repo.ZENODO or Repo.CNS).

  Returns:
      A tuple containing (edges, features, targets, splits, info)
  """
  if repo == Repo.ZENODO:
    # record_id is the Zenodo record ID for the GraphLand benchmark archive
    # compilation (https://zenodo.org/records/16895532)
    record_id = "16895532"
    url = f"https://zenodo.org/api/records/{record_id}/files/{name}.zip/content"

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
      content = response.read()

    z = zipfile.ZipFile(io.BytesIO(content))

    @contextlib.contextmanager
    def open_file(filename):
      assert z is not None
      with z.open(f"{name}/{filename}") as f:
        yield f

  elif repo == Repo.CNS:
    z = None

    @contextlib.contextmanager
    def open_file(filename):
      with fs.open_read(os.path.join(CNS_GRAPHLAND, name, filename)) as f:
        yield f

  else:
    raise ValueError(f"Unsupported repo for GraphLand: {repo}")

  try:
    with open_file("info.yaml") as f:
      info = yaml.safe_load(f)

    with open_file("edgelist.csv") as f:
      edges_df = pd.read_csv(f)
      edges = edges_df.to_numpy(dtype=np.int64).T

    with open_file("features.csv") as f:
      features_df = pd.read_csv(f)
      float_cols = features_df.select_dtypes(include="float").columns
      features_df[float_cols] = features_df[float_cols].astype(np.float32)
      if "node_id" in features_df.columns:
        features_df = features_df.drop(columns=["node_id"])
      features = {
          col: features_df[col].to_numpy() for col in features_df.columns
      }

    with open_file("targets.csv") as f:
      targets_df = pd.read_csv(f)
      if "node_id" in targets_df.columns:
        targets_df = targets_df.drop(columns=["node_id"])
      targets = targets_df.iloc[:, 0].to_numpy()

    def load_splits(filename):
      with open_file(filename) as f:
        mask_df = pd.read_csv(f)
        splits = np.full(mask_df.shape[0], "n/a", dtype="S5")
        for key, value in [
            ("train", "train"),
            ("val", "valid"),
            ("test", "test"),
        ]:
          splits[mask_df[key]] = value
      return splits

    splits = load_splits(f"split_masks_{mask_name}.csv")

  finally:
    if z is not None:
      z.close()

  return edges, features, targets, splits, info


def get_num_classes(targets: np.ndarray) -> int:
  """Computes the number of classes from the targets array."""
  return int(np.nanmax(targets) + 1)


def fetch_graphland_graph(
    name: str,
    cache_dir: str | None = "AUTO",
    verbose: bool = True,
    mask_name: str = "RL",
    repo: Repo | str = Repo.AUTO,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Downloads and loads a Graphland dataset into memory.

  This function fetches datasets from the Graphland benchmark
  (https://arxiv.org/abs/2409.14500).

  Usage example:

  ```python
  graph, schema = dgf.io.fetch_graphland_graph("tolokers-2")
  dgf.analyse.print_schema(schema)
  ```

  List of available graph land datasets (from smallest to largest):
    tolokers-2 (binary classification, 11.8K nodes, 519.0K edges, 16 features)
    hm-prices (regression, 46.5K nodes, 10.7M edges, 41 features)
    hm-categories (classification, 46.5K nodes, 10.7M edges, 35 features)
    artnet-views (regression, 50.4K nodes, 280.3K edges, 50 features)
    artnet-exp (binary classification, 50.4K nodes, 280.3K edges, 75 features)
    city-roads-M (regression, 57.1K nodes, 107.1K edges, 26 features)
    avazu-ctr (regression, 76.3K nodes, 11.0M edges, 260 features)
    city-roads-L (regression, 142.3K nodes, 231.6K edges, 26 features)
    city-reviews (binary classification, 148.8K nodes, 1.2M edges, 37 features)
    twitch-views (regression, 168.1K nodes, 6.8M edges, 4 features)
    pokec-regions (classification, 1.6M nodes, 22.3M edges, 56 features)
    web-topics (classification, 2.9M nodes, 12.4M edges, 263 features)
    web-fraud (binary classification, 2.9M nodes, 12.4M edges, 266 features)
    web-traffic (regression, 2.9M nodes, 12.4M edges, 267 features)

  Args:
    name: The name of the Graphland dataset.
    cache_dir: Optional. The directory to cache the graph in order to avoid to
      re-download it each time. If equal to "AUTO, use the OS default temporary
      directory. If None, does not cache the graph.
    verbose: Optional. Whether to print cache path information.
    mask_name: The name of the mask to use for splitting the data. Can be RL,
      RH, TH, THI. Note that not all datasets have all the options.
    repo: Define the source of the data.

  Returns:
    An InMemoryGraph instance representing the loaded dataset and the schema.
  """

  if cache_dir == "AUTO":
    cache_dir = os.path.join(tempfile.gettempdir(), "gf_fetch_graphland")

  if cache_dir is not None:
    fs.makedirs(cache_dir)
    cache_graph_path = os.path.join(cache_dir, f"{name}_{mask_name}.cache")
    if verbose:
      log.info(
          "Caching Graphland %s (%s) graph at %s",
          name,
          mask_name,
          cache_graph_path,
      )
  else:
    cache_graph_path = None

  if isinstance(repo, str):
    repo = Repo(repo)

  # Select the right repo.
  if repo == Repo.AUTO:
    repo = Repo.ZENODO

  def load_graph():
    edges, features, targets, splits, info = download_graphland_graph(
        name, mask_name, repo
    )
    task_type = info.get("task", "binary_classification")

    # Determine num_nodes by checking the size of the first feature array, or
    # targets
    if targets is not None:
      num_nodes = targets.shape[0]
    else:
      num_nodes = next(iter(features.values())).shape[0]

    is_classification = "classification" in task_type

    if is_classification:
      target_semantic = schema_lib.FeatureSemantic.CATEGORICAL
      num_classes = info.get("num_classes", get_num_classes(targets))
    else:
      target_semantic = schema_lib.FeatureSemantic.NUMERICAL
      num_classes = None
    target_format = feature_format_lib.NP_DTYPE_TO_FEATURE_FORMAT[
        targets.dtype.type
    ]

    features_schemas = {
        "#id": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BYTES,
            semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
        ),
        "#split": schema_lib.FeatureSchema(
            format=schema_lib.FeatureFormat.BYTES,
            semantic=schema_lib.FeatureSemantic.CATEGORICAL,
        ),
        "labels": schema_lib.FeatureSchema(
            format=target_format,
            semantic=target_semantic,
            num_categorical_values=num_classes,
        ),
    }

    # Prepare #split
    features_dict = {
        "#id": generate_ids("n", num_nodes),
        "#split": splits,
        "labels": targets,
    }

    # Determine semantics for features
    numerical_features = set(info.get("numerical_features_names", []))
    categorical_features = set(info.get("categorical_features_names", []))
    fraction_features = set(info.get("fraction_features_names", []))

    for feat_name, feat_values in features.items():
      feature_format = feature_format_lib.NP_DTYPE_TO_FEATURE_FORMAT[
          feat_values.dtype.type
      ]
      num_categorical_values = None
      if feat_name in categorical_features:
        semantic = schema_lib.FeatureSemantic.CATEGORICAL
        if feature_format.is_numerical():
          num_categorical_values = get_num_classes(feat_values)
      elif feat_name in numerical_features or feat_name in fraction_features:
        semantic = schema_lib.FeatureSemantic.NUMERICAL
      else:
        raise ValueError(f"Unknown feature type for {feat_name}")
      features_schemas[feat_name] = schema_lib.FeatureSchema(
          format=feature_format,
          semantic=semantic,
          shape=None,
          num_categorical_values=num_categorical_values,
      )
      features_dict[feat_name] = feat_values

    schema = schema_lib.GraphSchema(
        node_sets={"nodes": schema_lib.NodeSchema(features=features_schemas)},
        edge_sets={
            "edges": schema_lib.EdgeSchema(source="nodes", target="nodes")
        },
    )
    graph = in_memory_graph_lib.InMemoryGraph(
        node_sets={
            "nodes": in_memory_graph_lib.InMemoryNodeSet(
                num_nodes=num_nodes,
                features=features_dict,
            )
        },
        edge_sets={
            "edges": in_memory_graph_lib.InMemoryEdgeSet(adjacency=edges)
        },
    )
    return graph, schema

  if cache_graph_path is None:
    return load_graph()
  else:
    return cache_lib.cache(cache_graph_path, load_graph)


JENA_CLIMATE_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

JENA_CLIMATE_COLUMN_RENAME_MAP = {
    "p (mbar)": "p_mbar",
    "T (degC)": "t_degc",
    "Tpot (K)": "tpot_k",
    "Tdew (degC)": "tdew_degc",
    "rh (%)": "rh_percent",
    "VPmax (mbar)": "vpmax_mbar",
    "VPact (mbar)": "vpact_mbar",
    "VPdef (mbar)": "vpdef_mbar",
    "sh (g/kg)": "sh_g_per_kg",
    "H2OC (mmol/mol)": "h2oc_mmol_per_mol",
    "rho (g/m**3)": "rho_g_per_cubic_m",
    "wv (m/s)": "wv_m_per_s",
    "max. wv (m/s)": "max_wv_m_per_s",
    "wd (deg)": "wd_deg",
}

JENA_WEATHER_FEATURE_NAMES = list(JENA_CLIMATE_COLUMN_RENAME_MAP.values())


def download_jena_climate_csv(source: str | None = None) -> pd.DataFrame:
  """Downloads and parses the Jena Climate CSV.

  Cleans sentinel values (-9999.0 in wind velocities) and parses timestamps.

  Args:
    source: Optional URL or file path. If None, downloads from the official
      TensorFlow datasets public GCS archive.

  Returns:
    A cleaned pandas DataFrame with timestamps and renamed columns.
  """
  url_or_path = source or JENA_CLIMATE_URL
  if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
    log.info("Downloading Jena Climate dataset from %s", url_or_path)
    request = urllib.request.Request(
        url_or_path, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request) as response:
      content = response.read()
    with zipfile.ZipFile(io.BytesIO(content)) as zip_file:
      csv_names = [
          name for name in zip_file.namelist() if name.endswith(".csv")
      ]
      if not csv_names:
        raise ValueError(f"No CSV file found in archive from {url_or_path}")
      with zip_file.open(csv_names[0]) as csv_file:
        climate_df = pd.read_csv(csv_file)
  else:
    if url_or_path.endswith(".zip"):
      with zipfile.ZipFile(url_or_path) as zip_file:
        csv_names = [
            name for name in zip_file.namelist() if name.endswith(".csv")
        ]
        if not csv_names:
          raise ValueError(f"No CSV file found in {url_or_path}")
        with zip_file.open(csv_names[0]) as csv_file:
          climate_df = pd.read_csv(csv_file)
    else:
      climate_df = pd.read_csv(url_or_path)

  missing_columns = [
      column
      for column in ["Date Time", *JENA_CLIMATE_COLUMN_RENAME_MAP]
      if column not in climate_df.columns
  ]
  if missing_columns:
    raise ValueError(
        f"Jena Climate data from {url_or_path} is missing the columns"
        f" {missing_columns}."
    )

  # Clean sentinel values: wv (m/s) and max. wv (m/s) have -9999.0 for missing
  for column in ["wv (m/s)", "max. wv (m/s)"]:
    climate_df[column] = climate_df[column].replace(-9999.0, 0.0)

  # Parse Date Time to unix timestamp (seconds)
  date_times = pd.to_datetime(
      climate_df["Date Time"], format="%d.%m.%Y %H:%M:%S"
  )
  climate_df["timestamp"] = (
      date_times.to_numpy().astype("datetime64[s]").astype(np.int64)
  )

  return climate_df.rename(columns=JENA_CLIMATE_COLUMN_RENAME_MAP)


def build_jena_climate_graph(
    climate_df: pd.DataFrame,
    forecast_horizon_seconds: int = 3600,
    query_step: int = 6,
    subsample_station_step: int = 1,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Constructs a DGF InMemoryGraph and GraphSchema from Jena Climate data.

  Graph structure:
    - Single station node containing 14 time series weather metrics.
    - Query nodes at chronological intervals, each with `creation_time`
      and regression label `temperature` (at `creation_time + horizon`).
    - Bidirectional edges between query nodes and the single station node.

  Args:
    climate_df: Jena Climate pandas DataFrame, as returned by
      `download_jena_climate_csv`.
    forecast_horizon_seconds: Horizon into the future to predict (default: 3600s
      = 1 hour).
    query_step: Stride for sampling query nodes (default: 6, i.e. 1 query/hour
      for 10-minute data).
    subsample_station_step: Stride for station time series observations
      (default: 1).

  Returns:
    A tuple of (InMemoryGraph, GraphSchema).
  """
  missing_columns = [
      column
      for column in ["timestamp", *JENA_WEATHER_FEATURE_NAMES]
      if column not in climate_df.columns
  ]
  if missing_columns:
    raise ValueError(
        f"climate_df is missing the columns {missing_columns}. Expected the"
        " output of download_jena_climate_csv."
    )

  all_timestamps = climate_df["timestamp"].to_numpy(dtype=np.int64)
  all_temperatures = climate_df["t_degc"].to_numpy(dtype=np.float32)

  # Subsample station observations if requested
  station_timestamps = all_timestamps[::subsample_station_step]

  # Station features schema and data
  station_node_features: dict[str, np.ndarray] = {
      "#id": np.array([b"station_0"], dtype=np.bytes_),
  }
  station_schema_features: dict[str, schema_lib.FeatureSchema] = {
      "#id": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BYTES,
          semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
      ),
  }

  station_times = np.empty(1, dtype=object)
  station_times[0] = station_timestamps
  station_node_features["time"] = station_times
  station_schema_features["time"] = schema_lib.FeatureSchema(
      format=schema_lib.FeatureFormat.INTEGER_64,
      semantic=schema_lib.FeatureSemantic.TIMESTAMP,
      is_timeseries=True,
      is_creation_time=True,
      group="weather_ts",
      shape=(None,),
  )

  for column_name in JENA_WEATHER_FEATURE_NAMES:
    station_values = np.empty(1, dtype=object)
    station_values[0] = (
        climate_df[column_name]
        .iloc[::subsample_station_step]
        .to_numpy(dtype=np.float32)
    )
    station_node_features[column_name] = station_values
    station_schema_features[column_name] = schema_lib.FeatureSchema(
        format=schema_lib.FeatureFormat.FLOAT_32,
        semantic=schema_lib.FeatureSemantic.NUMERICAL,
        is_timeseries=True,
        group="weather_ts",
        shape=(None,),
    )

  station_node_set = in_memory_graph_lib.InMemoryNodeSet(
      num_nodes=1,
      features=station_node_features,
  )

  # Query nodes construction
  candidate_query_indices = np.arange(0, len(climate_df), query_step)
  candidate_query_times = all_timestamps[candidate_query_indices]
  target_times = candidate_query_times + forecast_horizon_seconds

  target_indices = np.searchsorted(all_timestamps, target_times)
  valid_mask = (target_indices < len(all_timestamps)) & (
      all_timestamps[np.minimum(target_indices, len(all_timestamps) - 1)]
      == target_times
  )

  valid_query_times = candidate_query_times[valid_mask]
  valid_target_temperatures = all_temperatures[target_indices[valid_mask]]
  num_queries = len(valid_query_times)

  if num_queries == 0:
    raise ValueError(
        f"No valid query points found with horizon {forecast_horizon_seconds}s."
    )

  # Chronological split: 70% train, 20% valid, 10% test
  num_train = int(num_queries * 0.7)
  num_valid = int(num_queries * 0.2)
  split_labels = np.full(num_queries, "n/a", dtype="S5")
  split_labels[:num_train] = b"train"
  split_labels[num_train : num_train + num_valid] = b"valid"
  split_labels[num_train + num_valid :] = b"test"

  query_ids = np.array(
      [f"query_{i}".encode("utf-8") for i in range(num_queries)],
      dtype=np.bytes_,
  )
  query_node_features: dict[str, np.ndarray] = {
      "#id": query_ids,
      "creation_time": valid_query_times,
      "temperature": valid_target_temperatures,
      "#split": split_labels,
  }

  query_schema_features: dict[str, schema_lib.FeatureSchema] = {
      "#id": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BYTES,
          semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
      ),
      "creation_time": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.TIMESTAMP,
          is_creation_time=True,
      ),
      "temperature": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
      ),
      "#split": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BYTES,
          semantic=schema_lib.FeatureSemantic.CATEGORICAL,
      ),
  }

  query_node_set = in_memory_graph_lib.InMemoryNodeSet(
      num_nodes=num_queries,
      features=query_node_features,
  )

  # Edge sets: bidirectional between queries and station node (0)
  query_indices = np.arange(num_queries, dtype=np.int64)
  zero_indices = np.zeros(num_queries, dtype=np.int64)

  query_to_station_edges = in_memory_graph_lib.InMemoryEdgeSet(
      adjacency=np.stack([query_indices, zero_indices], axis=0),
      features={},
  )
  station_to_query_edges = in_memory_graph_lib.InMemoryEdgeSet(
      adjacency=np.stack([zero_indices, query_indices], axis=0),
      features={},
  )

  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "queries": query_node_set,
          "station": station_node_set,
      },
      edge_sets={
          "query_to_station": query_to_station_edges,
          "station_to_query": station_to_query_edges,
      },
  )

  schema = schema_lib.GraphSchema(
      node_sets={
          "queries": schema_lib.NodeSchema(features=query_schema_features),
          "station": schema_lib.NodeSchema(features=station_schema_features),
      },
      edge_sets={
          "query_to_station": schema_lib.EdgeSchema(
              source="queries",
              target="station",
              features={},
          ),
          "station_to_query": schema_lib.EdgeSchema(
              source="station",
              target="queries",
              features={},
          ),
      },
  )

  return graph, schema


def fetch_jena_climate_graph(
    name: str = "jena_climate_1h",
    cache_dir: str | None = "AUTO",
    verbose: bool = True,
    forecast_horizon_seconds: int = 3600,
    query_step: int = 6,
    subsample_station_step: int = 1,
    repo: Repo | str = Repo.AUTO,
    source: str | None = None,
) -> tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Downloads and loads the Jena Climate time series benchmark into memory.

  This function loads the Jena Climate dataset
  (https://www.bgc-jena.mpg.de/wetter/) and represents it as an in-memory graph
  node prediction regression task with a single weather station node containing
  14 meteorological time-series features and query nodes representing prediction
  time points.

  Usage example:

  ```python
  graph, schema = dgf.io.fetch_jena_climate_graph()
  dgf.analyse.print_schema(schema)
  ```

  Args:
    name: The name of the dataset under CNS fetch_repo (e.g. 'jena_climate_1h'
      or 'jena_climate_24h').
    cache_dir: Optional. Directory to cache the graph in order to avoid
      re-downloading/re-parsing it each time. If "AUTO", uses OS default
      temporary directory. If None, does not cache the graph.
    verbose: Optional. Whether to print cache and download progress.
    forecast_horizon_seconds: Forecasting horizon in seconds (default: 3600 for
      1 hour ahead).
    query_step: Step size to subsample query nodes (default: 6, i.e. 1 query per
      hour for 10-minute data).
    subsample_station_step: Optional step to subsample the station time series
      (default: 1, full 10-minute resolution).
    repo: Define the source of the data (Repo.AUTO, Repo.CNS, Repo.WEB).
    source: Optional URL or file path to the Jena Climate zip/CSV.

  Returns:
    An InMemoryGraph instance and its GraphSchema.
  """
  if cache_dir == "AUTO":
    cache_dir = os.path.join(tempfile.gettempdir(), "gf_fetch_jena_climate")

  if cache_dir is not None:
    fs.makedirs(cache_dir)
    cache_key = (
        f"{name}_h{forecast_horizon_seconds}_qs{query_step}_"
        f"ss{subsample_station_step}.cache"
    )
    cache_graph_path = os.path.join(cache_dir, cache_key)
    if verbose:
      log.info("Caching Jena Climate graph at %s", cache_graph_path)
  else:
    cache_graph_path = None

  if isinstance(repo, str):
    repo = Repo(repo)

  # Select the right repo.
  if repo == Repo.AUTO:
    repo = Repo.WEB

  if repo == Repo.CNS:
    loader = functools.partial(load_from_cns, name=name)
  elif repo == Repo.WEB:

    def loader():
      if verbose:
        log.info(
            "Loading Jena Climate data from %s", source or JENA_CLIMATE_URL
        )
      climate_df = download_jena_climate_csv(source=source)
      return build_jena_climate_graph(
          climate_df=climate_df,
          forecast_horizon_seconds=forecast_horizon_seconds,
          query_step=query_step,
          subsample_station_step=subsample_station_step,
      )

  else:
    raise ValueError(f"Unsupported repo for Jena Climate: {repo}")

  if cache_graph_path is None:
    return loader()
  else:
    return cache_lib.cache(cache_graph_path, loader)

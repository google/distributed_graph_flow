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
import dataclasses
import enum
import functools
import io
import os
import tempfile
from typing import Any, Optional, Tuple, Union
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
from dgf.src.util.weak_dep.weak_dep_ogb import ogb_nodeproppred
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


def download_ogb_graph(name: str) -> Tuple[Any, Any, Any]:
  """Downloads an OGB graph dataset.

  Args:
    name: The name of the OGB dataset.
    cache_dir: The directory to cache the downloaded dataset.

  Returns:
    A tuple containing the graph data, labels, and index splits.
  """

  nodeproppred = ogb_nodeproppred

  # Download dataset using OGB's Library-Agnostic Loader.
  # TODO: b/449224186 - Temporarily, always clean up the cache directory
  # because the library has trouble loading the dataset from cache.
  # The problem might need to be fixed in the open-source library, and
  # sync back into /third_party.
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
    num_nodes: int, ogb_splits: Any, subdict: Optional[str] = None
) -> np.ndarray:
  """Given an OGB idx_split, generate the content of the #split feature."""
  splits = np.full(num_nodes, "n/a", dtype="S5")
  for name, idxs in ogb_splits.items():
    if subdict is not None:
      idxs = idxs[subdict]
    splits[idxs] = name
  return splits


def load_ogbn_arxiv() -> (
    Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]
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
    Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]
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
    Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]
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
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Loads the data from CNS."""
  path = os.path.join(CNS_GF_REPO, name)
  return gf_graph_in_memory.read_graph(path)


def fetch_ogb_graph(
    name: str,
    cache_dir: Optional[str] = "AUTO",
    verbose: bool = True,
    repo: Union[Repo, str] = Repo.AUTO,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
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
) -> Tuple[Any, Any, Any, Any, Any]:
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
    cache_dir: Optional[str] = "AUTO",
    verbose: bool = True,
    mask_name: str = "RL",
    repo: Union[Repo, str] = Repo.AUTO,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
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
    cache_graph_path = os.path.join(cache_dir, f"{name}.cache")
    if verbose:
      log.info("Caching Graphland %s graph at %s", name, cache_graph_path)
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
      format = feature_format_lib.NP_DTYPE_TO_FEATURE_FORMAT[
          feat_values.dtype.type
      ]
      num_categorical_values = None
      if feat_name in categorical_features:
        semantic = schema_lib.FeatureSemantic.CATEGORICAL
        if format.is_numerical():
          num_categorical_values = get_num_classes(feat_values)
      elif feat_name in numerical_features or feat_name in fraction_features:
        semantic = schema_lib.FeatureSemantic.NUMERICAL
      else:
        raise ValueError(f"Unknown feature type for {feat_name}")
      features_schemas[feat_name] = schema_lib.FeatureSchema(
          format=format,
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


JENA_CLIMATE_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
)

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


def download_jena_climate_csv(source: Optional[str] = None) -> pd.DataFrame:
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
  climate_df["timestamp"] = (date_times.astype("int64") // 10**9).astype(
      np.int64
  )

  return climate_df.rename(columns=JENA_CLIMATE_COLUMN_RENAME_MAP)


def build_jena_climate_graph(
    climate_df: pd.DataFrame,
    forecast_horizon_seconds: int = 3600,
    query_step: int = 6,
    subsample_station_step: int = 1,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
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
    cache_dir: Optional[str] = "AUTO",
    verbose: bool = True,
    forecast_horizon_seconds: int = 3600,
    query_step: int = 6,
    subsample_station_step: int = 1,
    repo: Union[Repo, str] = Repo.AUTO,
    source: Optional[str] = None,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
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


# The reference copies of the METR-LA and PEMS-BAY speed recordings are the HDF5
# frames of the Google Drive folders linked from the DCRNN repository
# (https://github.com/liyaguang/DCRNN). The URLs below point at the Zenodo
# record https://doi.org/10.5281/zenodo.5724362, which serves a csv export of
# those same frames.
METR_LA_SPEEDS_URL = (
    "https://zenodo.org/records/5724362/files/METR-LA.csv?download=1"
)
PEMS_BAY_SPEEDS_URL = (
    "https://zenodo.org/records/5724362/files/PEMS-BAY.csv?download=1"
)

# Road network distances and sensor coordinates, from the DCRNN repository.
DCRNN_SENSOR_GRAPH_URL = (
    "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph"
)

# Both datasets encode a missing speed recording with a zero.
TRAFFIC_MISSING_SPEED = 0.0


@dataclasses.dataclass(frozen=True)
class TrafficDatasetSpec:
  """Location and layout of the raw files of a traffic speed dataset.

  Attributes:
    speeds_url: URL of the csv with the speed recordings.
    distances_url: URL of the `from,to,cost` road network distances csv.
    distances_have_header: Whether the distances csv has a header row.
    locations_url: URL of the sensor coordinates csv.
    locations_have_header: Whether the locations csv has a header row.
  """

  speeds_url: str
  distances_url: str
  distances_have_header: bool
  locations_url: str
  locations_have_header: bool


TRAFFIC_DATASETS: dict[str, TrafficDatasetSpec] = {
    "metr_la": TrafficDatasetSpec(
        speeds_url=METR_LA_SPEEDS_URL,
        distances_url=f"{DCRNN_SENSOR_GRAPH_URL}/distances_la_2012.csv",
        distances_have_header=True,
        locations_url=f"{DCRNN_SENSOR_GRAPH_URL}/graph_sensor_locations.csv",
        locations_have_header=True,
    ),
    "pems_bay": TrafficDatasetSpec(
        speeds_url=PEMS_BAY_SPEEDS_URL,
        distances_url=f"{DCRNN_SENSOR_GRAPH_URL}/distances_bay_2017.csv",
        distances_have_header=False,
        locations_url=(
            f"{DCRNN_SENSOR_GRAPH_URL}/graph_sensor_locations_bay.csv"
        ),
        locations_have_header=False,
    ),
}


def traffic_dataset_spec(dataset: str) -> TrafficDatasetSpec:
  """Returns the spec of the traffic dataset called `dataset`."""
  if dataset not in TRAFFIC_DATASETS:
    raise ValueError(
        f"Unknown traffic dataset {dataset!r}. Available datasets:"
        f" {sorted(TRAFFIC_DATASETS)}."
    )
  return TRAFFIC_DATASETS[dataset]


def _is_url(url_or_path: str) -> bool:
  """Returns whether `url_or_path` is an http(s) URL instead of a path."""
  return url_or_path.startswith("http://") or url_or_path.startswith("https://")


def _download_bytes(url: str) -> bytes:
  """Returns the content served at `url`.

  The default `urllib` user agent is used on purpose: Zenodo answers 403 to
  requests that claim to come from a browser.

  Args:
    url: The URL to download.
  """
  with urllib.request.urlopen(url) as response:
    return response.read()


def _decode_sensor_id(sensor_id: Any) -> str:
  """Returns the sensor id as a string.

  Sensor ids are stored as bytes in METR-LA and as integers in PEMS-BAY.

  Args:
    sensor_id: The raw sensor id.
  """
  if isinstance(sensor_id, bytes):
    return sensor_id.decode("utf-8")
  return str(sensor_id)


def download_traffic_speeds(
    dataset: str, source: Optional[str] = None
) -> pd.DataFrame:
  """Downloads and parses the speed recordings of a traffic dataset.

  Args:
    dataset: Name of the dataset, e.g. "metr_la" or "pems_bay".
    source: Optional URL or file path of the csv. If None, downloads from the
      public mirror of the dataset.

  Returns:
    A frame of speeds in miles per hour, indexed by unix timestamp in seconds,
    with one column per sensor named after the sensor id.
  """
  spec = traffic_dataset_spec(dataset)
  url_or_path = source or spec.speeds_url
  csv_source: Any = url_or_path
  if _is_url(url_or_path):
    log.info("Downloading the %s speeds from %s", dataset, url_or_path)
    csv_source = io.BytesIO(_download_bytes(url_or_path))

  # The first column holds the recording time, as a "%Y-%m-%d %H:%M:%S" string,
  # and every other column holds the speeds of one sensor.
  speeds = pd.read_csv(csv_source, index_col=0)
  timestamps = (
      pd.to_datetime(speeds.index).astype("int64") // 10**9
  ).to_numpy(dtype=np.int64)
  return (
      speeds.set_axis(timestamps, axis="index")
      .set_axis(
          [_decode_sensor_id(sensor_id) for sensor_id in speeds.columns],
          axis="columns",
      )
      .astype(np.float32)
  )


def _read_traffic_csv(
    url_or_path: str, names: list[str], has_header: bool
) -> pd.DataFrame:
  """Reads a sensor graph csv, with or without a header row.

  Args:
    url_or_path: URL or file path of the csv.
    names: Names of the columns to return. When the csv has no header row, the
      columns are assumed to be in this order.
    has_header: Whether the csv has a header row.

  Returns:
    A frame with exactly the `names` columns.
  """
  source: Any = url_or_path
  if _is_url(url_or_path):
    log.info("Downloading %s", url_or_path)
    source = io.BytesIO(_download_bytes(url_or_path))

  if not has_header:
    return pd.read_csv(source, header=None, names=names)

  frame = pd.read_csv(source)
  missing_columns = [name for name in names if name not in frame.columns]
  if missing_columns:
    raise ValueError(
        f"The csv {url_or_path} is missing the columns {missing_columns}."
    )
  return frame[names]


def download_traffic_sensor_graph(
    dataset: str,
    distances_source: Optional[str] = None,
    locations_source: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Downloads the road network distances and the coordinates of the sensors.

  Args:
    dataset: Name of the dataset, e.g. "metr_la" or "pems_bay".
    distances_source: Optional URL or file path of the distances csv.
    locations_source: Optional URL or file path of the sensor locations csv.

  Returns:
    A tuple of (distances, locations). `distances` has the columns `from`, `to`
    and `cost`, where the cost is the driving distance in meters between two
    sensors. `locations` has the columns `sensor_id`, `latitude` and
    `longitude`. Sensor ids are strings in both frames.
  """
  spec = traffic_dataset_spec(dataset)
  distances = _read_traffic_csv(
      distances_source or spec.distances_url,
      ["from", "to", "cost"],
      spec.distances_have_header,
  )
  locations = _read_traffic_csv(
      locations_source or spec.locations_url,
      ["sensor_id", "latitude", "longitude"],
      spec.locations_have_header,
  )
  distances = distances.assign(
      **{
          "from": [_decode_sensor_id(value) for value in distances["from"]],
          "to": [_decode_sensor_id(value) for value in distances["to"]],
          "cost": distances["cost"].astype(np.float32),
      }
  )
  locations = locations.assign(
      sensor_id=[
          _decode_sensor_id(value) for value in locations["sensor_id"]
      ],
      latitude=locations["latitude"].astype(np.float32),
      longitude=locations["longitude"].astype(np.float32),
  )
  return distances, locations


def build_sensor_adjacency(
    distances: pd.DataFrame,
    sensor_ids: list[str],
    threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Builds the sensor graph from the pairwise road network distances.

  The edges follow the construction of DCRNN (Li et al., 2018): the weight of
  the edge between two sensors is `exp(-(distance / std)**2)`, where `std` is
  the standard deviation of the observed distances, and weights below
  `threshold` are dropped to make the graph sparse. Edges are directed: `cost`
  is the driving distance from the `from` sensor to the `to` sensor. Self loops
  are excluded, since each query node is already connected to its own sensor.

  Args:
    distances: Frame with the `from`, `to` and `cost` columns.
    sensor_ids: Sensor ids, in the node order of the sensor node set.
    threshold: Weights strictly below this value are dropped.

  Returns:
    A tuple of (sources, targets, weights) arrays of equal length, where
    sources and targets index into `sensor_ids`.
  """
  num_sensors = len(sensor_ids)
  sensor_index = {sensor_id: index for index, sensor_id in enumerate(sensor_ids)}

  pairwise_distances = np.full(
      (num_sensors, num_sensors), np.inf, dtype=np.float32
  )
  known_pairs = distances["from"].isin(sensor_index) & distances["to"].isin(
      sensor_index
  )
  known_distances = distances[known_pairs]
  if known_distances.empty:
    raise ValueError(
        "None of the distances relate two sensors of the dataset. The"
        " distances and the speed recordings are likely inconsistent."
    )
  sources = known_distances["from"].map(sensor_index).to_numpy(dtype=np.int64)
  targets = known_distances["to"].map(sensor_index).to_numpy(dtype=np.int64)
  pairwise_distances[sources, targets] = known_distances["cost"].to_numpy(
      dtype=np.float32
  )

  finite_distances = pairwise_distances[np.isfinite(pairwise_distances)]
  distance_std = finite_distances.std()
  if distance_std == 0.0:
    raise ValueError(
        "All the road network distances are identical, the Gaussian kernel is"
        " undefined."
    )
  weights = np.where(
      np.isfinite(pairwise_distances),
      np.exp(-np.square(pairwise_distances / distance_std)),
      0.0,
  )
  weights[weights < threshold] = 0.0
  np.fill_diagonal(weights, 0.0)

  edge_sources, edge_targets = np.nonzero(weights)
  return (
      edge_sources.astype(np.int64),
      edge_targets.astype(np.int64),
      weights[edge_sources, edge_targets].astype(np.float32),
  )


def build_traffic_graph(
    speeds: pd.DataFrame,
    distances: pd.DataFrame,
    locations: pd.DataFrame,
    forecast_horizon_seconds: int = 900,
    query_step: int = 1,
    adjacency_threshold: float = 0.1,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Constructs a DGF InMemoryGraph and GraphSchema from traffic recordings.

  Graph structure:
    - One sensor node per loop detector, holding the full speed time series and
      the coordinates of the detector.
    - Directed sensor to sensor edges, weighted by road network proximity.
    - Query nodes, one per (sensor, query time) pair, with `creation_time` and
      the regression label `speed` (the speed of the sensor at `creation_time +
      horizon`). Each query node is connected to its own sensor, so a model has
      to walk the sensor graph to see the neighboring detectors. Query times
      without a recording exactly one horizon ahead are dropped, and so are the
      queries whose target speed is missing (missing recordings are encoded as
      zeros).

  Args:
    speeds: Speed recordings, as returned by `download_traffic_speeds`.
    distances: Road network distances, as returned by
      `download_traffic_sensor_graph`.
    locations: Sensor coordinates, as returned by
      `download_traffic_sensor_graph`.
    forecast_horizon_seconds: Horizon into the future to predict (default: 900s
      = 15 minutes). Must be a multiple of the sampling period.
    query_step: Stride, in sampling periods, between two query times (default:
      1, i.e. one query per sensor and per recording, as in DCRNN).
    adjacency_threshold: Sensor edges with a weight below this value are
      dropped.

  Returns:
    A tuple of (InMemoryGraph, GraphSchema).
  """
  sensor_ids = [_decode_sensor_id(sensor_id) for sensor_id in speeds.columns]
  num_sensors = len(sensor_ids)
  timestamps = speeds.index.to_numpy(dtype=np.int64)
  values = speeds.to_numpy(dtype=np.float32)

  if len(timestamps) < 2:
    raise ValueError(
        f"The speed recordings only contain {len(timestamps)} time steps."
    )
  sampling_periods = np.diff(timestamps)
  if np.any(sampling_periods <= 0):
    raise ValueError(
        "The speed recordings are not sorted by strictly increasing time."
    )
  observed_periods, period_counts = np.unique(
      sampling_periods, return_counts=True
  )
  sampling_period_seconds = int(observed_periods[np.argmax(period_counts)])
  if forecast_horizon_seconds % sampling_period_seconds != 0:
    raise ValueError(
        f"The forecast horizon {forecast_horizon_seconds}s is not a multiple of"
        f" the {sampling_period_seconds}s sampling period."
    )
  num_gaps = int(np.sum(sampling_periods != sampling_period_seconds))
  if num_gaps:
    # PEMS-BAY is recorded in local time, so the daylight saving time change
    # leaves a one hour hole in the recordings.
    log.info(
        "The speed recordings have %d gap(s) in their %ds sampling period.",
        num_gaps,
        sampling_period_seconds,
    )

  # Sensor nodes.
  location_index = {
      sensor_id: index
      for index, sensor_id in enumerate(locations["sensor_id"].to_numpy())
  }
  sensors_without_location = [
      sensor_id for sensor_id in sensor_ids if sensor_id not in location_index
  ]
  if sensors_without_location:
    raise ValueError(
        f"{len(sensors_without_location)} sensors of the speed recordings are"
        f" missing from the locations, e.g. {sensors_without_location[:5]}."
    )
  location_positions = np.array(
      [location_index[sensor_id] for sensor_id in sensor_ids], dtype=np.int64
  )

  sensor_times = np.empty(num_sensors, dtype=object)
  sensor_speeds = np.empty(num_sensors, dtype=object)
  for index in range(num_sensors):
    sensor_times[index] = timestamps
    sensor_speeds[index] = np.ascontiguousarray(values[:, index])

  sensor_node_features: dict[str, np.ndarray] = {
      "#id": np.array(
          [f"sensor_{sensor_id}".encode("utf-8") for sensor_id in sensor_ids],
          dtype=np.bytes_,
      ),
      "time": sensor_times,
      "speed": sensor_speeds,
      "latitude": locations["latitude"].to_numpy(dtype=np.float32)[
          location_positions
      ],
      "longitude": locations["longitude"].to_numpy(dtype=np.float32)[
          location_positions
      ],
  }
  sensor_schema_features: dict[str, schema_lib.FeatureSchema] = {
      "#id": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BYTES,
          semantic=schema_lib.FeatureSemantic.PRIMARY_ID,
      ),
      "time": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.INTEGER_64,
          semantic=schema_lib.FeatureSemantic.TIMESTAMP,
          is_timeseries=True,
          is_creation_time=True,
          group="speed_ts",
          shape=(None,),
      ),
      "speed": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
          is_timeseries=True,
          group="speed_ts",
          shape=(None,),
      ),
      "latitude": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
      ),
      "longitude": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
      ),
  }

  # Query nodes. Queries are generated time major, i.e. all the sensors of a
  # query time are consecutive, so that a chronological split never splits the
  # queries of a same time between two splits.
  #
  # The target of a query is resolved by timestamp rather than by index, so
  # that a query right before a gap in the recordings is dropped instead of
  # being paired with a target further ahead than the forecast horizon.
  index_by_timestamp = {
      int(timestamp): index for index, timestamp in enumerate(timestamps)
  }
  candidate_time_indices = np.arange(0, len(timestamps), query_step)
  target_time_indices = np.array(
      [
          index_by_timestamp.get(
              int(timestamps[index]) + forecast_horizon_seconds, -1
          )
          for index in candidate_time_indices
      ],
      dtype=np.int64,
  )
  has_target = target_time_indices >= 0
  num_times_without_target = int(np.sum(~has_target))
  if num_times_without_target:
    log.info(
        "Dropping %d of the %d query times which have no recording %ds ahead.",
        num_times_without_target,
        len(candidate_time_indices),
        forecast_horizon_seconds,
    )
  query_time_indices = candidate_time_indices[has_target]
  target_time_indices = target_time_indices[has_target]
  num_query_times = len(query_time_indices)
  if num_query_times == 0:
    raise ValueError(
        f"No query time is left with a horizon of {forecast_horizon_seconds}s"
        f" and a query step of {query_step}."
    )

  query_times = np.repeat(timestamps[query_time_indices], num_sensors)
  query_sensors = np.tile(np.arange(num_sensors, dtype=np.int64),
                          num_query_times)
  target_speeds = values[target_time_indices].reshape(-1)

  # Chronological 70/10/20 split, matching the canonical METR-LA and PEMS-BAY
  # protocol introduced by DCRNN (https://arxiv.org/abs/1707.01926), so that
  # test metrics are comparable to published baselines.
  num_train_times = int(num_query_times * 0.7)
  num_valid_times = int(num_query_times * 0.1)
  time_splits = np.full(num_query_times, "n/a", dtype="S5")
  time_splits[:num_train_times] = b"train"
  time_splits[num_train_times : num_train_times + num_valid_times] = b"valid"
  time_splits[num_train_times + num_valid_times :] = b"test"
  split_labels = np.repeat(time_splits, num_sensors)

  # A missing recording is indistinguishable from a genuine zero speed, so a
  # query pointing at one cannot be labelled. This removes around 10% of the
  # METR-LA queries, and none of the PEMS-BAY ones.
  observed = target_speeds != TRAFFIC_MISSING_SPEED
  num_missing_targets = int(np.sum(~observed))
  if num_missing_targets:
    log.info(
        "Dropping %d of the %d queries (%.1f%%) whose target speed is missing.",
        num_missing_targets,
        len(target_speeds),
        100.0 * num_missing_targets / len(target_speeds),
    )
  query_times = query_times[observed]
  query_sensors = query_sensors[observed]
  target_speeds = target_speeds[observed]
  split_labels = split_labels[observed]

  num_queries = len(query_times)
  if num_queries == 0:
    raise ValueError("All the target speeds are missing.")

  query_node_features: dict[str, np.ndarray] = {
      "#id": generate_ids("query_", num_queries),
      "creation_time": query_times,
      "speed": target_speeds,
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
      "speed": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.FLOAT_32,
          semantic=schema_lib.FeatureSemantic.NUMERICAL,
      ),
      "#split": schema_lib.FeatureSchema(
          format=schema_lib.FeatureFormat.BYTES,
          semantic=schema_lib.FeatureSemantic.CATEGORICAL,
      ),
  }

  # Edges.
  query_indices = np.arange(num_queries, dtype=np.int64)
  sensor_sources, sensor_targets, sensor_weights = build_sensor_adjacency(
      distances, sensor_ids, threshold=adjacency_threshold
  )

  graph = in_memory_graph_lib.InMemoryGraph(
      node_sets={
          "queries": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=num_queries,
              features=query_node_features,
          ),
          "sensors": in_memory_graph_lib.InMemoryNodeSet(
              num_nodes=num_sensors,
              features=sensor_node_features,
          ),
      },
      edge_sets={
          "query_to_sensor": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.stack([query_indices, query_sensors], axis=0),
              features={},
          ),
          "sensor_to_query": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.stack([query_sensors, query_indices], axis=0),
              features={},
          ),
          "sensor_to_sensor": in_memory_graph_lib.InMemoryEdgeSet(
              adjacency=np.stack([sensor_sources, sensor_targets], axis=0),
              features={"weight": sensor_weights},
          ),
      },
  )

  schema = schema_lib.GraphSchema(
      node_sets={
          "queries": schema_lib.NodeSchema(features=query_schema_features),
          "sensors": schema_lib.NodeSchema(features=sensor_schema_features),
      },
      edge_sets={
          "query_to_sensor": schema_lib.EdgeSchema(
              source="queries",
              target="sensors",
              features={},
          ),
          "sensor_to_query": schema_lib.EdgeSchema(
              source="sensors",
              target="queries",
              features={},
          ),
          "sensor_to_sensor": schema_lib.EdgeSchema(
              source="sensors",
              target="sensors",
              features={
                  "weight": schema_lib.FeatureSchema(
                      format=schema_lib.FeatureFormat.FLOAT_32,
                      semantic=schema_lib.FeatureSemantic.NUMERICAL,
                  )
              },
          ),
      },
  )

  return graph, schema


def fetch_traffic_graph(
    dataset: str,
    name: str,
    cache_dir: Optional[str] = "AUTO",
    verbose: bool = True,
    forecast_horizon_seconds: int = 900,
    query_step: int = 1,
    adjacency_threshold: float = 0.1,
    repo: Union[Repo, str] = Repo.AUTO,
    speeds_source: Optional[str] = None,
    distances_source: Optional[str] = None,
    locations_source: Optional[str] = None,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Downloads and loads a traffic speed forecasting benchmark into memory.

  Both supported datasets record the speed of highway loop detectors every five
  minutes: METR-LA covers 207 detectors of the Los Angeles county highways over
  four months of 2012, and PEMS-BAY covers 325 detectors of the Bay Area over
  six months of 2017. They are the reference benchmarks for spatio-temporal
  forecasting, and combine a graph (the road network) with one time series per
  node.

  Usage example:

  ```python
  graph, schema = dgf.io.fetch_metr_la_graph()
  dgf.analyse.print_schema(schema)
  ```

  See `build_traffic_graph` for the structure of the returned graph.

  Args:
    dataset: Name of the dataset, e.g. "metr_la" or "pems_bay".
    name: The name of the dataset under the CNS fetch repo, e.g.
      "metr_la_15m".
    cache_dir: Optional. Directory to cache the graph in order to avoid
      re-downloading/re-parsing it each time. If "AUTO", uses the OS default
      temporary directory. If None, does not cache the graph.
    verbose: Optional. Whether to print cache and download progress.
    forecast_horizon_seconds: Forecasting horizon in seconds (default: 900 for
      15 minutes ahead).
    query_step: Stride, in sampling periods, between two query times (default:
      1, i.e. one query per sensor and per recording, as in DCRNN).
    adjacency_threshold: Sensor edges with a weight below this value are
      dropped.
    repo: Define the source of the data (Repo.AUTO, Repo.CNS, Repo.WEB).
    speeds_source: Optional URL or file path of the speed recordings.
    distances_source: Optional URL or file path of the road network distances.
    locations_source: Optional URL or file path of the sensor locations.

  Returns:
    An InMemoryGraph instance and its GraphSchema.
  """
  spec = traffic_dataset_spec(dataset)

  if cache_dir == "AUTO":
    cache_dir = os.path.join(tempfile.gettempdir(), "gf_fetch_traffic")

  if cache_dir is not None:
    fs.makedirs(cache_dir)
    cache_key = (
        f"{name}_h{forecast_horizon_seconds}_qs{query_step}_"
        f"at{adjacency_threshold}.cache"
    )
    cache_graph_path = os.path.join(cache_dir, cache_key)
    if verbose:
      log.info("Caching the %s graph at %s", dataset, cache_graph_path)
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
            "Loading the %s data from %s",
            dataset,
            speeds_source or spec.speeds_url,
        )
      speeds = download_traffic_speeds(dataset, source=speeds_source)
      distances, locations = download_traffic_sensor_graph(
          dataset,
          distances_source=distances_source,
          locations_source=locations_source,
      )
      return build_traffic_graph(
          speeds=speeds,
          distances=distances,
          locations=locations,
          forecast_horizon_seconds=forecast_horizon_seconds,
          query_step=query_step,
          adjacency_threshold=adjacency_threshold,
      )

  else:
    raise ValueError(f"Unsupported repo for the traffic datasets: {repo}")

  if cache_graph_path is None:
    return loader()
  else:
    return cache_lib.cache(cache_graph_path, loader)


def fetch_metr_la_graph(
    name: str = "metr_la_15m",
    cache_dir: Optional[str] = "AUTO",
    verbose: bool = True,
    forecast_horizon_seconds: int = 900,
    query_step: int = 1,
    adjacency_threshold: float = 0.1,
    repo: Union[Repo, str] = Repo.AUTO,
    speeds_source: Optional[str] = None,
    distances_source: Optional[str] = None,
    locations_source: Optional[str] = None,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Downloads and loads the METR-LA traffic benchmark into memory.

  METR-LA contains the speed of 207 loop detectors of the Los Angeles county
  highways, recorded every five minutes between March and June 2012, together
  with the road network distances between the detectors.

  Usage example:

  ```python
  graph, schema = dgf.io.fetch_metr_la_graph()
  dgf.analyse.print_schema(schema)
  ```

  See `build_traffic_graph` for the structure of the returned graph, and
  `fetch_traffic_graph` for the arguments, which are forwarded unchanged.

  Returns:
    An InMemoryGraph instance and its GraphSchema.
  """
  return fetch_traffic_graph(
      dataset="metr_la",
      name=name,
      cache_dir=cache_dir,
      verbose=verbose,
      forecast_horizon_seconds=forecast_horizon_seconds,
      query_step=query_step,
      adjacency_threshold=adjacency_threshold,
      repo=repo,
      speeds_source=speeds_source,
      distances_source=distances_source,
      locations_source=locations_source,
  )


def fetch_pems_bay_graph(
    name: str = "pems_bay_15m",
    cache_dir: Optional[str] = "AUTO",
    verbose: bool = True,
    forecast_horizon_seconds: int = 900,
    query_step: int = 1,
    adjacency_threshold: float = 0.1,
    repo: Union[Repo, str] = Repo.AUTO,
    speeds_source: Optional[str] = None,
    distances_source: Optional[str] = None,
    locations_source: Optional[str] = None,
) -> Tuple[in_memory_graph_lib.InMemoryGraph, schema_lib.GraphSchema]:
  """Downloads and loads the PEMS-BAY traffic benchmark into memory.

  PEMS-BAY contains the speed of 325 loop detectors of the San Francisco Bay
  Area highways, recorded every five minutes during the first six months of
  2017, together with the road network distances between the detectors.

  Usage example:

  ```python
  graph, schema = dgf.io.fetch_pems_bay_graph()
  dgf.analyse.print_schema(schema)
  ```

  See `build_traffic_graph` for the structure of the returned graph, and
  `fetch_traffic_graph` for the arguments, which are forwarded unchanged.

  Returns:
    An InMemoryGraph instance and its GraphSchema.
  """
  return fetch_traffic_graph(
      dataset="pems_bay",
      name=name,
      cache_dir=cache_dir,
      verbose=verbose,
      forecast_horizon_seconds=forecast_horizon_seconds,
      query_step=query_step,
      adjacency_threshold=adjacency_threshold,
      repo=repo,
      speeds_source=speeds_source,
      distances_source=distances_source,
      locations_source=locations_source,
  )

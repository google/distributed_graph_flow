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

"""Analysis of a trained NodePredictionModel for model interpretation.

The analysis measures how much the model relies on each component of the graph
(node features, edge features, edgesets and nodesets): each component is
perturbed in turn (see `dgf.src.transform.perturb`) and the degradation of the
model quality is recorded.

Main entry points:
  - `analyze_node_model`: Runs the analysis. Called by `model.analyze(graph)`.
  - `AnalyzeOptions`: Options of the analysis.
  - `ModelAnalysis`: Result of the analysis. Displayed as text in a terminal
    or as an interactive HTML report in a notebook.
  - `html_model_analysis` / `text_model_analysis`: Rendering of a
    `ModelAnalysis`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import dataclasses
import enum
import html
import time
from typing import Any, TYPE_CHECKING
import uuid

import dataclasses_json
from dgf.src.data import in_memory_graph
from dgf.src.data import schema as schema_lib
from dgf.src.learning.jax.layers import classification as classification_lib
from dgf.src.learning.jax.layers import regression as regression_lib
from dgf.src.learning.ten_lines import common
from dgf.src.learning.ten_lines import report
from dgf.src.sampling import in_memory_sampler as in_memory_sampler_lib
from dgf.src.transform import merge as merge_lib
from dgf.src.transform import perturb
from dgf.src.util import html_util
from dgf.src.util import log
from dgf.src.util import temporal as temporal_util
from dgf.src.util import util
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import optax
import pandas as pd
import tqdm

if TYPE_CHECKING:
  from dgf.src.learning.ten_lines import node_prediction_model

# Maximum number of seed nodes evaluated per perturbation when the number of
# seed nodes is selected automatically.
_MAX_AUTO_NUM_SEED_NODES = 512

# Maximum number of seed nodes for which the sampled sub-graphs of the baseline
# evaluation are kept in memory and re-used for the feature perturbations.
_MAX_CACHED_SEED_NODES = 2048


class ImportanceCategory(enum.Enum):
  """Type of an analyzed graph component."""

  NODE_FEATURE = "NODE_FEATURE"
  EDGE_FEATURE = "EDGE_FEATURE"
  EDGESET = "EDGESET"
  NODESET = "NODESET"


def parse_importance_category(
    category: ImportanceCategory | str,
) -> ImportanceCategory:
  """Parses a string or `ImportanceCategory` into an `ImportanceCategory`."""
  if isinstance(category, ImportanceCategory):
    return category
  if not isinstance(category, str):
    raise TypeError(
        f"Expected ImportanceCategory or str, got {type(category)}: {category}"
    )
  try:
    return ImportanceCategory(category)
  except ValueError as exc:
    raise ValueError(
        f"Unknown importance category: {category}. The supported values are:"
        f" {[item.value for item in ImportanceCategory]}."
    ) from exc


@dataclasses.dataclass
class AnalyzeOptions:
  """Options of `model.analyze(graph)`.

  Attributes:
    max_duration: Maximum duration of the analysis in seconds. Increase to
      improve the precision of the importances. If None, no time limit.
    num_seed_nodes: Number of seed nodes used to evaluate each perturbation.
      More nodes give more precise importances but take longer. If None,
      selected automatically to fit in `max_duration`.
    seed_node_idxs: Restrict the evaluation to these seed nodes of the target
      nodeset. If None, all the nodes are candidates.
    edge_perturbation_mode: How edgesets are perturbed. See
      `EdgePerturbationMode`.
    num_swaps_per_edge: Amount of rewiring for `MASLOV_SNEPPEN`. Larger values
      destroy more of the original topology.
    compute_gradient_importance: Whether to also compute the gradient saliency
      of the numerical features.
    include_nodesets: Whether to also compute the importance of nodesets.
    random_seed: Seed of the random perturbations and sampling.
    verbose: Verbosity level (0=silent, 1=logs, 2=logs and progress bar).
  """

  max_duration: float | None = 60.0
  num_seed_nodes: int | None = None
  seed_node_idxs: common.SeedNodeIdxs | None = None
  edge_perturbation_mode: perturb.EdgePerturbationMode | str = (
      perturb.EdgePerturbationMode.MASLOV_SNEPPEN
  )
  num_swaps_per_edge: int = 4
  compute_gradient_importance: bool = True
  include_nodesets: bool = True
  random_seed: int = 42
  verbose: int = 2

  def __post_init__(self) -> None:
    self.edge_perturbation_mode = perturb.parse_edge_perturbation_mode(
        self.edge_perturbation_mode
    )


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class ImportanceItem:
  """Importance of a single feature, edgeset or nodeset.

  Attributes:
    category: Type of the component.
    name: Name of the component, e.g. `"paper.feat"`, `"cites"` or `"author"`.
    delta_loss: Increase of the loss when the component is perturbed. This is
      the primary importance score: larger is more important.
    delta_metric: Degradation of the quality metric when the component is
      perturbed: drop of accuracy for classification, increase of RMSE for
        regression.
    perturbed_loss: Loss when the component is perturbed.
    perturbed_metric: Quality metric when the component is perturbed.
    gradient_importance: Gradient saliency of the feature, i.e. mean L1 norm of
      the gradient of the loss with respect to the feature values. None for
      non-numerical features, edgesets and nodesets.
  """

  category: ImportanceCategory
  name: str
  delta_loss: float
  delta_metric: float
  perturbed_loss: float
  perturbed_metric: float
  gradient_importance: float | None = None

  def __post_init__(self) -> None:
    self.category = parse_importance_category(self.category)


def _sort_items(items: list[ImportanceItem]) -> list[ImportanceItem]:
  """Sorts items by decreasing importance."""
  return sorted(
      items,
      key=lambda x: (x.delta_loss, x.delta_metric, x.gradient_importance or 0),
      reverse=True,
  )


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class ModelAnalysis:
  """Result of `model.analyze(graph)`.

  Attributes:
    task_type: `"NODE_CLASSIFICATION"` or `"NODE_REGRESSION"`.
    target_nodeset: Name of the target nodeset.
    target_column: Name of the target column.
    metric_name: Name of the quality metric (`"accuracy"` or `"rmse"`).
    baseline_metric: Quality metric on the unperturbed graph.
    baseline_loss: Loss on the unperturbed graph.
    num_seed_nodes: Number of seed nodes used to evaluate each perturbation.
    edge_perturbation_mode: How edgesets were perturbed.
    duration_seconds: Duration of the analysis in seconds.
    node_features: Importance of the node features, most important first.
    edge_features: Importance of the edge features, most important first.
    edgesets: Importance of the edgesets, most important first.
    nodesets: Importance of the nodesets, most important first.
  """

  task_type: str
  target_nodeset: str
  target_column: str
  metric_name: str
  baseline_metric: float
  baseline_loss: float
  num_seed_nodes: int
  edge_perturbation_mode: perturb.EdgePerturbationMode
  duration_seconds: float
  node_features: list[ImportanceItem] = dataclasses.field(default_factory=list)
  edge_features: list[ImportanceItem] = dataclasses.field(default_factory=list)
  edgesets: list[ImportanceItem] = dataclasses.field(default_factory=list)
  nodesets: list[ImportanceItem] = dataclasses.field(default_factory=list)

  def __post_init__(self) -> None:
    self.edge_perturbation_mode = perturb.parse_edge_perturbation_mode(
        self.edge_perturbation_mode
    )

  @property
  def features(self) -> list[ImportanceItem]:
    """All the node and edge features, most important first."""
    return _sort_items(self.node_features + self.edge_features)

  @property
  def all_items(self) -> list[ImportanceItem]:
    """All the features, edgesets and nodesets, most important first."""
    return _sort_items(self.features + self.edgesets + self.nodesets)

  def to_dataframe(self) -> pd.DataFrame:
    """Converts the importances into a pandas DataFrame."""
    rows = [
        {**dataclasses.asdict(item), "category": item.category.value}
        for item in self.all_items
    ]
    return pd.DataFrame(rows)

  def __repr__(self) -> str:
    return text_model_analysis(self)

  def _repr_html_(self) -> str:
    return html_model_analysis(self)

  def html(self) -> str:
    """Returns the HTML representation of the analysis."""
    return html_model_analysis(self)


# Rendering
# =========


def _delta_metric_label(metric_name: str) -> str:
  """Label of the metric degradation column, e.g. "Δ Accuracy"."""
  return (
      "Δ Accuracy" if metric_name == "accuracy" else f"Δ {metric_name.upper()}"
  )


def text_model_analysis(analysis: ModelAnalysis) -> str:
  """Text representation of a `ModelAnalysis`."""
  lines = [
      "ModelAnalysis(",
      (
          f"  target={analysis.target_nodeset}.{analysis.target_column},"
          f" task={analysis.task_type},"
          f" baseline_{analysis.metric_name}={analysis.baseline_metric:.4f},"
          f" baseline_loss={analysis.baseline_loss:.4f},"
          f" num_seed_nodes={analysis.num_seed_nodes},"
          f" edge_perturbation_mode={analysis.edge_perturbation_mode.value},"
          f" duration={util.format_duration(analysis.duration_seconds)}"
      ),
  ]
  for title, items in [
      ("Features", analysis.features),
      ("Edgesets", analysis.edgesets),
      ("Nodesets", analysis.nodesets),
  ]:
    if not items:
      continue
    lines.append(f"  [{title}]")
    for rank, item in enumerate(items, start=1):
      line = (
          f"    {rank:2d}. {item.name:<28}"
          f" delta_loss={item.delta_loss:+.4f},"
          f" delta_{analysis.metric_name}={item.delta_metric:+.4f}"
      )
      if item.gradient_importance is not None:
        line += f", grad_saliency={item.gradient_importance:.4f}"
      lines.append(line)
  lines.append(")")
  return "\n".join(lines)


def _html_details_table(analysis: ModelAnalysis) -> str:
  """HTML table with the details of the analysis."""
  rows = [
      (
          "Target",
          (
              f"{analysis.target_nodeset}.{analysis.target_column}"
              f" ({analysis.task_type})"
          ),
      ),
      (f"Baseline {analysis.metric_name}", f"{analysis.baseline_metric:.4f}"),
      ("Baseline loss", f"{analysis.baseline_loss:.4f}"),
      ("Number of seed nodes", str(analysis.num_seed_nodes)),
      (
          "Perturbation",
          (
              "Feature shuffle, edgeset"
              f" {analysis.edge_perturbation_mode.value.lower()}"
          ),
      ),
      ("Duration", util.format_duration(analysis.duration_seconds)),
  ]
  table_rows = "".join(
      f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>\n"
      for k, v in rows
  )
  return f'<table class="dgf-table"><tbody>\n{table_rows}</tbody></table>'


def _html_bar_cell(value: float | None, max_value: float, fmt: str) -> str:
  """HTML table cell showing a value and a proportional bar."""
  if value is None:
    return '<td data-sort-value="-Infinity"><span class="dgf-na">—</span></td>'
  pct = 100.0 * max(0.0, value) / max_value if max_value > 1e-12 else 0.0
  return (
      f'<td data-sort-value="{value:.8g}"><div class="dgf-bar-cell">'
      f'<span class="dgf-bar-val">{value:{fmt}}</span>'
      '<div class="dgf-bar-bg"><div class="dgf-bar-fill"'
      f' style="width: {pct:.1f}%;"></div></div></div></td>'
  )


def _html_importance_table(analysis: ModelAnalysis, component_id: str) -> str:
  """HTML with the category filter and the sortable table of importances."""
  items = analysis.all_items
  has_gradient = any(it.gradient_importance is not None for it in items)
  max_loss = max([it.delta_loss for it in items] + [0.0])
  max_metric = max([it.delta_metric for it in items] + [0.0])
  max_grad = max([it.gradient_importance or 0.0 for it in items] + [0.0])
  fn_suffix = component_id.replace("-", "_")

  # Filter buttons.
  filters = [("ALL", "All", len(items))]
  for category, label, category_items in [
      (
          ImportanceCategory.NODE_FEATURE,
          "Node features",
          analysis.node_features,
      ),
      (
          ImportanceCategory.EDGE_FEATURE,
          "Edge features",
          analysis.edge_features,
      ),
      (ImportanceCategory.EDGESET, "Edgesets", analysis.edgesets),
      (ImportanceCategory.NODESET, "Nodesets", analysis.nodesets),
  ]:
    if category_items:
      filters.append((category.value, label, len(category_items)))
  parts = ['<div class="dgf-filter-bar">']
  for idx, (value, label, count) in enumerate(filters):
    active = " active" if idx == 0 else ""
    parts.append(
        f'<button type="button" class="dgf-filter-btn{active}"'
        f" onclick=\"dgfFilter_{fn_suffix}('{value}', this)\">"
        f"{label} ({count})</button>"
    )
  parts.append("</div>")

  # Table header.
  columns = [
      ("Name", "str"),
      ("Type", "str"),
      ("Δ Loss", "num"),
      (_delta_metric_label(analysis.metric_name), "num"),
  ]
  if has_gradient:
    columns.append(("Gradient saliency", "num"))
  parts.append(
      f'<table id="{component_id}-table" class="dgf-table dgf-sortable">'
      "<thead><tr>"
  )
  for col_idx, (title, sort_type) in enumerate(columns):
    indicator = "▼" if col_idx == 2 else ""
    parts.append(
        f'<th data-sort-dir="{"desc" if col_idx == 2 else ""}"'
        f" onclick=\"dgfSort_{fn_suffix}({col_idx}, '{sort_type}')\">"
        f"{html.escape(title)}<span"
        f' class="dgf-sort-ind">{indicator}</span></th>'
    )
  parts.append("</tr></thead><tbody>")

  # Table rows.
  for item in items:
    category = item.category.value
    cells = [
        (
            f'<td data-sort-value="{html.escape(item.name.lower())}">'
            f"<b>{html.escape(item.name)}</b></td>"
        ),
        (
            f'<td data-sort-value="{category}">'
            f'<span class="dgf-badge dgf-badge-{category}">'
            f"{category.lower()}</span></td>"
        ),
        _html_bar_cell(item.delta_loss, max_loss, "+.4f"),
        _html_bar_cell(item.delta_metric, max_metric, "+.4f"),
    ]
    if has_gradient:
      cells.append(_html_bar_cell(item.gradient_importance, max_grad, ".4f"))
    parts.append(f'<tr data-category="{category}">{"".join(cells)}</tr>')
  parts.append("</tbody></table>")

  parts.append(f"""
<script>
(function() {{
  var table = document.getElementById("{component_id}-table");
  if (!table) return;
  window.dgfFilter_{fn_suffix} = function(category, button) {{
    var buttons = button.parentNode.getElementsByClassName("dgf-filter-btn");
    for (var i = 0; i < buttons.length; i++) {{
      buttons[i].classList.remove("active");
    }}
    button.classList.add("active");
    var rows = table.tBodies[0].rows;
    for (var i = 0; i < rows.length; i++) {{
      var visible = category === "ALL" ||
          rows[i].getAttribute("data-category") === category;
      rows[i].style.display = visible ? "" : "none";
    }}
  }};
  window.dgfSort_{fn_suffix} = function(colIdx, sortType) {{
    var headers = table.tHead.rows[0].cells;
    var dir = headers[colIdx].getAttribute("data-sort-dir") === "desc" ?
        "asc" : "desc";
    for (var i = 0; i < headers.length; i++) {{
      headers[i].setAttribute("data-sort-dir", i === colIdx ? dir : "");
      headers[i].querySelector(".dgf-sort-ind").textContent =
          i === colIdx ? (dir === "asc" ? "▲" : "▼") : "";
    }}
    var tbody = table.tBodies[0];
    var rows = Array.prototype.slice.call(tbody.rows);
    rows.sort(function(a, b) {{
      var va = a.cells[colIdx].getAttribute("data-sort-value");
      var vb = b.cells[colIdx].getAttribute("data-sort-value");
      var cmp = sortType === "num" ? parseFloat(va) - parseFloat(vb) :
          va.localeCompare(vb);
      return dir === "asc" ? cmp : -cmp;
    }});
    rows.forEach(function(r) {{ tbody.appendChild(r); }});
  }};
}})();
</script>
""")
  return "\n".join(parts)


def _html_glossary(analysis: ModelAnalysis) -> str:
  """HTML table defining the terms used in the report."""
  if analysis.metric_name == "accuracy":
    delta_metric_def = (
        "Drop of the accuracy when the component is perturbed. Larger is more"
        " important."
    )
  else:
    delta_metric_def = (
        f"Increase of the {analysis.metric_name.upper()} when the component is"
        " perturbed. Larger is more important."
    )
  terms = [
      (
          "Component",
          (
              "A node feature, an edge feature, an edgeset or a nodeset of the"
              " graph."
          ),
      ),
      (
          "Component importance",
          (
              "How much the model relies on a component to make its"
              " predictions. It is measured by perturbing the component and"
              " recording the degradation of the model quality. Components are"
              " ranked by Δ Loss."
          ),
      ),
      (
          "Perturbation",
          (
              "Destruction of the information carried by a component, while"
              " leaving the rest of the graph untouched. Feature values are"
              " shuffled across nodes or edges. Edgesets are rewired (by"
              " default, with degree-preserving Maslov-Sneppen swaps, so that"
              " every node keeps its number of neighbors). Nodesets have all"
              " their features shuffled and all their incident edgesets"
              " rewired."
          ),
      ),
      (
          "Baseline",
          (
              "Quality of the model on the unperturbed graph, on the same seed"
              " nodes."
          ),
      ),
      (
          "Δ Loss",
          (
              "Increase of the loss when the component is perturbed. Larger is"
              " more important. A value close to zero means the model can do"
              " without the component (possibly because other components carry"
              " the same information)."
          ),
      ),
      (_delta_metric_label(analysis.metric_name), delta_metric_def),
      (
          "Gradient saliency",
          (
              "Average magnitude of the gradient of the loss with respect to"
              " the values of a feature, on the unperturbed graph. It measures"
              " how sensitive the model is to small changes of the feature."
              " Only available for numerical features. A high saliency with a"
              " small Δ Loss indicates a feature used by the model but"
              " redundant with other features."
          ),
      ),
  ]
  table_rows = "".join(
      f"<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(v)}</td></tr>\n"
      for k, v in terms
  )
  return f'<table class="dgf-table"><tbody>\n{table_rows}</tbody></table>'


def html_model_analysis(analysis: ModelAnalysis) -> str:
  """HTML representation of a `ModelAnalysis`, with tabs."""
  component_id = f"dgf-analysis-{uuid.uuid4().hex[:8]}"
  style = f"""
<style>
{html_util.get_table_style(component_id)}
  #{component_id} .dgf-table th {{
    padding: 6px 8px;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
    color: #5f6368;
    border-bottom: 1px solid #e0e0e0;
  }}
  #{component_id} .dgf-sortable th {{ cursor: pointer; user-select: none; }}
  #{component_id} .dgf-sortable th:hover {{ color: #1a73e8; }}
  #{component_id} .dgf-sort-ind {{
    display: inline-block; width: 12px; margin-left: 4px; font-size: 10px;
  }}
  #{component_id} .dgf-table td b {{ display: inline; }}
  #{component_id} .dgf-filter-bar {{
    display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0;
  }}
  #{component_id} .dgf-filter-btn {{
    border: 1px solid #dadce0; background-color: #ffffff; color: #3c4043;
    border-radius: 16px; padding: 4px 12px; font-size: 12px; cursor: pointer;
  }}
  #{component_id} .dgf-filter-btn:hover {{ background-color: #f1f3f4; }}
  #{component_id} .dgf-filter-btn.active {{
    background-color: #1a73e8; color: #ffffff; border-color: #1a73e8;
  }}
  #{component_id} .dgf-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-family: 'Roboto Mono', monospace; font-size: 11px;
  }}
  #{component_id} .dgf-badge-NODE_FEATURE,
  #{component_id} .dgf-badge-EDGE_FEATURE {{
    background-color: #e8f0fe; color: #1967d2;
  }}
  #{component_id} .dgf-badge-EDGESET {{
    background-color: #e6f4ea; color: #137333;
  }}
  #{component_id} .dgf-badge-NODESET {{
    background-color: #fef7e0; color: #b06000;
  }}
  #{component_id} .dgf-bar-cell {{
    display: flex; align-items: center; gap: 8px; white-space: nowrap;
  }}
  #{component_id} .dgf-bar-val {{
    display: inline-block; min-width: 62px; text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  #{component_id} .dgf-na {{ color: #80868b; }}
  #{component_id} .dgf-bar-bg {{
    background-color: #e8eaed; border-radius: 3px; width: 80px; height: 10px;
    overflow: hidden; flex-shrink: 0;
  }}
  #{component_id} .dgf-bar-fill {{ height: 100%; background-color: #1a73e8; }}
</style>
"""
  tabs = [
      (
          "Component importance",
          "<b>Details</b>\n"
          + _html_details_table(analysis)
          + "<b>Importance</b>\n"
          + _html_importance_table(analysis, component_id),
      ),
      ("Glossary", _html_glossary(analysis)),
  ]
  return f'<div id="{component_id}">\n{style}\n{report.html_tabs(tabs)}\n</div>'


# Computation
# ===========


@dataclasses.dataclass
class _EvalResult:
  """Evaluation of the model on a (possibly perturbed) graph."""

  metric: float
  loss: float
  # Gradient saliency per feature name, e.g. {"paper.feat": 0.1}.
  gradient_saliency: dict[str, float] | None = None
  # Sampled sub-graphs, to re-evaluate the model on a graph with the same
  # topology but different node features.
  sampled_batches: list[list[in_memory_graph.InMemoryGraph]] | None = None


class _Evaluator:
  """Fast evaluation of a node prediction model on in-memory graphs."""

  def __init__(
      self,
      model: node_prediction_model.NodePredictionModel,
      options: AnalyzeOptions,
  ):
    data = model.data()
    if data.sampling_plan is None:
      raise ValueError(
          "`model.analyze(graph)` requires a model with a `sampling_plan`."
      )
    assert data.model_params is not None
    self._options = options
    self.data = data
    self.schema = data.schema
    self._live = model._get_live()  # pylint: disable=protected-access
    self._sampling_plan = data.sampling_plan
    self.params = data.model_params
    self.target_nodeset = data.task.target_nodeset
    self.target_column = data.task.target_column
    self.is_regression = data.task.task_type.value == "NODE_REGRESSION"
    self.batch_size = max(1, data.hparams.batch_size // 2)
    self._merger = merge_lib.GraphMerger(
        schema=self.schema, padding=data.padding
    )
    normalizer = self._live.normalizer
    self._target_norm_column = (
        data.task.normalized_target_column
        or normalizer.get_normalized_feature_names(
            self.target_nodeset, self.target_column
        )[0]
    )
    self._timestamp_feature = temporal_util.creation_time_feature_name(
        self.schema.node_sets[self.target_nodeset].features
    )

    # Mapping from (set name, normalized feature name) to the name of the
    # analyzed component (see `_component_name`), used to attribute the
    # gradient saliency.
    self._normalized_to_raw: dict[tuple[str, str], str] = {}
    for ns_name, ns_schema in self.schema.node_sets.items():
      for feature_name, feature_schema in ns_schema.features.items():
        for norm_name in normalizer.get_normalized_feature_names(
            ns_name, feature_name
        ):
          self._normalized_to_raw[(ns_name, norm_name)] = _component_name(
              ns_name, feature_name, feature_schema
          )
    for es_name, es_config in normalizer.config.edgesets.items():
      es_schema = self.schema.edge_sets[es_name]
      for feature_normalizer in es_config.normalizers:
        feature_name = feature_normalizer.input_feature
        for norm_name in feature_normalizer.output_schema():
          self._normalized_to_raw[(es_name, norm_name)] = _component_name(
              es_name, feature_name, es_schema.features[feature_name]
          )
    self._normalized_to_raw.pop(
        (self.target_nodeset, self._target_norm_column), None
    )

    self._eval_batch_fn = jax.jit(self._eval_batch)
    self._grad_batch_fn = jax.jit(jax.grad(self._masked_loss, argnums=(0, 1)))

  def _losses(
      self, logits: Any, labels: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    """Returns the (predictions, per-example losses) from the model logits."""
    if self.is_regression:
      head_config = self.data.core_model_config.head
      assert isinstance(head_config, regression_lib.RegressionHeadConfig)
      preds = head_config.logits_to_predictions(logits)
      losses = optax.squared_error(preds, labels)
      if losses.ndim > 1:
        # Multi-dimensional regression target: average over the dimensions.
        losses = jnp.mean(losses, axis=-1)
      return preds, losses
    probs = classification_lib.ClassificationHead.logits_to_probability(logits)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return probs, losses

  def _labels(self, batch: node_prediction_model.Batch) -> jax.Array:
    jax_graph, seed_idxs = batch
    return jax_graph.node_sets[self.target_nodeset].features[
        self._target_norm_column
    ][seed_idxs]

  def _eval_batch(
      self, params: jaxtyping.PyTree, batch: node_prediction_model.Batch
  ) -> tuple[jax.Array, jax.Array]:
    """Returns the (predictions, per-example losses) of a batch."""
    logits = self._live.core_model.apply(params, batch, training=False)
    return self._losses(logits, self._labels(batch))

  def _masked_loss(
      self,
      node_features: dict[tuple[str, str], jax.Array],
      edge_features: dict[tuple[str, str], jax.Array],
      params: jaxtyping.PyTree,
      batch: node_prediction_model.Batch,
      seed_mask: jax.Array,
  ) -> jax.Array:
    """Sum of the losses of a batch, differentiable w.r.t. the features."""
    jax_graph, seed_idxs = batch
    node_sets = {
        name: dataclasses.replace(
            ns,
            features={
                **ns.features,
                **{f: a for (n, f), a in node_features.items() if n == name},
            },
        )
        for name, ns in jax_graph.node_sets.items()
    }
    edge_sets = {
        name: dataclasses.replace(
            es,
            features={
                **es.features,
                **{f: a for (n, f), a in edge_features.items() if n == name},
            },
        )
        for name, es in jax_graph.edge_sets.items()
    }
    new_graph = dataclasses.replace(
        jax_graph, node_sets=node_sets, edge_sets=edge_sets
    )
    logits = self._live.core_model.apply(
        params, (new_graph, seed_idxs), training=False
    )
    _, losses = self._losses(logits, self._labels(batch))
    return jnp.sum(losses * seed_mask)

  def _seed_timestamps(
      self, graph: in_memory_graph.InMemoryGraph, seed_idxs: np.ndarray
  ) -> np.ndarray | None:
    """Creation timestamps of the seed nodes, if any."""
    if self._timestamp_feature is None:
      return None
    timestamps = graph.node_sets[self.target_nodeset].features[
        self._timestamp_feature
    ]
    return np.asarray(timestamps[seed_idxs]).reshape(-1)

  def create_sampler(
      self, graph: in_memory_graph.InMemoryGraph, return_node_idxs: bool = False
  ) -> in_memory_sampler_lib.Sampler:
    return in_memory_sampler_lib.create_sampler(
        graph=graph,
        plan=self._sampling_plan,
        schema=self.schema,
        batch_size=self.batch_size,
        seed=self._options.random_seed,
        return_node_idxs=return_node_idxs,
    )

  def _sample_batches(
      self,
      graph: in_memory_graph.InMemoryGraph,
      seed_idxs: np.ndarray,
      sampler: in_memory_sampler_lib.Sampler,
  ) -> Iterator[list[in_memory_graph.InMemoryGraph]]:
    """Samples the sub-graphs of the seed nodes, batch by batch."""
    for batch_idxs in util.batch_indices_generator(
        seed_idxs,
        batch_size=self.batch_size,
        drop_remainder=False,
        shuffle=False,
    ):
      if self.data.temporal_sampling:
        yield sampler.sample(
            batch_idxs,
            seed_timestamps=self._seed_timestamps(graph, batch_idxs),
        )
      else:
        yield sampler.sample(batch_idxs)

  def _to_jax_batch(
      self,
      merged_graph: in_memory_graph.InMemoryGraph,
      merge_offsets: dict[str, np.ndarray],
  ) -> tuple[node_prediction_model.Batch, np.ndarray, jax.Array]:
    """Normalizes a merged graph into a (batch, labels, seed mask)."""
    seed_idxs = merge_offsets[self.target_nodeset][:-1]
    num_seeds = len(seed_idxs)
    normalizer_kwargs = {}
    if "seed_timestamps" in self._live.normalizer.accepted_kwargs:
      # Temporal normalizers (e.g. "age" features) need the seed timestamps.
      seed_timestamps = self._seed_timestamps(merged_graph, seed_idxs)
      assert seed_timestamps is not None
      normalizer_kwargs["seed_timestamps"] = (
          temporal_util.expand_batch_seed_timestamps(
              sample=merged_graph,
              merge_offsets=merge_offsets,
              schema=self.schema,
              seed_timestamps=seed_timestamps,
          )
      )
    jax_graph = self._live.normalizer.normalize_numpy_to_jax(
        merged_graph, **normalizer_kwargs
    )
    labels = np.asarray(
        jax_graph.node_sets[self.target_nodeset].features[
            self._target_norm_column
        ]
    )[seed_idxs]
    # Pad the seed indices to a fixed size to avoid JIT re-compilations.
    padded_seeds = np.zeros(self.batch_size, dtype=seed_idxs.dtype)
    padded_seeds[:num_seeds] = seed_idxs
    seed_mask = np.zeros(self.batch_size, dtype=np.float32)
    seed_mask[:num_seeds] = 1.0
    return (
        (jax_graph, jnp.asarray(padded_seeds)),
        labels,
        jnp.asarray(seed_mask),
    )

  def evaluate(
      self,
      graph: in_memory_graph.InMemoryGraph,
      seed_idxs: np.ndarray,
      *,
      compute_gradients: bool = False,
      return_sampled_batches: bool = False,
      cached_sampled_batches: (
          list[list[in_memory_graph.InMemoryGraph]] | None
      ) = None,
      sampler: in_memory_sampler_lib.Sampler | None = None,
  ) -> _EvalResult:
    """Evaluates the model on the seed nodes of a graph.

    Args:
      graph: Graph to evaluate on.
      seed_idxs: Seed nodes to evaluate on.
      compute_gradients: Whether to compute the gradient saliency of the
        numerical features.
      return_sampled_batches: Whether to return the sampled sub-graphs.
      cached_sampled_batches: Sub-graphs previously sampled on a graph with the
        same topology. Only the node features are refreshed from `graph`.
      sampler: Sampler on `graph`. If None, a new sampler is created.

    Returns:
      The evaluation result.
    """
    if cached_sampled_batches is not None:
      # Refresh the node features from `graph` (the topology is unchanged).
      for samples in cached_sampled_batches:
        in_memory_sampler_lib.add_features_to_samples(
            graph, samples, return_features=True, return_node_idxs=True
        )
      batches: Iterator[list[in_memory_graph.InMemoryGraph]] = iter(
          cached_sampled_batches
      )
    else:
      # Sample the sub-graphs from scratch.
      if sampler is None:
        sampler = self.create_sampler(
            graph, return_node_idxs=return_sampled_batches
        )
      batches = self._sample_batches(graph, seed_idxs, sampler)

    all_preds = []
    all_labels = []
    total_loss = 0.0
    grad_sums: dict[str, float] = {}
    sampled_batches = [] if return_sampled_batches else None
    for samples in batches:
      if sampled_batches is not None:
        sampled_batches.append(samples)
      for merged_graph, merge_offsets, _ in self._merger.merge_sub_batches(
          samples, split_overflow_padding_error=True
      ):
        batch, labels, seed_mask = self._to_jax_batch(
            merged_graph, merge_offsets
        )
        num_seeds = len(labels)
        preds, losses = self._eval_batch_fn(self.params, batch)
        all_preds.append(np.asarray(preds)[:num_seeds])
        all_labels.append(labels)
        total_loss += float(np.sum(np.asarray(losses)[:num_seeds]))
        if compute_gradients:
          self._accumulate_gradients(batch, seed_mask, grad_sums)

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    num_examples = max(1, len(labels))
    if self.is_regression:
      metric = float(np.sqrt(np.mean(np.square(labels - preds))))
    else:
      metric = float(np.mean(np.argmax(preds, axis=-1) == labels))
    return _EvalResult(
        metric=metric,
        loss=total_loss / num_examples,
        gradient_saliency=(
            {k: v / num_examples for k, v in grad_sums.items()}
            if compute_gradients
            else None
        ),
        sampled_batches=sampled_batches,
    )

  def _accumulate_gradients(
      self,
      batch: node_prediction_model.Batch,
      seed_mask: jax.Array,
      grad_sums: dict[str, float],
  ) -> None:
    """Adds the L1 norm of the loss gradient of each numerical feature."""
    jax_graph = batch[0]
    node_features = {
        (ns_name, f_name): arr
        for ns_name, ns in jax_graph.node_sets.items()
        for f_name, arr in ns.features.items()
        if (ns_name, f_name) in self._normalized_to_raw
        and jnp.issubdtype(arr.dtype, jnp.floating)
    }
    edge_features = {
        (es_name, f_name): arr
        for es_name, es in jax_graph.edge_sets.items()
        for f_name, arr in es.features.items()
        if (es_name, f_name) in self._normalized_to_raw
        and jnp.issubdtype(arr.dtype, jnp.floating)
    }
    node_grads, edge_grads = self._grad_batch_fn(
        node_features, edge_features, self.params, batch, seed_mask
    )
    for key, grad in {**node_grads, **edge_grads}.items():
      raw_name = self._normalized_to_raw[key]
      grad_sums[raw_name] = grad_sums.get(raw_name, 0.0) + float(
          jnp.sum(jnp.abs(grad))
      )


def _component_name(
    set_name: str, feature_name: str, feature_schema: schema_lib.FeatureSchema
) -> str:
  """Name of the analyzed component a feature belongs to.

  The features of a timeseries group are perturbed together and form a single
  component named after the group, e.g. "paper.clicks".
  """
  group = perturb.timeseries_group(feature_name, feature_schema)
  return f"{set_name}.{group or feature_name}"


# A perturbation task: (category, name, function creating the perturbed graph).
_PerturbTask = tuple[
    ImportanceCategory,
    str,
    Callable[
        [in_memory_graph.InMemoryGraph, np.random.Generator],
        in_memory_graph.InMemoryGraph,
    ],
]


def _list_perturbations(
    evaluator: _Evaluator, options: AnalyzeOptions
) -> list[_PerturbTask]:
  """Lists the perturbations to evaluate, following the schema."""
  data = evaluator.data
  schema = evaluator.schema
  target_nodeset = evaluator.target_nodeset
  target_column = evaluator.target_column
  edge_mode = perturb.parse_edge_perturbation_mode(
      options.edge_perturbation_mode
  )
  nodeset_ts = {}
  edgeset_ts = {}
  if data.temporal_sampling:
    nodeset_ts = temporal_util.nodeset_timestamp_features(schema)
    edgeset_ts = temporal_util.edgeset_timestamp_features(schema)

  tasks: list[_PerturbTask] = []
  names: set[str] = set()
  for ns_name, ns_schema in schema.node_sets.items():
    for feature_name, feature_schema in ns_schema.features.items():
      name = _component_name(ns_name, feature_name, feature_schema)
      if (
          (ns_name == target_nodeset and feature_name == target_column)
          or feature_name == nodeset_ts.get(ns_name)
          or name in names
      ):
        # The label and the creation timestamps are not model inputs, and a
        # timeseries group is perturbed once.
        continue
      names.add(name)
      tasks.append((
          ImportanceCategory.NODE_FEATURE,
          name,
          lambda g, rng, ns=ns_name, f=feature_name: (
              perturb.perturb_node_feature(g, schema, ns, f, rng)
          ),
      ))
  for es_name, es_schema in schema.edge_sets.items():
    for feature_name, feature_schema in es_schema.features.items():
      name = _component_name(es_name, feature_name, feature_schema)
      if feature_name == edgeset_ts.get(es_name) or name in names:
        # The creation timestamps are not model inputs, and a timeseries group
        # is perturbed once.
        continue
      names.add(name)
      tasks.append((
          ImportanceCategory.EDGE_FEATURE,
          name,
          lambda g, rng, es=es_name, f=feature_name: (
              perturb.perturb_edge_feature(g, schema, es, f, rng)
          ),
      ))
  for es_name in schema.edge_sets:
    tasks.append((
        ImportanceCategory.EDGESET,
        es_name,
        lambda g, rng, es=es_name: perturb.perturb_edgeset(
            g,
            schema,
            es,
            rng,
            mode=edge_mode,
            num_swaps_per_edge=options.num_swaps_per_edge,
            preserve_timestamps=data.temporal_sampling,
        ),
    ))
  if options.include_nodesets:
    for ns_name in schema.node_sets:
      tasks.append((
          ImportanceCategory.NODESET,
          ns_name,
          lambda g, rng, ns=ns_name: perturb.perturb_nodeset(
              g,
              schema,
              ns,
              rng,
              target_nodeset=target_nodeset,
              target_column=target_column,
              edge_perturbation_mode=edge_mode,
              preserve_timestamps=data.temporal_sampling,
          ),
      ))
  return tasks


def _select_seed_nodes(
    evaluator: _Evaluator,
    graph: in_memory_graph.InMemoryGraph,
    options: AnalyzeOptions,
    tasks: list[_PerturbTask],
    start_time: float,
) -> np.ndarray:
  """Selects the seed nodes used to evaluate each perturbation.

  If the number of seed nodes is not specified by the user, it is selected
  so that all the perturbations fit in the time budget: the duration of a
  batch evaluation and of a sampler creation are measured on the model.
  """
  num_nodes = graph.node_sets[evaluator.target_nodeset].num_nodes
  assert num_nodes is not None
  if options.seed_node_idxs is None:
    candidates = np.arange(num_nodes, dtype=np.int32)
  else:
    candidates = np.asarray(options.seed_node_idxs, dtype=np.int32)

  if options.num_seed_nodes is not None:
    # The user specified the number of seed nodes.
    num_seed_nodes = max(1, options.num_seed_nodes)
  elif options.max_duration is None or len(candidates) <= evaluator.batch_size:
    # No time budget, or too few candidates to calibrate: use the default cap.
    num_seed_nodes = _MAX_AUTO_NUM_SEED_NODES
  else:
    # Calibrate the number of seed nodes on the time budget.
    warmup_idxs = candidates[: evaluator.batch_size]
    sampler_start = time.time()
    sampler = evaluator.create_sampler(graph)
    sampler_duration = time.time() - sampler_start
    # The first evaluation triggers the JIT compilation.
    evaluator.evaluate(graph, warmup_idxs, sampler=sampler)
    batch_start = time.time()
    evaluator.evaluate(graph, warmup_idxs, sampler=sampler)
    batch_duration = max(1e-4, time.time() - batch_start)

    # One evaluation pass per perturbation plus the baseline. Only the node
    # feature perturbations re-use the sub-graphs sampled for the baseline.
    remaining = max(1.0, options.max_duration - (time.time() - start_time))
    num_samplers = 1 + sum(
        1
        for category, _, _ in tasks
        if category != ImportanceCategory.NODE_FEATURE
    )
    budget_per_pass = (remaining * 0.65 - num_samplers * sampler_duration) / (
        len(tasks) + 1
    )
    num_batches = max(1, int(budget_per_pass // batch_duration))
    num_seed_nodes = min(
        num_batches * evaluator.batch_size, _MAX_AUTO_NUM_SEED_NODES
    )

  if num_seed_nodes >= len(candidates):
    # All the candidates are evaluated, keep their order.
    return candidates
  rng = np.random.default_rng(options.random_seed)
  return rng.choice(candidates, size=num_seed_nodes, replace=False)


def analyze_node_model(
    model: node_prediction_model.NodePredictionModel,
    graph: in_memory_graph.InMemoryGraph,
    options: AnalyzeOptions | None = None,
) -> ModelAnalysis:
  """Computes the importance of the features, edgesets and nodesets of a model.

  See `NodePredictionModel.analyze` for details.

  Args:
    model: Trained model to analyze.
    graph: Graph with the ground truth labels to evaluate the model on.
    options: Options of the analysis. If None, uses `AnalyzeOptions()`.

  Returns:
    The analysis.
  """
  if options is None:
    options = AnalyzeOptions()
  start_time = time.time()
  evaluator = _Evaluator(model, options)
  tasks = _list_perturbations(evaluator, options)
  seed_idxs = _select_seed_nodes(evaluator, graph, options, tasks, start_time)
  if options.verbose >= 1:
    log.info(
        "Analyzing model on %d seed nodes across %d perturbations",
        len(seed_idxs),
        len(tasks),
    )

  # Baseline evaluation on the unperturbed graph. The sampled sub-graphs are
  # re-used for the node feature perturbations (which keep the topology).
  cache_batches = len(
      seed_idxs
  ) <= _MAX_CACHED_SEED_NODES and not temporal_util.schema_has_timeseries_features(
      evaluator.schema
  )
  baseline = evaluator.evaluate(
      graph,
      seed_idxs,
      compute_gradients=options.compute_gradient_importance,
      return_sampled_batches=cache_batches,
  )

  items: dict[ImportanceCategory, list[ImportanceItem]] = {
      category: [] for category in ImportanceCategory
  }
  task_iter: Any = tasks
  if options.verbose >= 2:
    task_iter = tqdm.tqdm(tasks, desc="Analyzing model")
  for idx, (category, name, perturb_fn) in enumerate(task_iter):
    if (
        options.max_duration is not None
        and idx > 0
        and time.time() - start_time >= options.max_duration
    ):
      log.warning(
          "Stopping the analysis early after %d/%d perturbations because"
          " max_duration (%.1fs) was reached. Increase `max_duration` to"
          " analyze all the components.",
          idx,
          len(tasks),
          options.max_duration,
      )
      break

    perturbed_graph = perturb_fn(
        graph, np.random.default_rng(options.random_seed + 1 + idx)
    )
    perturbed = evaluator.evaluate(
        perturbed_graph,
        seed_idxs,
        cached_sampled_batches=(
            baseline.sampled_batches
            if category == ImportanceCategory.NODE_FEATURE
            else None
        ),
    )
    if evaluator.is_regression:
      # The RMSE increases when an important component is perturbed.
      delta_metric = perturbed.metric - baseline.metric
    else:
      # The accuracy drops when an important component is perturbed.
      delta_metric = baseline.metric - perturbed.metric
    gradient_importance = None
    if baseline.gradient_saliency is not None:
      # Only numerical features have a gradient saliency; others get None.
      gradient_importance = baseline.gradient_saliency.get(name)
    items[category].append(
        ImportanceItem(
            category=category,
            name=name,
            delta_loss=perturbed.loss - baseline.loss,
            delta_metric=delta_metric,
            perturbed_loss=perturbed.loss,
            perturbed_metric=perturbed.metric,
            gradient_importance=gradient_importance,
        )
    )

  return ModelAnalysis(
      task_type=model.data().task.task_type.value,
      target_nodeset=evaluator.target_nodeset,
      target_column=evaluator.target_column,
      metric_name="rmse" if evaluator.is_regression else "accuracy",
      baseline_metric=baseline.metric,
      baseline_loss=baseline.loss,
      num_seed_nodes=len(seed_idxs),
      edge_perturbation_mode=perturb.parse_edge_perturbation_mode(
          options.edge_perturbation_mode
      ),
      duration_seconds=time.time() - start_time,
      node_features=_sort_items(items[ImportanceCategory.NODE_FEATURE]),
      edge_features=_sort_items(items[ImportanceCategory.EDGE_FEATURE]),
      edgesets=_sort_items(items[ImportanceCategory.EDGESET]),
      nodesets=_sort_items(items[ImportanceCategory.NODESET]),
  )

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

"""Generation of reports (e.g., html) about models."""

import dataclasses
import html
import pprint
from typing import Any, Dict, List, Optional, Tuple
import uuid
import altair as alt
from dgf.src.analyse import padding as analyse_padding_lib
from dgf.src.analyse import print_schema as print_schema_lib
from dgf.src.analyse import sampling as analyse_sampling_lib
from dgf.src.data import evaluation
from dgf.src.data import padding as padding_data_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
from dgf.src.learning.ten_lines import common
from dgf.src.sampling import config as sampling_config_lib
from dgf.src.util import html_util
from dgf.src.util import log
import pandas as pd


def plot_html_training_logs(training_logs: common.TrainingLogs) -> str:
  """Creates an interactive HTML plot of training logs using Altair.

  The function generates a multi-faceted plot where each facet displays a
  different metric recorded during training. Within each metric plot, lines
  represent the training and validation values over steps.

  Args:
    training_logs: A TrainingLogs object containing lists of training and
      validation logs.

  Returns:
    An HTML string representing the interactive plot. If no logs are present,
    it returns a paragraph indicating that there's no data to display.
  """

  if not training_logs.train and not training_logs.valid:
    return "<p>No training logs to display.</p>"

  data = []
  for dataset_name, logs in [
      ("train", training_logs.train),
      ("valid", training_logs.valid),
  ]:
    for log in logs:
      for metric_name, value in log.metrics.items():
        data.append({
            "step": log.step,
            "dataset": dataset_name,
            "metric": metric_name,
            "value": value,
        })

  df = pd.DataFrame(data)

  chart = (
      alt.Chart(df)
      .mark_line()
      .encode(
          x="step:Q",
          y=alt.Y("value:Q", scale=alt.Scale(zero=False)),
          color="dataset:N",
      )
      .facet(
          facet="metric:N",
          columns=2,
      )
      .resolve_scale(y="independent")
  )

  return chart.to_html(fullhtml=False, output_div=f"vis-{uuid.uuid4().hex}")


def html_tabs(items: list[tuple[str, str]]) -> str:
  """Returns an HTML string that displays the given items in tabs.

  Args:
    items: A list of pairs (title, html_content).

  Returns:
    An HTML string with a tabbed interface.
  """
  if not items:
    return ""

  component_id = f"tabs-{uuid.uuid4().hex[:8]}"

  style = f"""
<style>
  #{component_id} {{
    font-family: 'Roboto', sans-serif;
    font-size: 13px;
    color: #202124;
    margin: 20px 0;
    line-height: 1.5;
  }}
  #{component_id} .tab-header {{
    display: flex;
    border-bottom: 1px solid #e0e0e0;
    margin-bottom: 15px;
    font-family: 'Roboto', sans-serif;
  }}
  #{component_id} .tab-btn {{
    padding: 10px 20px;
    cursor: pointer;
    border: none;
    background: none;
    font-family: 'Roboto', sans-serif;
    font-size: 13px;
    font-weight: normal;
    color: #5f6368;
    transition: color 0.2s, border-bottom 0.2s;
    margin-bottom: -1px;
  }}
  #{component_id} .tab-btn:hover {{
    color: #1a73e8;
  }}
  #{component_id} .tab-btn.active {{
    color: #1a73e8;
    border-bottom: 2px solid #1a73e8;
  }}
  #{component_id} .tab-content {{
    display: none;
    padding: 10px 0;
    animation: fadeIn 0.3s;
  }}
  #{component_id} .tab-content.active {{
    display: block;
  }}
  #{component_id} pre, #{component_id} code {{
    font-family: 'Roboto Mono', monospace;
    font-size: 13px;
  }}
  #{component_id} pre {{
    margin: 0 0 20px 0;
    white-space: pre-wrap;
    word-break: break-word;
  }}
  #{component_id} b {{
    font-weight: 600;
    display: block;
    margin-bottom: 4px;
    font-family: 'Roboto', sans-serif;
  }}
  #{component_id} ul {{
    list-style-type: none;
    padding-left: 0;
    margin: 0;
  }}
  #{component_id} li {{
    margin: 0;
    padding: 0;
  }}
  #{component_id} .lbl-key {{
    color: #5f6368;
    font-weight: 600;
  }}
  {html_util.get_table_style(component_id)}
  @keyframes fadeIn {{
    from {{ opacity: 0; }}
    to {{ opacity: 1; }}
  }}
</style>
"""

  html = [f'<div id="{component_id}">', style, '<div class="tab-header">']

  for i, (title, _) in enumerate(items):
    active_class = " active" if i == 0 else ""
    html.append(
        f'<button class="tab-btn{active_class}"'
        f' onclick="openTab_{component_id.replace("-", "_")}(event,'
        f" 'tab_{i}')\">{title}</button>"
    )

  html.append("</div>")

  for i, (_, content) in enumerate(items):
    active_class = " active" if i == 0 else ""
    html.append(
        f'<div id="tab_{i}" class="tab-content{active_class}">{content}</div>'
    )

  script = f"""
<script>
function openTab_{component_id.replace("-", "_")}(evt, tabId) {{
  var i, tabcontent, tablinks;
  var container = document.getElementById("{component_id}");
  tabcontent = container.getElementsByClassName("tab-content");
  for (i = 0; i < tabcontent.length; i++) {{
    tabcontent[i].style.display = "none";
  }}
  tablinks = container.getElementsByClassName("tab-btn");
  for (i = 0; i < tablinks.length; i++) {{
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }}
  document.getElementById(tabId).style.display = "block";
  evt.currentTarget.className += " active";
}}
</script>
"""
  html.append(script)
  html.append("</div>")

  return "\n".join(html)


def html_log_messages(log_messages: List[log.Message]) -> str:
  """Returns HTML rendering for log messages."""
  if not log_messages:
    return "<i>No logs were captured.</i>"

  table_rows = ""
  for msg in log_messages:
    if msg.severity == log.Severity.INFO:
      color = "blue"
    elif msg.severity == log.Severity.WARNING:
      color = "orange"
    elif msg.severity == log.Severity.ERROR:
      color = "red"
    else:
      assert False
    table_rows += (
        f"<tr>"
        f'<td style="color: {color}; font-weight:'
        f' 500;">{msg.severity.value}</td>'
        "<td style=\"font-family: 'Roboto Mono',"
        f' monospace;">{html.escape(msg.text)}</td>'
        f"</tr>\n"
    )

  return f"""
<table class="dgf-table">
  <tbody>
{table_rows}
  </tbody>
</table>
"""


def _get_training_logs_tab(
    training_logs: common.TrainingLogs,
    training_stats_summary: Optional[str],
) -> Tuple[str, str]:
  """Generates the Training logs tab."""
  if len(training_logs.train) >= 10:
    training_logs = dataclasses.replace(
        training_logs,
        train=training_logs.train[1:],
    )
    skip_msg = (
        "<p><i>Note: The logs for the first training step are not"
        " shown.</i></p>\n"
    )
  else:
    skip_msg = ""

  train_log_plots = plot_html_training_logs(training_logs)
  content = f"""
<table class="dgf-table">
  <tbody>
{training_stats_summary or ""}
    <tr><td>Number of training steps (final model)</td><td>{training_logs.num_train_step}</td></tr>
  </tbody>
</table>
<div style="width: 100%; margin-top: 15px;">{train_log_plots}</div>
{skip_msg}"""
  return "Training", content


def _get_hyper_parameters_tab(hparams: Any) -> Tuple[str, str]:
  """Generates the Hyper-parameters tab."""
  if dataclasses.is_dataclass(hparams):
    hparams_dict = dataclasses.asdict(hparams)
    table_rows = ""
    for k, v in hparams_dict.items():
      table_rows += (
          f"<tr><td>{html.escape(k)}</td><td>{html.escape(repr(v))}</td></tr>\n"
      )
    content = f"""
<table class="dgf-table">
  <tbody>
{table_rows}
  </tbody>
</table>
"""
  else:
    # Fallback for non-dataclass hparams.
    content = f"<pre>{html.escape(pprint.pformat(hparams))}</pre>"
  return "Hyper-parameters", content


def _get_schema_tab(
    schemas: dict[str, schema_lib.GraphSchema],
) -> Tuple[str, str]:
  """Generates the Schema tab."""
  txt_schemas = ""
  for name, schema in schemas.items():
    txt_schemas += f"<b>{name} schema</b>\n<pre>"
    txt_schemas += print_schema_lib.print_schema(  # pyrefly: ignore[unsupported-operation]
        schema, return_output=True, header=False
    )
    txt_schemas += "</pre>\n"
  return "Schema", txt_schemas


def _get_feature_stats_tab(
    feature_stats: dict[str, statistics_lib.GraphFeatureStatistics],
) -> Tuple[str, str]:
  """Generates the Feature statistics tab."""
  txt_feature_stats = ""
  for name, stats in feature_stats.items():
    txt_feature_stats += f"<b>{name} feature statistics</b>\n"
    txt_feature_stats += f"<pre>{repr(stats)}</pre>\n"
  return "Feature statistics", txt_feature_stats


def _get_graph_sampling_tab(
    sampling_plans: dict[str, sampling_config_lib.SamplingPlan],
) -> Tuple[str, str]:
  """Generates the Graph sampling tab."""
  txt_sampling_plan = ""
  for name, sampling_plan in sampling_plans.items():
    txt_sampling_plan += f"<b>{name} sampling plan</b>\n<pre>"
    txt_sampling_plan += analyse_sampling_lib.print_sampling_plan(  # pyrefly: ignore[unsupported-operation]
        sampling_plan, return_output=True, header=False
    )
    txt_sampling_plan += "</pre>\n"
  return "Graph sampling", txt_sampling_plan


def _get_architecture_tab(
    architecture: str, num_model_weights: Optional[Dict[str, int]] = None
) -> Tuple[str, str]:
  """Generates the Architecture tab."""
  num_weights_str = (
      pprint.pformat(num_model_weights)
      if num_model_weights is not None
      else "Unknown"
  )
  content = (
      f"<b>Model Structure</b>\n<pre>{architecture}</pre>\n"
      f"<b>Model Weights</b>\n<pre>{num_weights_str}</pre>"
  )
  return "Architecture", content


def _get_padding_tab(
    padding: dict[str, padding_data_lib.Padding],
) -> Tuple[str, str]:
  """Generates the Padding tab."""
  txt_padding = ""
  for name, pad in padding.items():
    txt_padding += f"<b>{name} padding</b>\n<pre>"
    txt_padding += analyse_padding_lib.print_padding(  # pyrefly: ignore[unsupported-operation]
        pad, return_output=True, header=False
    )
    txt_padding += "</pre>\n"
  return "Padding", txt_padding


def get_common_tabs(
    hparams: Any,
    schemas: dict[str, schema_lib.GraphSchema],
    feature_stats: Optional[
        dict[str, statistics_lib.GraphFeatureStatistics]
    ] = None,
    sampling_plans: Optional[
        dict[str, sampling_config_lib.SamplingPlan]
    ] = None,
    training_logs: Optional[common.TrainingLogs] = None,
    training_stats_summary: Optional[str] = None,
    padding: Optional[dict[str, padding_data_lib.Padding]] = None,
    architecture: Optional[str] = None,
    num_model_weights: Optional[Dict[str, int]] = None,
    log_messages: Optional[List[log.Message]] = None,
    final_evaluation: Optional[evaluation.Evaluation] = None,
) -> List[Tuple[str, str]]:
  """Generates common tabs for model description."""
  tabs = []

  if log_messages is not None:
    tabs.append(
        (f"Logs ({len(log_messages)})", html_log_messages(log_messages))
    )

  if final_evaluation is not None:
    tabs.append(("Evaluation", final_evaluation.html()))

  if training_logs is not None:
    tabs.append(
        _get_training_logs_tab(
            training_logs=training_logs,
            training_stats_summary=training_stats_summary,
        )
    )

  tabs.append(_get_hyper_parameters_tab(hparams))
  tabs.append(_get_schema_tab(schemas))

  if feature_stats is not None:
    tabs.append(_get_feature_stats_tab(feature_stats))

  if sampling_plans is not None:
    tabs.append(_get_graph_sampling_tab(sampling_plans))

  if architecture is not None:
    tabs.append(_get_architecture_tab(architecture, num_model_weights))

  if padding is not None:
    tabs.append(_get_padding_tab(padding))

  return tabs

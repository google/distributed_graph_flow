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
import pprint
from typing import Any, Dict, List, Optional, Tuple
import altair as alt
from dgf.src.analyse import padding as analyse_padding_lib
from dgf.src.analyse import print_schema as print_schema_lib
from dgf.src.analyse import sampling as analyse_sampling_lib
from dgf.src.data import padding as padding_data_lib
from dgf.src.data import schema as schema_lib
from dgf.src.data import statistics as statistics_lib
from dgf.src.learning.ten_lines import common
from dgf.src.sampling import config as sampling_config_lib
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
          y="value:Q",
          color="dataset:N",
      )
      .facet(
          facet="metric:N",
          columns=2,
      )
      .resolve_scale(y="independent")
  )

  return chart.to_html()


def html_tabs(items: list[tuple[str, str]]) -> str:
  """Returns an HTML string that displays the given items in tabs.

  Args:
    items: A list of pairs (title, html_content).

  Returns:
    An HTML string with a tabbed interface.
  """
  if not items:
    return ""

  import uuid

  component_id = f"tabs-{uuid.uuid4().hex[:8]}"

  style = f"""
<style>
  #{component_id} {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 20px 0;
  }}
  #{component_id} .tab-header {{
    display: flex;
    border-bottom: 2px solid #e0e0e0;
    margin-bottom: 15px;
  }}
  #{component_id} .tab-btn {{
    padding: 10px 20px;
    cursor: pointer;
    border: none;
    background: none;
    font-size: 16px;
    font-weight: 500;
    color: #5f6368;
    transition: color 0.2s, border-bottom 0.2s;
    margin-bottom: -2px;
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
    padding: 10px;
    animation: fadeIn 0.3s;
  }}
  #{component_id} .tab-content.active {{
    display: block;
  }}
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
) -> List[Tuple[str, str]]:
  """Generates common tabs for model description."""
  tabs = []

  if training_logs is not None:
    train_log_plots = plot_html_training_logs(training_logs)
    tabs.append((
        "Train logs",
        f"""
{training_stats_summary or ""}
<div style="width: 100%;">{train_log_plots}</div>
""",
    ))

  if dataclasses.is_dataclass(hparams):
    hparams_dict = dataclasses.asdict(hparams)
    lines = [f"{k}={repr(v)}" for k, v in hparams_dict.items()]
    txt_hyper_parameters = "\n".join(lines)
  else:
    # Fallback for non-dataclass hparams.
    txt_hyper_parameters = pprint.pformat(hparams)

  tabs.append((
      "Hyper-parameters",
      f"<pre>{txt_hyper_parameters}</pre>",
  ))

  txt_schemas = ""
  for name, schema in schemas.items():
    txt_schemas += f"<b>{name} schema</b><pre>\n"
    txt_schemas += print_schema_lib.print_schema(
        schema, return_output=True, header=False
    )
    txt_schemas += "</pre><br>\n"

  tabs.append((
      "Schemas",
      txt_schemas,
  ))

  if feature_stats is not None:
    txt_feature_stats = ""
    for name, stats in feature_stats.items():
      txt_feature_stats += f"<b>{name} feature statistics</b>\n"
      txt_feature_stats += f"<pre>{repr(stats)}</pre><br>\n"

    tabs.append((
        "Feature statistics",
        txt_feature_stats,
    ))

  if sampling_plans is not None:
    txt_sampling_plan = ""
    for name, sampling_plan in sampling_plans.items():
      txt_sampling_plan += f"<b>{name} sampling plan</b><pre>\n"
      txt_sampling_plan += analyse_sampling_lib.print_sampling_plan(
          sampling_plan, return_output=True, header=False
      )
      txt_sampling_plan += "</pre><br>\n"
    tabs.append(("Graph sampling", txt_sampling_plan))

  if architecture is not None:
    num_weights_str = (
        pprint.pformat(num_model_weights)
        if num_model_weights is not None
        else "Unknown"
    )
    tabs.append((
        "Architecture",
        (
            f"<b>Model Structure</b><br><pre>{architecture}</pre>"
            f"<b>Model Weights</b><br><pre>{num_weights_str}</pre>"
        ),
    ))

  if padding is not None:
    txt_padding = ""
    for name, pad in padding.items():
      txt_padding += f"<b>{name} padding</b><pre>\n"
      txt_padding += analyse_padding_lib.print_padding(
          pad, return_output=True, header=False
      )
      txt_padding += "</pre><br>\n"

    tabs.append((
        "Padding",
        txt_padding,
    ))

  return tabs

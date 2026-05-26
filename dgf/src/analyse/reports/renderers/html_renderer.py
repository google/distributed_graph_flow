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

"""HTML Renderer for Graph Flow EDA Reports."""

import base64
import json
import os
from typing import Any

from absl import logging
from dgf.src.analyse.reports import data_model
from dgf.src.analyse.reports import visual_utils
import jinja2


def _number_format(value: Any) -> str:
  """Formats a number with commas."""
  if isinstance(value, (int, float)):
    return f"{value:,}"
  return str(value)


class HtmlRenderer:
  """Renders the HTML report using Jinja2."""

  def __init__(self, template_dir: str | None = None):
    if template_dir is None:
      # Default to the templates directory relative to this file
      template_dir = os.path.join(os.path.dirname(__file__), "..", "templates")

    self._template_dir = template_dir
    self._env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    self._env.filters["number_format"] = _number_format
    self._env.filters["tojson"] = json.dumps

  def render(self, payload: data_model.GraphStatsPayload) -> str:
    """Renders the HTML report.

    Args:
      payload: The data payload.

    Returns:
      The rendered HTML string.
    """
    # Auto-convert subgraphs to visual_gallery_data if needed
    if not payload.visual_gallery_data and payload.subgraphs:
      logging.info(
          "Converting %d subgraphs to PyVis data...", len(payload.subgraphs)
      )
      payload.visual_gallery_data = [
          visual_utils.graph_to_pyvis_data(
              g,
              color_by_attribute=payload.color_by_attribute,
              node_label_attribute=payload.node_label_attribute,
              graph_schema=payload.graph_schema,
          )
          for g in payload.subgraphs
      ]
      # logging.info("Visual gallery data: %s", payload.visual_gallery_data)
      logging.info(
          "Converted %d subgraphs to PyVis data.", len(payload.subgraphs)
      )

    try:
      # Load and encode logo
      logo_path = os.path.join(self._template_dir, "graph-flow-icon-noback.png")
      logo_base64 = None
      if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
          logo_data = f.read()
          logo_base64 = base64.b64encode(logo_data).decode("utf-8")
      else:
        logging.warning("Logo file not found at: %s", logo_path)

      template = self._env.get_template("report.html")
      return template.render(
          payload=payload,
          logo_base64=logo_base64,
      )
    except jinja2.TemplateError as e:
      logging.error("Failed to render HTML template: %s", e)
      raise

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

"""Main entry point for generating Graph Flow EDA Reports."""

import os
from typing import Optional

from absl import logging
from dgf.src.analyse.reports import data_model
from dgf.src.analyse.reports.renderers import html_renderer
from dgf.src.analyse.reports.renderers import pdf_renderer


def generate_report(
    payload: data_model.GraphStatsPayload,
    output_dir: str,
    output_html_name: str = "report.html",
    output_pdf_name: str = "report.pdf",
    template_dir: Optional[str] = None,
) -> None:
  """Generates both HTML and PDF reports.

  Args:
    payload: The data payload containing graph statistics.
    output_dir: The directory where reports will be saved.
    output_html_name: The name of the HTML output file.
    output_pdf_name: The name of the PDF output file.
    template_dir: Optional custom directory for HTML templates.
  """
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Generate HTML
  try:
    logging.info("Generating HTML report...")
    h_renderer = html_renderer.HtmlRenderer(template_dir=template_dir)
    html_content = h_renderer.render(payload)
    html_path = os.path.join(output_dir, output_html_name)
    with open(html_path, "w", encoding="utf-8") as f:
      f.write(html_content)
    logging.info("HTML report saved to: %s", html_path)
  except Exception as e:
    logging.error("Failed to generate HTML report: %s", e)
    raise

  # Generate PDF
  try:
    logging.info("Generating PDF report...")
    p_renderer = pdf_renderer.PdfRenderer()
    pdf_bytes = p_renderer.render(payload)
    pdf_path = os.path.join(output_dir, output_pdf_name)
    with open(pdf_path, "wb") as f:
      f.write(pdf_bytes)
    logging.info("PDF report saved to: %s", pdf_path)
  except Exception as e:
    logging.error("Failed to generate PDF report: %s", e)
    raise

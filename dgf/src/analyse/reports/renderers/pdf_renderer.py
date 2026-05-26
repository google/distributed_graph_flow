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

"""PDF Renderer for Graph Flow EDA Reports."""

import io
from typing import List

from dgf.src.analyse.reports import data_model
from reportlab.lib import colors
from reportlab.lib import pagesizes
from reportlab.lib import styles
from reportlab.platypus import doctemplate
from reportlab.platypus import flowables
from reportlab.platypus import paragraph
from reportlab.platypus import tables


class PdfRenderer:
  """Renders the PDF report using ReportLab."""

  def __init__(self):
    self._styles = styles.getSampleStyleSheet()

    # Custom Styles
    self._title_style = styles.ParagraphStyle(
        "ReportTitle",
        parent=self._styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        textColor=colors.white,
        spaceAfter=10,
        alignment=1,  # Center
    )

    self._subtitle_style = styles.ParagraphStyle(
        "ReportSubtitle",
        parent=self._styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        textColor=colors.white,
        spaceAfter=20,
        alignment=1,  # Center
    )

    self._heading_style = styles.ParagraphStyle(
        "SectionHeading",
        parent=self._styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=colors.HexColor("#1a73e8"),  # Google Blue
        spaceBefore=20,
        spaceAfter=10,
        borderPadding=(0, 0, 0, 10),  # Left padding for border
        borderWidth=0,
        borderColor=colors.white,
    )

    # Metric Card Styles
    self._metric_value_style = styles.ParagraphStyle(
        "MetricValue",
        parent=self._styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        textColor=colors.HexColor("#1a73e8"),
        alignment=1,  # Center
        spaceAfter=4,
    )

    self._metric_label_style = styles.ParagraphStyle(
        "MetricLabel",
        parent=self._styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=colors.HexColor("#5f6368"),
        alignment=1,  # Center
    )

  def render(self, payload: data_model.GraphStatsPayload) -> bytes:
    """Renders the PDF report.

    Args:
      payload: The data payload.

    Returns:
      The rendered PDF bytes.
    """
    buffer = io.BytesIO()
    doc = doctemplate.SimpleDocTemplate(
        buffer,
        pagesize=pagesizes.LETTER,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    story: List[flowables.Flowable] = []

    # --- Header Section (Simulating a colored banner) ---
    # We use a table with a background color to create the header banner
    date_str = payload.generated_at.strftime("%Y-%m-%d")
    header_content = [
        [paragraph.Paragraph("Graph Flow Report", self._subtitle_style)],
        [paragraph.Paragraph(payload.dataset_name, self._title_style)],
        [
            paragraph.Paragraph(
                f"Task: {payload.task_type} | Date: {date_str}",
                self._subtitle_style,
            )
        ],
    ]

    header_table = tables.Table(
        header_content, colWidths=[530]
    )  # Approx full width
    header_table.setStyle(
        tables.TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1a73e8")),
            ("TOPPADDING", (0, 0), (-1, -1), 20),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ])
    )
    story.append(header_table)
    story.append(flowables.Spacer(1, 30))

    # --- Executive Summary ---
    story.append(
        paragraph.Paragraph("1. Executive Summary", self._heading_style)
    )
    story.append(flowables.Spacer(1, 10))

    # Metrics Grid
    # Formatted values
    ggt = payload.global_graph_topology
    total_nodes_str = f"{ggt.total_nodes:,}" if ggt else "N/A"
    total_edges_str = f"{ggt.total_edges:,}" if ggt else "N/A"

    # We construct a table where each cell acts as a card
    # To get spacing between "cards", we can use empty columns,
    # but standard tables share borders.
    # A cleaner look in PDF is a single grid with nice padding.

    metrics_values = [
        paragraph.Paragraph(total_nodes_str, self._metric_value_style),
        paragraph.Paragraph(total_edges_str, self._metric_value_style),
        paragraph.Paragraph(
            str(payload.feature_dimensionality)
            if payload.feature_dimensionality is not None
            else "N/A",
            self._metric_value_style,
        ),
    ]
    metrics_labels = [
        paragraph.Paragraph("Total Nodes", self._metric_label_style),
        paragraph.Paragraph("Total Edges", self._metric_label_style),
        paragraph.Paragraph("Vector Dim", self._metric_label_style),
    ]

    if payload.num_classes is not None:
      metrics_values.append(
          paragraph.Paragraph(
              str(payload.num_classes), self._metric_value_style
          )
      )
      metrics_labels.append(
          paragraph.Paragraph("Num Classes", self._metric_label_style)
      )

    metrics_data = [metrics_values, metrics_labels]

    # Calculate column width based on number of metrics to
    # keep them evenly spaced/sized
    # Total available width is approx 530.
    num_metrics = len(metrics_values)
    col_width = 530 / num_metrics
    col_widths = [col_width] * num_metrics

    metrics_table = tables.Table(metrics_data, colWidths=col_widths)
    metrics_table.setStyle(
        tables.TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, 0), 15),  # Padding above values
            ("BOTTOMPADDING", (0, 1), (-1, 1), 15),  # Padding below labels
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#e0e0e0"),
            ),  # Light grid
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ])
    )
    story.append(metrics_table)
    story.append(flowables.Spacer(1, 30))

    # --- Global Graph Topology ---
    story.append(
        paragraph.Paragraph("2. Global Graph Topology", self._heading_style)
    )
    story.append(flowables.Spacer(1, 10))

    # We use a table to show the metrics
    topology_data = [
        ["Metric", "Value"],
        [
            "Avg Degree",
            f"{ggt.avg_degree:.2f}"
            if ggt and ggt.avg_degree is not None
            else "N/A",
        ],
        [
            "Density",
            f"{ggt.graph_density:.4f}"
            if ggt and ggt.graph_density is not None
            else "N/A",
        ],
        [
            "Connected Components",
            f"{ggt.num_connected_components:,}"
            if ggt and ggt.num_connected_components is not None
            else "N/A",
        ],
        [
            "Largest Component Size",
            f"{ggt.largest_component_size:,}"
            if ggt and ggt.largest_component_size is not None
            else "N/A",
        ],
        [
            "Isolated Nodes",
            f"{ggt.isolated_nodes:,}"
            if ggt and ggt.isolated_nodes is not None
            else "N/A",
        ],
        [
            "Graph Diameter",
            (
                str(ggt.graph_diameter)
                if ggt and ggt.graph_diameter is not None
                else "N/A"
            ),
        ],
        [
            "Homophily Ratio",
            f"{ggt.homophily_ratio:.2f}"
            if ggt and ggt.homophily_ratio is not None
            else "N/A",
        ],
    ]

    topology_table = tables.Table(
        topology_data, colWidths=[200, 100], hAlign="LEFT"
    )
    topology_table.setStyle(
        tables.TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8f9fa")),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ])
    )
    story.append(topology_table)
    story.append(flowables.Spacer(1, 20))

    if ggt and ggt.degree_distribution:
      story.append(
          paragraph.Paragraph(
              "Degree Distribution available in HTML report.",
              styles.ParagraphStyle(
                  "Info2",
                  parent=self._styles["Normal"],
                  textColor=colors.gray,
                  fontSize=10,
              ),
          )
      )

    story.append(flowables.Spacer(1, 30))

    # Placeholder for future content
    story.append(
        paragraph.Paragraph(
            "Feature Analysis and Visual Inspection sections will be added"
            " here.",
            styles.ParagraphStyle(
                "Info",
                parent=self._styles["Italic"],
                textColor=colors.gray,
                alignment=1,
            ),
        )
    )

    doc.build(story)
    return buffer.getvalue()

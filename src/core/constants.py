"""
core/constants.py

Single source of truth for shared multimodal names:
- content types
- table statuses
- display preferences
- vision statuses
- metadata keys
"""

from __future__ import annotations

from enum import StrEnum


class ContentType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    IMAGE = "image"
    DIAGRAM = "diagram"
    FLOWCHART = "flowchart"


class TableStatus(StrEnum):
    GOOD = "good"
    WEAK = "weak"
    TEXT_FALLBACK = "text_fallback"
    CAPTION_ONLY = "caption_only"
    REJECTED = "rejected"


class DisplayPreference(StrEnum):
    HTML = "html"
    IMAGE = "image"
    TEXT = "text"
    NONE = "none"


class VisionStatus(StrEnum):
    NOT_GENERATED = "not_generated"
    GENERATED = "generated"
    FAILED = "failed"


TEXT_CONTENT_TYPES = {
    ContentType.TEXT.value,
}

TABLE_CONTENT_TYPES = {
    ContentType.TABLE.value,
}

VISUAL_CONTENT_TYPES = {
    ContentType.FIGURE.value,
    ContentType.IMAGE.value,
    ContentType.DIAGRAM.value,
    ContentType.FLOWCHART.value,
}

ARTIFACT_CONTENT_TYPES = TABLE_CONTENT_TYPES | VISUAL_CONTENT_TYPES

VISION_ELIGIBLE_TABLE_STATUSES = {
    TableStatus.WEAK.value,
    TableStatus.TEXT_FALLBACK.value,
    TableStatus.CAPTION_ONLY.value,
}

PLACEHOLDER_VISION_DESCRIPTIONS = {
    "",
    "Vision description not generated yet.",
    "vision description not generated yet.",
}


BASE_METADATA_KEYS = [
    "content_type",
    "paper_name",
    "author",
    "year",
    "source_file",
    "section",
    "page_number",
    "page_start",
    "page_end",
]


TABLE_METADATA_KEYS = [
    *BASE_METADATA_KEYS,
    "caption",
    "caption_confidence",
    "caption_link_method",
    "image_path",
    "csv_path",
    "html_path",
    "json_path",
    "markdown_path",
    "table_id",
    "extraction_method",
    "extraction_confidence",
    "structured_status",
    "display_preference",
    "quality_reason",
    "vision_status",
    "vision_model",
    "vision_description",
    "vision_markdown",
]


VISUAL_METADATA_KEYS = [
    *BASE_METADATA_KEYS,
    "caption",
    "caption_confidence",
    "caption_link_method",
    "image_path",
    "figure_id",
    "extraction_method",
    "extraction_confidence",
    "description",
    "vision_status",
    "vision_model",
    "vision_description",
    "mermaid",
    "mermaid_status",
]


SOURCE_TYPE_LABELS = {
    ContentType.TEXT.value: "Text",
    ContentType.TABLE.value: "Table",
    ContentType.FIGURE.value: "Figure / Image",
    ContentType.IMAGE.value: "Figure / Image",
    ContentType.DIAGRAM.value: "Diagram / Flowchart",
    ContentType.FLOWCHART.value: "Diagram / Flowchart",
}
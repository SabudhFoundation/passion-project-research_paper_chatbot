"""
pdf_artifacts.py

Reusable multimodal PDF artifact extraction helpers.

This file contains:
- table extraction helpers
- table quality checks
- table artifact writing
- caption-based table fallback
- accurate fallback table PNG cropping
- figure / image / diagram crop extraction

It does NOT:
- run the full PDF parser
- call any LLM
- create embeddings
- touch Streamlit
"""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pymupdf

from config import (
    ENABLE_BASIC_IMAGE_EXTRACTION,
    ENABLE_FIGURE_REGION_CROPPING,
    ENABLE_MERMAID_GENERATION,
    ENABLE_TABLE_EXTRACTION,
    ENABLE_VISION_DESCRIPTIONS,
    FIGURES_ARTIFACT_PATH,
    TABLES_ARTIFACT_PATH,
)
from core.constants import (
    ContentType,
    DisplayPreference,
    TableStatus,
    VisionStatus,
)
from utilities import save_json


# =========================================================
# Section helper
# =========================================================

_SECTION_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(Abstract|Introduction|Background|Related\s+Work|Methodology|Methods?|"
    r"Experiments?|Results?|Evaluation|Discussion|Conclusion|References?|"
    r"Appendix|Acknowledgements?)\b.*",
    re.IGNORECASE,
)


def infer_section_from_page_text(page_text: str) -> str:
    """Best-effort section label used only as artifact metadata."""
    for line in (page_text or "").splitlines():
        cleaned = line.strip()
        if 3 <= len(cleaned) <= 100 and _SECTION_RE.match(cleaned):
            return cleaned

    return "Unknown"


# =========================================================
# Table serialization helpers
# =========================================================

def normalize_matrix(rows: Iterable[Iterable[Any]]) -> list[list[str]]:
    matrix: list[list[str]] = []

    for row in rows or []:
        cleaned = [
            str(cell).strip() if cell is not None else ""
            for cell in row
        ]
        if any(cleaned):
            matrix.append(cleaned)

    if not matrix:
        return []

    max_cols = max(len(row) for row in matrix)
    return [row + [""] * (max_cols - len(row)) for row in matrix]


def split_columns_rows(matrix: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    if not matrix:
        return [], []

    first = matrix[0]
    has_header_text = any(re.search(r"[A-Za-z]", cell or "") for cell in first)

    if len(matrix) > 1 and has_header_text:
        columns = [
            cell or f"Column {index + 1}"
            for index, cell in enumerate(first)
        ]
        rows = matrix[1:]
    else:
        columns = [f"Column {index + 1}" for index in range(len(first))]
        rows = matrix

    seen: dict[str, int] = {}
    unique_columns: list[str] = []

    for index, column in enumerate(columns):
        column = (column or f"Column {index + 1}").strip()
        count = seen.get(column, 0)
        seen[column] = count + 1
        unique_columns.append(column if count == 0 else f"{column}_{count + 1}")

    return unique_columns, rows


def escape_markdown_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|").strip()


def table_to_markdown(columns: list[str], rows: list[list[str]]) -> str:
    if not columns:
        return ""

    header = "| " + " | ".join(escape_markdown_cell(col) for col in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    body = []
    for row in rows:
        padded = row + [""] * (len(columns) - len(row))
        body.append(
            "| "
            + " | ".join(escape_markdown_cell(cell) for cell in padded[:len(columns)])
            + " |"
        )

    return "\n".join([header, separator, *body])


def table_to_html(
    columns: list[str],
    rows: list[list[str]],
    caption: str = "",
) -> str:
    caption_html = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    thead = "".join(f"<th>{html.escape(str(col))}</th>" for col in columns)

    body_rows = []
    for row in rows:
        padded = row + [""] * (len(columns) - len(row))
        tds = "".join(
            f"<td>{html.escape(str(cell))}</td>"
            for cell in padded[:len(columns)]
        )
        body_rows.append(f"<tr>{tds}</tr>")

    return (
        "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        f"{caption_html}"
        f"<thead><tr>{thead}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def save_table_artifacts(
    table_obj: dict[str, Any],
    artifact_dir: Path,
    table_stem: str,
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    columns = table_obj.get("columns", []) or []
    rows = table_obj.get("rows", []) or []

    csv_path = artifact_dir / f"{table_stem}.csv"
    html_path = artifact_dir / f"{table_stem}.html"
    json_path = artifact_dir / f"{table_stem}.json"
    markdown_path = artifact_dir / f"{table_stem}.md"

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if columns:
            writer.writerow(columns)
        writer.writerows(rows)

    html_path.write_text(
        table_to_html(columns, rows, table_obj.get("caption", "")),
        encoding="utf-8",
    )

    markdown_path.write_text(
        table_obj.get("markdown", ""),
        encoding="utf-8",
    )

    table_for_json = dict(table_obj)
    table_for_json.update(
        {
            "csv_path": str(csv_path),
            "html_path": str(html_path),
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
        }
    )

    save_json(table_for_json, json_path)

    return {
        "csv_path": str(csv_path),
        "html_path": str(html_path),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


# =========================================================
# Rendering helpers
# =========================================================

def render_clip(
    page: pymupdf.Page,
    bbox: Any,
    output_path: Path,
    zoom: float = 2.0,
) -> str:
    """Render a page region to PNG. Returns path string or empty string."""
    try:
        rect = pymupdf.Rect(bbox) & page.rect

        if rect.is_empty or rect.width < 5 or rect.height < 5:
            return ""

        output_path.parent.mkdir(parents=True, exist_ok=True)

        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(zoom, zoom),
            clip=rect,
            alpha=False,
        )
        pix.save(str(output_path))

        return str(output_path)

    except Exception:
        return ""


# =========================================================
# Caption helpers
# =========================================================

_TABLE_CAPTION_RE = re.compile(
    r"^\s*(Table\s+\d+[A-Za-z]?(?:\s*[:.\-–]\s*|\s+).+)",
    re.IGNORECASE,
)

_FIGURE_CAPTION_RE = re.compile(
    r"^\s*((?:Fig\.?|Figure)\s+\d+[A-Za-z]?(?:\s*[:.\-–]\s*|\s+).+)",
    re.IGNORECASE,
)

_DIAGRAM_KEYWORDS = re.compile(
    r"\b(architecture|framework|pipeline|flowchart|workflow|model|network|method|"
    r"process|encoder|decoder|transformer|algorithm|system|structure|block\s+diagram)\b",
    re.IGNORECASE,
)


def normalize_caption_for_match(caption: str) -> str:
    caption = caption or ""
    caption = (
        caption.replace("ﬂ", "fl")
        .replace("ﬁ", "fi")
        .replace("ﬀ", "ff")
        .replace("–", "-")
    )
    caption = re.sub(r"\s+", " ", caption)
    caption = re.sub(r"\s*[:.\-–]\s*", ": ", caption, count=1)
    return caption.strip().lower()


def extract_caption_lines(page_text: str, kind: str) -> list[str]:
    pattern = _TABLE_CAPTION_RE if kind == "table" else _FIGURE_CAPTION_RE
    captions: list[str] = []

    for line in (page_text or "").splitlines():
        match = pattern.match(line.strip())
        if match:
            captions.append(match.group(1).strip())

    return captions


def get_caption_blocks(page: pymupdf.Page, kind: str) -> list[dict[str, Any]]:
    pattern = _TABLE_CAPTION_RE if kind == "table" else _FIGURE_CAPTION_RE
    blocks: list[dict[str, Any]] = []

    try:
        text_dict = page.get_text("dict")
    except Exception:
        return blocks

    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = " ".join(str(span.get("text", "")) for span in spans).strip()
            match = pattern.match(text)

            if match:
                blocks.append(
                    {
                        "caption": match.group(1).strip(),
                        "bbox": line.get("bbox") or block.get("bbox"),
                    }
                )

    return blocks


def choose_caption(
    caption_blocks: list[dict[str, Any]],
    index: int,
    bbox: Any | None = None,
) -> tuple[str, str]:
    if not caption_blocks:
        return "", "manual_unknown"

    if bbox is None:
        if index < len(caption_blocks):
            return caption_blocks[index]["caption"], "same_page_caption_order"
        return caption_blocks[-1]["caption"], "same_page_regex"

    try:
        target = pymupdf.Rect(bbox)
        best = None
        best_score = float("inf")

        for item in caption_blocks:
            caption_bbox = item.get("bbox")
            if not caption_bbox:
                continue

            caption_rect = pymupdf.Rect(caption_bbox)

            vertical_distance = min(
                abs(caption_rect.y1 - target.y0),
                abs(target.y1 - caption_rect.y0),
            )

            x_overlap = max(
                0,
                min(caption_rect.x1, target.x1) - max(caption_rect.x0, target.x0),
            )

            score = vertical_distance - (x_overlap / max(target.width, 1)) * 25

            if score < best_score:
                best_score = score
                best = item

        if best:
            return best["caption"], "nearest_caption_bbox"

    except Exception:
        pass

    if index < len(caption_blocks):
        return caption_blocks[index]["caption"], "same_page_caption_order"

    return caption_blocks[-1]["caption"], "same_page_regex"


def classify_figure_content_type(caption: str, nearby_text: str = "") -> str:
    combined = f"{caption}\n{nearby_text}"
    if _DIAGRAM_KEYWORDS.search(combined):
        return ContentType.DIAGRAM.value
    return ContentType.FIGURE.value


# =========================================================
# Table quality
# =========================================================

def build_table_object(
    *,
    paper_name: str,
    source_file: str,
    page_number: int,
    section: str,
    table_id: str,
    caption: str,
    columns: list[str],
    rows: list[list[str]],
    method: str,
    confidence: str,
    caption_method: str,
    image_path: str = "",
) -> dict[str, Any]:
    markdown = table_to_markdown(columns, rows)

    return {
        "table_id": table_id,
        "content_type": ContentType.TABLE.value,
        "paper_name": paper_name,
        "source_file": source_file,
        "page_number": page_number,
        "section": section,
        "caption": caption,
        "caption_confidence": "medium" if caption else "low",
        "caption_link_method": caption_method,
        "columns": columns,
        "rows": rows,
        "markdown": markdown,
        "csv_path": "",
        "html_path": "",
        "json_path": "",
        "markdown_path": "",
        "image_path": image_path,
        "extraction_method": method,
        "extraction_confidence": confidence,
        "structured_status": "",
        "display_preference": "",
        "quality_reason": "",
        "vision_status": "",
        "vision_model": "",
        "vision_description": "",
        "vision_markdown": "",
    }


def table_nonempty_cells(table_obj: dict[str, Any]) -> list[str]:
    cells: list[str] = []

    for row in table_obj.get("rows", []) or []:
        for cell in row:
            text = str(cell or "").strip()
            if text:
                cells.append(text)

    return cells


def looks_like_garbled_table_text(text: str) -> bool:
    cleaned = re.sub(r"\s+", "", text or "")

    if not cleaned:
        return True

    alpha_count = sum(ch.isalpha() for ch in cleaned)
    digit_count = sum(ch.isdigit() for ch in cleaned)

    if len(cleaned) <= 4 and alpha_count <= 1 and digit_count == 0:
        return True

    lowered = text.lower()
    false_fragments = [
        "flatten",
        "linear projection",
        "fla tten",
        "lin ear",
        "projec tion",
    ]

    return any(fragment in lowered for fragment in false_fragments)


def looks_like_table_reference_caption(caption: str) -> bool:
    lowered = (caption or "").lower()
    bad_phrases = [
        "note that",
        "as shown",
        "the table shows",
        "in table",
        "from table",
        "according to table",
    ]
    return any(phrase in lowered for phrase in bad_phrases)


def assess_table_quality(table_obj: dict[str, Any]) -> dict[str, str]:
    rows = table_obj.get("rows", []) or []
    columns = table_obj.get("columns", []) or []
    caption = table_obj.get("caption", "") or ""
    method = table_obj.get("extraction_method", "") or ""

    cells = table_nonempty_cells(table_obj)
    sample = " ".join(str(cell).replace("\n", " ") for cell in cells[:12])

    has_caption = bool(caption.strip())
    row_count = len(rows)
    col_count = len(columns)
    cell_count = len(cells)
    has_multiline_cell = any("\n" in str(cell) for cell in cells)

    if method == "caption_nearby_text":
        if row_count >= 2 and cell_count >= 2:
            return {
                "keep": "true",
                "structured_status": TableStatus.TEXT_FALLBACK.value,
                "display_preference": DisplayPreference.HTML.value,
                "quality_reason": "kept_caption_with_nearby_table_text",
            }

        return {
            "keep": "false",
            "structured_status": TableStatus.REJECTED.value,
            "display_preference": DisplayPreference.NONE.value,
            "quality_reason": "rejected_nearby_text_too_small",
        }

    if method == "caption_text":
        if looks_like_table_reference_caption(caption):
            return {
                "keep": "false",
                "structured_status": TableStatus.REJECTED.value,
                "display_preference": DisplayPreference.NONE.value,
                "quality_reason": "rejected_table_reference_sentence",
            }

        return {
            "keep": "true",
            "structured_status": TableStatus.CAPTION_ONLY.value,
            "display_preference": DisplayPreference.TEXT.value,
            "quality_reason": "caption_only_fallback",
        }

    if not has_caption:
        return {
            "keep": "false",
            "structured_status": TableStatus.REJECTED.value,
            "display_preference": DisplayPreference.NONE.value,
            "quality_reason": "rejected_no_caption",
        }

    if row_count <= 1 or col_count <= 1:
        return {
            "keep": "false",
            "structured_status": TableStatus.REJECTED.value,
            "display_preference": DisplayPreference.NONE.value,
            "quality_reason": "rejected_low_structure",
        }

    if cell_count <= 2 or looks_like_garbled_table_text(sample):
        return {
            "keep": "false",
            "structured_status": TableStatus.REJECTED.value,
            "display_preference": DisplayPreference.NONE.value,
            "quality_reason": "rejected_garbled_or_sparse",
        }

    if col_count <= 2 or has_multiline_cell:
        return {
            "keep": "true",
            "structured_status": TableStatus.WEAK.value,
            "display_preference": DisplayPreference.IMAGE.value,
            "quality_reason": "kept_captioned_but_structure_weak",
        }

    return {
        "keep": "true",
        "structured_status": TableStatus.GOOD.value,
        "display_preference": DisplayPreference.HTML.value,
        "quality_reason": "kept_structured_table",
    }


def apply_table_quality(table_obj: dict[str, Any]) -> dict[str, Any]:
    quality = assess_table_quality(table_obj)

    table_obj.update(
        {
            "structured_status": quality["structured_status"],
            "display_preference": quality["display_preference"],
            "quality_reason": quality["quality_reason"],
        }
    )

    if quality["structured_status"] == TableStatus.WEAK.value:
        table_obj["extraction_confidence"] = "medium"

    return table_obj


def should_keep_table(table_obj: dict[str, Any]) -> bool:
    return assess_table_quality(table_obj)["keep"] == "true"


def deduplicate_tables_by_caption(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    structured_caption_keys = set()

    for table in tables:
        if table.get("extraction_method") != "caption_text" and table.get("caption"):
            structured_caption_keys.add(
                normalize_caption_for_match(table.get("caption", ""))
            )

    deduped: list[dict[str, Any]] = []
    seen = set()

    for table in tables:
        caption_key = normalize_caption_for_match(table.get("caption", ""))
        method = table.get("extraction_method", "")

        if method == "caption_text" and caption_key in structured_caption_keys:
            continue

        identity = (
            table.get("page_number"),
            caption_key,
            method,
            table.get("table_id"),
        )

        if identity in seen:
            continue

        seen.add(identity)
        deduped.append(table)

    return deduped


def filter_and_finalize_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []

    for table in tables:
        table = apply_table_quality(table)
        if should_keep_table(table):
            kept.append(table)

    return deduplicate_tables_by_caption(kept)


# =========================================================
# PyMuPDF table extraction
# =========================================================

def extract_tables_with_pymupdf_page(
    *,
    page: pymupdf.Page,
    page_text: str,
    metadata: dict[str, Any],
    paper_slug: str,
    page_number: int,
    tables_artifact_path: Path = TABLES_ARTIFACT_PATH,
) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []

    if not ENABLE_TABLE_EXTRACTION:
        return tables

    finder = getattr(page, "find_tables", None)
    if not callable(finder):
        return tables

    section = infer_section_from_page_text(page_text)
    caption_blocks = get_caption_blocks(page, "table")
    artifact_dir = tables_artifact_path / paper_slug

    try:
        table_finder = finder()
        detected_tables = list(getattr(table_finder, "tables", []) or [])
    except Exception:
        detected_tables = []

    for table_index, table in enumerate(detected_tables, start=1):
        try:
            matrix = normalize_matrix(table.extract())
        except Exception:
            matrix = []

        if not matrix:
            continue

        bbox = getattr(table, "bbox", None)
        caption, caption_method = choose_caption(
            caption_blocks,
            table_index - 1,
            bbox,
        )

        columns, rows = split_columns_rows(matrix)
        table_id = f"{paper_slug}_page_{page_number}_table_{table_index}"
        table_stem = f"page_{page_number}_table_{table_index}"

        table_obj = build_table_object(
            paper_name=metadata["paper_name"],
            source_file=metadata["source_file"],
            page_number=page_number,
            section=section,
            table_id=table_id,
            caption=caption,
            columns=columns,
            rows=rows,
            method="pymupdf_find_tables",
            confidence="high",
            caption_method=caption_method,
        )

        table_obj = apply_table_quality(table_obj)

        if not should_keep_table(table_obj):
            continue

        if bbox:
            table_obj["image_path"] = render_clip(
                page,
                bbox,
                artifact_dir / f"{table_stem}.png",
            )

        table_obj.update(
            save_table_artifacts(
                table_obj,
                artifact_dir,
                table_stem,
            )
        )

        tables.append(table_obj)

    return tables


# =========================================================
# pdfplumber table fallback
# =========================================================

def extract_tables_with_pdfplumber(
    *,
    file_path: Path,
    pages_text: list[dict[str, Any]],
    metadata: dict[str, Any],
    paper_slug: str,
    existing_page_numbers: set[int],
    tables_artifact_path: Path = TABLES_ARTIFACT_PATH,
) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []

    if not ENABLE_TABLE_EXTRACTION:
        return tables

    try:
        import pdfplumber
    except ImportError:
        return tables

    page_text_map = {
        int(page.get("page_number", 0) or 0): page.get("text", "")
        for page in pages_text
    }

    artifact_dir = tables_artifact_path / paper_slug

    try:
        with pdfplumber.open(str(file_path)) as pdf:
            for page_index, plumber_page in enumerate(pdf.pages, start=1):
                if page_index in existing_page_numbers:
                    continue

                try:
                    extracted_tables = plumber_page.extract_tables() or []
                except Exception:
                    extracted_tables = []

                if not extracted_tables:
                    continue

                page_text = page_text_map.get(page_index, "")
                section = infer_section_from_page_text(page_text)
                captions = extract_caption_lines(page_text, "table")

                for table_index, raw_rows in enumerate(extracted_tables, start=1):
                    matrix = normalize_matrix(raw_rows)

                    if not matrix:
                        continue

                    columns, rows = split_columns_rows(matrix)
                    caption = (
                        captions[table_index - 1]
                        if table_index - 1 < len(captions)
                        else ""
                    )

                    table_id = (
                        f"{paper_slug}_page_{page_index}"
                        f"_pdfplumber_table_{table_index}"
                    )
                    table_stem = f"page_{page_index}_pdfplumber_table_{table_index}"

                    table_obj = build_table_object(
                        paper_name=metadata["paper_name"],
                        source_file=metadata["source_file"],
                        page_number=page_index,
                        section=section,
                        table_id=table_id,
                        caption=caption,
                        columns=columns,
                        rows=rows,
                        method="pdfplumber",
                        confidence="medium",
                        caption_method="same_page_regex" if caption else "manual_unknown",
                    )

                    table_obj = apply_table_quality(table_obj)

                    if not should_keep_table(table_obj):
                        continue

                    table_obj.update(
                        save_table_artifacts(
                            table_obj,
                            artifact_dir,
                            table_stem,
                        )
                    )

                    tables.append(table_obj)

    except Exception:
        return tables

    return tables


# =========================================================
# Caption + nearby-text table fallback
# =========================================================

def normalize_for_line_match(text: str) -> str:
    text = text or ""
    text = (
        text.replace("ﬂ", "fl")
        .replace("ﬁ", "fi")
        .replace("ﬀ", "ff")
        .replace("–", "-")
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def find_caption_line_index(lines: list[str], caption: str) -> int:
    caption_norm = normalize_for_line_match(caption)

    for index, line in enumerate(lines):
        line_norm = normalize_for_line_match(line)

        if not line_norm:
            continue

        if line_norm == caption_norm:
            return index

        if caption_norm in line_norm:
            return index

        if line_norm in caption_norm and len(line_norm) > 20:
            return index

    return -1


def looks_like_new_section(line: str) -> bool:
    text = line.strip()

    if not text:
        return False

    return bool(re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z]", text))


def looks_like_paragraph_after_table(line: str) -> bool:
    text = line.strip()

    if not text:
        return False

    lowered = text.lower()

    paragraph_starters = (
        "then,",
        "in addition",
        "furthermore",
        "besides",
        "comparing",
        "generally speaking",
        "it can be seen",
        "as can be seen",
        "to visualize",
        "therefore",
        "based on",
        "for a fair comparison",
    )

    if lowered.startswith(paragraph_starters):
        return True

    words = text.split()

    return len(words) >= 12 and text.endswith(".")


def extract_lines_after_caption(
    page_text: str,
    caption: str,
    max_lines: int = 45,
) -> list[str]:
    lines = [line.strip() for line in (page_text or "").splitlines()]
    start_index = find_caption_line_index(lines, caption)

    if start_index < 0:
        return []

    collected: list[str] = []

    for line in lines[start_index + 1:]:
        clean = line.strip()

        if not clean:
            continue

        if _TABLE_CAPTION_RE.match(clean) or _FIGURE_CAPTION_RE.match(clean):
            break

        if collected and looks_like_new_section(clean):
            break

        if collected and looks_like_paragraph_after_table(clean):
            break

        collected.append(clean)

        if len(collected) >= max_lines:
            break

    return collected


def looks_like_simple_value_line(line: str) -> bool:
    text = re.sub(r"\s+", " ", line or "").strip()

    if not text:
        return False

    lowered = text.lower()

    known_values = {
        "adam",
        "sgd",
        "rmsprop",
        "none",
        "true",
        "false",
        "1d",
        "2d",
    }

    if lowered in known_values:
        return True

    if re.fullmatch(r"\[[^\]]+\]", text):
        return True

    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?%?", text):
        return True

    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?e[+-]?\d+", text, flags=re.IGNORECASE):
        return True

    return len(text.split()) <= 2 and any(ch.isdigit() for ch in text)


def line_to_two_cells(line: str) -> list[str]:
    original = line.strip()
    compact = re.sub(r"\s+", " ", original).strip()

    if not compact:
        return ["", ""]

    parts = [part.strip() for part in re.split(r"\s{2,}", original) if part.strip()]
    if len(parts) >= 2:
        return [" ".join(parts[:-1]), parts[-1]]

    bracket_match = re.match(r"^(.*?)(\[[^\]]+\])$", compact)
    if bracket_match:
        return [
            bracket_match.group(1).strip(),
            bracket_match.group(2).strip(),
        ]

    tokens = compact.split()

    if len(tokens) >= 2:
        return [" ".join(tokens[:-1]), tokens[-1]]

    return [compact, ""]


def nearby_lines_to_table(
    caption: str,
    nearby_lines: list[str],
) -> tuple[list[str], list[list[str]], str]:
    cleaned_lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in nearby_lines
        if line and line.strip()
    ]

    if not cleaned_lines:
        columns = ["Extracted caption"]
        rows = [[caption]]
        return columns, rows, table_to_markdown(columns, rows)

    if cleaned_lines[0].strip().lower() == "value":
        body_lines = cleaned_lines[1:]
        columns = ["Parameter", "Value"]

        odd_lines = body_lines[1::2]
        value_like_count = sum(
            1 for line in odd_lines
            if looks_like_simple_value_line(line)
        )

        is_alternating_param_value = (
            len(body_lines) >= 4
            and len(body_lines) % 2 == 0
            and odd_lines
            and value_like_count >= max(2, int(len(odd_lines) * 0.6))
        )

        if is_alternating_param_value:
            rows = [
                [body_lines[index], body_lines[index + 1]]
                for index in range(0, len(body_lines), 2)
                if index + 1 < len(body_lines)
            ]
        else:
            rows = [line_to_two_cells(line) for line in body_lines]

    else:
        parsed_rows = [line_to_two_cells(line) for line in cleaned_lines]
        useful_two_col_rows = sum(
            1 for row in parsed_rows
            if len(row) >= 2 and row[1].strip()
        )

        if useful_two_col_rows >= max(2, len(parsed_rows) // 2):
            columns = ["Field", "Value"]
            rows = parsed_rows
        else:
            columns = ["Extracted table text"]
            rows = [[line] for line in cleaned_lines]

    markdown = table_to_markdown(columns, rows)
    return columns, rows, markdown


# =========================================================
# Accurate fallback table crop helpers
# =========================================================

def page_text_line_records(page: pymupdf.Page) -> list[dict[str, Any]]:
    """
    Return visible text lines with bounding boxes.

    Used to crop fallback tables more accurately by matching the same nearby
    lines that were used to rebuild the table text.
    """
    records: list[dict[str, Any]] = []

    try:
        text_dict = page.get_text("dict")
    except Exception:
        return records

    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = " ".join(str(span.get("text", "")) for span in spans).strip()
            bbox = line.get("bbox") or block.get("bbox")

            if not text or not bbox:
                continue

            records.append(
                {
                    "text": re.sub(r"\s+", " ", text).strip(),
                    "bbox": pymupdf.Rect(bbox),
                }
            )

    records.sort(key=lambda item: (item["bbox"].y0, item["bbox"].x0))
    return records


def line_text_matches_target(line_text: str, target_text: str) -> bool:
    """
    Robust line matching because PyMuPDF dict text and plain text extraction
    may have slightly different spacing/ligatures.
    """
    line_norm = normalize_for_line_match(line_text)
    target_norm = normalize_for_line_match(target_text)

    if not line_norm or not target_norm:
        return False

    if line_norm == target_norm:
        return True

    if target_norm in line_norm:
        return True

    if line_norm in target_norm and len(line_norm) >= 4:
        return True

    return False


def matched_nearby_line_rects(
    *,
    page: pymupdf.Page,
    caption_bbox: Any,
    nearby_lines: list[str],
) -> list[pymupdf.Rect]:
    """
    Find bounding boxes for the same nearby text lines used to build the
    fallback table.
    """
    if not nearby_lines:
        return []

    cap = pymupdf.Rect(caption_bbox)
    records = page_text_line_records(page)

    targets = [
        line.strip()
        for line in nearby_lines
        if line and line.strip()
    ]

    if not targets:
        return []

    matched_rects: list[pymupdf.Rect] = []
    used_target_indexes: set[int] = set()

    for record in records:
        rect = record["bbox"]
        text = record["text"]

        if rect.y0 <= cap.y1:
            continue

        if rect.y0 > cap.y1 + page.rect.height * 0.45:
            break

        for target_index, target in enumerate(targets):
            if target_index in used_target_indexes:
                continue

            if line_text_matches_target(text, target):
                matched_rects.append(rect)
                used_target_indexes.add(target_index)
                break

    return matched_rects


def table_caption_region_bbox(
    page: pymupdf.Page,
    caption_bbox: Any,
    nearby_lines: list[str] | None = None,
) -> pymupdf.Rect:
    """
    Accurate crop for fallback tables.

    Priority:
    1. Use bounding boxes of nearby table lines.
    2. If matching fails, use a smaller dynamic fallback.
    """
    cap = pymupdf.Rect(caption_bbox)
    page_rect = page.rect

    margin_x = 20
    margin_y = 8

    nearby_lines = nearby_lines or []

    matched_rects = matched_nearby_line_rects(
        page=page,
        caption_bbox=caption_bbox,
        nearby_lines=nearby_lines,
    )

    if matched_rects:
        y0 = max(0, cap.y0 - margin_y)
        y1 = min(
            page_rect.height,
            max(rect.y1 for rect in matched_rects) + margin_y,
        )

        return pymupdf.Rect(
            margin_x,
            y0,
            page_rect.width - margin_x,
            y1,
        )

    line_records = page_text_line_records(page)
    candidate_rects: list[pymupdf.Rect] = []

    max_y = min(page_rect.height, cap.y1 + page_rect.height * 0.30)

    for record in line_records:
        text = record["text"]
        rect = record["bbox"]

        if rect.y0 <= cap.y1:
            continue

        if rect.y0 > max_y:
            break

        clean = text.strip()

        if not clean:
            continue

        if _TABLE_CAPTION_RE.match(clean) or _FIGURE_CAPTION_RE.match(clean):
            break

        if candidate_rects and looks_like_new_section(clean):
            break

        if candidate_rects and looks_like_paragraph_after_table(clean):
            break

        candidate_rects.append(rect)

    if candidate_rects:
        y0 = max(0, cap.y0 - margin_y)
        y1 = min(
            page_rect.height,
            max(rect.y1 for rect in candidate_rects) + margin_y,
        )

        return pymupdf.Rect(
            margin_x,
            y0,
            page_rect.width - margin_x,
            y1,
        )

    return pymupdf.Rect(
        margin_x,
        max(0, cap.y0 - margin_y),
        page_rect.width - margin_x,
        min(page_rect.height, cap.y1 + page_rect.height * 0.18),
    )


def render_table_caption_crop(
    file_path: Path,
    page_number: int,
    caption: str,
    output_path: Path,
    nearby_lines: list[str] | None = None,
) -> str:
    """
    Render a PNG crop around a fallback table caption.

    For text_fallback tables, nearby_lines are used to make the crop accurate.
    """
    if not caption or page_number <= 0:
        return ""

    try:
        with pymupdf.open(str(file_path)) as doc:
            if page_number < 1 or page_number > doc.page_count:
                return ""

            page = doc[page_number - 1]
            caption_blocks = get_caption_blocks(page, "table")
            wanted_caption = normalize_caption_for_match(caption)

            selected_bbox = None

            for block in caption_blocks:
                block_caption = normalize_caption_for_match(block.get("caption", ""))

                if (
                    wanted_caption
                    and block_caption
                    and (
                        wanted_caption == block_caption
                        or wanted_caption in block_caption
                        or block_caption in wanted_caption
                    )
                ):
                    selected_bbox = block.get("bbox")
                    break

            if selected_bbox is None and caption_blocks:
                selected_bbox = caption_blocks[0].get("bbox")

            if selected_bbox is None:
                return ""

            crop_bbox = table_caption_region_bbox(
                page=page,
                caption_bbox=selected_bbox,
                nearby_lines=nearby_lines,
            )

            return render_clip(
                page,
                crop_bbox,
                output_path,
                zoom=2.0,
            )

    except Exception:
        return ""


def extract_caption_fallback_tables(
    *,
    file_path: Path,
    pages_text: list[dict[str, Any]],
    metadata: dict[str, Any],
    paper_slug: str,
    existing_caption_keys: set[tuple[int, str]],
    tables_artifact_path: Path = TABLES_ARTIFACT_PATH,
) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    artifact_dir = tables_artifact_path / paper_slug

    for page in pages_text:
        page_number = int(page.get("page_number", 0) or 0)
        page_text = page.get("text", "")
        section = infer_section_from_page_text(page_text)
        captions = extract_caption_lines(page_text, "table")

        for caption_index, caption in enumerate(captions, start=1):
            key = (page_number, normalize_caption_for_match(caption))

            if key in existing_caption_keys:
                continue

            table_id = f"{paper_slug}_page_{page_number}_caption_table_{caption_index}"
            table_stem = f"page_{page_number}_caption_table_{caption_index}"

            nearby_lines = extract_lines_after_caption(page_text, caption)
            columns, rows, markdown = nearby_lines_to_table(caption, nearby_lines)

            method = "caption_nearby_text" if nearby_lines else "caption_text"
            confidence = "medium" if nearby_lines else "low"
            caption_method = (
                "caption_plus_nearby_text"
                if nearby_lines
                else "same_page_regex"
            )

            table_obj = build_table_object(
                paper_name=metadata["paper_name"],
                source_file=metadata["source_file"],
                page_number=page_number,
                section=section,
                table_id=table_id,
                caption=caption,
                columns=columns,
                rows=rows,
                method=method,
                confidence=confidence,
                caption_method=caption_method,
            )

            table_obj["markdown"] = markdown
            table_obj = apply_table_quality(table_obj)

            if not should_keep_table(table_obj):
                continue

            table_obj["image_path"] = render_table_caption_crop(
                file_path=file_path,
                page_number=page_number,
                caption=caption,
                output_path=artifact_dir / f"{table_stem}.png",
                nearby_lines=nearby_lines,
            )

            table_obj.update(
                save_table_artifacts(
                    table_obj,
                    artifact_dir,
                    table_stem,
                )
            )

            tables.append(table_obj)

    return tables


# =========================================================
# Public table extraction API
# =========================================================

def extract_tables_from_document(
    *,
    file_path: Path,
    doc: pymupdf.Document,
    pages_text: list[dict[str, Any]],
    metadata: dict[str, Any],
    paper_slug: str,
    tables_artifact_path: Path = TABLES_ARTIFACT_PATH,
) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []

    page_text_map = {
        int(page.get("page_number", 0) or 0): page.get("text", "")
        for page in pages_text
    }

    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_number = page_index + 1
        page_text = page_text_map.get(page_number, "")

        tables.extend(
            extract_tables_with_pymupdf_page(
                page=page,
                page_text=page_text,
                metadata=metadata,
                paper_slug=paper_slug,
                page_number=page_number,
                tables_artifact_path=tables_artifact_path,
            )
        )

    existing_page_numbers = {
        int(table.get("page_number", 0) or 0)
        for table in tables
    }

    tables.extend(
        extract_tables_with_pdfplumber(
            file_path=file_path,
            pages_text=pages_text,
            metadata=metadata,
            paper_slug=paper_slug,
            existing_page_numbers=existing_page_numbers,
            tables_artifact_path=tables_artifact_path,
        )
    )

    tables = filter_and_finalize_tables(tables)

    existing_caption_keys = {
        (
            int(table.get("page_number", 0) or 0),
            normalize_caption_for_match(str(table.get("caption", ""))),
        )
        for table in tables
        if table.get("caption") and table.get("extraction_method") != "caption_text"
    }

    tables.extend(
        extract_caption_fallback_tables(
            file_path=file_path,
            pages_text=pages_text,
            metadata=metadata,
            paper_slug=paper_slug,
            existing_caption_keys=existing_caption_keys,
            tables_artifact_path=tables_artifact_path,
        )
    )

    return filter_and_finalize_tables(tables)


# =========================================================
# Figure / image / diagram extraction
# =========================================================

def build_figure_object(
    *,
    paper_name: str,
    source_file: str,
    page_number: int,
    section: str,
    figure_id: str,
    content_type: str,
    caption: str,
    image_path: str,
    method: str,
    confidence: str,
    caption_method: str,
) -> dict[str, Any]:
    vision_status = VisionStatus.NOT_GENERATED.value

    if ENABLE_VISION_DESCRIPTIONS:
        vision_status = "enabled_but_not_implemented"

    return {
        "figure_id": figure_id,
        "content_type": content_type,
        "paper_name": paper_name,
        "source_file": source_file,
        "page_number": page_number,
        "section": section,
        "caption": caption,
        "caption_confidence": "medium" if caption else "low",
        "caption_link_method": caption_method,
        "description": "Vision description not generated yet.",
        "image_path": image_path,
        "extraction_method": method,
        "extraction_confidence": confidence,
        "vision_status": vision_status,
        "vision_model": "",
        "vision_description": "",
        "mermaid": "",
        "mermaid_status": (
            "not_generated"
            if not ENABLE_MERMAID_GENERATION
            else "enabled_but_not_implemented"
        ),
    }


def caption_region_bbox(
    page: pymupdf.Page,
    caption_bbox: Any,
) -> pymupdf.Rect:
    cap = pymupdf.Rect(caption_bbox)
    page_rect = page.rect

    margin_x = 20
    margin_y = 20

    if cap.y0 > page_rect.height * 0.25:
        y0 = max(0, cap.y0 - page_rect.height * 0.45)
        y1 = min(page_rect.height, cap.y1 + margin_y)
    else:
        y0 = max(0, cap.y0 - margin_y)
        y1 = min(page_rect.height, cap.y1 + page_rect.height * 0.45)

    return pymupdf.Rect(
        margin_x,
        y0,
        page_rect.width - margin_x,
        y1,
    )


def extract_caption_region_figures_page(
    *,
    page: pymupdf.Page,
    page_text: str,
    metadata: dict[str, Any],
    paper_slug: str,
    page_number: int,
    figures_artifact_path: Path = FIGURES_ARTIFACT_PATH,
) -> list[dict[str, Any]]:
    figures: list[dict[str, Any]] = []

    if not ENABLE_FIGURE_REGION_CROPPING:
        return figures

    caption_blocks = get_caption_blocks(page, "figure")

    if not caption_blocks:
        return figures

    artifact_dir = figures_artifact_path / paper_slug
    section = infer_section_from_page_text(page_text)

    for figure_index, caption_block in enumerate(caption_blocks, start=1):
        caption = caption_block.get("caption", "")
        bbox = caption_block.get("bbox")

        if not bbox:
            continue

        region = caption_region_bbox(page, bbox)
        figure_stem = f"page_{page_number}_caption_fig_{figure_index}"

        image_path = render_clip(
            page,
            region,
            artifact_dir / f"{figure_stem}.png",
        )

        if not image_path:
            continue

        content_type = classify_figure_content_type(caption, page_text)
        figure_id = f"{paper_slug}_page_{page_number}_{content_type}_{figure_index}"

        figures.append(
            build_figure_object(
                paper_name=metadata["paper_name"],
                source_file=metadata["source_file"],
                page_number=page_number,
                section=section,
                figure_id=figure_id,
                content_type=content_type,
                caption=caption,
                image_path=image_path,
                method="caption_region_crop",
                confidence="medium",
                caption_method="same_page_regex",
            )
        )

    return figures


def extract_embedded_images_page(
    *,
    doc: pymupdf.Document,
    page: pymupdf.Page,
    page_text: str,
    metadata: dict[str, Any],
    paper_slug: str,
    page_number: int,
    skip_if_caption_regions_exist: bool = True,
    figures_artifact_path: Path = FIGURES_ARTIFACT_PATH,
) -> list[dict[str, Any]]:
    figures: list[dict[str, Any]] = []

    if not ENABLE_BASIC_IMAGE_EXTRACTION:
        return figures

    caption_blocks = get_caption_blocks(page, "figure")

    if skip_if_caption_regions_exist and caption_blocks:
        return figures

    artifact_dir = figures_artifact_path / paper_slug
    section = infer_section_from_page_text(page_text)

    seen_xrefs: set[int] = set()

    try:
        images = page.get_images(full=True) or []
    except Exception:
        images = []

    for image_index, image_info in enumerate(images, start=1):
        xref = int(image_info[0])

        if xref in seen_xrefs:
            continue

        seen_xrefs.add(xref)

        try:
            pix = pymupdf.Pixmap(doc, xref)

            if pix.width < 80 or pix.height < 80:
                continue

            if pix.n - pix.alpha > 3:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

            image_stem = f"page_{page_number}_embedded_image_{image_index}"
            image_path_obj = artifact_dir / f"{image_stem}.png"
            image_path_obj.parent.mkdir(parents=True, exist_ok=True)

            pix.save(str(image_path_obj))

        except Exception:
            continue

        caption, caption_method = choose_caption(
            caption_blocks,
            image_index - 1,
            None,
        )

        content_type = classify_figure_content_type(caption, page_text)
        figure_id = f"{paper_slug}_page_{page_number}_embedded_{content_type}_{image_index}"

        figures.append(
            build_figure_object(
                paper_name=metadata["paper_name"],
                source_file=metadata["source_file"],
                page_number=page_number,
                section=section,
                figure_id=figure_id,
                content_type=content_type,
                caption=caption,
                image_path=str(image_path_obj),
                method="pymupdf_embedded_image",
                confidence="medium" if caption else "low",
                caption_method=caption_method,
            )
        )

    return figures


def extract_figures_from_document(
    *,
    doc: pymupdf.Document,
    pages_text: list[dict[str, Any]],
    metadata: dict[str, Any],
    paper_slug: str,
    figures_artifact_path: Path = FIGURES_ARTIFACT_PATH,
) -> list[dict[str, Any]]:
    figures: list[dict[str, Any]] = []

    page_text_map = {
        int(page.get("page_number", 0) or 0): page.get("text", "")
        for page in pages_text
    }

    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_number = page_index + 1
        page_text = page_text_map.get(page_number, "")

        figures.extend(
            extract_caption_region_figures_page(
                page=page,
                page_text=page_text,
                metadata=metadata,
                paper_slug=paper_slug,
                page_number=page_number,
                figures_artifact_path=figures_artifact_path,
            )
        )

        figures.extend(
            extract_embedded_images_page(
                doc=doc,
                page=page,
                page_text=page_text,
                metadata=metadata,
                paper_slug=paper_slug,
                page_number=page_number,
                skip_if_caption_regions_exist=True,
                figures_artifact_path=figures_artifact_path,
            )
        )

    return figures
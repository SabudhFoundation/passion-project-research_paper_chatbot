"""
chunking.py

Creates typed chunks for text and multimodal artifacts.

- text chunks are split normally
- each table is kept as one atomic chunk
- each figure / image / diagram / flowchart is kept as one atomic chunk

Run:
python src/chunking.py --combined_json data/parsed_combined/combined_papers.json --output_folder data/chunks --chunk_size 800 --chunk_overlap 150
"""

from __future__ import annotations

import argparse

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from core.constants import (
    ContentType,
    TABLE_METADATA_KEYS,
    VISUAL_CONTENT_TYPES,
    VISUAL_METADATA_KEYS,
)
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_SECTION_CHARS,
)
from utilities import save_json

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    class RecursiveCharacterTextSplitter:
        """Fallback splitter used when langchain-text-splitters is not installed."""

        def __init__(self, chunk_size: int, chunk_overlap: int, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        def split_text(self, text: str) -> list[str]:
            text = text or ""

            if len(text) <= self.chunk_size:
                return [text]

            chunks = []
            start = 0

            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                window = text[start:end]

                if end < len(text):
                    split_positions = [
                        window.rfind(separator)
                        for separator in self.separators
                        if separator
                    ]
                    split_at = max(split_positions) if split_positions else -1

                    if split_at > self.chunk_size * 0.5:
                        end = start + split_at
                        window = text[start:end]

                chunks.append(window.strip())

                if end >= len(text):
                    break

                start = max(end - self.chunk_overlap, start + 1)

            return [chunk for chunk in chunks if chunk]


# =========================================================
# Config
# =========================================================

class ChunkConfig(BaseModel):
    combined_json: Path
    output_folder: Path
    chunk_size: int = Field(default=CHUNK_SIZE, gt=100)
    chunk_overlap: int = Field(default=CHUNK_OVERLAP, ge=0)
    min_section_chars: int = Field(default=MIN_SECTION_CHARS, ge=0)

    @field_validator("combined_json")
    @classmethod
    def file_must_exist(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Combined JSON not found: {value}")
        return value

def parse_args() -> ChunkConfig:
    parser = argparse.ArgumentParser(description="Create text + artifact chunks")
    parser.add_argument("--combined_json", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--min_section_chars", type=int, default=MIN_SECTION_CHARS)

    args = parser.parse_args()

    return ChunkConfig(
        combined_json=Path(args.combined_json),
        output_folder=Path(args.output_folder),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_section_chars=args.min_section_chars,
    )


# =========================================================
# Section detection
# =========================================================

_RE_APPENDIX_SUB = re.compile(r"^[A-Z]\.\d+(?:\.\d+)?\s+[A-Z]")
_RE_NUMBERED_SECTION = re.compile(r"^(?:\d+\.?|\d+\.\d+\.?)\s+[A-Z][a-zA-Z]")
_RE_ROMAN = re.compile(r"^(?:I{1,3}|IV|V|VI{0,3}|IX|X)\.?\s+[A-Z][a-zA-Z]")

_KNOWN_SECTION_NAMES = re.compile(
    r"^(Abstract|Introduction|Background|Methodology|Methods|"
    r"Experiments?|Results?|Evaluation|Discussion|Conclusion|"
    r"Limitations?|Future\s+Work|References?|Bibliography|"
    r"Appendix|Acknowledgements?|Related\s+Work)$",
    re.IGNORECASE,
)

_RE_AUTHOR_LINE = re.compile(
    r"(university|institute|department|laboratory|\binc\b|\bltd\b)",
    re.IGNORECASE,
)
_RE_CITATION = re.compile(r"\[\d+\]|\(\w[\w\s]+,\s*\d{4}\)")
_RE_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_RE_EMAIL = re.compile(
    r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}",
    re.IGNORECASE,
)
_RE_DOI = re.compile(r"\bdoi\s*:?\s*10\.\d{4,}", re.IGNORECASE)
_RE_COPYRIGHT = re.compile(r"©|Copyright|All rights reserved", re.IGNORECASE)
_RE_PAGE_NUM = re.compile(r"^\s*(Page\s*)?\d{1,4}\s*$", re.IGNORECASE)
_RE_LONE_SECTION_NUM = re.compile(r"^(?:\d+|[A-Z]|\d+\.\d+)$")


def is_noise_line(text: str) -> bool:
    text = text.strip()

    if not text or len(text) < 4:
        return True
    if _RE_URL.search(text):
        return True
    if _RE_EMAIL.search(text):
        return True
    if _RE_DOI.search(text):
        return True
    if _RE_COPYRIGHT.search(text):
        return True
    if _RE_PAGE_NUM.match(text):
        return True
    if sum(char.isalpha() for char in text) < 3:
        return True

    return False


def is_hard_rejected_header(text: str) -> bool:
    text = text.strip()

    if not text or len(text) < 2:
        return True
    if len(text) > 90:
        return True
    if text.count(",") > 3:
        return True
    if _RE_AUTHOR_LINE.search(text):
        return True
    if _RE_CITATION.search(text):
        return True
    if _RE_URL.search(text):
        return True
    if _RE_EMAIL.search(text):
        return True
    if not any(char.isalpha() for char in text):
        return True

    return False


def looks_like_sentence_fragment(text: str) -> bool:
    words = text.split()

    if len(words) > 12:
        return True
    if words and words[0][0].islower():
        return True
    if text.endswith(","):
        return True

    return False


def classify_header(line: str) -> Optional[str]:
    text = line.strip()

    if is_hard_rejected_header(text):
        return None

    if looks_like_sentence_fragment(text):
        return None

    score = 0

    if _RE_APPENDIX_SUB.match(text):
        score += 4
    if _RE_NUMBERED_SECTION.match(text):
        score += 3
    if _RE_ROMAN.match(text):
        score += 3
    if _KNOWN_SECTION_NAMES.match(text):
        score += 3
    if text.istitle():
        score += 1
    if len(text.split()) <= 6:
        score += 1
    if not text.endswith("."):
        score += 1

    return text if score >= 4 else None


def merge_split_headers(lines: list[str]) -> list[str]:
    result = []
    index = 0

    while index < len(lines):
        current = lines[index].strip()
        next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""

        is_lone = bool(_RE_LONE_SECTION_NUM.match(current))
        next_is_header = bool(
            _KNOWN_SECTION_NAMES.match(next_line)
            or _RE_APPENDIX_SUB.match(next_line)
        )

        if is_lone and next_is_header:
            result.append(current + " " + next_line)
            index += 2
        else:
            result.append(lines[index])
            index += 1

    return result


def split_into_sections(full_text: str) -> list[dict[str, str]]:
    raw_lines = merge_split_headers(full_text.split("\n"))
    lines = [line.rstrip() for line in raw_lines if not is_noise_line(line)]

    sections = []
    current_header = "Body"
    current_body = []

    def flush() -> None:
        text = "\n".join(current_body).strip()
        if text:
            sections.append(
                {
                    "section": current_header,
                    "text": text,
                }
            )
        current_body.clear()

    for line in lines:
        header = classify_header(line)

        if header:
            flush()
            current_header = header
        else:
            current_body.append(line)

    flush()

    if not sections:
        fallback = "\n".join(lines).strip()
        if fallback:
            sections.append({"section": "Body", "text": fallback})

    return sections


def accumulate_sections(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accumulated = []
    current_label = None
    current_parts = []
    page_start = None
    page_end = None

    def flush() -> None:
        if current_label and current_parts:
            accumulated.append(
                {
                    "section": current_label,
                    "text": "\n".join(current_parts).strip(),
                    "page_start": page_start,
                    "page_end": page_end,
                }
            )

    for page in pages:
        page_number = page.get("page_number", "?")
        text = page.get("text", "")

        if not text.strip():
            continue

        for section in split_into_sections(text):
            label = section["section"]
            body = section["text"]

            if label == current_label:
                current_parts.append(body)
            else:
                flush()
                current_label = label
                current_parts = [body]
                page_start = page_number

            page_end = page_number

    flush()
    return accumulated


# =========================================================
# Chunk helper functions
# =========================================================

def compact_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def clean_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    return {
        key: compact_value(value)
        for key, value in metadata.items()
        if value is not None
    }


def build_table_embedding_text(table: dict[str, Any]) -> str:
    columns = table.get("columns") or []
    rows = table.get("rows") or []
    markdown = table.get("markdown") or ""

    if not markdown and rows:
        markdown = json.dumps(rows[:20], ensure_ascii=False)

    vision_description = table.get("vision_description", "") or ""
    vision_markdown = table.get("vision_markdown", "") or ""

    return f"""Content type: table
Paper: {table.get('paper_name', '')}
Section: {table.get('section', '')}
Page: {table.get('page_number', '')}
Caption: {table.get('caption', '')}
Columns: {', '.join(str(column) for column in columns)}
Markdown table:
{markdown}

Vision description:
{vision_description}

Vision markdown:
{vision_markdown}

CSV path: {table.get('csv_path', '')}
HTML path: {table.get('html_path', '')}
JSON path: {table.get('json_path', '')}
Image path: {table.get('image_path', '')}
Structured status: {table.get('structured_status', '')}
Display preference: {table.get('display_preference', '')}
Quality reason: {table.get('quality_reason', '')}
Extraction method: {table.get('extraction_method', '')}
Extraction confidence: {table.get('extraction_confidence', '')}
Vision status: {table.get('vision_status', '')}
Vision model: {table.get('vision_model', '')}
""".strip()


def build_visual_embedding_text(visual: dict[str, Any]) -> str:
    content_type = visual.get("content_type", ContentType.FIGURE.value)
    description = visual.get("vision_description") or visual.get("description", "")

    return f"""Content type: {content_type}
Paper: {visual.get('paper_name', '')}
Section: {visual.get('section', '')}
Page: {visual.get('page_number', '')}
Caption: {visual.get('caption', '')}
Description:
{description}
Image path: {visual.get('image_path', '')}
Mermaid status: {visual.get('mermaid_status', '')}
Mermaid:
{visual.get('mermaid', '')}
Extraction method: {visual.get('extraction_method', '')}
Extraction confidence: {visual.get('extraction_confidence', '')}
Vision status: {visual.get('vision_status', '')}
Vision model: {visual.get('vision_model', '')}
""".strip()


def build_table_metadata(table: dict[str, Any]) -> dict[str, str]:
    return clean_metadata(
        {
            key: table.get(key, "")
            for key in TABLE_METADATA_KEYS
        }
    )


def build_visual_metadata(visual: dict[str, Any]) -> dict[str, str]:
    return clean_metadata(
        {
            key: visual.get(key, "")
            for key in VISUAL_METADATA_KEYS
        }
    )


def make_table_chunk(
    chunk_id: int,
    paper: dict[str, Any],
    table: dict[str, Any],
) -> dict[str, Any]:
    page_number = table.get("page_number", "")
    table = {
        **table,
        "paper_name": table.get("paper_name") or paper.get("paper_name", "unknown"),
        "author": paper.get("author", "Unknown"),
        "year": paper.get("year", "Unknown"),
        "source_file": table.get("source_file") or paper.get("source_file", ""),
        "page_start": page_number,
        "page_end": page_number,
    }

    embedding_text = build_table_embedding_text(table)
    metadata = build_table_metadata(table)

    return {
        "chunk_id": chunk_id,
        "content_type": ContentType.TABLE.value,
        "paper_name": table.get("paper_name", "unknown"),
        "author": table.get("author", "Unknown"),
        "year": table.get("year", "Unknown"),
        "source_file": table.get("source_file", ""),
        "section": table.get("section", "Unknown"),
        "text": embedding_text,
        "embedding_text": embedding_text,
        "char_count": len(embedding_text),
        "page_start": page_number,
        "page_end": page_number,
        "metadata": metadata,
    }


def make_visual_chunk(
    chunk_id: int,
    paper: dict[str, Any],
    visual: dict[str, Any],
) -> dict[str, Any]:
    page_number = visual.get("page_number", "")
    content_type = visual.get("content_type", ContentType.FIGURE.value)

    visual = {
        **visual,
        "paper_name": visual.get("paper_name") or paper.get("paper_name", "unknown"),
        "author": paper.get("author", "Unknown"),
        "year": paper.get("year", "Unknown"),
        "source_file": visual.get("source_file") or paper.get("source_file", ""),
        "page_start": page_number,
        "page_end": page_number,
    }

    embedding_text = build_visual_embedding_text(visual)
    metadata = build_visual_metadata(visual)

    return {
        "chunk_id": chunk_id,
        "content_type": content_type,
        "paper_name": visual.get("paper_name", "unknown"),
        "author": visual.get("author", "Unknown"),
        "year": visual.get("year", "Unknown"),
        "source_file": visual.get("source_file", ""),
        "section": visual.get("section", "Unknown"),
        "text": embedding_text,
        "embedding_text": embedding_text,
        "char_count": len(embedding_text),
        "page_start": page_number,
        "page_end": page_number,
        "metadata": metadata,
    }


# =========================================================
# Pipeline
# =========================================================

def chunk_papers(config: ChunkConfig) -> Path:
    config.output_folder.mkdir(parents=True, exist_ok=True)

    with config.combined_json.open("r", encoding="utf-8") as file:
        papers = json.load(file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    chunk_id = 0

    for paper in papers:
        paper_name = paper.get("paper_name", "unknown")
        paper_author = paper.get("author", "Unknown")
        paper_year = paper.get("year", "Unknown")
        paper_source_file = paper.get("source_file", "")
        pages = paper.get("pages", []) or []

        print(f"\n{paper_name} ({len(pages)} pages)")

        doc_sections = accumulate_sections(pages)
        section_summary: dict[str, int] = {}

        for section in doc_sections:
            label = section["section"]
            body = section["text"]

            if len(body) < config.min_section_chars:
                continue

            for part in splitter.split_text(body):
                part = part.strip()

                if len(part) < 50:
                    continue

                metadata = clean_metadata(
                    {
                        "content_type": ContentType.TEXT.value,
                        "paper_name": paper_name,
                        "author": paper_author,
                        "year": paper_year,
                        "section": label,
                        "page_start": section["page_start"],
                        "page_end": section["page_end"],
                        "source_file": paper_source_file,
                    }
                )

                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "content_type": ContentType.TEXT.value,
                        "paper_name": paper_name,
                        "author": paper_author,
                        "year": paper_year,
                        "source_file": paper_source_file,
                        "section": label,
                        "text": part,
                        "embedding_text": part,
                        "char_count": len(part),
                        "page_start": section["page_start"],
                        "page_end": section["page_end"],
                        "metadata": metadata,
                    }
                )

                chunk_id += 1
                section_summary[label] = section_summary.get(label, 0) + 1

        table_count = 0
        for table in paper.get("tables", []) or []:
            all_chunks.append(make_table_chunk(chunk_id, paper, table))
            chunk_id += 1
            table_count += 1

        visual_counts: dict[str, int] = {}
        for visual in paper.get("figures", []) or []:
            content_type = visual.get("content_type", ContentType.FIGURE.value)

            if content_type not in VISUAL_CONTENT_TYPES:
                content_type = ContentType.FIGURE.value
                visual["content_type"] = content_type

            all_chunks.append(make_visual_chunk(chunk_id, paper, visual))
            chunk_id += 1
            visual_counts[content_type] = visual_counts.get(content_type, 0) + 1

        print("Text sections:", section_summary)
        print(f"Artifact chunks: tables={table_count}, visuals={visual_counts}")

    output_path = config.output_folder / "chunks.json"
    save_json(all_chunks, output_path)

    print(f"\nSaved: {output_path}")
    print(f"Total chunks: {len(all_chunks)}")

    return output_path


if __name__ == "__main__":
    config = parse_args()
    print("CONFIG:", config)
    chunk_papers(config)
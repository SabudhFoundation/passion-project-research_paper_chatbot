"""
python src/chunking.py --combined_json data/parsed_combined/combined_papers.json --output_folder data/chunks --chunk_size 800 --chunk_overlap 150
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, field_validator


# CONFIG


class ChunkConfig(BaseModel):
    combined_json: Path
    output_folder: Path
    chunk_size: int = Field(default=1000, gt=100)
    chunk_overlap: int = Field(default=200, ge=0)
    min_section_chars: int = Field(default=60, ge=0)

    @field_validator("combined_json")
    def file_must_exist(cls, v):
        if not v.exists():
            raise ValueError(f"Combined JSON not found: {v}")
        return v


def parse_args() -> ChunkConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined_json", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    parser.add_argument("--min_section_chars", type=int, default=60)

    args = parser.parse_args()

    return ChunkConfig(
        combined_json=Path(args.combined_json),
        output_folder=Path(args.output_folder),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_section_chars=args.min_section_chars,
    )


# HEADER PATTERNS


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


# SAFETY FILTERS


_RE_AUTHOR_LINE = re.compile(r"(university|institute|department|laboratory|\binc\b|\bltd\b)", re.I)
_RE_CITATION = re.compile(r"\[\d+\]|\(\w[\w\s]+,\s*\d{4}\)")
_RE_URL = re.compile(r"https?://|www\.", re.I)
_RE_EMAIL = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", re.I)


def _is_hard_rejected(t: str) -> bool:
    if not t or len(t) < 2:
        return True
    if len(t) > 90:
        return True
    if t.count(",") > 3:
        return True
    if _RE_AUTHOR_LINE.search(t):
        return True
    if _RE_CITATION.search(t):
        return True
    if _RE_URL.search(t):
        return True
    if _RE_EMAIL.search(t):
        return True
    if not any(c.isalpha() for c in t):
        return True
    return False


def _looks_like_sentence_fragment(t: str) -> bool:
    words = t.split()

    if len(words) > 12:
        return True
    if words and words[0][0].islower():
        return True
    if t.endswith(","):
        return True

    return False


# HEADER CLASSIFICATION (FIXED)


def _classify_header(line: str) -> Optional[str]:
    t = line.strip()

    if _is_hard_rejected(t):
        return None

    if _looks_like_sentence_fragment(t):
        return None

    score = 0

    if _RE_APPENDIX_SUB.match(t):
        score += 4
    if _RE_NUMBERED_SECTION.match(t):
        score += 3
    if _RE_ROMAN.match(t):
        score += 3
    if _KNOWN_SECTION_NAMES.match(t):
        score += 3

    if t.istitle():
        score += 1
    if len(t.split()) <= 6:
        score += 1
    if not t.endswith("."):
        score += 1

    return t if score >= 4 else None


# NOISE FILTER


_RE_DOI = re.compile(r"\bdoi\s*:?\s*10\.\d{4,}", re.I)
_RE_COPYRIGHT = re.compile(r"©|Copyright|All rights reserved", re.I)
_RE_PAGE_NUM = re.compile(r"^\s*(Page\s*)?\d{1,4}\s*$", re.I)


def is_noise_line(text: str) -> bool:
    t = text.strip()

    if not t or len(t) < 4:
        return True
    if _RE_URL.search(t):
        return True
    if _RE_EMAIL.search(t):
        return True
    if _RE_DOI.search(t):
        return True
    if _RE_COPYRIGHT.search(t):
        return True
    if _RE_PAGE_NUM.match(t):
        return True
    if sum(c.isalpha() for c in t) < 3:
        return True

    return False


# HEADER MERGING FIX

_RE_LONE_SECTION_NUM = re.compile(r"^(?:\d+|[A-Z]|\d+\.\d+)$")


def _merge_split_headers(lines: List[str]) -> List[str]:
    result = []
    i = 0

    while i < len(lines):
        cur = lines[i].strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""

        is_lone = bool(_RE_LONE_SECTION_NUM.match(cur))
        next_is_header = bool(_KNOWN_SECTION_NAMES.match(nxt) or _RE_APPENDIX_SUB.match(nxt))

        if is_lone and next_is_header:
            result.append(cur + " " + nxt)
            i += 2
        else:
            result.append(lines[i])
            i += 1

    return result


# SECTION SPLITTER


def split_into_sections(full_text: str) -> List[Dict[str, str]]:
    raw_lines = full_text.split("\n")

    # FIX: merge first, THEN filter
    raw_lines = _merge_split_headers(raw_lines)
    lines = [l.rstrip() for l in raw_lines if not is_noise_line(l)]

    sections = []
    current_header = "Body"
    current_body = []

    def flush():
        text = "\n".join(current_body).strip()
        if text:
            sections.append({"section": current_header, "text": text})
        current_body.clear()

    for line in lines:
        header = _classify_header(line)

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



# ACCUMULATE DOCUMENT


def accumulate_sections(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    accumulated = []
    current_label = None
    current_parts = []
    page_start = None
    page_end = None

    def flush():
        if current_label and current_parts:
            accumulated.append({
                "section": current_label,
                "text": "\n".join(current_parts).strip(),
                "page_start": page_start,
                "page_end": page_end,
            })

    for page in pages:
        pnum = page.get("page_number", "?")
        text = page.get("text", "")

        if not text.strip():
            continue

        for sec in split_into_sections(text):
            label = sec["section"]
            body = sec["text"]

            if label == current_label:
                current_parts.append(body)
            else:
                flush()
                current_label = label
                current_parts = [body]
                page_start = pnum

            page_end = pnum

    flush()
    return accumulated



# CHUNKING PIPELINE


def chunk_papers(config: ChunkConfig) -> Path:
    config.output_folder.mkdir(parents=True, exist_ok=True)

    with open(config.combined_json, "r", encoding="utf-8") as f:
        papers = json.load(f)

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
        pages = paper.get("pages", [])

        print(f"\n{paper_name} ({len(pages)} pages)")

        doc_sections = accumulate_sections(pages)
        summary = {}

        for sec in doc_sections:
            label = sec["section"]
            body = sec["text"]

            if len(body) < config.min_section_chars:
                continue

            for part in splitter.split_text(body):
                part = part.strip()
                if len(part) < 50:
                    continue

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "paper_name": paper_name,
                    "author": paper_author,
                    "year": paper_year,
                    "source_file": paper_source_file,
                    "section": label,
                    "text": part,
                    "char_count": len(part),
                    "page_start": sec["page_start"],
                    "page_end": sec["page_end"],
                })

                chunk_id += 1
                summary[label] = summary.get(label, 0) + 1

        print("Sections:", summary)

    output_path = config.output_folder / "chunks.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {output_path}")
    return output_path



# MAIN


if __name__ == "__main__":
    cfg = parse_args()
    print("CONFIG:", cfg)
    chunk_papers(cfg)
"""
chunking.py
Goal: Chunk parsed paper text (from combined JSON) and store chunks as JSON.

Arguments:
    --combined_json  : Path to combined_papers.json produced by parse_pdf.py
    --output_folder  : Folder to save chunks JSON
    --chunk_size     : Max characters per chunk (default 1000)
    --chunk_overlap  : Overlap between consecutive chunks (default 200)

Returns:
    - chunks.json in output_folder/
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, validator


# -------------------- SCHEMA --------------------

class ChunkConfig(BaseModel):
    combined_json: Path
    output_folder: Path
    chunk_size:    int = Field(default=1000, gt=100)
    chunk_overlap: int = Field(default=200,  ge=0)

    @validator("combined_json")
    def file_must_exist(cls, v):
        if not v.exists():
            raise ValueError(f"Combined JSON not found: {v}")
        return v


# -------------------- ARG PARSER --------------------

def parse_args() -> ChunkConfig:
    parser = argparse.ArgumentParser(description="Chunk combined paper JSON")
    parser.add_argument("--combined_json",  type=str, required=True)
    parser.add_argument("--output_folder",  type=str, required=True)
    parser.add_argument("--chunk_size",     type=int, default=1000)
    parser.add_argument("--chunk_overlap",  type=int, default=200)
    args = parser.parse_args()

    return ChunkConfig(
        combined_json=Path(args.combined_json),
        output_folder=Path(args.output_folder),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


# -------------------- SECTION DETECTION --------------------

# Simple heuristic: lines that are short, Title-Cased or ALL-CAPS → section headers
def _is_section_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    return stripped.istitle() or stripped.isupper()


def split_into_sections(full_text: str) -> List[Dict[str, str]]:
    """
    Try to split text into labelled sections using header detection.
    Falls back to a single section if no headers are detected.
    """
    lines = full_text.split("\n")
    sections: List[Dict[str, str]] = []
    current_header = "Introduction"
    current_lines: List[str] = []

    for line in lines:
        if _is_section_header(line):
            if current_lines:
                sections.append({
                    "section": current_header,
                    "text":    "\n".join(current_lines).strip()
                })
            current_header = line.strip()
            current_lines  = []
        else:
            current_lines.append(line)

    # Flush last section
    if current_lines:
        sections.append({
            "section": current_header,
            "text":    "\n".join(current_lines).strip()
        })

    return sections if sections else [{"section": "Full Text", "text": full_text}]


# -------------------- CHUNKING --------------------

def chunk_papers(config: ChunkConfig) -> Path:
    config.output_folder.mkdir(parents=True, exist_ok=True)

    with open(config.combined_json, "r", encoding="utf-8") as f:
        papers: List[Dict[str, Any]] = json.load(f)

    print(f"Loaded {len(papers)} paper(s) from {config.combined_json}\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for paper in papers:
        paper_name  = paper.get("paper_name", "Unknown")
        author      = paper.get("author",     "Unknown")
        year        = paper.get("year",       "Unknown")
        source_file = paper.get("source_file","Unknown")
        full_text   = paper.get("full_text",  "")

        print(f"  Paper : {paper_name}")

        # Split into sections first, then chunk each section
        sections = split_into_sections(full_text)
        paper_chunk_count = 0

        for sec in sections:
            section_label = sec["section"]
            section_text  = sec["text"]

            if not section_text.strip():
                continue

            raw_chunks = splitter.split_text(section_text)

            for raw in raw_chunks:
                all_chunks.append({
                    "chunk_id":    chunk_id,
                    "paper_name":  paper_name,
                    "author":      author,
                    "year":        year,
                    "source_file": source_file,
                    "section":     section_label,
                    "text":        raw,
                    "char_count":  len(raw),
                })
                chunk_id += 1
                paper_chunk_count += 1

        print(f"    → {paper_chunk_count} chunks\n")

    output_path = config.output_folder / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"✓ Total chunks : {len(all_chunks)}")
    print(f"✓ Saved        : {output_path}")
    return output_path


# -------------------- ENTRY --------------------

if __name__ == "__main__":
    config = parse_args()
    print("\nCONFIG:", config)
    chunk_papers(config)

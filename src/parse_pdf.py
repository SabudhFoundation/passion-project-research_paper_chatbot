"""
parse_pdf.py

Parse PDFs and save individual + combined JSON files.

Clean responsibility:
- open PDF
- extract metadata
- extract text pages
- call pdf_artifacts.py for tables / figures / diagrams
- save JSON output

No LLM calls.
No embeddings.
No Streamlit code.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pymupdf
from pydantic import BaseModel, Field, field_validator

from config import (
    FIGURES_ARTIFACT_PATH,
    TABLES_ARTIFACT_PATH,
)
from core.paths import safe_slug
from pdf_artifacts import (
    extract_figures_from_document,
    extract_tables_from_document,
)
from utilities import save_json


class ParseConfig(BaseModel):
    pdf_folder: Path
    num_pdfs: int = Field(default=0, ge=0)
    individual_out: Path
    combined_out: Path
    tables_artifact_path: Path = TABLES_ARTIFACT_PATH
    figures_artifact_path: Path = FIGURES_ARTIFACT_PATH

    @field_validator("pdf_folder")
    @classmethod
    def folder_must_exist(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"PDF folder does not exist: {value}")
        return value


def parse_args() -> ParseConfig:
    parser = argparse.ArgumentParser(description="Parse PDFs to JSON")
    parser.add_argument(
        "--pdf_folder",
        type=str,
        required=True,
        help="Path to PDF folder",
    )
    parser.add_argument(
        "--num_pdfs",
        type=int,
        default=0,
        help="Number of PDFs to process. Use 0 for all PDFs.",
    )
    parser.add_argument(
        "--individual_out",
        type=str,
        required=True,
        help="Folder to save individual JSON files",
    )
    parser.add_argument(
        "--combined_out",
        type=str,
        required=True,
        help="Folder to save combined JSON file",
    )

    args = parser.parse_args()

    return ParseConfig(
        pdf_folder=Path(args.pdf_folder),
        num_pdfs=args.num_pdfs,
        individual_out=Path(args.individual_out),
        combined_out=Path(args.combined_out),
    )


def extract_year(date_str: str, file_path: Path) -> str:
    """
    Extract year from PDF metadata date string.

    Examples:
    D:20210401120000
    2021-04-01
    """
    if not date_str:
        return "Unknown"

    match = re.search(r"D:(\d{4})", date_str)
    if match:
        return match.group(1)

    match = re.search(r"(\d{4})", date_str)
    if match:
        return match.group(1)

    try:
        return str(datetime.fromtimestamp(file_path.stat().st_ctime).year)
    except Exception:
        return "Unknown"


def extract_metadata_from_doc(doc: pymupdf.Document, file_path: Path) -> dict[str, Any]:
    """
    Extract available PDF metadata.
    """
    metadata = doc.metadata or {}
    creation_date = metadata.get("creationDate", "")

    return {
        "paper_name": metadata.get("title") or file_path.stem,
        "author": metadata.get("author") or "Unknown",
        "year": extract_year(creation_date, file_path),
        "source_file": file_path.name,
        "num_pages": doc.page_count,
        "subject": metadata.get("subject") or "",
        "keywords": metadata.get("keywords") or "",
    }


def extract_pages_text(doc: pymupdf.Document) -> list[dict[str, Any]]:
    """
    Extract text from every page.

    Output format is intentionally simple because chunking.py consumes this.
    """
    pages_text: list[dict[str, Any]] = []

    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_number = page_index + 1
        text = page.get_text("text").strip()

        if text:
            pages_text.append(
                {
                    "page_number": page_number,
                    "text": text,
                }
            )

    return pages_text


def extract_text_from_pdf(
    file_path: Path,
    *,
    tables_artifact_path: Path = TABLES_ARTIFACT_PATH,
    figures_artifact_path: Path = FIGURES_ARTIFACT_PATH,
) -> Optional[dict[str, Any]]:
    """
    Parse one PDF into the JSON schema used by the RAG pipeline.

    This function keeps the old name `extract_text_from_pdf` for backward
    compatibility, but it now also extracts multimodal artifacts.
    """
    try:
        with pymupdf.open(str(file_path)) as doc:
            metadata = extract_metadata_from_doc(doc, file_path)
            paper_slug = safe_slug(Path(metadata["source_file"]).stem)

            pages_text = extract_pages_text(doc)
            full_text = "\n\n".join(page["text"] for page in pages_text)

            tables = extract_tables_from_document(
                file_path=file_path,
                doc=doc,
                pages_text=pages_text,
                metadata=metadata,
                paper_slug=paper_slug,
                tables_artifact_path=tables_artifact_path,
            )

            figures = extract_figures_from_document(
                doc=doc,
                pages_text=pages_text,
                metadata=metadata,
                paper_slug=paper_slug,
                figures_artifact_path=figures_artifact_path,
            )

            return {
                **metadata,
                "parser_used": "pymupdf",
                "full_text": full_text,
                "pages": pages_text,
                "tables": tables,
                "figures": figures,
                "total_chars": len(full_text),
            }

    except Exception as exc:
        print(f"  [ERROR] Could not parse {file_path.name}: {exc}")
        return None


def parse_pdfs(config: ParseConfig) -> Path | None:
    """
    Parse PDFs from a folder and save:
    - individual JSON files
    - one combined_papers.json
    """
    config.individual_out.mkdir(parents=True, exist_ok=True)
    config.combined_out.mkdir(parents=True, exist_ok=True)

    config.tables_artifact_path.mkdir(parents=True, exist_ok=True)
    config.figures_artifact_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(config.pdf_folder.glob("**/*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return None

    if config.num_pdfs > 0:
        pdf_files = pdf_files[: config.num_pdfs]

    print(f"Processing {len(pdf_files)} PDF(s)...\n")

    combined: list[dict[str, Any]] = []
    saved_individual = 0

    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{index}/{len(pdf_files)}] {pdf_path.name}")

        paper_data = extract_text_from_pdf(
            pdf_path,
            tables_artifact_path=config.tables_artifact_path,
            figures_artifact_path=config.figures_artifact_path,
        )

        if paper_data is None:
            continue

        safe_name = safe_slug(pdf_path.stem)
        individual_path = config.individual_out / f"{safe_name}.json"

        save_json(paper_data, individual_path)

        table_count = len(paper_data.get("tables", []) or [])
        figure_count = len(paper_data.get("figures", []) or [])

        print(
            f"  Saved → {individual_path} "
            f"| pages={len(paper_data.get('pages', []) or [])} "
            f"| tables={table_count} "
            f"| figures={figure_count}"
        )

        saved_individual += 1
        combined.append(paper_data)

    combined_path = config.combined_out / "combined_papers.json"
    save_json(combined, combined_path)

    print()
    print(f"✓ Individual JSONs saved : {saved_individual} → {config.individual_out}")
    print(f"✓ Combined JSON saved    : {combined_path}")

    return combined_path


if __name__ == "__main__":
    config = parse_args()
    print("\nCONFIG:", config)
    parse_pdfs(config)
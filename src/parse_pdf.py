"""
parse_pdf.py
Goal: Parse PDFs and save individual + combined JSON files.

Arguments:
    --pdf_folder     : Path to folder containing PDFs
    --num_pdfs       : Number of PDFs to process (0 = all)
    --individual_out : Folder to save individual paper JSONs
    --combined_out   : Folder to save combined JSON

python src/parse_pdf.py --pdf_folder data/raw/papers --individual_out data/parsed_individual --combined_out data/parsed_combined --num_pdfs 0

Returns:
    - Individual JSON per paper in individual_out/
    - combined_papers.json in combined_out/
"""

import argparse
import re
from pathlib import Path
from typing import Optional
import pymupdf  # PyMuPDF
from pydantic import BaseModel, Field, validator
from datetime import datetime

from utilities import save_json


# -------------------- SCHEMA --------------------

class ParseConfig(BaseModel):
    pdf_folder: Path
    num_pdfs: int = Field(default=0, ge=0)
    individual_out: Path
    combined_out: Path

    @validator("pdf_folder")
    def folder_must_exist(cls, v):
        if not v.exists():
            raise ValueError(f"PDF folder does not exist: {v}")
        return v


# -------------------- ARG PARSER --------------------

def parse_args() -> ParseConfig:
    parser = argparse.ArgumentParser(description="Parse PDFs to JSON")
    parser.add_argument("--pdf_folder",     type=str, required=True, help="Path to PDF folder")
    parser.add_argument("--num_pdfs",       type=int, default=0,     help="Number of PDFs to process (0 = all)")
    parser.add_argument("--individual_out", type=str, required=True, help="Folder to save individual JSONs")
    parser.add_argument("--combined_out",   type=str, required=True, help="Folder to save combined JSON")
    args = parser.parse_args()

    return ParseConfig(
        pdf_folder=Path(args.pdf_folder),
        num_pdfs=args.num_pdfs,
        individual_out=Path(args.individual_out),
        combined_out=Path(args.combined_out),
    )


# -------------------- HELPERS --------------------

def extract_year(date_str: str, file_path: Path) -> str:
    """Extract year from PDF metadata date string."""
    if not date_str:
        return "Unknown"

    # Match formats like D:20230405120000
    match = re.search(r"D:(\d{4})", date_str)
    if match:
        return match.group(1)

    # Fallback: any 4-digit year
    match = re.search(r"(\d{4})", date_str)
    if match:
        return match.group(1)

    # Final fallback: file creation time
    try:

        return str(datetime.fromtimestamp(file_path.stat().st_ctime).year)
    except Exception:
        return "Unknown"


def extract_metadata_from_doc(doc: pymupdf.Document, file_path: Path) -> dict:
    """Extract available metadata from a PDF document."""
    meta = doc.metadata or {}
    creation_date = meta.get("creationDate", "")

    return {
        "paper_name": meta.get("title") or file_path.stem,
        "author":     meta.get("author") or "Unknown",
        "year":       extract_year(creation_date, file_path),
        "source_file": file_path.name,
        "num_pages":  doc.page_count,
        "subject":    meta.get("subject") or "",
        "keywords":   meta.get("keywords") or "",
    }


def extract_text_from_pdf(file_path: Path) -> Optional[dict]:
    """
    Open a PDF and extract full text + metadata.
    Returns a dict ready for JSON serialisation, or None on failure.
    """
    try:
        # Context manager ensures file is always closed
        with pymupdf.open(str(file_path)) as doc:
            pages_text = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text("text").strip()

                if text:
                    pages_text.append({
                        "page_number": page_num + 1,
                        "text": text
                    })

            full_text = "\n\n".join(p["text"] for p in pages_text)
            metadata  = extract_metadata_from_doc(doc, file_path)

        return {
            **metadata,
            "full_text":   full_text,
            "pages":       pages_text,
            "total_chars": len(full_text),
        }

    except Exception as e:
        print(f"  [ERROR] Could not parse {file_path.name}: {e}")
        return None


# -------------------- MAIN LOGIC --------------------

def parse_pdfs(config: ParseConfig):
    # Prepare output directories
    config.individual_out.mkdir(parents=True, exist_ok=True)
    config.combined_out.mkdir(parents=True, exist_ok=True)

    # Collect PDF files
    pdf_files = sorted(config.pdf_folder.glob("**/*.pdf"))
    if not pdf_files:
        print("No PDF files found.")
        return

    if config.num_pdfs > 0:
        pdf_files = pdf_files[: config.num_pdfs]

    print(f"Processing {len(pdf_files)} PDF(s)...\n")

    combined = []
    saved_individual = 0

    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{idx}/{len(pdf_files)}] {pdf_path.name}")
        paper_data = extract_text_from_pdf(pdf_path)

        if paper_data is None:
            continue

        # Save individual JSON
        safe_name = pdf_path.stem.replace(" ", "_")
        individual_path = config.individual_out / f"{safe_name}.json"

        save_json(paper_data, individual_path)

        print(f"  Saved → {individual_path}")
        saved_individual += 1
        combined.append(paper_data)

    # Save combined JSON
    combined_path = config.combined_out / "combined_papers.json"

    save_json(combined, combined_path)

    print(f"\n✓ Individual JSONs saved : {saved_individual}  →  {config.individual_out}")
    print(f"✓ Combined JSON saved    : {combined_path}")

    return combined_path


# -------------------- ENTRY --------------------

if __name__ == "__main__":
    config = parse_args()
    print("\nCONFIG:", config)
    parse_pdfs(config)
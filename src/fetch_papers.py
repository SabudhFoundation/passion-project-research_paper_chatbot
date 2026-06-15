'''
python src/fetch_papers.py --sources arxiv semantic_scholar --query_file data/external/queries.txt --folder_path data/raw --num_pdfs 20 --year 2023
'''

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import requests
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    import arxiv
except ImportError:
    arxiv = None


SOURCE_ORDER = ["arxiv", "semantic_scholar", "openalex", "core"]
VALID_SOURCES = set(SOURCE_ORDER)

REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_QUERIES = 0.5

CSV_COLUMNS = [
    "source",
    "title",
    "authors",
    "year",
    "category",
    "paper_id",
    "doi",
    "pdf_url",
    "pdf_path",
    "published_date",
    "abstract",
    "time_logged",
]


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data").exists() or (parent / "pyproject.toml").exists():
            return parent
    return here.parent


PROJECT_ROOT = find_project_root()

PAPERS_DIR = PROJECT_ROOT / "papers"
LOG_CSV = PROJECT_ROOT / "papers_log.csv"

logger = logging.getLogger("fetch_papers")


def setup_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Keep output focused on this script's logs, not third-party INFO chatter.
    logger.handlers.clear()
    logger.setLevel(log_level)
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)

    for noisy_logger in ("arxiv", "urllib3", "requests"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


@dataclass(slots=True)
class Paper:
    source: str
    title: str
    authors: list[str]
    year: int
    category: str
    paper_id: str
    doi: str = ""
    pdf_url: str = ""
    published_date: str = ""
    abstract: str = ""

    def authors_str(self) -> str:
        return ", ".join(a for a in self.authors if a)

    def to_csv_row(self, pdf_path: str) -> dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "authors": self.authors_str(),
            "year": self.year,
            "category": self.category,
            "paper_id": self.paper_id,
            "doi": self.doi,
            "pdf_url": self.pdf_url,
            "pdf_path": pdf_path,
            "published_date": self.published_date,
            "abstract": self.abstract,
            "time_logged": datetime.now(timezone.utc).isoformat(),
        }


class CLIArgs(BaseModel):
    sources: list[str]
    query_file: Path
    folder_path: Path
    num_pdfs: int = Field(..., ge=1)
    year: str
    year_start: int = 0
    year_end: int = 0
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("sources", mode="before")
    @classmethod
    def normalize_sources(cls, v: Any) -> list[str]:
        if not isinstance(v, list) or not v:
            raise ValueError("--sources must contain at least one source")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in v:
            s = str(item).strip().lower()
            if not s:
                continue
            if s not in seen:
                normalized.append(s)
                seen.add(s)

        if not normalized:
            raise ValueError("--sources must contain at least one valid source")

        if "all" in normalized:
            return SOURCE_ORDER.copy()

        return normalized

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: list[str]) -> list[str]:
        bad = set(v) - VALID_SOURCES
        if bad:
            raise ValueError(f"Unknown source(s): {sorted(bad)}. Valid: {SOURCE_ORDER} or all")
        return v

    @field_validator("query_file")
    @classmethod
    def validate_query_file(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Query file not found: {v}")
        if not v.is_file():
            raise ValueError(f"Query path is not a file: {v}")

        lines = [
            line.strip()
            for line in v.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not lines:
            raise ValueError(f"Query file is empty: {v}")

        return v

    @field_validator("folder_path")
    @classmethod
    def validate_folder_path(cls, v: Path) -> Path:
        if v.exists() and not v.is_dir():
            raise ValueError(f"Folder path is not a directory: {v}")
        return v

    @field_validator("year")
    @classmethod
    def validate_year_format(cls, v: str) -> str:
        raw = v.strip()
        if not re.fullmatch(r"\d{4}(-\d{4})?", raw):
            raise ValueError("--year must be YYYY or YYYY-YYYY")
        return raw

    @model_validator(mode="after")
    def parse_year_range(self) -> "CLIArgs":
        current_year = datetime.now(timezone.utc).year

        if "-" in self.year:
            start_str, end_str = self.year.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                raise ValueError(f"Year range start ({start}) must be <= end ({end})")
        else:
            start = end = int(self.year)

        if start < 1900 or end > current_year:
            raise ValueError(f"Year values must be between 1900 and {current_year}")

        self.year_start = start
        self.year_end = end
        return self


def parse_cli() -> CLIArgs:
    parser = argparse.ArgumentParser(
        prog="fetch_papers.py",
        description="Fetch research papers from multiple sources and store PDFs + CSV logs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        metavar="SOURCE",
        help="One or more of: arxiv  semantic_scholar  openalex  core  all",
    )
    parser.add_argument(
        "--query_file",
        required=True,
        metavar="PATH",
        help="Path to a text file with one search query per line.",
    )
    parser.add_argument(
        "--folder_path",
        required=True,
        metavar="PATH",
        help="Base output directory (for example: data/raw). Creates papers/ and papers_log.csv inside it.",
    )
    parser.add_argument(
        "--num_pdfs",
        required=True,
        type=int,
        metavar="N",
        help="Total number of PDFs to download.",
    )
    parser.add_argument(
        "--year",
        required=True,
        metavar="YYYY or YYYY-YYYY",
        help="Single year (2023) or range (2020-2024).",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    raw = parser.parse_args()

    try:
        return CLIArgs(
            sources=raw.sources,
            query_file=Path(raw.query_file),
            folder_path=Path(raw.folder_path),
            num_pdfs=raw.num_pdfs,
            year=raw.year,
            log_level=raw.log_level,
        )
    except Exception as exc:
        parser.error(str(exc))


def safe_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r'[\\/:*?"<>|]+', "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return (text[:max_len] or "untitled").strip("_") or "untitled"


def build_pdf_path(paper_id: str, category: str, year: int) -> Path:
    pid_part = safe_filename(paper_id, 60)
    category_part = safe_filename(category, 40)
    year_part = safe_filename(str(year), 10)
    return PAPERS_DIR / f"{pid_part}__{category_part}__{year_part}.pdf"


def load_queries(path: Path) -> list[str]:
    queries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            queries.append(line)
    if not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


def ensure_dirs() -> None:
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_csv() -> None:
    if not LOG_CSV.exists():
        with LOG_CSV.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def load_existing_fingerprints(csv_path: Path) -> set[str]:
    fingerprints: set[str] = set()
    if not csv_path.exists():
        return fingerprints

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for key in ("paper_id", "doi", "pdf_url", "pdf_path"):
                value = (row.get(key) or "").strip()
                if value:
                    fingerprints.add(value)
    return fingerprints


def add_fingerprints(paper: Paper, fingerprints: set[str], pdf_path: str) -> None:
    for value in (paper.paper_id, paper.doi, paper.pdf_url, pdf_path):
        if value:
            fingerprints.add(value)


def is_duplicate(paper: Paper, fingerprints: set[str], pdf_path: Path) -> bool:
    for value in (paper.paper_id, paper.doi, paper.pdf_url, str(pdf_path)):
        if value and value in fingerprints:
            return True
    return pdf_path.exists()


def append_row(paper: Paper, pdf_path: str) -> None:
    with LOG_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writerow(paper.to_csv_row(pdf_path))


def in_year_filter(year: Any, year_set: Optional[set[int]]) -> bool:
    if year_set is None:
        return True
    try:
        return int(year) in year_set
    except (TypeError, ValueError):
        return False


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _match_score(query: str, paper: Paper) -> float:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0

    # Weight title heavily and abstract lightly for relevance ranking.
    title_counter = Counter(_tokenize(paper.title))
    abstract_counter = Counter(_tokenize(paper.abstract))

    title_hits = sum(title_counter[t] for t in q_tokens)
    abstract_hits = sum(abstract_counter[t] for t in q_tokens)

    return (2.0 * title_hits) + (0.5 * abstract_hits)


def _download_file(url: str, dest: Path) -> bool:
    if not url:
        return False

    temp_path = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 ResearchFetcher/1.0"},
            allow_redirects=True,
            stream=True,
        ) as resp:
            resp.raise_for_status()

            content_type = (resp.headers.get("Content-Type") or "").lower()
            if content_type and "pdf" not in content_type and "octet-stream" not in content_type:
                logger.debug("Unexpected Content-Type '%s' for %s", content_type, url)
                return False
            with temp_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        fh.write(chunk)

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            temp_path.unlink(missing_ok=True)
            return False

        with temp_path.open("rb") as fh:
            header = fh.read(8)
            if not header.startswith(b"%PDF"):
                temp_path.unlink(missing_ok=True)
                return False

        temp_path.replace(dest)
        return True

    except Exception as exc:
        logger.debug("Download failed for %s: %s", url, exc)
        temp_path.unlink(missing_ok=True)
        return False


def download_pdf(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    return _download_file(url, dest)


def _unpaywall_url(doi: str) -> str:
    if not doi:
        return ""
    email = os.getenv("OPENALEX_EMAIL", "researchbot@example.com")
    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": email},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            best = resp.json().get("best_oa_location") or {}
            return best.get("url_for_pdf") or best.get("url") or ""
    except Exception as exc:
        logger.debug("Unpaywall lookup failed for %s: %s", doi, exc)
    return ""


def _arxiv_url(paper_id: str, doi: str) -> str:
    if paper_id.startswith("arxiv:"):
        arxiv_id = paper_id.replace("arxiv:", "").strip()
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    if doi and "arxiv" in doi.lower():
        match = re.search(r"arXiv[.:](.+)$", doi, flags=re.IGNORECASE)
        if match:
            arxiv_id = match.group(1).strip()
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return ""


def _semantic_scholar_url(doi: str) -> str:
    if not doi:
        return ""
    try:
        resp = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            params={"fields": "openAccessPdf"},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            return (resp.json().get("openAccessPdf") or {}).get("url", "") or ""
    except Exception as exc:
        logger.debug("Semantic Scholar fallback failed for %s: %s", doi, exc)
    return ""


def download_with_fallbacks(primary_url: str, paper_id: str, doi: str, dest: Path) -> tuple[bool, str]:
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []

    def add(method: str, url: str) -> None:
        if url and url not in seen:
            seen.add(url)
            candidates.append((method, url))

    add("primary", primary_url)
    add("arxiv", _arxiv_url(paper_id, doi))
    add("unpaywall", _unpaywall_url(doi))
    add("semantic_scholar", _semantic_scholar_url(doi))

    for method, url in candidates:
        logger.debug("Trying %s: %s", method, url)
        if download_pdf(url, dest):
            logger.info("Saved via %s -> %s", method, dest.name)
            return True, url

    return False, ""


def _reconstruct_openalex_abstract(inverted_index: Optional[dict]) -> str:
    if not inverted_index:
        return ""
    try:
        positions: list[tuple[int, str]] = []
        for word, indexes in inverted_index.items():
            for pos in indexes:
                positions.append((pos, word))
        positions.sort(key=lambda item: item[0])
        return " ".join(word for _, word in positions)
    except Exception:
        return ""


def fetch_arxiv(query: str, max_results: int, year_set: Optional[set[int]]) -> list[Paper]:
    papers: list[Paper] = []
    if arxiv is None:
        logger.warning("arxiv package not installed. Run: pip install arxiv")
        return papers

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )

        for result in client.results(search):
            year = result.published.year if result.published else None
            if not in_year_filter(year, year_set):
                continue

            raw_id = result.get_short_id() or ""
            paper_id = f"arxiv:{raw_id}"
            doi = result.doi or ""
            title = (result.title or "untitled").strip()
            pdf_url = result.pdf_url or f"https://arxiv.org/pdf/{raw_id.split('v')[0]}.pdf"

            papers.append(
                Paper(
                    source="arxiv",
                    title=title,
                    authors=[str(a) for a in (result.authors or [])],
                    year=int(year) if year is not None else datetime.now(timezone.utc).year,
                    category=query,
                    paper_id=paper_id,
                    doi=doi,
                    pdf_url=pdf_url,
                    published_date=result.published.strftime("%Y-%m-%d") if result.published else "",
                    abstract=(result.summary or "").replace("\n", " ").strip(),
                )
            )
    except Exception as exc:
        logger.error("[arXiv] Error fetching '%s': %s", query, exc)

    return papers


def fetch_semantic_scholar(
    query: str,
    max_results: int,
    year_set: Optional[set[int]],
) -> list[Paper]:
    papers: list[Paper] = []
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "paperId,title,authors,year,externalIds,openAccessPdf,publicationDate,abstract"

    try:
        resp = requests.get(
            url,
            params={"query": query, "fields": fields, "limit": min(max_results, 100)},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("[Semantic Scholar] API error for '%s': %s", query, exc)
        return papers

    for item in resp.json().get("data", []):
        year = item.get("year")
        if not in_year_filter(year, year_set):
            continue

        paper_id = item.get("paperId", "")
        title = (item.get("title") or "untitled").strip()
        doi = (item.get("externalIds") or {}).get("DOI", "") or ""
        pdf_url = (item.get("openAccessPdf") or {}).get("url", "") or ""
        authors = [a.get("name", "") for a in (item.get("authors") or []) if isinstance(a, dict)]

        papers.append(
            Paper(
                source="semantic_scholar",
                title=title,
                authors=authors,
                year=int(year) if year is not None else datetime.now(timezone.utc).year,
                category=query,
                paper_id=paper_id,
                doi=doi,
                pdf_url=pdf_url,
                published_date=item.get("publicationDate") or str(year or ""),
                abstract=(item.get("abstract") or "").replace("\n", " ").strip(),
            )
        )

    return papers


def fetch_openalex(
    query: str,
    max_results: int,
    year_set: Optional[set[int]],
) -> list[Paper]:
    papers: list[Paper] = []
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": min(max_results, 200),
        "select": (
            "id,title,authorships,publication_year,doi,open_access,"
            "best_oa_location,publication_date,abstract_inverted_index"
        ),
    }
    email = os.getenv("OPENALEX_EMAIL", "")
    if email:
        params["mailto"] = email

    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("[OpenAlex] API error for '%s': %s", query, exc)
        return papers

    for item in resp.json().get("results", []):
        year = item.get("publication_year")
        if not in_year_filter(year, year_set):
            continue

        raw_id = (item.get("id") or "").replace("https://openalex.org/", "")
        if not raw_id:
            continue

        doi = (item.get("doi") or "").replace("https://doi.org/", "")
        title = (item.get("title") or "untitled").strip()

        best_oa = item.get("best_oa_location") or {}
        pdf_url = best_oa.get("pdf_url") or ""
        if not pdf_url:
            pdf_url = (item.get("open_access") or {}).get("oa_url") or ""

        authors = [
            a.get("author", {}).get("display_name", "")
            for a in (item.get("authorships") or [])
            if isinstance(a, dict)
        ]

        papers.append(
            Paper(
                source="openalex",
                title=title,
                authors=authors,
                year=int(year) if year is not None else datetime.now(timezone.utc).year,
                category=query,
                paper_id=f"openalex:{raw_id}",
                doi=doi,
                pdf_url=pdf_url,
                published_date=item.get("publication_date") or str(year or ""),
                abstract=_reconstruct_openalex_abstract(item.get("abstract_inverted_index")),
            )
        )

    return papers


def fetch_core(
    query: str,
    max_results: int,
    year_set: Optional[set[int]],
) -> list[Paper]:
    papers: list[Paper] = []
    api_key = os.getenv("CORE_API_KEY", "")
    if not api_key:
        logger.warning("[CORE] CORE_API_KEY is not set. Skipping CORE.")
        return papers

    url = "https://api.core.ac.uk/v3/search/works"
    payload = {"q": query, "limit": min(max_results, 100)}
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("[CORE] API error for '%s': %s", query, exc)
        return papers

    for item in resp.json().get("results", []):
        year = item.get("yearPublished")
        if not in_year_filter(year, year_set):
            continue

        raw_id = str(item.get("id", "")).strip()
        if not raw_id:
            continue

        title = (item.get("title") or "untitled").strip()
        doi = item.get("doi") or ""
        pdf_url = item.get("downloadUrl") or ""
        pub_date = item.get("publishedDate") or (str(year) if year else "")
        authors_raw = item.get("authors") or []
        if authors_raw and isinstance(authors_raw[0], dict):
            authors = [a.get("name", "") for a in authors_raw]
        else:
            authors = [str(a) for a in authors_raw]

        papers.append(
            Paper(
                source="core",
                title=title,
                authors=authors,
                year=int(year) if year is not None else datetime.now(timezone.utc).year,
                category=query,
                paper_id=f"core:{raw_id}",
                doi=doi,
                pdf_url=pdf_url,
                published_date=pub_date,
                abstract=(item.get("abstract") or "").replace("\n", " ").strip(),
            )
        )

    return papers


SOURCE_FETCHERS = {
    "arxiv": fetch_arxiv,
    "semantic_scholar": fetch_semantic_scholar,
    "openalex": fetch_openalex,
    "core": fetch_core,
}


def fetch_from_source(
    source: str,
    query: str,
    year_set: Optional[set[int]],
    max_results: int,
) -> list[Paper]:
    fetcher = SOURCE_FETCHERS.get(source)
    if fetcher is None:
        logger.warning("Unknown source '%s'; skipping.", source)
        return []
    return fetcher(query, max_results, year_set)


def get_next_valid_paper(
    source: str,
    query: str,
    year_set: set[int],
    fingerprints: set[str],
) -> Optional[Paper]:
    papers = fetch_from_source(
        source=source,
        query=query,
        year_set=year_set,
        max_results=30,
    )

    ranked_candidates: list[tuple[float, Paper, Path]] = []
    for paper in papers:
        pdf_path = build_pdf_path(paper.paper_id, paper.category, paper.year)

        if is_duplicate(paper, fingerprints, pdf_path):
            continue

        if not paper.pdf_url:
            continue

        ranked_candidates.append((_match_score(query, paper), paper, pdf_path))

    ranked_candidates.sort(key=lambda item: item[0], reverse=True)

    for _, paper, pdf_path in ranked_candidates:

        saved = download_pdf(paper.pdf_url, pdf_path)
        if not saved:
            ok, used_url = download_with_fallbacks(paper.pdf_url, paper.paper_id, paper.doi, pdf_path)
            if not ok:
                continue
            paper.pdf_url = used_url

        return paper

    return None


def run(args: CLIArgs) -> None:
    global BASE_OUTPUT_DIR, PAPERS_DIR, LOG_CSV

    BASE_OUTPUT_DIR = args.folder_path if args.folder_path.is_absolute() else (PROJECT_ROOT / args.folder_path)
    PAPERS_DIR = BASE_OUTPUT_DIR / "papers"
    LOG_CSV = BASE_OUTPUT_DIR / "papers_log.csv"

    ensure_dirs()
    ensure_csv()

    queries = load_queries(args.query_file)
    year_set = set(range(args.year_start, args.year_end + 1))

    fingerprints = load_existing_fingerprints(LOG_CSV)

    logger.info(
        "Start fetch | sources=%s | queries=%d | target=%d | year=%s",
        ",".join(args.sources),
        len(queries),
        args.num_pdfs,
        f"{args.year_start}-{args.year_end}" if args.year_start != args.year_end else args.year_start,
    )

    downloaded_count = 0
    skipped_count = 0

    round_index = 0
    while downloaded_count < args.num_pdfs:
        progress_made = False
        round_index += 1
        logger.debug("=== ROUND %d ===", round_index)

        for query in queries:
            if downloaded_count >= args.num_pdfs:
                break

            logger.debug("--- QUERY: %s ---", query)

            for source in args.sources:
                if downloaded_count >= args.num_pdfs:
                    break

                paper = get_next_valid_paper(
                    source=source,
                    query=query,
                    year_set=year_set,
                    fingerprints=fingerprints,
                )
                if paper is None:
                    skipped_count += 1
                    continue

                pdf_path = build_pdf_path(paper.paper_id, paper.category, paper.year)
                append_row(paper, str(pdf_path))
                add_fingerprints(paper, fingerprints, str(pdf_path))
                downloaded_count += 1
                progress_made = True

                logger.info("[%d/%d] %s (%s)", downloaded_count, args.num_pdfs, paper.title[:90], source)
                time.sleep(SLEEP_BETWEEN_QUERIES)

        if not progress_made:
            break

    logger.info(
        "Done. Downloaded=%d, skipped=%d, csv=%s, pdf_dir=%s",
        downloaded_count,
        skipped_count,
        LOG_CSV,
        PAPERS_DIR,
    )


def main() -> None:
    args = parse_cli()
    setup_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
"""
uploaded_pdf_pipeline.py

Clean uploaded-PDF RAG pipeline for Streamlit.

Purpose:
- Save one or more uploaded PDFs.
- Reuse existing parse_pdf.py, chunking.py, embedding.py.
- Build a separate vector store for uploaded PDFs.
- Answer questions only from uploaded PDFs.
- Use the same intent classification style as Knowledge Base Chat.
- Support separate uploaded-PDF chat memory.
- Save uploaded-PDF answers and metrics to uploaded-PDF-specific paths.
- Support paper-balanced retrieval for multi-PDF uploads.
- Reset uploaded-PDF processing data safely on Windows.

Important:
All uploaded-PDF paths come from config.py.
Shared RAG logic comes from rag_common.py.
"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from config import (
    EMBED_MODEL,
    TEMPERATURE,
    UPLOADS_PATH,
    UPLOAD_CHUNK_OVERLAP,
    UPLOAD_CHUNK_SIZE,
    UPLOAD_BASE_PATH,
    UPLOAD_PARSED_INDIVIDUAL_PATH,
    UPLOAD_PARSED_COMBINED_PATH,
    UPLOAD_CHUNKS_PATH,
    UPLOAD_VECTOR_STORE_PATH,
    UPLOAD_OUTPUT_JSON_PATH,
    UPLOAD_METRICS_JSON_PATH,
    UPLOAD_COMBINED_JSON_PATH,
    UPLOAD_CHUNKS_JSON_PATH,
    UPLOAD_VECTOR_MARKER_PATH,
    UPLOAD_INDEX_MANIFEST_PATH,
    UPLOAD_TABLES_ARTIFACT_PATH,
    UPLOAD_FIGURES_ARTIFACT_PATH,
)
from core.paths import safe_slug
from parse_pdf import ParseConfig, parse_pdfs
from chunking import ChunkConfig, chunk_papers
from embedding import EmbedConfig, run_embedding
from llm import build_llm, normalize_provider_name
from metrics import (
    Timer,
    save_metrics,
)
from rag_common import (
    build_empty_generation_metrics,
    build_overall_from_stage_metrics,
    classify_route,
    decide_content_types,
    format_chunk_for_context,
    generate_non_rag_answer_with_metrics,
    generate_rag_answer_with_metrics,
    serialize_retrieved_chunk,
)
from retriever import retrieve
from utilities import save_result


# =========================================================
# Data model
# =========================================================

@dataclass
class UploadedPdfIndex:
    upload_id: str
    file_names: list[str]
    workspace_dir: Path
    raw_dir: Path
    parsed_individual_dir: Path
    parsed_combined_dir: Path
    combined_json_path: Path
    chunks_dir: Path
    chunks_json_path: Path
    vector_store_path: Path
    results_dir: Path
    pdf_count: int
    chunk_count: int


# =========================================================
# Upload file helpers
# =========================================================

def normalize_uploaded_files(uploaded_files: Any) -> list[Any]:
    """
    Streamlit can return:
    - None
    - one UploadedFile
    - list[UploadedFile]

    This normalizes it to a list.
    """
    if uploaded_files is None:
        return []

    if isinstance(uploaded_files, list):
        return uploaded_files

    return [uploaded_files]


def uploaded_file_bytes(uploaded_file: Any) -> bytes:
    """
    Read Streamlit UploadedFile bytes safely.
    """
    return bytes(uploaded_file.getbuffer())


def uploaded_files_signature(uploaded_files: Sequence[Any]) -> str:
    """
    Stable hash for the uploaded file set.

    Used so Streamlit does not rebuild the uploaded-PDF index on every rerun.
    """
    hasher = hashlib.sha256()

    for uploaded_file in uploaded_files:
        name = getattr(uploaded_file, "name", "uploaded.pdf")
        data = uploaded_file_bytes(uploaded_file)

        hasher.update(name.encode("utf-8", errors="ignore"))
        hasher.update(str(len(data)).encode("utf-8"))
        hasher.update(data)

    return hasher.hexdigest()


def upload_id_from_signature(signature: str) -> str:
    return f"upload_{signature[:16]}"


def workspace_paths(upload_id: str) -> dict[str, Path]:
    """
    Return all uploaded-PDF paths.

    Raw uploaded PDFs are stored per upload_id.
    Parsed/chunks/vector/results are stored in the central uploaded_pdf folder
    defined in config.py.

    The uploaded vector store intentionally uses one fixed folder:
    data/uploaded_pdf/vector_store

    On Process/Rebuild, that folder is reset and rebuilt fresh.
    """
    raw_dir = UPLOADS_PATH / upload_id

    return {
        "workspace_dir": UPLOAD_BASE_PATH,
        "raw_dir": raw_dir,
        "parsed_individual_dir": UPLOAD_PARSED_INDIVIDUAL_PATH,
        "parsed_combined_dir": UPLOAD_PARSED_COMBINED_PATH,
        "chunks_dir": UPLOAD_CHUNKS_PATH,
        "vector_store_path": UPLOAD_VECTOR_STORE_PATH,
        "results_dir": UPLOAD_OUTPUT_JSON_PATH.parent,
    }


def expected_saved_pdf_names(uploaded_files: Sequence[Any]) -> list[str]:
    """
    Compute the exact normalized file names that save_uploaded_pdfs() will write.
    """
    names: list[str] = []

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        original_name = getattr(uploaded_file, "name", f"uploaded_{index}.pdf")
        stem = safe_slug(Path(original_name).stem)
        names.append(f"{index:02d}_{stem}.pdf")

    return names


def save_uploaded_pdfs(
    uploaded_files: Sequence[Any],
    raw_dir: Path,
) -> list[str]:
    """
    Save uploaded PDFs into the upload-specific raw directory.

    File names are normalized to avoid unsafe characters and duplicate names.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved_names: list[str] = []

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        original_name = getattr(uploaded_file, "name", f"uploaded_{index}.pdf")
        stem = safe_slug(Path(original_name).stem)
        file_name = f"{index:02d}_{stem}.pdf"

        output_path = raw_dir / file_name
        output_path.write_bytes(uploaded_file_bytes(uploaded_file))

        saved_names.append(file_name)

    return saved_names


# =========================================================
# Count / readiness helpers
# =========================================================

def count_chunks(chunks_json_path: Path) -> int:
    if not chunks_json_path.exists():
        return 0

    try:
        with chunks_json_path.open("r", encoding="utf-8") as file:
            chunks = json.load(file)

        return len(chunks) if isinstance(chunks, list) else 0

    except Exception:
        return 0


def count_papers(combined_json_path: Path) -> int:
    if not combined_json_path.exists():
        return 0

    try:
        with combined_json_path.open("r", encoding="utf-8") as file:
            papers = json.load(file)

        return len(papers) if isinstance(papers, list) else 0

    except Exception:
        return 0


def load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:
        return None


def write_empty_json_array(path: Path) -> None:
    """
    Create or reset a JSON file to an empty list.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump([], file, ensure_ascii=False, indent=2)


def safe_unlink(path: Path) -> None:
    """
    Delete a file if it exists.
    Ignore missing files.
    """
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        pass


# =========================================================
# Chroma / Windows cleanup helpers
# =========================================================

def release_chroma_resources() -> None:

    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    gc.collect()
    time.sleep(0.5)


def safe_reset_dir(path: Path) -> None:
    
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return

    last_error: Exception | None = None

    for _attempt in range(6):
        release_chroma_resources()

        try:
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            return

        except OSError as exc:
            last_error = exc
            time.sleep(0.8)

    raise RuntimeError(
        f"Could not reset directory because it is locked: {path}. "
        "Stop Streamlit/Python, delete the locked folder manually, "
        "then rebuild the uploaded PDF index."
    ) from last_error


# =========================================================
# Manifest helpers
# =========================================================

def save_index_manifest(
    *,
    upload_id: str,
    signature: str,
    saved_file_names: list[str],
    combined_json_path: Path,
    chunks_json_path: Path,
    vector_store_path: Path,
    pdf_count: int,
    chunk_count: int,
) -> None:
    """
    Save a small manifest so fixed uploaded-PDF paths are not accidentally reused
    for the wrong uploaded file set.
    """
    UPLOAD_INDEX_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "upload_id": upload_id,
        "signature": signature,
        "file_names": saved_file_names,
        "combined_json_path": str(combined_json_path),
        "chunks_json_path": str(chunks_json_path),
        "vector_store_path": str(vector_store_path),
        "pdf_count": pdf_count,
        "chunk_count": chunk_count,
        "created_at": time.time(),
    }

    with UPLOAD_INDEX_MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    UPLOAD_VECTOR_MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_VECTOR_MARKER_PATH.write_text(
        json.dumps(
            {
                "status": "ready",
                "upload_id": upload_id,
                "signature": signature,
                "created_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def manifest_matches_upload(
    *,
    upload_id: str,
    signature: str,
    expected_file_names: list[str],
) -> bool:
    manifest = load_json_if_exists(UPLOAD_INDEX_MANIFEST_PATH)

    if not isinstance(manifest, dict):
        return False

    if manifest.get("upload_id") != upload_id:
        return False

    if manifest.get("signature") != signature:
        return False

    if manifest.get("file_names") != expected_file_names:
        return False

    return True


def existing_index_if_ready(
    *,
    upload_id: str,
    signature: str,
    file_names: list[str],
) -> UploadedPdfIndex | None:
    """
    Reuse existing uploaded PDF index only if:
    - raw upload folder exists
    - combined JSON exists
    - chunks JSON exists
    - Chroma marker exists
    - manifest matches the current upload signature

    This prevents reusing the wrong fixed uploaded-PDF vector store.
    """
    paths = workspace_paths(upload_id)

    combined_json_path = UPLOAD_COMBINED_JSON_PATH
    chunks_json_path = UPLOAD_CHUNKS_JSON_PATH
    vector_marker = UPLOAD_VECTOR_MARKER_PATH

    if not paths["raw_dir"].exists():
        return None

    if not combined_json_path.exists():
        return None

    if not chunks_json_path.exists():
        return None

    if not vector_marker.exists():
        return None

    if not manifest_matches_upload(
        upload_id=upload_id,
        signature=signature,
        expected_file_names=file_names,
    ):
        return None

    if not paths["vector_store_path"].exists():
        return None

    return UploadedPdfIndex(
        upload_id=upload_id,
        file_names=file_names,
        workspace_dir=paths["workspace_dir"],
        raw_dir=paths["raw_dir"],
        parsed_individual_dir=paths["parsed_individual_dir"],
        parsed_combined_dir=paths["parsed_combined_dir"],
        combined_json_path=combined_json_path,
        chunks_dir=paths["chunks_dir"],
        chunks_json_path=chunks_json_path,
        vector_store_path=paths["vector_store_path"],
        results_dir=paths["results_dir"],
        pdf_count=count_papers(combined_json_path),
        chunk_count=count_chunks(chunks_json_path),
    )


# =========================================================
# Cleanup helpers
# =========================================================

def reset_uploaded_processing_dirs(paths: dict[str, Path]) -> None:
    """
    Clear processing directories for the current uploaded-PDF index.

    Important:
    Do NOT physically delete the Chroma vector_store folder here.
    On Windows, ChromaDB can keep SQLite/HNSW files locked while Streamlit
    is running, which can corrupt the DB.

    The old Chroma collection is reset safely by run_embedding(reset=True).
    """
    dirs_to_reset = [
        paths["raw_dir"],
        paths["parsed_individual_dir"],
        paths["parsed_combined_dir"],
        paths["chunks_dir"],
        UPLOAD_TABLES_ARTIFACT_PATH,
        UPLOAD_FIGURES_ARTIFACT_PATH,
    ]

    for path in dirs_to_reset:
        safe_reset_dir(path)

    paths["workspace_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    paths["vector_store_path"].mkdir(parents=True, exist_ok=True)

    safe_unlink(UPLOAD_INDEX_MANIFEST_PATH)
    safe_unlink(UPLOAD_VECTOR_MARKER_PATH)

    write_empty_json_array(UPLOAD_OUTPUT_JSON_PATH)
    write_empty_json_array(UPLOAD_METRICS_JSON_PATH)


def reset_uploaded_pdf_workspace(*, clear_results: bool = True) -> None:
    """
    Clear uploaded-PDF workspace data.

    Intended only for optional/manual cleanup. Do not call this automatically
    on Streamlit startup.

    Clears:
    - uploaded raw PDFs
    - parsed individual JSON
    - parsed combined JSON
    - chunks
    - uploaded table artifacts
    - uploaded figure artifacts
    - manifest / vector marker

    Does NOT clear uploaded vector store here because ChromaDB files can be
    locked on Windows during Streamlit startup.

    If clear_results=True:
    - output.json becomes []
    - metrics.json becomes []
    """
    dirs_to_reset = [
        UPLOADS_PATH,
        UPLOAD_PARSED_INDIVIDUAL_PATH,
        UPLOAD_PARSED_COMBINED_PATH,
        UPLOAD_CHUNKS_PATH,
        UPLOAD_TABLES_ARTIFACT_PATH,
        UPLOAD_FIGURES_ARTIFACT_PATH,
    ]

    for path in dirs_to_reset:
        safe_reset_dir(path)

    UPLOAD_BASE_PATH.mkdir(parents=True, exist_ok=True)
    UPLOAD_OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    safe_unlink(UPLOAD_INDEX_MANIFEST_PATH)
    safe_unlink(UPLOAD_VECTOR_MARKER_PATH)

    if clear_results:
        write_empty_json_array(UPLOAD_OUTPUT_JSON_PATH)
        write_empty_json_array(UPLOAD_METRICS_JSON_PATH)


# =========================================================
# Optional vision preprocessing
# =========================================================

def maybe_run_upload_vision(
    *,
    combined_json_path: Path,
    generate_vision: bool,
    provider: str,
    model_name: str,
) -> None:
    """
    Optional uploaded-PDF vision preprocessing.

    This can be slow and may hit rate limits, so the Streamlit UI keeps it optional.
    """
    if not generate_vision:
        return

    from vision_describe import VisionConfig, run_vision_describe

    config = VisionConfig(
        combined_json=combined_json_path,
        output_json=combined_json_path,
        provider=provider,
        model_name=model_name,
        artifact_types="all",
        retries=1,
        retry_delay=8.0,
        dry_run=False,
        force=False,
    )

    run_vision_describe(config)


# =========================================================
# Build uploaded PDF index
# =========================================================

def build_uploaded_pdf_index(
    uploaded_files: Sequence[Any],
    *,
    chunk_size: int = UPLOAD_CHUNK_SIZE,
    chunk_overlap: int = UPLOAD_CHUNK_OVERLAP,
    embed_model: str = EMBED_MODEL,
    force_rebuild: bool = False,
    generate_vision: bool = False,
    vision_provider: str | None = None,
    vision_model_name: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> UploadedPdfIndex:
    """
    Build or reuse a vector store for uploaded PDFs.

    Pipeline:
    uploaded PDFs
    -> reset uploaded processing folders
    -> save uploaded PDFs
    -> parse_pdfs()
    -> optional vision_describe()
    -> chunk_papers()
    -> run_embedding()
    """
    files = normalize_uploaded_files(uploaded_files)
    emit_progress(progress_callback, "Checking uploaded PDF files...")

    if not files:
        raise ValueError("No uploaded PDF files were provided.")

    signature = uploaded_files_signature(files)
    upload_id = upload_id_from_signature(signature)
    paths = workspace_paths(upload_id)

    saved_file_names_expected = expected_saved_pdf_names(files)

    if not force_rebuild:
        existing = existing_index_if_ready(
            upload_id=upload_id,
            signature=signature,
            file_names=saved_file_names_expected,
        )

        if existing is not None:
            return existing

    emit_progress(progress_callback, "Preparing uploaded PDF workspace...")
    reset_uploaded_processing_dirs(paths)

    emit_progress(progress_callback, "Saving uploaded PDF files...")
    saved_file_names = save_uploaded_pdfs(
        files,
        paths["raw_dir"],
    )

    combined_json_path = UPLOAD_COMBINED_JSON_PATH
    chunks_json_path = UPLOAD_CHUNKS_JSON_PATH

    parse_config = ParseConfig(
        pdf_folder=paths["raw_dir"],
        num_pdfs=0,
        individual_out=paths["parsed_individual_dir"],
        combined_out=paths["parsed_combined_dir"],
        tables_artifact_path=UPLOAD_TABLES_ARTIFACT_PATH,
        figures_artifact_path=UPLOAD_FIGURES_ARTIFACT_PATH,
    )

    emit_progress(progress_callback, "Extracting text, tables, and figures...")
    parse_pdfs(parse_config)

    if not combined_json_path.exists():
        raise RuntimeError("PDF parsing failed: combined_papers.json was not created.")

    if generate_vision and (not vision_provider or not vision_model_name):
        raise ValueError(
            "Vision preprocessing is enabled, but no vision-capable model was selected."
        )

    if generate_vision:
        emit_progress(progress_callback, "Generating descriptions for tables and figures...")
    else:
        emit_progress(progress_callback, "Skipping vision descriptions...")
    maybe_run_upload_vision(
        combined_json_path=combined_json_path,
        generate_vision=generate_vision,
        provider=vision_provider or "",
        model_name=vision_model_name or "",
    )

    chunk_config = ChunkConfig(
        combined_json=combined_json_path,
        output_folder=paths["chunks_dir"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    emit_progress(progress_callback, "Creating searchable chunks...")
    chunk_papers(chunk_config)

    if not chunks_json_path.exists():
        raise RuntimeError("Chunking failed: chunks.json was not created.")

    chunk_count = count_chunks(chunks_json_path)

    if chunk_count == 0:
        raise RuntimeError("No chunks were created from the uploaded PDF(s).")

    release_chroma_resources()
    paths["vector_store_path"].mkdir(parents=True, exist_ok=True)

    embed_config = EmbedConfig(
        chunk_file=chunks_json_path,
        vector_folder=paths["vector_store_path"],
        model_name=embed_model,
        reset=True,
    )

    emit_progress(progress_callback, "Building uploaded PDF vector index...")
    run_embedding(embed_config)

    pdf_count = count_papers(combined_json_path)

    save_index_manifest(
        upload_id=upload_id,
        signature=signature,
        saved_file_names=saved_file_names,
        combined_json_path=combined_json_path,
        chunks_json_path=chunks_json_path,
        vector_store_path=paths["vector_store_path"],
        pdf_count=pdf_count,
        chunk_count=chunk_count,
    )

    emit_progress(progress_callback, "Uploaded PDF index is ready.")
    return UploadedPdfIndex(
        upload_id=upload_id,
        file_names=saved_file_names,
        workspace_dir=paths["workspace_dir"],
        raw_dir=paths["raw_dir"],
        parsed_individual_dir=paths["parsed_individual_dir"],
        parsed_combined_dir=paths["parsed_combined_dir"],
        combined_json_path=combined_json_path,
        chunks_dir=paths["chunks_dir"],
        chunks_json_path=chunks_json_path,
        vector_store_path=paths["vector_store_path"],
        results_dir=paths["results_dir"],
        pdf_count=pdf_count,
        chunk_count=chunk_count,
    )


# =========================================================
# Paper-balanced retrieval helpers
# =========================================================

def load_uploaded_chunks(chunks_json_path: Path) -> list[dict[str, Any]]:
    """
    Load uploaded-PDF chunks from chunks.json.

    Used for paper-balanced retrieval so multi-PDF questions get context
    from every uploaded paper, not only the most semantically similar one.
    """
    if not chunks_json_path.exists():
        return []

    try:
        with chunks_json_path.open("r", encoding="utf-8") as file:
            chunks = json.load(file)

        return chunks if isinstance(chunks, list) else []

    except Exception:
        return []


def chunk_paper_key(chunk: dict[str, Any]) -> str:
    """
    Return a stable paper key from a chunk.

    Different pipeline stages may store paper identity in slightly different
    places, so this checks multiple fields.
    """
    metadata = chunk.get("metadata", {}) or {}

    return (
        str(metadata.get("source_file") or "").strip()
        or str(metadata.get("paper_name") or "").strip()
        or str(chunk.get("source_file") or "").strip()
        or str(chunk.get("paper_name") or "").strip()
        or "unknown"
    )


def group_chunks_by_paper(
    chunks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group uploaded chunks by paper/source file.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}

    for chunk in chunks:
        paper_key = chunk_paper_key(chunk)
        grouped.setdefault(paper_key, []).append(chunk)

    return grouped


def chunk_identity(chunk: dict[str, Any]) -> str:
    """
    Build a stable identity for deduplicating chunks.

    Prefer explicit IDs if present. Fall back to paper/page/content.
    """
    metadata = chunk.get("metadata", {}) or {}

    explicit_id = (
        metadata.get("chunk_id")
        or metadata.get("id")
        or chunk.get("chunk_id")
        or chunk.get("id")
    )

    if explicit_id:
        return str(explicit_id)

    paper_key = chunk_paper_key(chunk)
    page = (
        metadata.get("page_number")
        or metadata.get("page_start")
        or chunk.get("page_number")
        or chunk.get("page_start")
        or ""
    )
    content = str(
        chunk.get("content")
        or chunk.get("text")
        or chunk.get("embedding_text")
        or ""
    )

    return f"{paper_key}|{page}|{content[:300]}"


def normalize_chunk_for_context(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Convert raw chunks.json items into the same shape returned by retriever.retrieve().
    """
    metadata = dict(chunk.get("metadata", {}) or {})

    for key in [
        "paper_name",
        "source_file",
        "section",
        "page_number",
        "page_start",
        "page_end",
        "content_type",
        "caption",
        "image_path",
        "csv_path",
        "html_path",
        "json_path",
        "markdown_path",
        "table_id",
        "figure_id",
        "structured_status",
        "display_preference",
        "vision_status",
        "vision_model",
        "vision_description",
        "mermaid",
        "mermaid_status",
    ]:
        if key in chunk and key not in metadata:
            metadata[key] = chunk.get(key)

    content = (
        chunk.get("content")
        or chunk.get("text")
        or chunk.get("embedding_text")
        or ""
    )

    content_type = (
        chunk.get("content_type")
        or metadata.get("content_type")
        or "text"
    )

    return {
        "content": content,
        "metadata": metadata,
        "content_type": content_type,
        "similarity": chunk.get("similarity", 0.0),
    }


def merge_deduplicate_chunks(
    primary_chunks: list[dict[str, Any]],
    extra_chunks: list[dict[str, Any]],
    *,
    max_chunks: int,
) -> list[dict[str, Any]]:
    """
    Merge two chunk lists and remove duplicates.

    The first list gets priority.
    """
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for chunk in [*primary_chunks, *extra_chunks]:
        normalized = normalize_chunk_for_context(chunk)
        identity = chunk_identity(normalized)

        if identity in seen:
            continue

        seen.add(identity)
        merged.append(normalized)

        if len(merged) >= max_chunks:
            break

    return merged


def select_balanced_paper_chunks(
    *,
    chunks_json_path: Path,
    chunks_per_paper: int = 4,
) -> list[dict[str, Any]]:
    """
    Select a small number of chunks from each uploaded paper.

    This guarantees multi-PDF uploaded chat has at least some context
    from every uploaded PDF.
    """
    raw_chunks = load_uploaded_chunks(chunks_json_path)
    grouped = group_chunks_by_paper(raw_chunks)

    balanced_chunks: list[dict[str, Any]] = []

    for _paper_key, paper_chunks in grouped.items():
        selected_count = 0
        seen_for_this_paper: set[str] = set()

        # Prefer text chunks because they usually contain title/abstract/method.
        for chunk in paper_chunks:
            normalized = normalize_chunk_for_context(chunk)
            content_type = normalized.get("content_type", "text")

            if content_type != "text":
                continue

            identity = chunk_identity(normalized)
            if identity in seen_for_this_paper:
                continue

            balanced_chunks.append(normalized)
            seen_for_this_paper.add(identity)
            selected_count += 1

            if selected_count >= chunks_per_paper:
                break

        # If a paper has fewer text chunks, fill with any remaining chunks.
        if selected_count < chunks_per_paper:
            for chunk in paper_chunks:
                normalized = normalize_chunk_for_context(chunk)
                identity = chunk_identity(normalized)

                if identity in seen_for_this_paper:
                    continue

                balanced_chunks.append(normalized)
                seen_for_this_paper.add(identity)
                selected_count += 1

                if selected_count >= chunks_per_paper:
                    break

    return balanced_chunks


def retrieve_uploaded_pdf_chunks(
    *,
    question: str,
    index: UploadedPdfIndex,
    top_k: int,
    embed_model: str,
    content_types: list[str],
) -> list[dict[str, Any]]:
    """
    Retrieve uploaded-PDF chunks.

    Single PDF:
        normal semantic retrieval.

    Multiple PDFs:
        semantic retrieval + guaranteed paper-balanced seed chunks.
        Balanced chunks are placed first so they cannot be dropped.
    """
    semantic_chunks = retrieve(
        question=question,
        top_k=top_k,
        vector_store_folder_path=str(index.vector_store_path),
        embed_model=embed_model,
        content_types=content_types,
    )

    if index.pdf_count <= 1:
        return semantic_chunks

    balanced_chunks = select_balanced_paper_chunks(
        chunks_json_path=index.chunks_json_path,
        chunks_per_paper=4,
    )

    if not balanced_chunks:
        return semantic_chunks

    return merge_deduplicate_chunks(
        primary_chunks=balanced_chunks,
        extra_chunks=semantic_chunks,
        max_chunks=top_k,
    )


# =========================================================
# Uploaded PDF question answering + saving
# =========================================================

def uploaded_pdf_index_payload(index: UploadedPdfIndex) -> dict[str, Any]:
    return {
        "upload_id": index.upload_id,
        "pdf_count": index.pdf_count,
        "chunk_count": index.chunk_count,
        "files": index.file_names,
        "workspace_dir": str(index.workspace_dir),
        "raw_dir": str(index.raw_dir),
        "combined_json_path": str(index.combined_json_path),
        "chunks_json_path": str(index.chunks_json_path),
        "vector_store_path": str(index.vector_store_path),
    }


def save_uploaded_pdf_result_if_requested(
    *,
    result: dict[str, Any],
    output_json_path: Path | None = UPLOAD_OUTPUT_JSON_PATH,
    metrics_json_path: Path | None = UPLOAD_METRICS_JSON_PATH,
) -> None:
    """
    Save uploaded-PDF result and metrics.

    Defaults come from config.py.
    """
    if result.get("error"):
        return

    if output_json_path is not None:
        save_result(
            output_json_path,
            result,
            append=True,
        )

    if metrics_json_path is not None:
        metrics_payload = {
            "question": result.get("question", ""),
            "route": result.get("route", "uploaded_pdf"),
            "intent_route": result.get("intent_route", ""),
            "provider": result.get("provider", ""),
            "model_name": result.get("model_name", ""),
            "uploaded_pdf_index": result.get("uploaded_pdf_index", {}),
            "routing_metrics": result.get("routing_metrics", {}),
            "generation_metrics": result.get("generation_metrics", {}),
            "overall_metrics": result.get("overall_metrics", {}),
        }

        save_metrics(
            metrics_payload,
            metrics_json_path,
        )

def emit_progress(
    progress_callback: Callable[[str], None] | None,
    message: str,
) -> None:
    """
    Send progress messages to Streamlit UI when a callback is provided.
    """
    if progress_callback is not None:
        progress_callback(message)

def run_uploaded_pdf_question(
    *,
    question: str,
    index: UploadedPdfIndex,
    provider: str,
    model_name: str,
    temperature: float = TEMPERATURE,
    top_k: int = 10,
    embed_model: str = EMBED_MODEL,
    memory: list[dict[str, Any]] | None = None,
    output_json_path: Path | None = UPLOAD_OUTPUT_JSON_PATH,
    metrics_json_path: Path | None = UPLOAD_METRICS_JSON_PATH,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Answer uploaded-PDF questions.

    Flow:
    - classify route using same router
    - if non_rag: answer normally
    - if rag: retrieve only from uploaded PDF vector store
    - for multi-PDF uploads, add paper-balanced seed chunks
    - use uploaded-PDF memory
    - build routing/generation/overall metrics
    - save result and metrics to uploaded-PDF paths by default
    """
    start_timer = Timer()
    question = question.strip()

    if not question:
        raise ValueError("Question is empty.")

    recent_memory = memory or []
    resolved_provider = normalize_provider_name(
        provider,
        model_name,
    )

    llm = build_llm(
        model_name=model_name,
        temperature=temperature,
        provider=provider,
    )

    emit_progress(
        progress_callback,
        "Analyzing the question and selecting the best answer route...",
    )

    route, routing_metrics = classify_route(
        question=question,
        llm=llm,
        provider=resolved_provider,
        model_name=model_name,
        memory=recent_memory,
    )

    if route == "non_rag":
        emit_progress(
            progress_callback,
            "No uploaded-PDF retrieval needed. Generating a direct answer...",
        )
    else:
        emit_progress(
            progress_callback,
            "Retrieving relevant evidence from the uploaded PDF index...",
        )

    if route == "non_rag":
        answer, generation_metrics = generate_non_rag_answer_with_metrics(
            question=question,
            memory=recent_memory,
            llm=llm,
            provider=resolved_provider,
            model_name=model_name,
        )

        overall_metrics = build_overall_from_stage_metrics(
            routing_metrics=routing_metrics,
            generation_metrics=generation_metrics,
            total_pipeline_time=start_timer.stop(),
        )

        result = {
            "question": question,
            "route": "non_rag",
            "intent_route": route,
            "answer": answer,
            "provider": resolved_provider,
            "model_name": model_name,
            "temperature": temperature,
            "top_k": top_k,
            "retrieved_chunks": [],
            "routing_metrics": routing_metrics,
            "generation_metrics": generation_metrics,
            "overall_metrics": overall_metrics,
            "uploaded_pdf_index": uploaded_pdf_index_payload(index),
        }

        save_uploaded_pdf_result_if_requested(
            result=result,
            output_json_path=output_json_path,
            metrics_json_path=metrics_json_path,
        )

        return result

    content_types = decide_content_types(question)

    retrieved_chunks = retrieve_uploaded_pdf_chunks(
        question=question,
        index=index,
        top_k=top_k,
        embed_model=embed_model,
        content_types=content_types,
    )

    if not retrieved_chunks:
        answer = "No relevant information was found in the uploaded PDF(s)."

        generation_metrics = build_empty_generation_metrics()

        overall_metrics = build_overall_from_stage_metrics(
            routing_metrics=routing_metrics,
            generation_metrics=generation_metrics,
            total_pipeline_time=start_timer.stop(),
        )

        result = {
            "question": question,
            "route": "uploaded_pdf",
            "intent_route": route,
            "answer": answer,
            "provider": resolved_provider,
            "model_name": model_name,
            "temperature": temperature,
            "top_k": top_k,
            "retrieved_chunks": [],
            "routing_metrics": routing_metrics,
            "generation_metrics": generation_metrics,
            "overall_metrics": overall_metrics,
            "uploaded_pdf_index": uploaded_pdf_index_payload(index),
        }

        save_uploaded_pdf_result_if_requested(
            result=result,
            output_json_path=output_json_path,
            metrics_json_path=metrics_json_path,
        )

        return result

    context = "\n\n".join(
        format_chunk_for_context(chunk)
        for chunk in retrieved_chunks
    )

    emit_progress(
        progress_callback,
        "Generating a grounded answer from the retrieved uploaded-PDF context...",
    )

    answer, generation_metrics = generate_rag_answer_with_metrics(
        question=question,
        context=context,
        memory=recent_memory,
        llm=llm,
        provider=resolved_provider,
        model_name=model_name,
    )

    overall_metrics = build_overall_from_stage_metrics(
        routing_metrics=routing_metrics,
        generation_metrics=generation_metrics,
        total_pipeline_time=start_timer.stop(),
    )

    result = {
        "question": question,
        "route": "uploaded_pdf",
        "intent_route": route,
        "answer": answer,
        "provider": resolved_provider,
        "model_name": model_name,
        "temperature": temperature,
        "top_k": top_k,
        "vector_store_path": str(index.vector_store_path),
        "retrieved_chunks": [
            serialize_retrieved_chunk(chunk)
            for chunk in retrieved_chunks
        ],
        "routing_metrics": routing_metrics,
        "generation_metrics": generation_metrics,
        "overall_metrics": overall_metrics,
        "uploaded_pdf_index": uploaded_pdf_index_payload(index),
    }

    save_uploaded_pdf_result_if_requested(
        result=result,
        output_json_path=output_json_path,
        metrics_json_path=metrics_json_path,
    )

    return result
"""
retriever.py

Retrieve relevant typed chunks from ChromaDB.

Supports:
- normal text retrieval
- content type filtering
- exact Table/Figure reranking

No prompts.
No LLM calls.
No answer generation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from core.constants import ContentType
from config import COLLECTION_NAME

_embedder_cache: Dict[str, SentenceTransformer] = {}


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Cache embedding model so repeated retrieval calls are faster.
    """
    if model_name not in _embedder_cache:
        print(f"[retriever] Loading embedding model: {model_name}", flush=True)
        _embedder_cache[model_name] = SentenceTransformer(model_name)

    return _embedder_cache[model_name]


def build_where_filter(content_types: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    """
    Build Chroma metadata filter for content_type.
    """
    if not content_types:
        return None

    cleaned = [
        str(content_type).strip()
        for content_type in content_types
        if str(content_type).strip()
    ]

    if not cleaned:
        return None

    if len(cleaned) == 1:
        return {"content_type": cleaned[0]}

    return {"content_type": {"$in": cleaned}}


def rows_from_query_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert Chroma query output into clean retrieved chunk dictionaries.
    """
    chunks: List[Dict[str, Any]] = []

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    distances = results.get("distances") or []

    if not documents or not documents[0]:
        return chunks

    for document, metadata, distance in zip(
        documents[0],
        metadatas[0],
        distances[0],
    ):
        metadata = metadata or {}
        content_type = metadata.get("content_type", ContentType.TEXT.value)

        chunks.append(
            {
                "content": document,
                "metadata": metadata,
                "content_type": content_type,
                "similarity": round(1 - float(distance), 4),
            }
        )

    return chunks


def extract_artifact_refs(question: str) -> dict[str, set[str]]:
    """
    Extract exact references like:
    - Table 5
    - Tab. 2
    - Figure 4
    - Fig. 3
    """
    question = question or ""

    table_refs = {
        match.group(1).lower()
        for match in re.finditer(
            r"\b(?:table|tab\.?)\s*([0-9]+[a-z]?)\b",
            question,
            flags=re.IGNORECASE,
        )
    }

    figure_refs = {
        match.group(1).lower()
        for match in re.finditer(
            r"\b(?:fig\.?|figure)\s*([0-9]+[a-z]?)\b",
            question,
            flags=re.IGNORECASE,
        )
    }

    return {
        "table": table_refs,
        "figure": figure_refs,
    }


def caption_has_ref(caption: str, label: str, number: str) -> bool:
    """
    Check whether caption contains Table/Figure number.
    """
    caption = caption or ""

    if label == "table":
        label_pattern = r"(?:table|tab\.?)"
    else:
        label_pattern = r"(?:fig\.?|figure)"

    pattern = rf"\b{label_pattern}\s*{re.escape(number)}\b"

    return bool(re.search(pattern, caption, flags=re.IGNORECASE))


def has_exact_artifact_ref(question: str) -> bool:
    refs = extract_artifact_refs(question)
    return bool(refs["table"] or refs["figure"])


def rerank_exact_artifact_refs(
    question: str,
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    If user asks "Table 5", boost the chunk whose caption contains Table 5.
    If user asks "Figure 4", boost figure/diagram chunks whose caption contains Figure 4.
    """
    refs = extract_artifact_refs(question)

    if not refs["table"] and not refs["figure"]:
        return chunks

    boosted: list[tuple[float, float, int, Dict[str, Any]]] = []

    for index, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {}) or {}
        content_type = chunk.get("content_type") or metadata.get("content_type", "")
        caption = metadata.get("caption", "")

        boost = 0.0

        if content_type == ContentType.TABLE.value:
            for number in refs["table"]:
                if caption_has_ref(caption, "table", number):
                    boost += 1.0

        if content_type in {
            ContentType.FIGURE.value,
            ContentType.IMAGE.value,
            ContentType.DIAGRAM.value,
            ContentType.FLOWCHART.value,
        }:
            for number in refs["figure"]:
                if caption_has_ref(caption, "figure", number):
                    boost += 1.0

        similarity = float(chunk.get("similarity", 0.0) or 0.0)

        boosted.append((boost, similarity, -index, chunk))

    boosted.sort(
        key=lambda item: (item[0], item[1], item[2]),
        reverse=True,
    )

    return [item[3] for item in boosted]


def retrieve(
    question: str,
    top_k: int,
    vector_store_folder_path: str,
    embed_model: str = "all-MiniLM-L6-v2",
    collection_name: str = COLLECTION_NAME,
    content_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from ChromaDB.

    Args:
        question:
            User question.

        top_k:
            Number of final chunks to return.

        vector_store_folder_path:
            Path to Chroma vector store folder.

        embed_model:
            SentenceTransformer model used during embedding.

        collection_name:
            Chroma collection name.

        content_types:
            Optional content-type filter.

            Examples:
            None
            ["text"]
            ["table", "text"]
            ["figure", "diagram", "text"]
    """
    embedder = get_embedder(embed_model)
    client = chromadb.PersistentClient(path=vector_store_folder_path)

    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"[retriever] Collection '{collection_name}' not found in "
            f"'{vector_store_folder_path}'. Run embedding.py first."
        ) from exc

    doc_count = collection.count()

    if doc_count == 0:
        raise RuntimeError(
            f"[retriever] Collection '{collection_name}' exists but contains no documents. "
            "Run embedding.py again."
        )

    print(f"[retriever] Vector store loaded. Docs in DB: {doc_count}", flush=True)

    safe_top_k = min(top_k, doc_count)

    if safe_top_k < top_k:
        print(
            f"[retriever] Warning: top_k={top_k} exceeds collection size "
            f"({doc_count}). Using top_k={safe_top_k}.",
            flush=True,
        )

    candidate_k = safe_top_k

    # For exact references like "Table 5", retrieve more candidates first,
    # then rerank by caption match.
    if content_types and has_exact_artifact_ref(question):
        candidate_k = min(max(safe_top_k * 5, 40), doc_count)

    query_embedding = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    where_filter = build_where_filter(content_types)

    try:
        if where_filter:
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=candidate_k,
                where=where_filter,
            )
        else:
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=candidate_k,
            )

        chunks = rows_from_query_results(results)

    except Exception as exc:
        # Some Chroma versions differ in metadata filter behavior.
        # Fallback: broad retrieval, then Python-side filtering.
        if not content_types:
            raise

        print(
            f"[retriever] Metadata filter failed. Falling back to Python filter: {exc}",
            flush=True,
        )

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=candidate_k,
        )

        allowed = set(content_types)
        chunks = [
            chunk
            for chunk in rows_from_query_results(results)
            if chunk.get("content_type") in allowed
        ]

    chunks = rerank_exact_artifact_refs(question, chunks)
    chunks = chunks[:safe_top_k]

    print(
        f"[retriever] Retrieved {len(chunks)} chunk(s)."
        + (f" content_types={content_types}" if content_types else ""),
        flush=True,
    )

    return chunks
"""
embedding.py

Generates embeddings from chunks.json and stores them in ChromaDB.

Supports typed multimodal chunks by embedding:
- chunk["embedding_text"] if present
- otherwise chunk["text"]

Run:
python src/embedding.py --chunk_file data/chunks/chunks.json --vector_folder data/vector_store --reset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import numpy as np
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer
from config import COLLECTION_NAME

DISTANCE_METRIC = "cosine"

_META_MODEL = "embedding_model"
_META_METRIC = "distance_metric"


class EmbedConfig(BaseModel):
    chunk_file: Path
    vector_folder: Path
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = Field(default=64, gt=0)
    reset: bool = False

    @field_validator("chunk_file")
    @classmethod
    def file_must_exist(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Chunk file not found: {value}")
        return value


def parse_args() -> EmbedConfig:
    parser = argparse.ArgumentParser(
        description="Generate embeddings and store in ChromaDB",
    )

    parser.add_argument("--chunk_file", type=str, required=True)
    parser.add_argument("--vector_folder", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the Chroma collection from scratch.",
    )

    args = parser.parse_args()

    return EmbedConfig(
        chunk_file=Path(args.chunk_file),
        vector_folder=Path(args.vector_folder),
        model_name=args.model_name,
        batch_size=args.batch_size,
        reset=args.reset,
    )


def embedding_text_for_chunk(chunk: Dict[str, Any]) -> str:
    """
    Prefer explicit embedding_text, then fallback to text.
    """
    return str(chunk.get("embedding_text") or chunk.get("text") or "").strip()


def stable_chunk_id(chunk: Dict[str, Any], index: int) -> str:
    """
    Stable ID based on chunk identity and content.
    """
    metadata = chunk.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    identity = (
        f"{chunk.get('content_type', 'text')}::"
        f"{chunk.get('paper_name', '')}::"
        f"{chunk.get('section', '')}::"
        f"{chunk.get('page_start', '')}::"
        f"{chunk.get('page_end', '')}::"
        f"{metadata.get('table_id', '')}::"
        f"{metadata.get('figure_id', '')}::"
        f"{embedding_text_for_chunk(chunk)}::"
        f"{index}"
    )

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def scalar_metadata_value(value: Any) -> str:
    """
    Chroma metadata supports scalar values.
    Stringify lists/dicts safely.
    """
    if value is None:
        return ""

    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def metadata_for_chunk(chunk: Dict[str, Any]) -> Dict[str, str]:
    """
    Merge chunk metadata with top-level fields.
    """
    metadata: Dict[str, Any] = {}

    if isinstance(chunk.get("metadata"), dict):
        metadata.update(chunk["metadata"])

    for key in [
        "content_type",
        "paper_name",
        "author",
        "year",
        "section",
        "page_start",
        "page_end",
        "source_file",
    ]:
        metadata.setdefault(key, chunk.get(key, ""))

    metadata["distance_metric"] = DISTANCE_METRIC
    metadata["similarity_formula"] = "1 - cosine_distance"

    return {
        str(key): scalar_metadata_value(value)
        for key, value in metadata.items()
        if value is not None
    }


class EmbeddingManager:
    def __init__(self, model_name: str):
        self.model_name = model_name

        print(f"[embedding] Loading embedding model: {model_name}", flush=True)
        self.model = SentenceTransformer(model_name)

        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[embedding] Embedding dimension: {self.dimension}", flush=True)

    def generate(self, texts: List[str], batch_size: int) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


class VectorStore:
    """
    Small ChromaDB wrapper with collection metadata validation.
    """

    def __init__(self, persist_dir: Path, model_name: str, reset: bool):
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.max_batch_size = self.resolve_max_batch_size()

        if reset:
            self.drop_collection()

        self.collection = self.open_or_create_collection()
        self.validate_collection_metadata()

        print(f"[vector] Collection '{COLLECTION_NAME}' ready.", flush=True)
        print(f"[vector] Embedding model: {model_name}", flush=True)
        print(f"[vector] Distance metric: {DISTANCE_METRIC}", flush=True)
        print(f"[vector] Existing vectors: {self.collection.count()}", flush=True)

    def resolve_max_batch_size(self) -> int:
        getter = getattr(self.client, "get_max_batch_size", None)
        if callable(getter):
            try:
                value = int(getter())
                if value > 0:
                    return value
            except Exception:
                pass

        value = getattr(self.client, "max_batch_size", None)
        if isinstance(value, int) and value > 0:
            return value

        return 5461

    def drop_collection(self) -> None:
        try:
            self.client.delete_collection(COLLECTION_NAME)
            print(f"[vector] Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    def open_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                _META_MODEL: self.model_name,
                _META_METRIC: DISTANCE_METRIC,
                "hnsw:space": DISTANCE_METRIC,
            },
        )

    def validate_collection_metadata(self) -> None:
        stored = self.collection.metadata or {}

        stored_model = stored.get(_META_MODEL)
        stored_metric = stored.get(_META_METRIC)

        if stored_model and stored_model != self.model_name:
            raise ValueError(
                "Embedding model mismatch. "
                f"Stored={stored_model}, requested={self.model_name}. "
                "Run again with --reset."
            )

        if stored_metric and stored_metric != DISTANCE_METRIC:
            raise ValueError(
                "Distance metric mismatch. "
                f"Stored={stored_metric}, expected={DISTANCE_METRIC}. "
                "Run again with --reset."
            )

    def upsert(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> Dict[str, int]:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})."
            )

        submitted = 0

        for start in range(0, len(chunks), self.max_batch_size):
            end = min(start + self.max_batch_size, len(chunks))

            ids: List[str] = []
            documents: List[str] = []
            embedding_values: List[List[float]] = []
            metadatas: List[Dict[str, str]] = []

            for index in range(start, end):
                chunk = chunks[index]
                text = embedding_text_for_chunk(chunk)

                if not text:
                    continue

                ids.append(stable_chunk_id(chunk, index))
                documents.append(text)
                embedding_values.append(embeddings[index].tolist())
                metadatas.append(metadata_for_chunk(chunk))

            if not ids:
                continue

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embedding_values,
                metadatas=metadatas,
            )

            submitted += len(ids)

        return {
            "total_submitted": submitted,
            "collection_size": self.collection.count(),
        }


def run_embedding(config: EmbedConfig) -> None:
    start_time = time.time()

    print("[START] Embedding pipeline initializing...", flush=True)
    print(f"[LOAD] Reading chunks from {config.chunk_file}", flush=True)

    with config.chunk_file.open("r", encoding="utf-8") as file:
        raw_chunks: List[Dict[str, Any]] = json.load(file)

    chunks = [
        chunk
        for chunk in raw_chunks
        if embedding_text_for_chunk(chunk)
    ]

    skipped = len(raw_chunks) - len(chunks)

    print(f"[LOAD] Loaded chunks: {len(raw_chunks)}", flush=True)
    if skipped:
        print(f"[LOAD] Skipped empty chunks: {skipped}", flush=True)

    if not chunks:
        print("[DONE] Nothing to embed.", flush=True)
        return

    manager = EmbeddingManager(config.model_name)

    texts = [embedding_text_for_chunk(chunk) for chunk in chunks]

    print(f"[EMBED] Encoding {len(texts)} chunks...", flush=True)
    embeddings = manager.generate(texts, config.batch_size)

    print("[STORE] Opening vector store...", flush=True)
    store = VectorStore(
        persist_dir=config.vector_folder,
        model_name=config.model_name,
        reset=config.reset,
    )

    print("[STORE] Writing embeddings to ChromaDB...", flush=True)
    summary = store.upsert(chunks, embeddings)

    elapsed = round(time.time() - start_time, 2)

    print()
    print("─" * 50)
    print(f"Chunks submitted : {summary['total_submitted']}")
    print(f"Collection size  : {summary['collection_size']}")
    print(f"Elapsed          : {elapsed}s")
    print("─" * 50)
    print(f"Done. Vectors stored at: {config.vector_folder}")


if __name__ == "__main__":
    config = parse_args()
    print("\nCONFIG:", config, flush=True)
    run_embedding(config)
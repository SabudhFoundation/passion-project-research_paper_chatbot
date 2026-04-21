"""
embedding.py

Generates embeddings from a chunks JSON file and stores them in ChromaDB.
python src/embedding.py --chunk_file data/chunks/chunks.json --vector_folder data/vector_store --reset

"""

import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field, field_validator



# CONSTANTS


COLLECTION_NAME = "pdf_chunks"

# The only supported distance metric.
# ChromaDB key: hnsw:space   Interpretation: similarity = 1 - distance
DISTANCE_METRIC = "cosine"


# CONFIG

class EmbedConfig(BaseModel):
    chunk_file: Path
    vector_folder: Path
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = Field(default=64, gt=0)
    reset: bool = False

    @field_validator("chunk_file")
    def file_must_exist(cls, v):
        if not v.exists():
            raise ValueError(f"Chunk file not found: {v}")
        return v


# ARG PARSER


def parse_args() -> EmbedConfig:
    parser = argparse.ArgumentParser(
        description="Generate embeddings and store in ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First run or incremental update (safe to repeat):
  python embedding.py --chunk_file chunks.json --vector_folder ./db

  # Switch to a new embedding model (wipes and rebuilds the collection):
  python embedding.py --chunk_file chunks.json --vector_folder ./db \\
      --model_name all-mpnet-base-v2 --reset
        """,
    )
    parser.add_argument("--chunk_file",    type=str, required=True)
    parser.add_argument("--vector_folder", type=str, required=True)
    parser.add_argument("--model_name",    type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the collection from scratch. "
             "Required when changing --model_name.",
    )
    args = parser.parse_args()
    return EmbedConfig(
        chunk_file=Path(args.chunk_file),
        vector_folder=Path(args.vector_folder),
        model_name=args.model_name,
        batch_size=args.batch_size,
        reset=args.reset,
    )

# STABLE CHUNK ID 


def stable_chunk_id(chunk: Dict[str, Any], idx: int) -> str:
    identity = (
        f"{chunk.get('paper_name','')}::"
        f"{chunk.get('section','')}::"
        f"{chunk.get('page_start','')}::"
        f"{chunk.get('text','')}::"
        f"{idx}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


# EMBEDDING MANAGER

class EmbeddingManager:
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension : {self.dim}")

    def generate(self, texts: List[str], batch_size: int) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # required for cosine via dot-product
        )


# VECTOR STORE  


# Metadata keys written into the ChromaDB collection
_META_MODEL  = "embedding_model"
_META_METRIC = "distance_metric"


class VectorStore:
    """
    Wraps a ChromaDB persistent collection with:
      - Enforced embedding model consistency  (Issue 6)
      - Enforced distance metric consistency  (Issue 7)
      - Content-addressed upsert (no duplicates) (Issue 5)

    Similarity score interpretation
    ────────────────────────────────
    ChromaDB returns distances, not similarities.
    With distance_metric = "cosine":
        cosine_distance = 1 - cosine_similarity
        cosine_similarity = 1 - cosine_distance
        Range: distance ∈ [0, 2], similarity ∈ [-1, 1]
        Higher similarity → more relevant.
    Always use: similarity = 1 - result["distances"][0][i]
    """

    def __init__(self, persist_dir: Path, model_name: str, reset: bool):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self.client = chromadb.PersistentClient(path=str(persist_dir))

        if reset:
            self._drop_collection()

        self.collection = self._open_or_create_collection()
        self._validate_collection_metadata()

        print(f"Collection '{COLLECTION_NAME}' ready.")
        print(f"  Embedding model : {model_name}")
        print(f"  Distance metric : {DISTANCE_METRIC}  (similarity = 1 - distance)")
        print(f"  Existing vectors: {self.collection.count()}")

    # ── private helpers ────────────────────────────────────────────────

    def _drop_collection(self) -> None:
        try:
            self.client.delete_collection(COLLECTION_NAME)
            print(f"Existing collection '{COLLECTION_NAME}' deleted (--reset).")
        except Exception:
            pass  # collection did not exist yet

    def _open_or_create_collection(self):
        """
        Always open with get_or_create_collection.

        IMPORTANT: the metadata dict is only written on CREATION.
        On subsequent opens ChromaDB silently ignores the metadata
        argument — that is why we validate separately in
        _validate_collection_metadata().
        """
        return self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                _META_MODEL:  self.model_name,
                _META_METRIC: DISTANCE_METRIC,
                "hnsw:space": DISTANCE_METRIC,   # ChromaDB internal key
            },
        )

    def _validate_collection_metadata(self) -> None:
        """
        Read the metadata that is actually stored on disk and compare
        against what we expect. Raises ValueError on any mismatch so the
        user knows to pass --reset rather than silently poisoning the DB.

        This is the correct fix for Issues 6 and 7: we cannot trust the
        metadata argument to get_or_create_collection on re-opens, so we
        always validate after opening.
        """
        stored: Dict[str, Any] = self.collection.metadata or {}

        stored_model  = stored.get(_META_MODEL)
        stored_metric = stored.get(_META_METRIC)

        # Validate model
        if stored_model and stored_model != self.model_name:
            raise ValueError(
                f"\n[Issue 6] Embedding model mismatch — cannot safely add vectors.\n"
                f"  Stored in collection : {stored_model}\n"
                f"  Requested now        : {self.model_name}\n"
                f"  → Re-run with --reset to rebuild the collection with the new model."
            )

        # Validate metric
        if stored_metric and stored_metric != DISTANCE_METRIC:
            raise ValueError(
                f"\n[Issue 7] Distance metric mismatch — similarity scores would be wrong.\n"
                f"  Stored in collection : {stored_metric}\n"
                f"  Expected             : {DISTANCE_METRIC}\n"
                f"  → Re-run with --reset to rebuild the collection."
            )

    # ── public API ─────────────────────────────────────────────────────

    def upsert(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> Dict[str, int]:
        """
        Insert or update chunks using content-addressed IDs.

        Returns a summary dict with keys: total_submitted, upserted.
        ChromaDB upsert semantics:
          - New ID   → inserted as a new vector.
          - Known ID → the document, embedding, and metadata are updated.
          - Unchanged text → same ID → update is a no-op at the storage level.
        """
        ids:  List[str]        = []
        docs: List[str]        = []
        embs: List[List[float]] = []
        metas: List[Dict]      = []

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            ids.append(stable_chunk_id(chunk, i))
            docs.append(chunk["text"])
            embs.append(emb.tolist())
            metas.append({
                "paper_name":  str(chunk.get("paper_name",  "") or ""),
                "author":      str(chunk.get("author",      "") or ""),
                "year":        str(chunk.get("year",        "") or ""),
                "section":     str(chunk.get("section",     "") or ""),
                "page_start":  str(chunk.get("page_start",  "") or ""),
                "page_end":    str(chunk.get("page_end",    "") or ""),
                "source_file": str(chunk.get("source_file", "") or ""),
                # Store similarity interpretation alongside data
                "distance_metric": DISTANCE_METRIC,
                "similarity_formula": "1 - cosine_distance",
            })

        self.collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=embs,
            metadatas=metas,
        )

        after_count = self.collection.count()
        return {
            "total_submitted": len(ids),
            "collection_size": after_count,
        }



# MAIN


def run_embedding(config: EmbedConfig) -> None:
    start = time.time()

    # ── Load chunks ───────────────────────────────────────────────────
    with open(config.chunk_file, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {config.chunk_file}")

    if not chunks:
        print("Nothing to embed. Exiting.")
        return

    # ── Generate embeddings ───────────────────────────────────────────
    manager = EmbeddingManager(config.model_name)
    texts = [c["text"] for c in chunks]
    embeddings = manager.generate(texts, config.batch_size)

    # ── Store in ChromaDB ─────────────────────────────────────────────
    store = VectorStore(
        persist_dir=config.vector_folder,
        model_name=config.model_name,
        reset=config.reset,
    )

    summary = store.upsert(chunks, embeddings)

    # ── Report ────────────────────────────────────────────────────────
    elapsed = round(time.time() - start, 2)
    print(f"\n{'─'*50}")
    print(f"  Chunks submitted : {summary['total_submitted']}")
    print(f"  Collection size  : {summary['collection_size']}")
    print(f"  Elapsed          : {elapsed}s")
    print(f"{'─'*50}")
    print(f" Done. Vectors stored at: {config.vector_folder}")



# ENTRY


if __name__ == "__main__":
    cfg = parse_args()
    print("\nCONFIG:", cfg)
    run_embedding(cfg)
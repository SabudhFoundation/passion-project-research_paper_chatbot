"""
retriever.py
Goal: Retrieve relevant chunks from ChromaDB.
      No prompts, no LLM calls, no answer generation here.
"""

from typing import Any, Dict, List

import chromadb
from sentence_transformers import SentenceTransformer


_embedder_cache: Dict[str, SentenceTransformer] = {}


def _get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    if model_name not in _embedder_cache:
        print(f"[retriever] Loading embedding model: {model_name}")
        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]


class Retriever:
    def __init__(self, vector_folder: str, embed_model: str = "all-MiniLM-L6-v2"):
        self.vector_folder = vector_folder
        self.embed_model = embed_model

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return retrieve(
            question=question,
            top_k=top_k,
            vector_store_folder_path=self.vector_folder,
            embed_model=self.embed_model
        )


def retrieve(
    question: str,
    top_k: int,
    vector_store_folder_path: str,
    embed_model: str = "all-MiniLM-L6-v2",
    collection_name: str = "pdf_chunks",
) -> List[Dict[str, Any]]:
    """
    Embed the question and retrieve the top_k most relevant chunks
    from the ChromaDB vector store.

    Args:
        question:                 The user query.
        top_k:                    Maximum number of chunks to retrieve.
        vector_store_folder_path: Path to the persisted ChromaDB folder.
        embed_model:              Sentence-transformer model name.
        collection_name:          ChromaDB collection to query.

    Returns:
        List of dicts, each containing:
            - content    (str)   : The chunk text.
            - metadata   (dict)  : Paper name, section, author, year, etc.
            - similarity (float) : 1 - cosine distance (higher = more relevant).
    """
    embedder = _get_embedder(embed_model)

    client = chromadb.PersistentClient(path=vector_store_folder_path)

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        raise RuntimeError(
            f"[retriever] Collection '{collection_name}' not found in "
            f"'{vector_store_folder_path}'. "
            f"Run embedding.py first to build the vector store."
        )

    doc_count = collection.count()
    if doc_count == 0:
        raise RuntimeError(
            f"[retriever] Collection '{collection_name}' exists but contains no "
            f"documents. Re-run embedding.py to populate it."
        )

    print(f"[retriever] Vector store loaded. Docs in DB: {doc_count}")

    safe_top_k = min(top_k, doc_count)
    if safe_top_k < top_k:
        print(
            f"[retriever] Warning: top_k={top_k} exceeds collection size "
            f"({doc_count}). Clamping to {safe_top_k}."
        )

    query_emb = embedder.encode([question], convert_to_numpy=True)[0]

    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=safe_top_k,
    )

    chunks: List[Dict[str, Any]] = []

    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "content":    doc,
                "metadata":   meta,
                "similarity": round(1 - dist, 4),
            })

    print(f"[retriever] Retrieved {len(chunks)} chunk(s).")
    return chunks